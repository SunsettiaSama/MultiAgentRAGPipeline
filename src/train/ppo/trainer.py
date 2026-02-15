# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Callable
import numpy as np
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from llamafactory.extras import logging
from llamafactory.extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from llamafactory.train.callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead
    from llamafactory.hparams import FinetuningArguments, GeneratingArguments, ModelArguments

    # 我们的部分
    from RAG_modules import AERR
    from config import MyTrainConfig

from reward import format_reward_func
from dataset_evaluation import Evaluator

logger = logging.get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        # reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        our_pipeline: "AERR" = None, 
        our_config: "MyTrainConfig" = None, 
        our_reward_func: "Callable" = None
    ) -> None: 
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed configW
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        # 这一步的dataset已经转化成功了，但是它的labels似乎还在
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        # self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # if finetuning_args.reward_model_type == "full":
        #     if self.is_deepspeed_enabled:
        #         if not (
        #             getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
        #             or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
        #         ):  # quantized models are already set on the correct device
        #             self.reward_model = self._prepare_deepspeed(self.reward_model)
        #     else:
        #         self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.our_pipeline = our_pipeline
        self.our_config = our_config
        self.our_reward_func = our_reward_func

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        # 居然还没有实现这个功能，那我只好手动实现一下了，虽然实现很简陋
        if resume_from_checkpoint is not None: 
            self.state.global_step = resume_from_checkpoint

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        max_tree_length = self.our_config.max_tree_length
        self.callback_handler.on_train_begin(self.args, self.state, self.control)
        # 硬编码一个tensorboard进去
        writer = SummaryWriter(self.our_config.tensorboard_log)
        evaluator = Evaluator()
        with tqdm(range(max_steps), disable=not self.is_local_process_zero()) as pbar:
            for step in range(max_steps):
                pbar.set_description("Get Data")
                try:
                    batch = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    batch = next(dataiter)

                # 获取完数据就赶紧润啊，避免迭代多了
                if resume_from_checkpoint is not None and step < resume_from_checkpoint:
                    pbar.update()
                    continue # 最前面已经对resume_ckpt进行了处理，不需要在这里进行额外处理

                pbar.set_description("Get Model Response")
                # Get inputs
                self.model.eval()
                self.processing_class.padding_side = "left"  # change padding side
                queries, responses, rewards = [], [], []

                mini_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                golden_answers = batch["labels"]
                golden_answers = [self.processing_class.decode([token for token in golden_answers[i] if token not in [-100, 151645, 198]], skep_special_tokens = True) # 这里的198, 151645纯粹是为了截取末尾的padding符号准备的，因为一般的指令调优都会做右侧符号的终止填充
                                    for i in range(self.config.batch_size)]
                
                #######################
                # 获取训练数据
                #######################
                # Decision Model的输入 | Decision Model的输出 | 最终输出 | 时间长度
                pbar.set_description("Get AERR Response")
                if self.our_config.sample_mode == "normal":
                    questions, mini_batch_queries, mini_batch_responses, pipeline_outputs, time_lis = self.get_inputs(mini_batch, max_tree_length = max_tree_length)
                    mini_batch_rewards = self.our_reward_func(questions = questions, golden_answers = golden_answers, predictions = pipeline_outputs, time_lis = time_lis)
                    # 分配reward，展平结构
                    for mini_batch_index in range(len(mini_batch_queries)):
                        for interaction_index in range(len(mini_batch_queries[mini_batch_index])):
                            # 去掉适中的部分，防止优势震荡
                            # 这方面我还是太不懂了，RL应该有充分的研究，我该去学习一下的
                            # if mini_batch_rewards[mini_batch_index] >= self.our_config.baseline_reward or mini_batch_rewards[mini_batch_index] == 0:
                            if mini_batch_queries[mini_batch_index][interaction_index] and mini_batch_responses[mini_batch_index][interaction_index]:
                                queries.append(torch.tensor(self.processing_class.encode(mini_batch_queries[mini_batch_index][interaction_index])))
                                responses.append(torch.tensor(self.processing_class.encode(mini_batch_responses[mini_batch_index][interaction_index])))
                                rewards.append(torch.tensor(
                                    format_reward_func(model_output = mini_batch_responses[mini_batch_index][interaction_index], 
                                                       reward = mini_batch_rewards[mini_batch_index])
                                    )) # 维度为0，不可展开，使用append
                
                #################
                # 梯度传播
                #################
                pbar.set_description("Compute and Train by safe step")
                self.processing_class.padding_side = "right"  # change padding side
                # Run PPO step
                self.model.train()
                # 大概是这一部分在计算的时候遇到了问题，爆显存了
                # 写了一个safe_step防爆机制，遇到爆显存了，每次随机削减20%的batch，然后再训练
                stats = self.safe_step(queries, responses, rewards, keep_ratio = 0.8) # stats中应该有我们想要的数据
                if stats == None:
                    pbar.update(1)
                    self.state.global_step += 1
                    self.callback_handler.on_step_end(self.args, self.state, self.control)
                    continue
                
                self.processing_class.padding_side = "left"  # restore padding side
                loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
                reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

                pbar.set_description("Visualize on tensorboard")
                # 可视化
                for key, value in stats.items():
                    writer.add_scalar(key, scalar_value = np.around(np.mean(value), 4), global_step = step)
                writer.add_scalar("reward/Reward Mean (Every 10 Step)", scalar_value = round(reward_meter.avg, 4), global_step = step)
                writer.add_scalar("reward/Reward Mean (Current Step)", scalar_value = round(stats["ppo/mean_scores"], 4), global_step = step)
                
                pbar.set_description("Update stats in global")
                if self.config.log_with is not None:
                    try:
                        batch["query"] = self.processing_class.batch_decode(queries, skip_special_tokens=True)
                        batch["response"] = self.processing_class.batch_decode(responses, skip_special_tokens=True)
                        self.log_stats(stats, batch, rewards)
                    except Exception:
                        logger.warning_rank0("Failed to save stats due to unknown errors.")

                self.state.global_step += 1
                pbar.update(1) # 这一步被跳过了？
                self.callback_handler.on_step_end(self.args, self.state, self.control)
                # 打印结果
                if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                    logs = dict(
                        loss=round(loss_meter.avg, 4),
                        reward=round(reward_meter.avg, 4), # 这玩意其实是10个轮次的平均Reward？？？？？有点意思
                        learning_rate=stats["ppo/learning_rate"],
                        epoch=round(step / steps_in_epoch, 2),
                    )
                    tqdm.write(str(logs))
                    logs["step"] = step
                    self.state.log_history.append(logs)
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)
                    loss_meter.reset()
                    reward_meter.reset()

                # 保存模型
                if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                    self.save_model(
                        os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}") # 字符串没有问题
                    )
                    self.callback_handler.on_save(self.args, self.state, self.control) # 这一步是用来将它转化成价值头的
                    
                # 验证结果：
                if (step + 1) % self.our_config.eval_interval == 0:
                    pbar.set_description("Eval Model")
                    evaluator.evaluate_pipelines(
                        ambig_qa_path = "/root/autodl-tmp/data/hotpotqa/light/hotpot_validation.parquet", 
                        pipeline = self.our_pipeline, 
                        results_save_dir = '/root/autodl-tmp/QA_Evaluation/AERR/',
                        batch_size = 16, 
                        pipeline_mode = "AERR", 
                        extract_context_from_template = False, 
                        ckpt = self.state.global_step, 
                        max_tree_length = self.our_config.max_tree_length, 
                        sample_mode = "normal", 
                        eval_mode = "sample", 
                        model = self.model, 
                        tokenizer = self.processing_class, 
                        activate_sampling = True, 
                        # 这里要放上AERR评估用的参数
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"], max_tree_length = None) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.processing_class.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]
        
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)
            
            # 所以实际上，这里只是一句话而已，而不是一个batch的输入
            # 因此才叫mini_batch？
            # 考虑到字符串操作的方便性，这里需要进行解码
            input_prompts = [self.processing_class.decode(batch["input_ids"][i], skip_special_tokens = True) for i in range(len(batch["input_ids"]))]

            from functools import partial
            generate_func = partial(unwrapped_model.generate,
                                 generation_config=self.generation_config,
                                 logits_processor=get_logits_processor())
            if self.our_config.sample_mode == "normal":
                pipeline_outputs, model_inputs, model_outputs, time_costs, questions = self.our_pipeline.generate_batch(
                                                                        input_prompts = input_prompts,
                                                                        extract_context_from_template = True, 
                                                                        return_interaction_state = True, 
                                                                        max_tree_length = self.our_config.max_tree_length if max_tree_length is None else max_tree_length, 
                                                                        model = unwrapped_model, 
                                                                        generate_func = generate_func, 
                                                                        tokenizer = self.processing_class, 
                                                                        sample_mode = self.our_config.sample_mode, # 如果是森林采样，那么必须传入训练配置
                                                                        our_config = self.our_config)
                if self.model_args.upcast_layernorm:
                    restore_layernorm(unwrapped_model, layernorm_params)

                queries: list["str"] = model_inputs
                responses: list["str"] = model_outputs
                return questions, queries, responses, pipeline_outputs, time_costs

            elif self.our_config.sample_mode == "forest":
                pipeline_outputs, forest, questions = self.our_pipeline.generate_batch(
                                                                        input_prompts = input_prompts,
                                                                        extract_context_from_template = True, 
                                                                        return_interaction_state = True, 
                                                                        max_tree_length = self.our_config.max_tree_length if max_tree_length is None else max_tree_length, 
                                                                        model = unwrapped_model, 
                                                                        generate_func = generate_func, 
                                                                        tokenizer = self.processing_class, 
                                                                        sample_mode = self.our_config.sample_mode, # 如果是森林采样，那么必须传入训练配置
                                                                        our_config = self.our_config)
                if self.model_args.upcast_layernorm:
                    restore_layernorm(unwrapped_model, layernorm_params)

                # queries: list["str"] = model_inputs
                # responses: list["str"] = model_outputs
                return questions, forest, pipeline_outputs
            
            # generate_output: torch.Tensor = unwrapped_model.generate(
            #     generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            # )


    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type
  
    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        # 原来是这一行出了问题，因为没办法捕获才导致的问题！！！！
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):

            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            # 空列表保护机制
            if any((logits.numel() == 0, values.numel() == 0, logprobs.numel() == 0, masks.numel() == 0)):
                continue

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        # 也就是这一部分没能正常保存模型对吧
        # LoRA适配器保存失败，也就是解析失败
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())

    def safe_step(self, queries, responses, rewards, keep_ratio = 0.8):
        """每次抛弃的batch数量为10%"""
        max_loop = 5
        curr_loop = 0
        while curr_loop <= max_loop:
            try:
                batch_size = len(queries)
                return self.step(queries, responses, rewards, batch_size = batch_size)
            except Exception as e:
                logger.info_rank0(f"[Warning] Step Faild: {str(e)}")

            queries, responses, rewards = reduce_lists(queries, responses, rewards, keep_ratio = keep_ratio)
            curr_loop += 1

        logger.info_rank0(f"[Warning] Max retry attempts ({max_loop}) exceeded. Returning None.")
        return None
    
# 爆显存时，随机抛弃特定比例的结果
import random
def reduce_lists(queries, responses, rewards, keep_ratio = 0.9):
    """如果每次都能正常削减的话，那么就不会报错"""
    assert len(queries) == len(responses) == len(rewards), "All lists must have the same length"
    n = len(queries) # 如果返回0，那么下面的抽样自然不成立 如果返回1，那么
    keep_n = max(1, int(n * keep_ratio))  # 至少保留一个元素

    # 哪怕是keep_n = 1，n = 1，那么也还是会正常返回一个
    # 除非说，keep_n = 0，但这种情况不可能
    indices = random.sample(range(n), keep_n)
    queries = [queries[i] for i in indices]
    responses = [responses[i] for i in indices]
    rewards = [rewards[i] for i in indices]
    return queries, responses, rewards



