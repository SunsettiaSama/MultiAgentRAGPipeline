# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/ppo.py
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

from typing import TYPE_CHECKING, Optional

from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import fix_valuehead_checkpoint
from llamafactory.train.trainer_utils import create_ref_model, create_reward_model
from .trainer import CustomPPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from config import AERRConfig, MyTrainConfig
from RAG_modules import AERR
from reward import RewardManager
from large_language_model import END_TAG

def run_ppo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    our_config: "MyTrainConfig", 
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 添加一个结束标记
    tokenizer.add_special_tokens({"eos_token": END_TAG})
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    # 本质上是AutoModelForCausalLMWithValueHead类
    # 试试看能不能用LLM进行初始化
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    # 这一部分是我们的部分，加到其中
    pipelineconfig = our_config.to_AERRConfig()
    pipelineconfig.decision.load_without_model = True # 不加载模型，使用外部模型
    pipeline = AERR(pipelineconfig)
    our_reward_func = RewardManager(alpha = our_config.alpha, time_avg = our_config.mean_time_baseline)

    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, model=model, **tokenizer_module)

    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    # reward_model = create_reward_model(model, model_args, finetuning_args)

    # Initialize our Trainer
    ppo_trainer: CustomPPOTrainer = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
#       reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        our_pipeline = pipeline, 
        our_config = our_config, 
        our_reward_func = our_reward_func, 
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        ppo_trainer.ppo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ppo_trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        ppo_trainer.save_state()  # must be called after save_model to have a folder
        if ppo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])
