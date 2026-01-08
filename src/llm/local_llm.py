import torch
from typing import Union, List, Tuple, Dict, TYPE_CHECKING
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
from peft import PeftModel, PeftConfig
from concurrent.futures import ThreadPoolExecutor
from trl.models.utils import unwrap_model_for_generation
from openai import OpenAI
import json
from typing import Union, List, Optional, TYPE_CHECKING, Callable
import copy


class llm_transformer_based:
    """
    
    """
    def __init__(self, 
                 local_dir='./qwen2.5_1.5B/', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model: "AutoModelForCausalLM" = None, 
                 tokenizer: "AutoTokenizer" = None, 
                 without_model: bool = False, 
                 batch_size: int = 4, 
                 with_value_head: bool = True, 
                 **kwargs, 
                 ):
        self.has_value_head = with_value_head
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

        if not without_model:
            if not model:
                self.model = AutoModelForCausalLM.from_pretrained(local_dir)
                self.model.to(device)

            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = local_dir, 
                                                            padding_side='left')
            else:
                self.tokenizer = tokenizer
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 存储系统提示词：每个线程的系统提示
        self.system_prompts = []
        self.complement_prompt = []

    def init_llm(self, 
                 system_prompt: Union[str, List[str]]):
        """
        初始化系统提示词，保存为str
        """
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]

        # 清空之前的记录
        self.system_prompts.clear()

        # 编码系统提示为张量
        for prompt in system_prompt:
            self.system_prompts.append(prompt)  # 保留字符串

    def init_llm_complement_prompt(self, 
                                complement_prompt: Union[str, List[str]]):
        """
        初始化补足提示词，用于在用户输入后追加输入
        """
        if isinstance(complement_prompt, str):
            complement_prompt = [complement_prompt]

        # 清空之前的记录
        self.complement_prompt.clear()

        # 保存补足提示为字符串列表
        for prompt in complement_prompt:
            self.complement_prompt.append(prompt)

    def _normalize_input(
        self,
        input_data: Union[str, Dict, List], 
        tokenizer: "AutoTokenizer" = None, 
    ) -> List[str]:
        """
        预处理输入数据，将所有输入对齐为字符串列表。
        支持字符串、张量字典、字符串列表、张量列表。
        如果发现张量传入，则使用 tokenizer 解码为字符串。
        """
        # 使用传入的 tokenizer 或者默认的 tokenizer
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        # 检查 tokenizer 是否存在
        if tokenizer is None and isinstance(input_data[0], torch.Tensor):
            raise ValueError("大模型正则化传入数据错误: 需要传入一个Tokenizer进行解码")

        # 处理字符串输入
        if isinstance(input_data, str):
            return [input_data]
        
        # 处理列表输入
        elif isinstance(input_data, list):
            normalized = []
            for item in input_data:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    if "input_ids" in item:
                        # 解码张量为字符串
                        decoded = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
                        normalized.append(decoded)
                    else:
                        raise KeyError("Input dictionary must contain 'input_ids'.")
                elif isinstance(item, torch.Tensor):
                    # 假设是 input_ids 张量
                    decoded = tokenizer.decode(item, skip_special_tokens=True)
                    normalized.append(decoded)
                else:
                    raise ValueError(f"Invalid item type in list: {type(item)}")
            return normalized
        
        # 处理张量字典
        elif isinstance(input_data, dict) and "input_ids" in input_data:
            decoded = tokenizer.decode(input_data["input_ids"], skip_special_tokens=True)
            return [decoded]
        
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _split_into_batches(self, tensors: List[Dict[str, torch.Tensor]]) -> List[List[Dict[str, torch.Tensor]]]:
        """
        将输入张量列表分割为多个 batch，每个 batch 不超过 self.batch_size。
        """
        batched = []
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            batched.append(batch)
        return batched

    def __generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 1000, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        include_system: bool = True,  # 新增参数：是否包含系统提示
        include_complement: bool =True, 
        return_input_prompts: bool = True, 
        model: "AutoModelForCausalLMWithValueHead" = None, 
        generate_func = None, 
        tokenizer: "AutoTokenizer" = None, 
        **kwargs
    ) -> Union[str, List[str]]:
        """
        生成文本，支持单个字符串或字符串列表输入。
        参数：
            include_system: 是否在生成时包含系统提示（默认 True）
        """

        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        is_batch = isinstance(prompt, list)
        input_batch = prompt if is_batch else [prompt]
        batch_size = len(input_batch)

        if include_system:
            if len(self.system_prompts) < batch_size:
                if not self.system_prompts:
                    raise ValueError("No system prompts initialized. Call init_llm first.")
                delta = batch_size - len(self.system_prompts)
                last_prompt = copy.deepcopy(self.system_prompts[-1])
                self.system_prompts.extend([last_prompt for i in range(delta)])

        if include_complement: 
            if len(self.complement_prompt) < batch_size:
                delta = batch_size - len(self.complement_prompt)
                if len(self.complement_prompt) == 0:
                    self.complement_prompt.extend([""] * delta)
                else:
                    self.complement_prompt.extend([self.complement_prompt[-1]] * delta)

        processed_inputs = []
        for i in range(batch_size):
            system_prompt = self.system_prompts[i] if include_system else ""
            user_input = input_batch[i]
            complement_prompt = self.complement_prompt[i] if include_complement else ""
            full_input = f"{system_prompt}\n{user_input}\n{complement_prompt}" if system_prompt else user_input
            processed_inputs.append(full_input)

        # 📥 Tokenize 处理
        inputs = tokenizer(
            processed_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)
        
        # 🚀 模型生成
        with torch.no_grad():
            if generate_func is not None:
                outputs = generate_func(
                    input_ids = input_ids,
                    attention_mask=attention_mask)
            else:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p = top_p, 
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, 
                    **kwargs
                ) # 需要将结果移到内存中，也即CPU上，不能指向GPU内存，不够用

        # 📤 解码结果
        results = []
        for i in range(len(outputs)):
            input_length = input_ids[i].shape[0]
            generated_tokens = outputs[i][input_length:]
            decoded = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip() # + END_TAG # 这个END_TAG是为了后续训练时，也能正常进行解码么？
            results.append(decoded)
            # 释放 GPU 上的张量
        
        returns = [results] if is_batch else [results[0]]
        if return_input_prompts:
            returns.append(processed_inputs)

        # 强制清理缓存
        del input_ids, attention_mask
        torch.cuda.empty_cache()  # 强制清理缓存

        return *returns, 

    def generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 600, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        include_system: bool = True,
        include_complement: bool = True, 
        return_time: bool = False, 
        return_input_prompts: bool = False, 
        model: "AutoModelForCausalLMWithValueHead" = None, 
        generate_func = None, 
        tokenizer: "AutoTokenizer" = None, 
        **kwargs
    ) -> Tuple[Union[str, List[str]], List[float]]:
        """
        用于处理大批次输入的生成函数，通过分批次调用 generate 来避免显存溢出。

        ##Time功能可以正常使用##
        """

        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        # 返回时间列表
        if return_time:
            time_lis = []

        # 处理输入为统一的列表形式
        is_original_batch = isinstance(prompt, list)
        input_list = prompt if is_original_batch else [prompt]

        total_results = []
        input_prompt_lis = []
        # 按 batch_size 分块处理
        for i in range(0, len(input_list), self.batch_size):
            batch = input_list[i:i + self.batch_size]
            now = time.time()
            # 调用 generate 处理当前小 batch
            batch_result, input_prompts = self.__generate(
                prompt=batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p = top_p, 
                include_system=include_system,
                include_complement = include_complement, 
                return_input_prompts = return_input_prompts, 
                model = model, 
                generate_func = generate_func, 
                tokenizer = tokenizer, 
                **kwargs
            )
            input_prompt_lis.extend(input_prompts)
            after = time.time()
            time_delta = after - now
            if return_time:
                time_lis.extend([time_delta] * len(batch))
            
            # 添加结果到总结果中
            if isinstance(batch_result, list):
                total_results.extend(batch_result)
            else:
                total_results.append(batch_result)
        
        # 最终响应
        returns = [total_results] if is_original_batch else [total_results[0]]
        # 时间
        if return_time:
            returns.append(time_lis)
            
        # 输入模型的prompt
        if return_input_prompts:
            returns.append(input_prompt_lis)
        
        # 输出接口：total_results, time_lis, input_prompt_lis
        # 输出接口：List         , List    , List          
        return *returns, 

    def release(self):

        self.model.to("cpu")
        del self.model  # 删除模型实例
        # 清除引用
        self.model = None  
        torch.cuda.empty_cache()
        gc.collect()  # 触发垃圾回收
        return 
    
    def reload(self, config):
        """重加载模型"""
        self.batch_size = config.batch_size
        self.model = AutoModelForCausalLM.from_pretrained(config.model_dir)
        self.model.to(config.device)
        self.model.eval()

    def reload_lora(self, config):
        """重新加载模型，并装配lora权重矩阵""" 
        
        # decisionConfig的配置
        self.batch_size = config.batch_size
        base_model_name = config.model_dir  
        device = config.device

        # 加载基础模型（根据需求选择设备）
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            return_dict=True, 
            device_map=device
        )
        if isinstance(config.lora_dir, str):
            model = PeftModel.from_pretrained(model, 
                                            config.lora_dir, 
                                            device_map=device)
        elif isinstance(config.lora_dir, list):
            for dir in config.lora_dir:
                model = PeftModel.from_pretrained(model, 
                                                    dir, 
                                                    device_map=device)
            
                model = model.merge_and_unload()

        self.model = model
        # 现在 model 就是加载了 LoRA 权重的模型，可直接用于推理或评估
        self.model.eval()
        self.model.to(config.device)


    def save_model(self, dir):
        """如果发现存在适配器，则优先保存适配器"""

        # 检查是否使用了适配器（如 Peft 库的 LoRA 适配器）
        try:
            self.model = self.model.merge_and_unload()  # 合并适配器权重
        except:
            pass

        # 保存模型
        self.model.save_pretrained(dir)
        # 保存分词器
        self.tokenizer.save_pretrained(dir)
        return 
