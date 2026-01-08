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
# from .llamafactory.train.ppo.ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm # 使用绝对路径


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

END_TAG = "<END>"

class Large_Language_Model(torch.nn.Module):
    def __init__(self, 
                 local_dir='./qwen2.5_1.5B/', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model: "AutoModelForCausalLM" = None, 
                 tokenizer: torch.nn.Module = None, 
                 batch_size: int = 4, 
                 **kwargs, 
                 ):
        
        super(Large_Language_Model, self).__init__()
        self.batch_size = batch_size
        
        if not model:
            self.model = AutoModelForCausalLM.from_pretrained(local_dir)
            self.model.to(device)
        else:
            self.model = model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = local_dir, 
                                                           padding_side='left')
        else:
            self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

        # 存储系统提示词：每个线程的系统提示
        self.system_prompts = []
        self.complement_prompt = []

    def init_llm(self, system_prompt: Union[str, List[str]]):
        """
        初始化系统提示词。支持两种模式：
        - 单个字符串：初始化一个线程
        - 字符串列表：每个字符串初始化一个线程
        """
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]

        # 清空之前的线程记录
        self.system_prompts.clear()

        # 为每个系统提示词创建一个线程的系统提示
        self.system_prompts.extend(system_prompt)

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

    def __generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 1000, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        top_k: int = 0, 
        include_system: bool = True,  # 新增参数：是否包含系统提示
        include_complement: bool =True, 
        return_input_prompts: bool = True, 
        model: "AutoModelForCausalLMWithValueHead" = None, 
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
                last_prompt = self.system_prompts[-1]
                self.system_prompts.extend([last_prompt] * delta)

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
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                top_k = top_k, 
                temperature=temperature,
                top_p = top_p, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache = True, 
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
            ).strip() 
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
            results = self.__generate(
                prompt=batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p = top_p, 
                include_system=include_system,
                include_complement = include_complement, 
                return_input_prompts = return_input_prompts, 
                model = model, 
                tokenizer = tokenizer, 
                **kwargs
            )
            if return_input_prompts:
                batch_result, input_prompts = results
                input_prompt_lis.extend(input_prompts)
            else:
                batch_result, = results
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
    


class Large_Language_Model_API:
    def __init__(self, 
                 api_key: str = 'sk-NHZNQX83inCXYa8a2tL2yMnGST1ls64CFVdhBlira9sM8qBI',
                 base_url: str = 'https://api.openai-proxy.org/v1',
                 model: str = "gpt-4o-mini",
                 timeout: Optional[float] = None,
                 **kwargs, 
                 ):
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.timeout = timeout

        # 存储系统提示词：每个线程的系统提示
        self.system_prompts = []
        self.complement_prompt = []

    def init_llm(self, system_prompt: Union[str, List[str]]):
        """
        初始化系统提示词。支持两种模式：
        - 单个字符串：初始化一个线程
        - 字符串列表：每个字符串初始化一个线程
        """
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]

        # 清空之前的线程记录
        self.system_prompts.clear()

        # 为每个系统提示词创建一个线程的系统提示
        self.system_prompts.extend(system_prompt)

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

    def require(
        self, 
        messages: Union[List[dict], List[List[dict]]],  # 支持单个或多个 messages 列表
        max_tokens: int = 2000, 
        temperature: float = 1.0, 
        **kwargs
    ) -> Union[str, List[str]]:
        """
        批量通过 API 生成文本，输入为符合 API 要求的 messages 列表（支持单个或多个请求）。
        注意：Generate方法不包含任何历史信息的输入输出记录，只有chat才包含历史信息的输入输出
        """
        # 判断是否为批量输入
        is_batch = isinstance(messages[0], list) if messages else False
        if not is_batch:
            # 如果输入是单个 messages 列表，则包装为 [messages] 以统一处理
            messages = [messages]

        results = []
        for i, msg_list in enumerate(messages):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=msg_list,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                    **kwargs
                )
                result = response.choices[0].message.content.strip()
                results.append(result)
                 
            except Exception as e:
                results.append(f"Error in request {i}: {str(e)}")  # 单个请求失败不影响后续

        return results if is_batch else results[0]
    
    def generate(self, input_text: Union[str, List[str]], 
                 max_tokens: int = 2000, 
                 temperature: float = 1.0, 
                 return_time: bool = False,  
                 return_input_prompts: bool = False, 
                 include_system: bool = True,
                 **kwargs) -> Union[str, List[str], Tuple[List[str], List[float]]]:
        """
        基于系统提示的对话生成，支持批量输入。
        参数：
            include_system: 是否在生成时包含系统提示（默认 True）
        """
        is_batch = isinstance(input_text, list)
        input_batch = input_text if is_batch else [input_text]
        batch_size = len(input_batch)

        # 自动扩展 system_prompts 长度以匹配输入 batch 数量
        if len(self.system_prompts) < batch_size:
            if not self.system_prompts:
                raise ValueError("No system prompts initialized. Call init_llm first.")
            delta = batch_size - len(self.system_prompts)
            last_prompt = self.system_prompts[-1] 
            self.system_prompts.extend([last_prompt] * delta)

        if len(self.complement_prompt) < batch_size:
            delta = batch_size - len(self.complement_prompt)
            if len(self.complement_prompt) == 0:
                self.complement_prompt.extend([""] * delta)
            else:
                self.complement_prompt.extend([self.complement_prompt[-1]] * delta)

        # 构造 messages 列表
        messages_list = []
        for i in range(batch_size):
            messages = []
            if include_system:
                messages.append({"role": "system", "content": self.system_prompts[i]})
            messages.append({"role": "user", "content": input_batch[i] + self.complement_prompt[i]})
            messages_list.append(messages)

        # 并发执行所有请求
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_request, messages, max_tokens, temperature, kwargs)
                for messages in messages_list
            ]
            results_with_time = [future.result() for future in futures]

        # 返回结果
        # 不影响之前的接口
        if not (return_time or return_input_prompts):
            return [rt[0] for rt in results_with_time]

        returns = [[rt[0] for rt in results_with_time]]
        if return_time:
            returns.append([rt[1] for rt in results_with_time])
        if return_input_prompts:
            returns.append(input_text)
        
        return *returns, 


    def _process_request(self, messages, max_tokens, temperature, kwargs):
        """
        单个请求的处理逻辑，封装在子线程中执行。
        """
        before = time.time()
        try:
            response = self.require(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
        except Exception as e:
            after = time.time()
            return str(e), after - before  # 返回错误信息及耗时
        after = time.time()
        return response, after - before
    

class Large_Language_ModelV2:
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
