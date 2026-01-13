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
from typing import Union, List, Optional, TYPE_CHECKING

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

    def __generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 1000, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        include_system: bool = True,  # 新增参数：是否包含系统提示
        include_complement: bool =True, 
        mode: str = "CasualLM", 
        **kwargs
    ) -> Union[str, List[str]]:
        """
        生成文本，支持单个字符串或字符串列表输入。
        参数：
            include_system: 是否在生成时包含系统提示（默认 True）
        """
        is_batch = isinstance(prompt, list)
        input_batch = prompt if is_batch else [prompt]
        batch_size = len(input_batch)

        # 🔁 自动扩展 system_prompts 长度以匹配输入 batch 数量
        if len(self.system_prompts) < batch_size:
            if not self.system_prompts:
                raise ValueError("No system prompts initialized. Call init_llm first.")
            delta = batch_size - len(self.system_prompts)
            last_prompt = self.system_prompts[-1]
            self.system_prompts.extend([last_prompt] * delta)


        # 自动扩展 complement_prompt 长度以匹配输入 batch 数量
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
        inputs = self.tokenizer(
            processed_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 🚀 模型生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p = top_p, 
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            ) # 需要将结果移到内存中，也即CPU上，不能指向GPU内存，不够用

        # 📤 解码结果
        results = []
        for i in range(len(outputs)):
            input_length = input_ids[i].shape[0]
            generated_tokens = outputs[i][input_length:]
            decoded = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip() 
            results.append(decoded)
            # 释放 GPU 上的张量
        
        # 强制清理缓存
        del input_ids, attention_mask
        torch.cuda.empty_cache()  # 强制清理缓存

        return results if is_batch else results[0]

    def encode(self, prompt: str, return_tensors: str = None, **kwargs) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            return_tensors=return_tensors,
            **kwargs
        )
        return inputs["input_ids"]

    def decode(self, tokens: torch.Tensor, **kwargs) -> str:
        decode_kwargs = {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": True
        }
        decode_kwargs.update(kwargs)
        return self.tokenizer.decode(tokens, **decode_kwargs)

    def generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 600, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        include_system: bool = True,
        return_time: bool = False, 
        return_type: type = str, 
        **kwargs
    ) -> Tuple[Union[str, List[str]], List[float]]:
        """
        用于处理大批次输入的生成函数，通过分批次调用 generate 来避免显存溢出。

        ##Time功能可以正常使用##
        """

        # 返回时间列表
        if return_time:
            time_lis = []

        # 处理输入为统一的列表形式
        is_original_batch = isinstance(prompt, list)
        input_list = prompt if is_original_batch else [prompt]

        total_results = []

        # 按 batch_size 分块处理
        for i in range(0, len(input_list), self.batch_size):
            batch = input_list[i:i + self.batch_size]
            now = time.time()
            # 调用 generate 处理当前小 batch
            batch_result = self.__generate(
                prompt=batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p = top_p, 
                include_system=include_system,
                **kwargs
            )
            after = time.time()
            time_delta = after - now
            if return_time:
                time_lis.extend([time_delta] * len(batch))
            
            # 添加结果到总结果中
            if isinstance(batch_result, list):
                total_results.extend(batch_result)
            else:
                total_results.append(batch_result)
        
        # 区分是否需要返回时间
        if return_time:
            return (total_results, time_lis) if is_original_batch else (total_results[0], time_lis)
        else:
            return total_results if is_original_batch else total_results[0]

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

    def require(
        self, 
        messages: Union[List[dict], List[List[dict]]],  # 支持单个或多个 messages 列表
        max_tokens: int = 1000, 
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
                 max_tokens: int = 1000, 
                 temperature: float = 1.0, 
                 return_time: bool = False,  
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

        # 构造 messages 列表
        messages_list = []
        for i in range(batch_size):
            messages = []
            if include_system:
                messages.append({"role": "system", "content": self.system_prompts[i]})
            messages.append({"role": "user", "content": input_batch[i]})
            messages_list.append(messages)

        # 并发执行所有请求
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_request, messages, max_tokens, temperature, kwargs)
                for messages in messages_list
            ]
            results_with_time = [future.result() for future in futures]

        # 提取响应和耗时
        results = [rt[0] for rt in results_with_time]
        time_lis = [rt[1] for rt in results_with_time]

        # 返回结果
        if not return_time:
            return results if is_batch else results[0]
        else:
            return (results, time_lis) if is_batch else (results[0], time_lis[0])

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
    

class CasualLMWithValueHead(torch.nn.Module):
    """
    正常运作，没有问题
    """
    def __init__(self, 
                 local_dir='./qwen2.5_1.5B/', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model: "AutoModelForCausalLM" = None, 
                 tokenizer: torch.nn.Module = None, 
                 without_model: bool = False, 
                 batch_size: int = 4, 
                 with_value_head: bool = True, 
                 **kwargs, 
                 ):
        
        super(CasualLMWithValueHead, self).__init__()
        self.has_value_head = with_value_head
        self.batch_size = batch_size
        self.device = device

        if not without_model:
            if not model:
                self.model = AutoModelForCausalLM.from_pretrained(local_dir)
                self.model.to(device)
            else:
                self.model = model
        # Tokenizer还是必须要初始化的
        # 和model配套的tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = local_dir, 
                                                        padding_side='left')
        else:
            self.tokenizer = tokenizer

            self.tokenizer.pad_token = self.tokenizer.eos_token


        # 存储系统提示词：每个线程的系统提示
        self.system_prompts = []
        self.system_tensors = []
        self.complement_prompt = []

    def init_llm(self, system_prompt: Union[str, List[str]]):
        """
        初始化系统提示词，保存为张量（input_ids, attention_mask）
        """
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]

        # 清空之前的记录
        self.system_prompts.clear()
        self.system_tensors.clear()

        # 编码系统提示为张量
        for prompt in system_prompt:
            # 编码为张量（padding=False 表示不填充，保留原始长度）
            encoded = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
            self.system_prompts.append(prompt)  # 保留字符串（可选）
            self.system_tensors.append(encoded)  # 保存张量

    def concatenate_system_tensor(
        self,
        input_data: Union[str, Dict[str, torch.Tensor], List[Union[str, Dict[str, torch.Tensor]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        拼接输入与系统张量。支持字符串、张量字典或张量字典列表。
        参数：
            input_data: 字符串、单个张量字典或张量字典列表。
        返回：
            拼接后的张量字典（input_ids, attention_mask）。
        """
        # 步骤 1: 将输入统一转换为张量字典列表
        input_list = self._normalize_input(input_data)

        # 步骤 2: 检查系统张量是否存在
        if not self.system_tensors:
            raise ValueError("System tensors are empty. Call init_llm() first.")

        # 步骤 3: 检查输入长度是否匹配系统张量长度
        system_length = len(self.system_tensors)
        input_length = len(input_list)

        if input_length != system_length:
            # 如果不匹配，复制输入元素以匹配系统张量长度
            input_list = self._repeat_input_to_match_system_length(input_list, system_length)

        # 步骤 4: 拼接每个输入与对应的系统张量
        combined_input_ids = []
        combined_attention_mask = []

        for i in range(system_length):
            system_tensor = self.system_tensors[i]
            user_tensor = input_list[i]

            # 拼接 input_ids 和 attention_mask
            input_ids = torch.cat([system_tensor["input_ids"], user_tensor["input_ids"]], dim=1)
            attention_mask = torch.cat([system_tensor["attention_mask"], user_tensor["attention_mask"]], dim=1)

            combined_input_ids.append(input_ids)
            combined_attention_mask.append(attention_mask)

        # 合并所有输入为 batch
        final_input_ids = torch.stack(combined_input_ids)
        final_attention_mask = torch.stack(combined_attention_mask)

        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask
        }

    def _normalize_input(
        self,
        input_data: Union[str, Dict[str, torch.Tensor], List[Union[str, Dict[str, torch.Tensor]]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        预处理输入数据：
            字符串的；
            没有在正确设备上的，都转移到正确的设备上；
            不是列表的字符串或者张量
        统统都转化成：
            正确设备上的张量列表
        """
        if isinstance(input_data, str):
            # 字符串 -> 单个张量字典
            encoded = self.tokenizer(input_data, return_tensors="pt", padding=False, truncation=True)
            # 确保张量在正确设备上
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            return [encoded]
        elif isinstance(input_data, dict):
            # 单个张量字典 -> 列表
            if "input_ids" in input_data and "attention_mask" in input_data:
                # 确保张量在正确设备上
                input_ids = input_data["input_ids"].to(self.device)
                attention_mask = input_data["attention_mask"].to(self.device)
                return [{"input_ids": input_ids, "attention_mask": attention_mask}]
            else:
                raise KeyError("Input dictionary must contain 'input_ids' and 'attention_mask' keys.")
        elif isinstance(input_data, list):
            # 张量字典列表 -> 验证每个元素
            normalized = []
            for item in input_data:
                if isinstance(item, str):
                    # 字符串 -> 张量字典
                    encoded = self.tokenizer(item, return_tensors="pt", padding=False, truncation=True)
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    normalized.append(encoded)
                elif isinstance(item, dict) and "input_ids" in item and "attention_mask" in item:
                    # 确保张量在正确设备上
                    item["input_ids"] = item["input_ids"].to(self.device)
                    item["attention_mask"] = item["attention_mask"].to(self.device)
                    normalized.append(item)
                else:
                    raise ValueError(f"Invalid item in input list: {item}")
            return normalized
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _repeat_input_to_match_system_length(
        self,
        input_list: List[Dict[str, torch.Tensor]],
        target_length: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        循环复制输入列表，直到其长度与目标长度一致。
        """
        input_len = len(input_list)
        if input_len == 0:
            raise ValueError("Input list is empty.")
        return [input_list[i % input_len] for i in range(target_length)]
    
    def _split_into_batches(self, tensors: List[Dict[str, torch.Tensor]]) -> List[List[Dict[str, torch.Tensor]]]:
        """
        将输入张量列表分割为多个 batch，每个 batch 不超过 self.batch_size。
        """
        batched = []
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            batched.append(batch)
        return batched

    def generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 600, 
        temperature: float = 1.0, 
        top_p: float = 0.9, 
        include_system: bool = True,
        return_time: bool = False, 
        return_type: type = torch.Tensor, 
        model: "AutoModelForCausalLMWithValueHead" = None, 
        tokenizer: "AutoTokenizer" = None, 
        **kwargs
    ) -> Tuple[Union[str, List[str]], List[float]]:
        """
        生成文本。支持字符串或字符串列表输入，也接受张量的直接传入
        参数：
            prompt: 输入字符串或字符串列表。
            max_tokens: 最大生成 token 数量。
            temperature: 温度参数。
            top_p: Top-p 抽样参数。
            include_system: 是否包含系统提示。
            return_time: 是否返回耗时。
            return_type: 返回类型（str 或 torch.Tensor）。
            **kwargs: 其他生成参数。
        返回：
            生成结果（str 或 tensor）和耗时列表（如果 return_time=True）。
        """

        # 输入归一化
        user_tensors = self._normalize_input(prompt)

        logits = None
        # 步骤 3: 如果包含系统提示，与系统张量拼接
        if include_system and self.system_tensors:
            user_tensors = self.concatenate_system_tensor(user_tensors) 

        # 步骤 4: 分割为多个 batch
        batches = self._split_into_batches(user_tensors)
        results = []
        times = []

        for batch in batches:
            # 准备输入张量
            input_ids = torch.cat([t["input_ids"] for t in batch], dim=0)
            attention_mask = torch.cat([t["attention_mask"] for t in batch], dim=0)

            # 生成参数
            gen_kwargs = {
                "max_length": input_ids.size(1) + max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }

            # 记录时间
            start_time = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                ) if model is None else model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            end_time = time.time()
            times.extend([end_time - start_time] * len(input_ids))

            # 【修改】提取 logits（如果模型包含 Value Head，则从输出中提取）
            if self.has_value_head:
                # 假设输出格式为 (logits, value)，具体需根据模型实现调整
                if isinstance(output_ids, tuple) and len(output_ids) >= 2:
                    logits = output_ids[0]
                    # value = output_ids[1]  # 可选处理
                else:
                    logits = output_ids
            else:
                logits = output_ids

            # 解码或保留为张量
            if return_type == str:
                decoded = self.tokenizer.batch_decode(output_ids, 
                                                      skip_special_tokens=True) if tokenizer is None\
                                                      else tokenizer.batch_decode(output_ids, 
                                                                                  skip_special_tokens=True)
                results.extend(decoded)
            elif return_type == torch.Tensor:
                if logits == None:
                    results.append(output_ids)
                else:
                    results.append(logits)

            elif return_type == "all":
                decoded = self.tokenizer.batch_decode(output_ids, 
                                                      skip_special_tokens=True) if tokenizer is None\
                                                      else tokenizer.batch_decode(output_ids, 
                                                                                  skip_special_tokens=True)
                results.append({"ids": output_ids, 
                                "str": decoded})
            else:
                raise ValueError("return_type must be str or torch.Tensor")

        # 合并结果
        if return_type == str:
            final_results = results
        else:
            final_results = torch.cat(results, dim=0)

        # 返回结果（是否包含时间）
        if return_time:
            return (final_results, times)
        else:
            return (final_results,)










