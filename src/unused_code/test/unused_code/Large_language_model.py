import torch
from typing import Union, List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 新版本的LLM
class Large_Language_Model(torch.nn.Module):
    def __init__(self, 
                 model_name=None, 
                 local_dir='./qwen2.5_1.5B/', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model: torch.nn.Module = None, 
                 tokenizer: torch.nn.Module = None, 
                 ):
        super(Large_Language_Model, self).__init__()
        
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir, padding_side='left')
        else:
            self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if not model:
            self.model = AutoModelForCausalLM.from_pretrained(local_dir)
            self.model.to(device)
        else:
            self.model = model

        self.model.eval()

        # 新增：用于保存生成历史记录
        self.history = []

        # 用于保存采样得到的历史记录
        self.sample_memory: List[List[List]] = []

    def init_llm(self, 
                 system_prompt: Union[str, List[str]], 
                 thread_idx: int = 0, 
                 **kwargs):
        """
        初始化系统提示词。支持两种模式：
        1. 单线程模式：传入字符串和 thread_idx（默认为 0）
        2. 批量多线程模式：传入字符串列表，每个元素对应一个线程的系统提示

        Args:
            system_prompt (str | List[str]): 系统提示词内容
            thread_idx (int): 单线程模式下指定线程索引，默认为 0
        """
        if isinstance(system_prompt, list):
            # 批量多线程模式
            for i, prompt in enumerate(system_prompt):
                if len(self.history) <= i:
                    self.history.append([{
                    "from": "system",
                    "value": prompt
                }])

        else:
            # 单线程模式
            if len(self.history) <= thread_idx:
                self.history.extend([[] for _ in range(thread_idx - len(self.history) + 1)])
            self.history[thread_idx] = [{
                "from": "system",
                "value": system_prompt
            }]

    def generate(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 1000, 
        temperature: float = 1.0, 
        add_to_history: bool = True, 
        **kwargs
    ) -> Union[str, List[str]]:
        """
        生成文本，支持单个字符串或字符串列表输入。
        """

        is_batch = isinstance(prompt, list)
        # 获取输入的 batch_size
        batch_size = len(prompt) if is_batch else 1
        if not isinstance(prompt, (str, list)):
            raise TypeError("prompt must be a string or a list of strings")

        # 动态扩展 history 线程数
        if len(self.history) < batch_size:
            if not self.history:
                raise ValueError("No system prompt initialized. Call init_llm first.")
            last_system_prompt = self.history[-1][0]["value"]  # 获取最后一个线程的系统提示
            # 扩展 history
            for i in range(len(self.history), batch_size):
                self.history.append([{"from": "system", "value": last_system_prompt}])


        # 动态扩展 history 线程数
        if len(self.history) < batch_size:
            if not self.history:
                raise ValueError("No system prompt initialized. Call init_llm first.")
            last_system_prompt = self.history[-1][0]["value"]  # 获取最后一个线程的系统提示
            # 扩展 history
            for i in range(len(self.history), batch_size):
                self.history.append([{"from": "system", "value": last_system_prompt}])

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
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

        if add_to_history:
            original_prompts = prompt if isinstance(prompt, list) else [prompt]
            for input_prompt, generated_text in zip(original_prompts, results):
                self.history.append({
                    "from": "human",
                    "value": input_prompt
                })

                self.history.append({
                    "from": "gpt",
                    "value": generated_text
                })

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

    def save_history(self, file_path: str):
        """
        将历史记录保存为 JSON 文件。支持多线程历史记录保存。

        结构说明：
        - 第一层列表：每个元素对应一个对话线程
        - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

        示例：
        [
            [  # 线程 0 的对话历史
                {"from": "system", "value": "..."},
                {"from": "human", "value": "xxx"},
                ...
            ],
            [  # 线程 1 的对话历史
                {"from": "system", "value": "..."},
                {"from": "human", "value": "xxx"},
                ...
            ],
            ...
        ]
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
    
    def clear_history(self):
        """清空历史记录"""
        # 保留历史记录中的system prompt
        self.history = [self.history[i][0] for i in range(len(self.history))]

    def get_history(self):
        """获取当前历史记录的副本"""
        return self.history.copy()

    def chat(self, input_text: Union[str, List[str]], max_tokens: int = 1000, temperature: float = 1.0, is_test: bool = False, **kwargs) -> Union[str, List[str]]:
        """
        支持带历史记录的对话生成，可处理批量输入。所有提示词和格式由外部传入的 System prompt 决定。
        """
        is_batch = isinstance(input_text, list)
        input_batch = input_text if is_batch else [input_text]
        batch_size = len(input_batch)

        # 确保 self.history 的长度足够
        if len(self.history) < batch_size:
            self.history.extend([[] for _ in range(batch_size - len(self.history))])

        # 构造每个输入的上下文
        prompts = []
        for i in range(batch_size):
            # 检查是否已初始化系统提示
            if not self.history[i]:  # 如果尚未初始化 System prompt
                raise ValueError(f"Thread {i} has no system prompt initialized. Call init_system_prompt first.")
            system_prompt = self.history[i][0]['value']
            conversation_history = self.history[i][1:]  # 跳过系统提示

            # 构建上下文
            context_parts = [system_prompt]
            for entry in conversation_history:

                for key, value in entry.items():
                    if key == 'from':
                        context_parts.append("Assistant" + ":" if value == "gpt" else "Human" + ":")
                    elif key == 'value':
                        context_parts.append(value)
                    
                    

            context = " ".join(context_parts)

            # 构建最终提示词
            prompt = f"{context} User: {input_batch[i]}"
            prompts.append(prompt)

        
        # 生成回复
        responses = self.generate(prompts, max_tokens=max_tokens, temperature=temperature, **kwargs)
        if is_test:
            print(prompts)
            print(responses)
        
        # 提取模型输出中的 assistant 部分
        extracted_responses = []
        for i in range(batch_size):
            full_response = responses[i]
            assistant_index = full_response.find("Assistant: ")
            if assistant_index != -1:
                extracted = full_response[assistant_index + len("Assistant: "):].strip()
            else:
                extracted = full_response.strip()
            extracted_responses.append(extracted)

        # 更新历史记录
        for i in range(batch_size):
            self.history[i].append({
                "from": "human",
                "value": input_batch[i]
            })

            self.history[i].append({
                "from": "gpt",
                "value": extracted_responses[i]
            })

        return extracted_responses if is_batch else extracted_responses[0]
  
    def chat_without_history(self, input_text: Union[str, List[str]], 
                             max_tokens: int = 1000, 
                             temperature: float = 1.0, 
                             need_print: bool = False, 
                             **kwargs) -> Union[str, List[str]]:
        """
        支持带系统提示的对话生成，但不使用过往的历史对话信息。所有提示词和格式由外部传入的 System prompt 决定。

        Args:
            input_text (str | List[str]): 用户输入，可为单个字符串或字符串列表
            max_tokens (int): 生成的最大新token数
            temperature (float): 控制生成的随机性
            **kwargs: 其他传递给generate的参数

        Returns:
            str 或 List[str]: 生成的回复文本
        """
        # 处理输入类型
        is_batch = isinstance(input_text, list)
        input_batch = input_text if is_batch else [input_text]
        batch_size = len(input_batch)

        # 确保 self.history 的长度足够
        if len(self.history) < batch_size:
            self.history.extend([[] for _ in range(batch_size - len(self.history))])
        
        # 构造每个输入的上下文
        prompts = []

        for i in range(batch_size):
            # 检查是否已初始化系统提示
            if not self.history[i]:  # 如果尚未初始化 System prompt
                raise ValueError(f"Thread {i} has no system prompt initialized. Call init_system_prompt first.")
            system_prompt = self.history[i][0]['value']  # 修改点1：提取系统提示
            prompt = f"System: {system_prompt} User: {input_batch[i]}\nYour Output: "
            prompts.append(prompt)

        # 生成回复
        responses = self.generate(prompts, 
                                  max_tokens=max_tokens, 
                                  temperature=temperature, 
                                  **kwargs)
        
        if need_print:
            print('=' * 20)
            print('Current Input Prompt:' )
            print('\n'.join(prompts))
            print('=' * 20)
            print('Current Input Prompt:' )
            print('\n'.join(responses))


        # 提取模型输出中的 assistant 部分
        extracted_responses = []
        for i in range(batch_size):
            full_response = responses[i]
            # 假设模型输出格式为 "Assistant: [response]"
            assistant_index = full_response.find("Your Output: ")
            if assistant_index != -1:
                extracted = full_response[assistant_index + len("Your Output: "):].strip()
            else:
                extracted = full_response.strip()
            extracted_responses.append(extracted)

        # 更新历史记录
        for i in range(batch_size):
            self.history[i].append({
                "from": "human",
                "value": input_batch[i]
            })

            self.history[i].append({
                "from": "gpt",
                "value": extracted_responses[i]
            })

        return extracted_responses if is_batch else extracted_responses[0]
    
    def sample_from_batch(
        self, 
        prompt: Union[str, List[str]], 
        max_tokens: int = 1000, 
        temperature: float = 1.0,
        do_sample: bool = False,           # 是否启用采样
        top_k: int = 10,                  # Top-k 参数
        top_p: float = 0.9,               # Top-p 参数
        num_return_sequences: int = 1,    # 每个输入生成多少个样本
        add_to_history: bool = False,
        **kwargs
    ) -> Union[List[List[str]], List[str], str]:
        """
        生成文本，支持单个字符串或字符串列表输入。
        - 如果 do_sample=True，则返回 List[List[str]]：每个输入对应多个生成样本。
        - 如果 do_sample=False，则返回 List[str] 或 str（单个输入）。
        """
        is_batch = isinstance(prompt, list)
        if not isinstance(prompt, (str, list)):
            raise TypeError("prompt must be a string or a list of strings")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # 根据 do_sample 决定生成参数
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        if do_sample:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences
            })
        else:
            generate_kwargs.update({
                "do_sample": False,  # 明确关闭采样（默认为贪心搜索）
            })

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        results = []
        for i in range(len(input_ids)):  # 遍历每个输入
            input_length = input_ids[i].shape[0]
            # 每个输入生成 num_return_sequences 个样本
            batch_samples = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j  # 计算样本在 outputs 中的索引
                generated_tokens = outputs[idx][input_length:]
                decoded = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                batch_samples.append(decoded)
            results.append(batch_samples)


        if add_to_history:
            self.add_to_history(input_prompts = prompt, responses = results)
        return results if is_batch else results[0]
    

    def sample_based_generate(
        self,
        prompt: List[List[str]],  # 输入格式：List[List[str]]，每个子列表是一个对话历史
        max_tokens: int = 1000,
        num_return_sequences: int = 3,  # 每个对话历史生成多少个采样响应
        add_to_memory: bool = True,
        **kwargs
    ) -> List[List[List[str]]]:
        """
        支持多轮对话的采样生成，调用 self.generate 实现采样逻辑。
        - 扁平化输入：将所有输入合并为一个大张量，一次性输入模型。
        - 重组输出：将模型输出拆分为原始 batch 结构。
        """
        if not isinstance(prompt, list) or not all(isinstance(h, list) for h in prompt):
            raise TypeError("输入应符合 List[List[str]] 格式")

        batch_size = len(prompt)
        total_samples = batch_size * num_return_sequences

        # 1. 扁平化输入：将每个对话历史复制 num_return_sequences 次
        flat_prompt = []
        for history in prompt:
            for _ in range(num_return_sequences):
                flat_prompt.append("\n".join(history))  # 将对话历史拼接为模型输入

        # 2. 调用 self.generate 实现批量生成
        # 注意：self.generate 返回 List[str]，每个输入对应 1 个输出
        sampled_responses = self.generate(
            prompt=flat_prompt,
            max_tokens=max_tokens,
            **kwargs  # 通过 **kwargs 传递所有采样参数（如 do_sample, temperature, top_k, top_p）
        )

        # 3. 重组输出为原始 batch 结构
        results = []
        for i in range(batch_size):
            base_history = prompt[i]
            start_idx = i * num_return_sequences
            end_idx = (i + 1) * num_return_sequences
            # 提取该 batch 对应的 num_return_sequences 个样本
            batch_samples = [
                base_history + [sampled_responses[j]]  # 构造完整的对话链
                for j in range(start_idx, end_idx)
            ]
            results.append(batch_samples)

            # 4. 更新 sample_memory
            if add_to_memory:
                if len(self.sample_memory) <= i:
                    self.sample_memory.append([])
                self.sample_memory[i].extend(batch_samples)

        return results

    def add_to_history(self, input_prompts, responses: Union[str, List[str]]):

        if not len(input_prompts) == len(responses):
            raise ValueError("更新历史记录时，获取的输入Input_prompts和输出Responses长度不一致！")
        
        if isinstance(responses[0], list) and isinstance(input_prompts[0], str):
            is_sample = True

        batch_size = len(input_prompts)
        sample_nums = len(responses[0]) if is_sample else 1

        for i in range(batch_size):
            for k in range(sample_nums):

                self.sample_memory[i][k].append({
                    "from": "human",
                    "value": input_prompts[i]
                })

                self.sample_memory[i][k].append({
                    "from": "gpt",
                    "value": responses[i][k]
                })

        return 

    def load_history_from_file(self, file_path: str):
        """
        从 JSON 文件加载历史记录，覆盖当前 self.history。
        要求文件结构与 save_history 方法保存的格式一致：
        - 第一层列表：每个元素对应一个对话线程
        - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

        示例文件结构：
        [
            [  # 线程 0 的对话历史
                {"input": "System Prompt", "output": ""},
                {"input": "用户问题1", "output": "模型回答1"},
                ...
            ],
            [  # 线程 1 的对话历史
                {"input": "另一个 System Prompt", "output": ""},
                {"input": "用户问题2", "output": "模型回答2"},
                ...
            ],
            ...
        ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"History file not found at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")

        # 验证结构
        if not isinstance(loaded_history, list):
            raise ValueError("Loaded history is not a list of threads.")
        
        for thread_idx, thread in enumerate(loaded_history):
            if not isinstance(thread, list):
                raise ValueError(f"Thread {thread_idx} is not a list of conversation entries.")
            for entry_idx, entry in enumerate(thread):
                if not isinstance(entry, dict) or 'input' not in entry or 'output' not in entry:
                    raise ValueError(f"Entry {entry_idx} in thread {thread_idx} is missing 'input' or 'output' field.")

        # 加载历史记录
        self.history = loaded_history

    def load_history_from_dict(self, history: List[List[Dict[str, str]]]):
        """
        从内存中的数据结构加载历史记录，覆盖当前 self.history。

        Args:
            history (List[List[Dict[str, str]]]): 历史记录数据结构，格式要求如下：
                - 第一层列表：每个元素对应一个对话线程
                - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

        示例：
        [
            [  # 线程 0 的对话历史
                {"input": "System Prompt", "output": ""},
                {"input": "用户问题1", "output": "模型回答1"},
                ...
            ],
            [  # 线程 1 的对话历史
                {"input": "另一个 System Prompt", "output": ""},
                {"input": "用户问题2", "output": "模型回答2"},
                ...
            ],
            ...
        ]
        """
        # 验证数据结构
        if not isinstance(history, list):
            raise ValueError("History must be a list of threads.")
        
        for thread_idx, thread in enumerate(history):
            if not isinstance(thread, list):
                raise ValueError(f"Thread {thread_idx} must be a list of conversation entries.")
            for entry_idx, entry in enumerate(thread):
                if not isinstance(entry, dict) or 'input' not in entry or 'output' not in entry:
                    raise ValueError(f"Entry {entry_idx} in thread {thread_idx} is missing 'input' or 'output' field.")

        # 加载历史记录
        self.history = history

from openai import OpenAI
import json
from typing import Union, List, Optional

class Large_Language_Model_API:

    """

    Args:
        history (List[List[Dict[str, str]]]): 历史记录数据结构，格式要求如下：
            - 第一层列表：每个元素对应一个对话线程
            - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

    示例：
    [
        [  # 线程 0 的对话历史
            {"input": "System Prompt", "output": ""},
            {"input": "用户问题1", "output": "模型回答1"},
            ...
        ],
        [  # 线程 1 的对话历史
            {"input": "另一个 System Prompt", "output": ""},
            {"input": "用户问题2", "output": "模型回答2"},
            ...
        ],
        ...
    ]
    """

    def __init__(self, 
                 api_key: str = 'sk-NHZNQX83inCXYa8a2tL2yMnGST1ls64CFVdhBlira9sM8qBI',
                 base_url: str = 'https://api.openai-proxy.org/v1',
                 model: str = "gpt-4o-mini",
                 timeout: Optional[float] = None):
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.timeout = timeout

        # 历史记录：每个线程的对话历史
        self.history = []

    def init_llm(self, system_prompt: Union[str, List[str]], thread_idx: int = 0):
        """
        初始化系统提示词。支持两种模式：
        1. 单线程模式：传入字符串和 thread_idx（默认为 0）
        2. 批量多线程模式：传入字符串列表，每个元素对应一个线程的系统提示

        Args:
            system_prompt (str | List[str]): 系统提示词内容
            thread_idx (int): 单线程模式下指定线程索引，默认为 0
        """
        if isinstance(system_prompt, list):
            # 批量多线程模式
            for i, prompt in enumerate(system_prompt):
                if len(self.history) <= i:
                    self.history.append([])
                self.history[i] = [{
                    "from": "system",
                    "value": prompt
                }]
        else:
            # 单线程模式
            if len(self.history) <= thread_idx:
                self.history.extend([[] for _ in range(thread_idx - len(self.history) + 1)])
            self.history[thread_idx] = [{
                "from": "system",
                "value": system_prompt
            }]

    def generate(
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
    
    def chat(self, input_text: Union[str, List[str]], 
             max_tokens: int = 1000, 
             temperature: float = 1.0, 
             is_test: bool = False, 
             **kwargs) -> Union[str, List[str]]:
        """
        带历史记录的对话生成，支持批量输入。
        """
        is_batch = isinstance(input_text, list)
        input_batch = input_text if is_batch else [input_text]
        batch_size = len(input_batch)

        # 确保 history 长度足够
        if len(self.history) < batch_size:
            self.history.extend([[] for _ in range(batch_size - len(self.history))])

        # 构造每个输入的上下文
        results = []
        for i in range(batch_size):
            if not self.history[i]:  # 检查是否初始化系统提示
                raise ValueError(f"Thread {i} has no system prompt initialized. Call init_llm first.")

            # 构造 messages 列表（符合 API 格式）
            messages = []
            system_prompt = self.history[i][0]['value']
            messages.append({"role": "system", "content": system_prompt})
            
            for entry in self.history[i][1:]:  # 跳过系统提示
                for key, value in entry.items():
                    if key == 'from':
                        role = "assistant"  if value == "gpt" else "user" 
                    elif key == 'value':
                        value = value

                    messages.append({"role": role, "content": value})

            messages.append({"role": "user", "content": input_batch[i]})

            # 调用 API 生成回复
            response = self.generate(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
            results.append(response)

        # 更新历史记录
        for i in range(batch_size):
            self.history[i].append({
                "from": "human",
                "value": input_batch[i]
            })

            self.history[i].append({
                "from": "gpt",
                "value": results[i]
            })

        return results if is_batch else results[0]

    def chat_without_history(self, input_text: Union[str, List[str]], 
                            max_tokens: int = 1000, 
                            temperature: float = 1.0, 
                            need_print: bool = False, 
                            save_history: bool = False, 
                            **kwargs) -> Union[str, List[str]]:
        """
        带系统提示的对话生成，但不使用过往的历史对话信息。
        每次调用后会记录当前对话到 history，供后续调用使用。
        """
        is_batch = isinstance(input_text, list)
        input_batch = input_text if is_batch else [input_text]
        batch_size = len(input_batch)

        
        # 确保 history 长度足够（用于存储系统提示）
        if len(self.history) < batch_size:
            self.history.extend([[] for _ in range(batch_size - len(self.history))])

        results = []
        for i in range(batch_size):
            if not self.history[i]:  # 检查是否初始化系统提示
                raise ValueError(f"Thread {i} has no system prompt initialized. Call init_llm first.")
            
            system_prompt = self.history[i][0]['value']
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_batch[i]}
            ]

            # 调用 API 生成回复
            response = self.generate(messages, 
                                     max_tokens=max_tokens, 
                                     temperature=temperature, 
                                     **kwargs)
            results.append(response)

            if save_history:
                # 更新历史记录（将当前对话记录添加到 history 中）
                self.history[i].append({
                    "from": "human",
                    "value": input_batch[i]
                })

                self.history[i].append({
                    "from": "gpt",
                    "value": results[i]
                })


        return results if is_batch else results[0]

    def save_history(self, file_path: str):
        """将历史记录保存为 JSON 文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
    
    def clear_history(self):
        """清空历史记录"""
        self.history = [self.history[i][0] for i in range(len(self.history))]

    def get_history(self):
        """获取当前历史记录的副本"""
        return self.history.copy()

    def load_history_from_file(self, file_path: str):
        """
        从 JSON 文件加载历史记录，覆盖当前 self.history。
        要求文件结构与 save_history 方法保存的格式一致：
        - 第一层列表：每个元素对应一个对话线程
        - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

        示例文件结构：
        [
            [  # 线程 0 的对话历史
                {"input": "System Prompt", "output": ""},
                {"input": "用户问题1", "output": "模型回答1"},
                ...
            ],
            [  # 线程 1 的对话历史
                {"input": "另一个 System Prompt", "output": ""},
                {"input": "用户问题2", "output": "模型回答2"},
                ...
            ],
            ...
        ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"History file not found at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")

        # 验证结构
        if not isinstance(loaded_history, list):
            raise ValueError("Loaded history is not a list of threads.")
        
        for thread_idx, thread in enumerate(loaded_history):
            if not isinstance(thread, list):
                raise ValueError(f"Thread {thread_idx} is not a list of conversation entries.")
            for entry_idx, entry in enumerate(thread):
                if not isinstance(entry, dict) or 'input' not in entry or 'output' not in entry:
                    raise ValueError(f"Entry {entry_idx} in thread {thread_idx} is missing 'input' or 'output' field.")

        # 加载历史记录
        self.history = loaded_history

    def load_history_from_dict(self, history: List[List[Dict[str, str]]]):
        """
        从内存中的数据结构加载历史记录，覆盖当前 self.history。

        Args:
            history (List[List[Dict[str, str]]]): 历史记录数据结构，格式要求如下：
                - 第一层列表：每个元素对应一个对话线程
                - 第二层列表：每个元素是该线程的一次对话记录，格式为 {'input': ..., 'output': ...}

        示例：
        [
            [  # 线程 0 的对话历史
                {"input": "System Prompt", "output": ""},
                {"input": "用户问题1", "output": "模型回答1"},
                ...
            ],
            [  # 线程 1 的对话历史
                {"input": "另一个 System Prompt", "output": ""},
                {"input": "用户问题2", "output": "模型回答2"},
                ...
            ],
            ...
        ]
        """
        # 验证数据结构
        if not isinstance(history, list):
            raise ValueError("History must be a list of threads.")
        
        for thread_idx, thread in enumerate(history):
            if not isinstance(thread, list):
                raise ValueError(f"Thread {thread_idx} must be a list of conversation entries.")
            for entry_idx, entry in enumerate(thread):
                if not isinstance(entry, dict) or 'input' not in entry or 'output' not in entry:
                    raise ValueError(f"Entry {entry_idx} in thread {thread_idx} is missing 'input' or 'output' field.")

        # 加载历史记录
        self.history = history

class Execution_Specified_Model_API(Large_Language_Model_API):

    def __init__(self):
        super().__init__()
        self.input_text_lists = []


    def clear_input_texts(self):
        self.input_text_lists = []

    def receive_inputs(self, system_prompts, input_prompts, **kwargs):
        """
        获取单条信息，等待后续输入到API中得到回复
        
        """
        if len(system_prompts) != len(input_prompts):
            raise ValueError("Execution模型接受的System Prompts输入和Input Prompts输入不等长")
        

        is_batch = isinstance(system_prompts, List)
        system_prompts = [system_prompts] if is_batch else system_prompts
        input_prompts = [input_prompts] if is_batch else input_prompts

        for i in range(len(system_prompts)):
            self.input_text_lists.append(
                [
                    {"role": "system", "content": system_prompts[i]},
                    {"role": "user", "content": input_prompts[i]}
                ]
            )

    def chat_without_history(self, 
                            max_tokens: int = 1000, 
                            temperature: float = 1.0, 
                            need_print: bool = False, 
                            **kwargs) -> Union[str, List[str]]:
                             
        """
        带系统提示的对话生成，但不使用过往的历史对话信息。
        每次调用后会记录当前对话到 history，供后续调用使用。
        """

        results = []
        for i in range(len(self.input_text_lists)):
            # 调用 API 生成回复
            response = self.generate(self.input_text_lists, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature, 
                                        **kwargs)
        results.append(response)

        return results if len(results) != 1 else results[0]


def test():


    model = Large_Language_Model()

    # 第一轮对话（批量输入）
    inputs = [
        "How were you today?",
        "What's your favorite color?"
    ]
    outputs = model.chat(inputs)
    print("第一轮回复:", outputs)

    # 第二轮对话（继续对话）
    inputs = [
        "Not bad, Not good. Life is like this right?",
        "I think blue is the best!",
    ]
    outputs = model.chat(inputs)
    print("第二轮回复:", outputs)

    # 查看完整对话历史
    print("对话历史:", model.get_history())

"""
[
[(A, R1)],
[(B, R2)],
[(C, R3)]
]
第二次调用inputs2 = [D, E, F]，生成回复R4、R5、R6，此时history变为：
[
[(A, R1), (D, R4)],
[(B, R2), (E, R5)],
[(C, R3), (F, R6)]
]
因此，每个索引i对应的历史是连续的对话线程。所以，inputs1[0]和inputs2[0]确实属于同一个对话线程，因为它们在同一个索引位置，历史记录被连续追加。

"""


def test_API():


    model = Large_Language_Model_API()
    model.init_llm(
        system_prompt = ['', ''], 
    )
    # 第一轮对话（批量输入）
    inputs = [
        "How were you today?",
        "What's your favorite color?"
    ]
    outputs = model.chat_without_history(inputs)
    print("第一轮回复:", outputs)

    # 第二轮对话（继续对话）
    inputs = [
        "Not bad, Not good. Life is like this right?",
        "I think blue is the best!",
    ]
    outputs = model.chat_without_history(inputs)
    print("第二轮回复:", outputs)

    # 查看完整对话历史
    print("对话历史:", model.get_history())



if __name__ == '__main__':
    test_API()