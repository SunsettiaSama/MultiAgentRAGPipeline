from typing import Union, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

class Large_Language_Model_API:
    def __init__(self, 
                 api_key: str = '',
                 base_url: str = '',
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
    
