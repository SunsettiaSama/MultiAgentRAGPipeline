from typing import List, Dict, Union, Optional
import os
import yaml
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# API配置文件路径（相对于项目根目录）
API_YAML_PATH = r""


class MessageProcessor:
    """消息处理类 - 处理用户消息并生成回复"""
    
    def __init__(self):
        """初始化消息处理器"""
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        从YAML配置文件加载API配置
        
        配置文件位置：由常量 API_YAML_PATH 指定
        
        返回:
            配置字典，如果文件不存在或读取失败，返回空字典
        """
        try:
            if os.path.exists(API_YAML_PATH):
                with open(API_YAML_PATH, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                # 提取api配置部分
                return config.get('api', {})
            else:
                return {}
        except Exception as e:
            print(f"[警告] 读取配置文件失败: {str(e)}")
            return {}
    
    def get_default_api_key(self) -> str:
        """获取默认API密钥"""
        return self.config.get('api_key', '')
    
    def get_default_base_url(self) -> str:
        """获取默认API基础URL"""
        return self.config.get('base_url', 'https://api.openai-proxy.org/v1')
    
    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return self.config.get('model', 'gpt-4o-mini')
    
    def get_default_timeout(self) -> Optional[float]:
        """获取默认超时时间"""
        return self.config.get('timeout', None)
    
    def _api_call(self, messages: List[Dict[str, str]], api_key: str, 
                  max_tokens: int, temperature: float, 
                  base_url: str = "https://api.openai-proxy.org/v1",
                  model: str = "gpt-4o-mini") -> str:
        """
        调用API获取回复
        
        参数:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            api_key: API密钥
            max_tokens: 最大token数
            temperature: 温度参数
            base_url: API基础URL
            model: 模型名称
            
        返回:
            API返回的回复内容
        """
        if OpenAI is None:
            return "[错误] 未安装openai库，请运行: pip install openai"
        
        if not api_key or not api_key.strip():
            return "[错误] API密钥未配置"
        
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[API调用错误] {str(e)}"
    
    def _placeholder_call(self, message: str, max_tokens: int, 
                         temperature: float, system_prompt: str) -> str:
        """
        占位符调用，返回模拟回复
        
        参数:
            message: 用户消息
            max_tokens: 最大token数
            temperature: 温度参数
            system_prompt: 系统提示词
            
        返回:
            占位符回复
        """
        placeholder_response = f"[占位符回复] 已收到消息：「{message}」\n\n" \
                              f"参数：max_tokens={max_tokens}, temperature={temperature}\n" \
                              f"系统提示词：{system_prompt[:50]}..."
        return placeholder_response
    
    def process(self, message: str, history: List[List[str]], 
                max_tokens: int, temperature: float, system_prompt: str,
                enable_api: bool = False, api_key: str = "",
                api_base_url: str = "https://api.openai-proxy.org/v1",
                api_model: str = "gpt-4o-mini") -> List[List[str]]:
        """
        处理用户消息并生成回复（符合Gradio Chatbot接口格式）
        
        参数:
            message: 用户输入的消息
            history: 对话历史，Gradio Chatbot格式为 [[用户消息, 助手回复], ...]
            max_tokens: 最大token数
            temperature: 温度参数
            system_prompt: 系统提示词
            enable_api: 是否启用API调用
            api_key: API密钥
            api_base_url: API基础URL
            api_model: API模型名称
            
        返回:
            更新后的对话历史，Gradio Chatbot格式为 [[用户消息, 助手回复], ...]
        """
        if not message.strip():
            return history
        
        # 确保history是列表格式
        if history is None:
            history = []
        
        # 根据配置选择调用方式
        if enable_api:
            # 构建API调用的messages格式（包含系统提示词和历史记录）
            api_messages = []
            if system_prompt and system_prompt.strip():
                api_messages.append({"role": "system", "content": system_prompt})
            
            # 转换history格式为OpenAI API格式
            for user_msg, assistant_msg in history:
                api_messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    api_messages.append({"role": "assistant", "content": assistant_msg})
            
            # 添加当前用户消息
            api_messages.append({"role": "user", "content": message})
            
            # 调用API
            response = self._api_call(
                messages=api_messages,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=api_base_url,
                model=api_model
            )
        else:
            # 使用占位符回复
            response = self._placeholder_call(message, max_tokens, temperature, system_prompt)
        
        # 添加用户消息和助手回复对到历史记录
        history.append([message, response])
        
        return history
