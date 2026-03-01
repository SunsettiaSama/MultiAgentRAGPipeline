import gradio as gr
from typing import List, Tuple, Dict
import os
import sys

# 添加当前目录到路径，确保可以导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from message_proposser import MessageProcessor


class LLMWebUI:
    """LLM对话WebUI类 - 主体框架"""
    
    def __init__(self):
        """初始化WebUI"""
        self.demo = None
        # UI组件属性
        self.chatbot = None
        self.msg_input = None
        self.send_btn = None
        self.clear_btn = None
        self.export_btn = None
        self.max_tokens = None
        self.temperature = None
        self.system_prompt = None
        self.export_output = None
        # API配置组件
        self.enable_api = None
        self.api_key = None
        self.api_base_url = None
        self.api_model = None
        # 消息处理器
        self.message_processor = MessageProcessor()
        # 构建UI
        self._build_ui()
    
    def _get_css(self) -> str:
        """获取CSS样式"""
        return """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chatbot {
            min-height: 500px;
        }
        """
    
    def _create_header(self):
        """创建页面标题"""
        # 使用markdown语法构筑页面标题
        gr.Markdown("""
        # 🤖 LLM对话WebUI
        
        基于Gradio搭建的LLM对话界面框架。
        """)
    
    def _create_chatbot_component(self):
        """创建对话窗口组件"""
        self.chatbot = gr.Chatbot(
            label="对话窗口",
            height=500,
        )
    
    def _create_input_component(self):
        """创建输入组件"""
        self.msg_input = gr.Textbox(
            label="输入消息",
            placeholder="在这里输入你的问题，按Enter键发送（Shift+Enter换行），或点击发送按钮...",
            scale=9,
            lines=2,
            max_lines=5,
        )
    
    def _create_send_button(self):
        """创建发送按钮"""
        self.send_btn = gr.Button("发送", variant="primary", scale=1, size="lg")
    
    def _create_clear_button(self):
        """创建清空按钮"""
        self.clear_btn = gr.Button("🗑️ 清空对话", scale=1)
    
    def _create_export_button(self):
        """创建导出按钮"""
        self.export_btn = gr.Button("📥 导出对话", scale=1)
    
    def _create_chat_tab(self):
        """创建对话标签页"""
        with gr.Tab("💬 对话"):
            self._create_chatbot_component()
            
            with gr.Row():
                self._create_input_component()
                self._create_send_button()
            
            with gr.Row():
                self._create_clear_button()
                self._create_export_button()
            
            # 导出输出组件（初始隐藏，点击导出后显示）
            self._create_export_output_component()
    
    def _create_config_sliders(self):
        """创建配置滑块组件"""
        self.max_tokens = gr.Slider(
            label="最大Token数",
            minimum=100,
            maximum=4000,
            value=2000,
            step=100
        )
        self.temperature = gr.Slider(
            label="Temperature (创造性)",
            minimum=0.0,
            maximum=2.0,
            value=0.7,
            step=0.1
        )
    
    def _create_system_prompt_component(self):
        """创建系统提示词组件"""
        self.system_prompt = gr.Textbox(
            label="系统提示词 (System Prompt)",
            placeholder="例如：你是一个专业的AI助手，擅长回答技术问题。",
            value="你是一个有用的AI助手。",
            lines=3
        )
    
    def _create_api_config_components(self):
        """创建API配置组件"""
        # 从配置文件中读取默认值
        default_api_key = self.message_processor.get_default_api_key()
        default_base_url = self.message_processor.get_default_base_url()
        default_model = self.message_processor.get_default_model()
        
        self.enable_api = gr.Checkbox(
            label="启用API调用",
            value=False,
            info="勾选后使用API调用，否则使用占位符回复"
        )
        
        # 如果配置文件中有API密钥，则预填；否则留空，提示用户填写
        api_key_placeholder = "请输入API密钥" if not default_api_key else "已从配置文件加载（可修改）"
        self.api_key = gr.Textbox(
            label="API Key",
            placeholder=api_key_placeholder,
            value=default_api_key,
            type="password",
            visible=True,
            info="如果配置文件中没有API密钥，请在此处填写"
        )
        
        self.api_base_url = gr.Textbox(
            label="API Base URL",
            placeholder="API基础URL",
            value=default_base_url,
            visible=True,
            info="默认值来自配置文件"
        )
        
        self.api_model = gr.Textbox(
            label="API Model",
            placeholder="模型名称",
            value=default_model,
            visible=True,
            info="默认值来自配置文件"
        )
    
    def _create_config_tab(self):
        """创建配置标签页"""
        with gr.Tab("⚙️ 配置"):
            gr.Markdown("### 🔧 配置选项")
            
            with gr.Row():
                self._create_config_sliders()
            
            self._create_system_prompt_component()
            
            gr.Markdown("### 🌐 API配置")
            self._create_api_config_components()
    
    def _create_export_output_component(self):
        """创建导出输出组件"""
        self.export_output = gr.Textbox(label="导出的对话内容", lines=20)
    
    def _bind_send_message_event(self):
        """绑定发送消息事件"""
        def send_message(msg, hist, max_tok, temp, sys_prompt, enable_api, api_key, api_base_url, api_model):
            """发送消息处理函数"""
            if not msg.strip():
                return hist, ""  # 返回历史记录和清空后的输入框
            
            # 处理消息并更新历史记录
            updated_history = self.message_processor.process(
                msg, hist, max_tok, temp, sys_prompt,
                enable_api=enable_api,
                api_key=api_key,
                api_base_url=api_base_url,
                api_model=api_model
            )
            
            # 返回更新后的历史记录和清空后的输入框（视觉上突出消息已发送）
            return updated_history, ""
        
        # 绑定发送按钮点击事件
        self.send_btn.click(
            fn=send_message,
            inputs=[
                self.msg_input, self.chatbot, self.max_tokens, self.temperature, self.system_prompt,
                self.enable_api, self.api_key, self.api_base_url, self.api_model
            ],
            outputs=[self.chatbot, self.msg_input]  # 同时更新对话窗口和清空输入框
        )
        
        # 绑定输入框回车事件（Enter键）
        # 注意：Gradio的submit事件默认绑定Enter键，但需要Shift+Enter换行
        self.msg_input.submit(
            fn=send_message,
            inputs=[
                self.msg_input, self.chatbot, self.max_tokens, self.temperature, self.system_prompt,
                self.enable_api, self.api_key, self.api_base_url, self.api_model
            ],
            outputs=[self.chatbot, self.msg_input]  # 同时更新对话窗口和清空输入框
        )
    
    def _bind_clear_chat_event(self):
        """绑定清空对话事件"""
        self.clear_btn.click(
            fn=self._clear_chat,
            outputs=[self.chatbot]
        )
    
    def _bind_export_chat_event(self):
        """绑定导出对话事件"""
        self.export_btn.click(
            fn=self._export_chat,
            inputs=[self.chatbot],
            outputs=[self.export_output]
        )
    
    def _bind_all_events(self):
        """绑定所有事件"""
        self._bind_send_message_event()
        self._bind_clear_chat_event()
        self._bind_export_chat_event()
    
    def _build_ui(self):
        """构建UI界面"""
        with gr.Blocks(
            title="LLM对话WebUI",
            theme=gr.themes.Soft(),
            css=self._get_css()
        ) as demo:
            self._create_header()
            
            with gr.Tabs():
                self._create_chat_tab()
                self._create_config_tab()
            
            self._bind_all_events()
            
            self.demo = demo
    
    def _clear_chat(self) -> List[List[str]]:
        """清空对话历史"""
        return []
    
    def _export_chat(self, history: List[List[str]]) -> str:
        """导出对话历史为文本"""
        if not history:
            return "对话历史为空，无法导出。"
        
        export_text = "# 对话历史导出\n\n"
        
        # 遍历历史记录，每条记录包含[用户消息, 助手回复]
        for i, (user_msg, assistant_msg) in enumerate(history, 1):
            export_text += f"## 对话 {i}\n\n"
            export_text += f"**用户**: {user_msg}\n\n"
            export_text += f"**助手**: {assistant_msg}\n\n"
            export_text += "---\n\n"
        
        return export_text
    
    def run(self, server_name: str = "0.0.0.0", server_port: int = 7860, 
            share: bool = False, inbrowser: bool = True, show_error: bool = True):
        """
        启动WebUI服务
        
        参数:
            server_name: 服务器地址
            server_port: 端口号
            share: 是否创建公网链接
            inbrowser: 是否自动打开浏览器
            show_error: 是否显示错误信息
        """
        if self.demo is None:
            raise RuntimeError("UI未初始化，请先调用_build_ui()")
        
        self.demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            inbrowser=inbrowser,
            show_error=show_error
        )


def main():
    """主函数：创建并运行WebUI"""
    webui = LLMWebUI()
    webui.run()


if __name__ == "__main__":
    main()
