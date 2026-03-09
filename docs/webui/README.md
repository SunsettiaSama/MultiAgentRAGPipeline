# Web UI 模块文档

## 概述

`src/webui/` 模块提供了基于 **Gradio** 的交互式 Web 用户界面，用于与 LLM 进行对话和配置模型参数。

## 模块结构

```
src/webui/
├── demo.py                  # 主对话界面
├── message_proposser.py     # 消息处理器
└── rag_demo.py              # RAG 演示界面
```

## 快速开始

### 启动主对话界面

```bash
python -m src.webui.demo
```

或：

```python
from src.webui.demo import LLMWebUI

webui = LLMWebUI()
webui.run(server_name="0.0.0.0", server_port=7860)
```

打开浏览器访问 `http://localhost:7860` 即可使用。

## 核心功能

### 1. 对话界面 (💬 对话标签)

- **实时对话**：与 LLM 进行多轮对话
- **对话历史**：自动保存对话记录
- **消息输入**：支持多行输入（Shift+Enter 换行）
- **导出功能**：支持导出对话历史为 Markdown 格式

**界面组件：**

| 组件 | 功能 | 说明 |
|------|------|------|
| 对话窗口 | 显示对话 | 实时显示用户消息和助手回复 |
| 输入框 | 输入消息 | Shift+Enter 换行，Enter 发送 |
| 发送按钮 | 提交消息 | 点击发送消息 |
| 清空按钮 | 清空历史 | 清空所有对话记录 |
| 导出按钮 | 导出对话 | 导出为 Markdown 格式 |

### 2. 配置界面 (⚙️ 配置标签)

#### 生成配置

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| 最大Token数 | 100-4000 | 2000 | 单次生成的最大 token 数 |
| Temperature | 0.0-2.0 | 0.7 | 控制生成的随机性 |

#### 系统提示词

设置模型的行为方式和角色定位，例如：
- "你是一个专业的技术顾问"
- "你是一个有趣的故事编写者"
- "你是一个严格的代码审查员"

#### API 配置

| 参数 | 类型 | 说明 |
|------|------|------|
| 启用API调用 | 复选框 | 选中使用 API，否则使用占位符 |
| API Key | 密码 | OpenAI API 密钥 |
| API Base URL | 文本 | API 服务地址 |
| API Model | 文本 | 使用的模型名称 |

**API 配置从配置文件加载：**

从 `src/webui/message_proposser.py` 中配置 YAML 文件路径后，Web UI 会自动加载默认配置。

## 核心类

### LLMWebUI

主 Web UI 类，管理所有界面和事件。

**主要方法：**

- `__init__()`: 初始化 UI
- `run(server_name, server_port, share, inbrowser)`: 启动 Web 服务
- `_build_ui()`: 构建 UI 布局
- `_bind_all_events()`: 绑定所有事件处理器

**使用示例：**

```python
from src.webui.demo import LLMWebUI

# 创建 Web UI
webui = LLMWebUI()

# 在本地运行
webui.run(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    inbrowser=True
)

# 或在服务器上运行
webui.run(
    server_name="0.0.0.0",
    server_port=8080,
    share=True  # 生成公网链接
)
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| server_name | "0.0.0.0" | 服务器地址 |
| server_port | 7860 | 端口号 |
| share | False | 是否创建公网链接 |
| inbrowser | True | 是否自动打开浏览器 |
| show_error | True | 是否显示错误信息 |

### MessageProcessor

消息处理和 API 调用器。

**主要方法：**

- `process(message, history, ...)`: 处理消息并生成回复
- `_api_call(messages, ...)`: 调用 API
- `_placeholder_call(...)`: 返回占位符回复
- `get_default_api_key()`: 获取默认 API 密钥
- `get_default_base_url()`: 获取默认 API URL
- `get_default_model()`: 获取默认模型

**使用示例：**

```python
from src.webui.message_proposser import MessageProcessor

processor = MessageProcessor()

# 初始化
history = []

# 处理用户消息
history = processor.process(
    message="你好，请介绍一下自己",
    history=history,
    max_tokens=1000,
    temperature=0.7,
    system_prompt="你是一个有用的助手",
    enable_api=True,
    api_key="your-key",
    api_base_url="https://api.openai.com/v1",
    api_model="gpt-4"
)

print(history)
```

## 配置文件管理

### 自动配置加载

如果提供了配置文件路径，Web UI 会自动加载以下参数：

```yaml
# config/api_config.yaml
api:
  api_key: ""  # 从环境变量读取
  base_url: "https://api.openai-proxy.org/v1"
  model: "gpt-4o-mini"
  timeout: 60
```

**设置步骤：**

1. 编辑 `src/webui/message_proposser.py` 中的 `API_YAML_PATH`
2. 创建相应的 YAML 配置文件
3. 填入 API 配置（API Key 应从环境变量读取）

## 事件处理

### 发送消息事件

当点击"发送"按钮或按 Enter 键时：

1. 获取用户输入和所有参数
2. 调用 `MessageProcessor.process()`
3. 更新对话历史
4. 清空输入框
5. 显示新的助手回复

### 清空对话事件

点击"清空对话"按钮：
- 清除所有对话历史
- 重置为空列表

### 导出对话事件

点击"导出对话"按钮：
- 格式化对话历史为 Markdown
- 在导出框中显示可复制的文本

## 样式定制

修改 `LLMWebUI._get_css()` 方法自定义界面样式：

```python
def _get_css(self) -> str:
    return """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chatbot {
        min-height: 500px;
    }
    """
```

## 高级用法

### 集成本地模型

在 `MessageProcessor` 中替换 API 调用为本地模型：

```python
from src.large_language_model import Large_Language_Model

class LocalMessageProcessor(MessageProcessor):
    def __init__(self):
        super().__init__()
        self.llm = Large_Language_Model(local_dir='./model/')
        
    def _api_call(self, messages, **kwargs):
        # 转换为本地模型格式
        text = messages[-1]["content"]
        response = self.llm.generate(text, **kwargs)
        return response
```

### 多用户隔离

使用会话管理隔离不同用户的对话：

```python
from gradio import Session

def process_with_session(message, session):
    user_id = session.id
    # 基于 user_id 隔离对话历史
    return response
```

## 常见问题

### Q: 如何修改端口号？
A: 调用 `run()` 方法时修改 `server_port` 参数。

### Q: 如何在生产环境部署？
A: 设置 `server_name="0.0.0.0"` 并配置反向代理（如 Nginx）。

### Q: 如何添加新的参数？
A: 在 `_create_config_tab()` 中添加新的 Gradio 组件，并在 `send_message()` 中处理。

### Q: 如何实现流式输出？
A: 使用 Gradio 的流式生成器功能，修改事件处理函数返回值为生成器。

## 最佳实践

1. **安全性**：API Key 使用密码字段，不在日志中显示
2. **可用性**：支持 Enter/Shift+Enter 快捷键，提供导出功能
3. **可扩展性**：使用 `MessageProcessor` 便于替换不同后端
4. **错误处理**：API 错误显示为用户消息，不导致崩溃
