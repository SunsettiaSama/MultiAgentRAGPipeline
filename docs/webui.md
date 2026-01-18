# WebUI 使用文档

## 概述

WebUI 是基于 Gradio 搭建的 LLM 对话界面，支持通过 API 调用或占位符模式进行对话交互。

## 快速开始

### 1. 基本使用

#### 方式一：直接运行脚本

```bash
cd D:\ProjectsOnPython\MultiAgentRAGPipeline
python src/webui/demo.py
```

#### 方式二：在代码中调用

```python
from src.webui.demo import LLMWebUI

# 创建WebUI实例
webui = LLMWebUI()

# 启动服务（使用默认配置）
webui.run()

# 或自定义配置
webui.run(
    server_name="0.0.0.0",  # 服务器地址，允许局域网访问
    server_port=7860,       # 端口号
    share=False,            # 是否创建公网链接
    inbrowser=True,         # 是否自动打开浏览器
    show_error=True         # 是否显示错误信息
)
```

### 2. 配置文件设置

WebUI 会自动从配置文件读取 API 配置。配置文件位置：

```
config/api_config.yaml
```

#### 配置文件格式

```yaml
# API配置默认文件
# 用于WebUI的API调用配置

api:
  # API密钥（如果为空，则需要在界面中手动填写）
  api_key: "your-api-key-here"
  
  # API服务器地址
  base_url: "https://api.openai-proxy.org/v1"
  
  # 默认模型名称
  model: "gpt-4o-mini"
  
  # 请求超时时间（秒），None表示使用默认值
  timeout: null
```

#### 配置说明

- **api_key**: API 密钥，如果配置文件中为空，需要在 WebUI 界面中手动填写
- **base_url**: API 服务器地址，默认为 OpenAI 兼容的代理地址
- **model**: 默认使用的模型名称
- **timeout**: 请求超时时间，设置为 `null` 使用默认值（60秒）

## 功能说明

### 1. 对话功能

- **发送消息**：
  - 点击"发送"按钮
  - 或按 `Enter` 键发送（`Shift+Enter` 换行）
  
- **清空对话**：点击"🗑️ 清空对话"按钮

- **导出对话**：点击"📥 导出对话"按钮，导出为 Markdown 格式

### 2. 配置功能

#### 生成参数配置

- **最大Token数**：控制生成回复的最大长度（100-4000）
- **Temperature**：控制回复的创造性（0.0-2.0）
  - 较低值：更保守、确定性回复
  - 较高值：更创造性、随机性回复
- **系统提示词**：设置 AI 助手的角色和行为

#### API配置

- **启用API调用**：勾选后使用真实的 API 调用，否则使用占位符回复
- **API Key**：API 密钥（如果配置文件中已设置，会自动预填）
- **API Base URL**：API 服务器地址（默认从配置文件读取）
- **API Model**：模型名称（默认从配置文件读取）

### 3. 工作模式

#### 占位符模式（默认）

- 不勾选"启用API调用"
- 返回模拟回复，用于测试界面功能
- 不需要 API 配置

#### API模式

- 勾选"启用API调用"
- 需要配置有效的 API 密钥
- 调用真实的 LLM API 获取回复

## 代码结构

### 主要类

#### `LLMWebUI`

WebUI 主类，负责构建和启动界面。

**主要方法：**

- `__init__()`: 初始化 WebUI，自动构建界面
- `run()`: 启动 WebUI 服务

#### `MessageProcessor`

消息处理类，负责处理用户消息并生成回复。

**主要方法：**

- `process()`: 处理用户消息，根据配置选择调用方式
- `get_default_api_key()`: 获取默认 API 密钥
- `get_default_base_url()`: 获取默认 API 基础 URL
- `get_default_model()`: 获取默认模型名称

### 文件结构

```
src/webui/
├── demo.py                 # WebUI主文件
├── message_proposser.py    # 消息处理类
└── __init__.py

config/
└── api_config.yaml         # API配置文件
```

## 使用示例

### 示例1：基本使用

```python
from src.webui.demo import LLMWebUI

# 创建并启动WebUI
webui = LLMWebUI()
webui.run()
```

### 示例2：自定义端口

```python
from src.webui.demo import LLMWebUI

webui = LLMWebUI()
webui.run(server_port=8080)  # 使用8080端口
```

### 示例3：允许局域网访问

```python
from src.webui.demo import LLMWebUI

webui = LLMWebUI()
webui.run(
    server_name="0.0.0.0",  # 允许所有网络接口访问
    server_port=7860
)
```

### 示例4：创建公网链接

```python
from src.webui.demo import LLMWebUI

webui = LLMWebUI()
webui.run(
    share=True,  # 创建临时公网链接（需要网络连接）
    server_port=7860
)
```

## 注意事项

1. **配置文件路径**：确保 `config/api_config.yaml` 文件存在且格式正确
2. **API密钥安全**：不要将包含真实 API 密钥的配置文件提交到版本控制系统
3. **网络要求**：使用 API 模式需要网络连接
4. **依赖安装**：确保已安装必要的依赖包
   ```bash
   pip install gradio openai pyyaml
   ```

## 故障排除

### 问题1：无法读取配置文件

**症状**：界面中 API 配置项为空

**解决方案**：
- 检查 `config/api_config.yaml` 文件是否存在
- 检查文件格式是否正确（YAML 格式）
- 检查文件路径是否正确

### 问题2：API调用失败

**症状**：显示 "[API调用错误]" 消息

**解决方案**：
- 检查 API 密钥是否正确
- 检查网络连接是否正常
- 检查 API 服务器地址是否正确
- 检查模型名称是否有效

### 问题3：端口被占用

**症状**：启动时提示端口已被占用

**解决方案**：
- 修改 `server_port` 参数使用其他端口
- 或关闭占用该端口的其他程序

## 更新日志

- **v1.0**: 初始版本
  - 支持占位符模式和 API 模式
  - 支持配置文件读取
  - 支持对话历史管理
  - 支持对话导出功能
