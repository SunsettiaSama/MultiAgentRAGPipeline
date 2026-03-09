# LLM 模块文档

## 概述

`src/llm/` 模块提供了统一的语言模型接口，支持**本地模型推理**和 **API 调用**两种方式。

## 模块结构

```
src/llm/
├── local_llm.py          # 本地模型推理
├── api_llm.py            # API 模型调用
└── kv_cache_manager.py   # KV 缓存管理
```

## 核心类

### 1. Large_Language_Model (本地推理)

位置：`src/large_language_model.py`

用于加载和推理本地模型（如 Qwen、Llama）。

**主要方法：**

- `__init__(local_dir, device, model, tokenizer, batch_size)`: 初始化模型
- `init_llm(system_prompt)`: 初始化系统提示词
- `generate(prompt, max_tokens, temperature, ...)`: 生成文本
- `release()`: 释放模型资源
- `reload_lora(config)`: 加载 LoRA 权重

**使用示例：**

```python
from src.large_language_model import Large_Language_Model

# 初始化模型
llm = Large_Language_Model(
    local_dir='./qwen2.5_1.5B/',
    device=torch.device('cuda')
)

# 设置系统提示词
llm.init_llm("你是一个有用的AI助手")

# 生成文本（单个）
result = llm.generate("What is RAG?")
print(result)

# 批量生成
prompts = ["What is RAG?", "Explain LLM"]
results = llm.generate(prompts)

# 返回生成时间
results, times = llm.generate(
    prompts, 
    return_time=True
)

# 释放资源
llm.release()
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| local_dir | str | './qwen2.5_1.5B/' | 本地模型路径 |
| device | torch.device | cuda/cpu | 推理设备 |
| model | AutoModelForCausalLM | None | 预加载的模型 |
| tokenizer | AutoTokenizer | None | 预加载的分词器 |
| batch_size | int | 4 | 批处理大小 |

### 2. Large_Language_Model_API (API 调用)

位置：`src/llm/api_llm.py` 和 `src/large_language_model.py`

用于调用 OpenAI 兼容的 API 模型。

**主要方法：**

- `__init__(api_key, base_url, model, timeout)`: 初始化 API 客户端
- `init_llm(system_prompt)`: 初始化系统提示词
- `require(messages, max_tokens, temperature)`: 发送原始消息列表
- `generate(input_text, max_tokens, temperature, ...)`: 生成文本

**使用示例：**

```python
from src.llm.api_llm import Large_Language_Model_API

# 初始化 API 模型
llm = Large_Language_Model_API(
    api_key="your-api-key",  # 从环境变量获取
    base_url="https://api.openai-proxy.org/v1",
    model="gpt-4o-mini"
)

# 设置系统提示词
llm.init_llm("你是一个专业的技术助手")

# 生成文本
result = llm.generate("What is retrieval augmented generation?")
print(result)

# 批量调用
prompts = ["What is RAG?", "Explain LLM"]
results = llm.generate(prompts)

# 使用原始消息格式
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
]
response = llm.require(messages)
```

**API 配置：**

通过环境变量设置 API 密钥：

```bash
export OPENAI_API_KEY="your-api-key"
```

或在代码中传入：

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
```

## 模型特性

### 支持的操作

| 操作 | 本地模型 | API 模型 |
|------|---------|---------|
| 单条生成 | ✓ | ✓ |
| 批量生成 | ✓ | ✓ |
| 返回耗时 | ✓ | ✓ |
| 返回输入提示 | ✓ | ✓ |
| LoRA 微调 | ✓ | ✗ |
| 流式输出 | ✗ | ✗ |
| 模型保存 | ✓ | ✗ |

### 性能优化

**本地模型：**

- 支持批处理，降低推理延迟
- 自动 GPU 显存管理
- 支持 LoRA 权重加载
- KV 缓存优化

**API 模型：**

- 自动重试和速率限制处理
- 并发请求优化
- 超时配置

## 系统提示词管理

两种模型都支持系统提示词（system prompt）的管理：

```python
# 单个系统提示
llm.init_llm("你是一个助手")

# 多个系统提示（用于批处理）
llm.init_llm([
    "你是一个技术助手",
    "你是一个创意写手"
])
```

## 补足提示词

支持在用户输入后追加提示词：

```python
llm.init_llm_complement_prompt("\n请用中文回答")
result = llm.generate("What is AI?")
# 实际输入会变成: "What is AI?\n请用中文回答"
```

## 错误处理

**API 调用错误：**

- 自动重试机制处理临时错误
- 返回错误消息而不是抛出异常

**本地模型错误：**

- 检查 GPU 显存
- 验证模型路径存在
- 确保 CUDA 版本兼容

## 性能对比

| 指标 | 本地模型 | API 模型 |
|------|---------|---------|
| 推理速度 | 快（GPU） | 取决于网络 |
| 显存占用 | 高 | 无 |
| 调用成本 | 免费 | 按 token 计费 |
| 隐私性 | 完全本地 | 数据上传 |
| 自定义度 | 高 | 低 |

## 常见问题

### Q: 如何切换模型？
A: 创建新实例或修改参数后重新初始化。

### Q: 如何处理显存溢出？
A: 减少 `batch_size` 或使用更小的模型。

### Q: API 调用超时怎么办？
A: 增加 `timeout` 参数或检查网络连接。

### Q: 如何使用本地 Ollama 模型？
A: 配置 `base_url` 指向本地 Ollama 服务。

## 最佳实践

1. **生产环境**：使用 API 模型以减少资源占用
2. **研究环发**：使用本地模型以获得完全控制
3. **批处理**：充分利用 `batch_size` 提高吞吐量
4. **资源管理**：及时调用 `release()` 释放资源
5. **提示工程**：优化系统提示词以获得更好结果
