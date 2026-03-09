# LLMs 研究项目

## 项目概述

这是一个全面的大语言模型 (LLM) 研究框架，专注于**检索增强生成 (RAG)** 和**强化学习 (RL)** 方法。项目集成了多个先进技术，包括 Self-RAG、AERR 决策代理、本地模型推理和 API 调用等。

### 核心特性

- **多模型支持**：支持本地模型（Qwen、Llama 等）和 API 模型（GPT 系列）
- **RAG 架构**：实现了多种 RAG 方案（Self-RAG、AERR 等）
- **强化学习训练**：基于 PPO 的模型微调和奖励优化
- **Web UI**：提供基于 Gradio 的对话界面
- **数据处理**：完整的数据预处理、生成和评估管道

## 项目结构

```
f:\Research\LLMs\
├── README.md                 # 本文件
├── main.py                  # 项目主入口
├── src/                     # 源代码目录
│   ├── llm/                 # LLM 接口和本地模型
│   ├── webui/               # Web 用户界面
│   ├── train/               # 训练框架（PPO、RL）
│   ├── indexer/             # 索引和嵌入
│   ├── Template/            # 模板类和处理器
│   ├── Forest/              # 对话树和数据集结构
│   ├── RAG_Modules/         # RAG 相关模块（Self-RAG、AERR）
│   ├── self_rag_main/       # Self-RAG 参考实现
│   ├── large_language_model.py    # LLM 核心类
│   ├── RAG_modules.py            # RAG 管道
│   ├── dataset.py                # 数据集定义
│   ├── dataset_evaluation.py     # 评估指标
│   ├── reward.py                 # 奖励计算
│   ├── config.py                 # 配置类
│   └── train.py                  # 训练脚本
├── docs/                    # 文档目录
│   ├── llm/                 # LLM 模块文档
│   ├── webui/               # Web UI 文档
│   ├── train/               # 训练框架文档
│   ├── indexer/             # 索引文档
│   ├── RAG_Modules/         # RAG 文档
│   ├── dataset/             # 数据集文档
│   └── Template/            # 模板文档
└── requirements.txt         # 项目依赖
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 Web UI

启动基于 Gradio 的对话界面：

```bash
python -m src.webui.demo
```

访问 `http://localhost:7860` 即可使用对话界面。

### 本地模型推理

```python
from src.large_language_model import Large_Language_Model

llm = Large_Language_Model(local_dir='./qwen2.5_1.5B/')
llm.init_llm("你是一个有用的AI助手")
result = llm.generate("What is RAG?")
print(result)
```

### API 模型调用

```python
from src.llm.api_llm import Large_Language_Model_API

llm = Large_Language_Model_API(
    api_key="your-api-key",
    base_url="https://api.openai-proxy.org/v1",
    model="gpt-4o-mini"
)
llm.init_llm("你是一个专业的技术助手")
result = llm.generate("Explain retrieval augmented generation")
print(result)
```

## 核心模块说明

### 1. LLM 模块 (`src/llm/`)
- **local_llm.py**: 本地模型推理接口
- **api_llm.py**: API 模型调用接口
- **kv_cache_manager.py**: KV 缓存管理

详见 [LLM 模块文档](docs/llm/README.md)

### 2. Web UI 模块 (`src/webui/`)
- **demo.py**: 主对话界面
- **message_proposser.py**: 消息处理器
- **rag_demo.py**: RAG 演示界面

详见 [Web UI 模块文档](docs/webui/README.md)

### 3. 训练框架 (`src/train/`)
- **trainer.py**: 模型训练器
- **workflow.py**: 训练工作流
- **ppo_utils.py**: PPO 算法工具

详见 [训练模块文档](docs/train/README.md)

### 4. RAG 模块 (`src/RAG_Modules/`)
- **Self-RAG**: 自适应检索增强生成
- **AERR**: 动态决策的 RAG 管道

详见 [RAG 模块文档](docs/RAG_Modules/README.md)

### 5. 数据集模块 (`src/`)
- **dataset.py**: 数据集定义与构建
- **dataset_evaluation.py**: 评估指标
- **reward.py**: 奖励计算

详见 [数据集模块文档](docs/dataset/README.md)

### 6. 索引模块 (`src/indexer/`)
- **indexer.py**: 索引管理
- **embedder.py**: 嵌入模型

详见 [索引模块文档](docs/indexer/README.md)

### 7. 模板模块 (`src/Template/`)
- **llm_template.py**: LLM 提示模板
- **AERRTemplate.py**: AERR 专用模板

详见 [模板模块文档](docs/Template/README.md)

## 关键概念

### 检索增强生成 (RAG)
结合信息检索和神经文本生成，改进 LLM 的事实性和可溯源性。

### Self-RAG
动态判断何时需要检索信息，并评估检索的相关性和生成文本的质量。

### AERR (Agent-Enhanced Retrieval Reasoning)
基于决策代理的 RAG 框架，支持复杂的推理和多步骤检索。

### 强化学习微调
使用 PPO 等 RL 算法，基于人工反馈或自动奖励信号优化模型。

## 配置管理

项目使用 dataclass 进行配置管理，详见 `src/config.py`：

```python
from src.config import MyTrainConfig

config = MyTrainConfig()
config.model_path = "path/to/model"
config.batch_size = 32
```

## 常见问题

### Q: 如何使用自己的数据？
A: 查看 `src/dataset.py` 中的数据集定义，继承 `ConversationTree` 类并实现自己的数据加载逻辑。

### Q: 如何训练模型？
A: 使用 `src/train.py` 中的 `Trainer` 类，参考 `main.py` 中的示例。

### Q: 如何添加新的 RAG 策略？
A: 在 `src/RAG_Modules/` 中继承现有的 RAG 类，实现自己的检索和生成逻辑。

### Q: 如何配置 API 密钥？
A: 通过环境变量或配置文件（不要硬编码在源文件中）传入 API 密钥。

## 贡献指南

- 遵循现有的代码结构和命名规范
- 为新功能添加相应的文档
- 提交 PR 前请确保代码通过 lint 检查
- 不要提交包含个人密钥或敏感信息的代码

## 许可证

根据项目需要指定许可证

## 相关资源

- [Self-RAG 论文](https://arxiv.org/abs/2310.11511)
- [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b)
- [Qwen 模型](https://huggingface.co/Qwen)
- [OpenAI API 文档](https://platform.openai.com/docs)

## 联系方式

如有问题，请通过以下方式联系：
- 提交 Issue
- 发送 PR
