# RAG 模块文档

## 概述

`src/RAG_Modules/` 包含多种先进的**检索增强生成 (RAG)** 实现，包括 **Self-RAG** 和 **AERR** 等方案，用于增强 LLM 的事实性和可溯源性。

## 模块结构

```
src/RAG_Modules/
├── selfrag/          # Self-RAG 实现
│   ├── retrieval_lm/ # 检索语言模型
│   └── data_creation/# 数据生成和处理
├── AERR/             # AERR 决策代理
│   └── pipeline.py   # AERR 管道
└── README.md
```

## RAG 基本原理

### 什么是 RAG？

检索增强生成将外部知识与神经生成结合：

```
用户查询
    ↓
检索相关文档
    ↓
构建增强上下文
    ↓
LLM 生成回复
    ↓
返回答案
```

### 核心优势

- **事实性**：使用真实数据库信息
- **时效性**：支持外部数据实时更新
- **可溯源性**：答案可追踪到源文档
- **降低幻觉**：减少模型的虚假生成

## RAG 架构

### 1. Self-RAG

自适应检索增强生成，动态判断何时需要检索。

**核心特性：**

- **按需检索**：根据任务判断是否需要检索
- **质量评估**：评估检索和生成的质量
- **自我反思**：模型评估自己的输出
- **迭代改进**：根据反馈调整检索策略

**架构：**

```
输入查询
    ↓
[RETRIEVE] 判断是否需要检索
    ↓
若需要 → 检索文档
    ↓
[RELEVANT] 评估检索的相关性
    ↓
生成候选答案
    ↓
[SUPPORT] 评估答案支持度
    ↓
[UTILITY] 评估答案有用性
    ↓
最终答案
```

### 2. AERR (Agent-Enhanced Retrieval Reasoning)

基于决策代理的 RAG 框架。

**核心特性：**

- **代理决策**：智能决策代理选择检索策略
- **多步推理**：支持复杂的多步骤检索和推理
- **动态规划**：根据中间结果调整后续步骤
- **上下文管理**：有效管理对话历史

**决策流程：**

```
当前状态
    ↓
代理分析
    ↓
选择行动：
  - 检索
  - 生成
  - 推理
  - 结束
    ↓
执行行动
    ↓
更新状态
    ↓
判断是否完成
```

## 核心类

### Self-RAG 实现

位置：`src/RAG_Modules/selfrag/`

**主要类：**

- `SelfRAGPipeline`: 主管道类
- `RetrievalLM`: 检索语言模型
- `CriticModule`: 评估模块

**使用示例：**

```python
from src.RAG_Modules.selfrag import SelfRAGPipeline

# 初始化 Self-RAG
rag = SelfRAGPipeline(
    model_name="selfrag-model",
    retriever_path="path/to/retriever",
    corpus_path="path/to/corpus"
)

# 执行 RAG
query = "What is the capital of France?"
result = rag.generate(query)

print(result)
# {
#     "answer": "Paris",
#     "source": "retrieved_doc.txt",
#     "confidence": 0.95
# }
```

### AERR 实现

位置：`src/RAG_Modules/AERR/pipeline.py`

**主要类：**

- `AERRPipeline`: 主管道
- `DecisionAgent`: 决策代理
- `ContextManager`: 上下文管理

**使用示例：**

```python
from src.RAG_Modules.AERR.pipeline import AERRPipeline

# 初始化 AERR
aerr = AERRPipeline(
    model_path="path/to/model",
    retriever_path="path/to/retriever",
    max_steps=5
)

# 执行多步推理
query = "Compare the capital of France and Germany"
result = aerr.run(query)

print(result)
# {
#     "reasoning_steps": [...],
#     "final_answer": "...",
#     "retrieved_docs": [...],
#     "confidence": 0.88
# }
```

## 检索系统

### 向量检索

使用嵌入和相似度搜索：

```python
from src.indexer.embedder import Embedder
from src.indexer.indexer import Indexer

# 创建嵌入器
embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 创建索引
indexer = Indexer(embedder=embedder)

# 索引文档
docs = ["Paris is the capital of France", "Berlin is the capital of Germany"]
doc_ids = indexer.index(docs)

# 检索相关文档
query = "What is the capital of France?"
top_k = indexer.retrieve(query, top_k=5)

print(top_k)
# [
#     {"id": 0, "text": "Paris is...", "score": 0.92},
#     {"id": 1, "text": "Berlin is...", "score": 0.45}
# ]
```

### BM25 检索

基于词频的检索方法：

```python
from src.RAG_Modules.selfrag.retrieval_lm.src.beir_utils import BM25Retriever

bm25 = BM25Retriever(corpus)
results = bm25.retrieve(query, top_k=10)
```

## 数据创建流程

### 1. 数据生成

使用 LLM 生成合成数据：

```python
from src.RAG_Modules.selfrag.data_creation.generator import DataGenerator

generator = DataGenerator(llm_model="gpt-3.5-turbo")

# 生成检索数据
data = generator.generate_retrieval_data(
    num_samples=1000,
    topics=["science", "history", "technology"]
)
```

### 2. 质量评估

使用 GPT-4 或其他评估模型：

```python
from src.RAG_Modules.selfrag.data_creation.critic import CriticModule

critic = CriticModule(model="gpt-4")

# 评估相关性
relevance = critic.evaluate_relevance(
    query="What is AI?",
    document="Artificial Intelligence is..."
)

# 评估支持度
support = critic.evaluate_support(
    answer="Paris is the capital of France",
    document="Paris, the capital of France, is..."
)
```

### 3. 数据后处理

整理和验证数据：

```python
from src.RAG_Modules.selfrag.data_creation.generator.postprocess_data import postprocess

# 清理和验证
clean_data = postprocess(
    raw_data=generated_data,
    remove_duplicates=True,
    validate=True
)

# 保存
import json
with open("training_data.jsonl", "w") as f:
    for item in clean_data:
        f.write(json.dumps(item) + "\n")
```

## 评估指标

### RAGAS 评估框架

```python
from src.RAG_Modules.selfrag.retrieval_lm.src.evaluation import evaluate_ragas

metrics = evaluate_ragas(
    predictions=rag_outputs,
    references=ground_truth,
    documents=retrieved_docs
)

print(metrics)
# {
#     "faithfulness": 0.85,
#     "answer_relevancy": 0.92,
#     "context_recall": 0.78,
#     "context_precision": 0.88
# }
```

### 评估维度

| 指标 | 说明 | 范围 |
|------|------|------|
| Faithfulness | 答案对检索文档的忠实度 | 0-1 |
| Answer Relevancy | 答案与查询的相关度 | 0-1 |
| Context Recall | 检索文档包含答案信息的比例 | 0-1 |
| Context Precision | 检索文档的相关性精度 | 0-1 |

## 最佳实践

### 1. 检索质量优化

```python
# 优化检索参数
rag = SelfRAGPipeline(
    top_k=5,              # 检索文档数
    threshold=0.7,        # 相似度阈值
    use_reranking=True    # 使用重排序
)
```

### 2. 上下文窗口管理

```python
# 控制上下文长度
context_limit = 2000  # token 数
retrieved_docs = truncate_documents(
    docs=retrieved_docs,
    max_length=context_limit
)
```

### 3. 多步推理

```python
# AERR 多步推理
aerr = AERRPipeline(
    max_steps=5,
    step_timeout=10,      # 单步超时时间
    use_cache=True        # 缓存中间结果
)
```

### 4. 混合检索

```python
# 结合向量检索和 BM25
results_vector = vector_retriever.retrieve(query, top_k=10)
results_bm25 = bm25_retriever.retrieve(query, top_k=10)

# 融合结果
merged_results = merge_and_rerank(
    results_vector,
    results_bm25,
    weights=[0.6, 0.4]
)
```

## 常见问题

### Q: 检索的文档太多导致 token 溢出怎么办？
A: 使用重排序器（reranker）筛选最相关的 top-k 文档。

### Q: 如何处理检索到的噪声文档？
A: 设置相似度阈值，过滤低分文档；使用质量评估器验证。

### Q: Self-RAG 和 AERR 的区别是什么？
A: Self-RAG 关注**何时**检索和**质量评估**；AERR 关注**多步决策**和**复杂推理**。

### Q: 如何加速检索速度？
A: 使用向量数据库（Faiss、Milvus）、添加索引、使用 GPU 加速。

### Q: 能否离线运行 RAG 系统？
A: 可以，将检索库离线索引，使用本地模型进行生成。

## 参考资源

- [Self-RAG 论文](https://arxiv.org/abs/2310.11511)
- [RAG 论文](https://arxiv.org/abs/2005.11401)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/qa_structured_sources/integrations/openai_embeddings)
- [Hugging Face RAGAS](https://github.com/explodinggradients/ragas)
