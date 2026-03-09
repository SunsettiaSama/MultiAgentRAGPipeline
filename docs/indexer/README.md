# 索引模块文档

## 概述

`src/indexer/` 模块提供了文本检索和向量索引的功能，支持快速相似度搜索和信息检索。

## 模块结构

```
src/indexer/
├── indexer.py      # 索引管理
├── embedder.py     # 嵌入模型
└── __init__.py
```

## 核心类

### Embedder（嵌入器）

将文本转换为向量表示。

**主要方法：**

- `__init__(model_name)`: 初始化嵌入模型
- `encode(texts)`: 编码文本为向量
- `encode_batch(texts, batch_size)`: 批量编码
- `get_dimension()`: 获取向量维度

**使用示例：**

```python
from src.indexer.embedder import Embedder

# 初始化嵌入器
embedder = Embedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 编码单个文本
text = "Paris is the capital of France"
embedding = embedder.encode(text)
print(embedding.shape)  # (384,)

# 批量编码
texts = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "London is the capital of England"
]
embeddings = embedder.encode_batch(texts, batch_size=2)
print(embeddings.shape)  # (3, 384)

# 获取维度
dim = embedder.get_dimension()
print(dim)  # 384
```

**支持的模型：**

- `sentence-transformers/all-MiniLM-L6-v2`: 轻量级，384维
- `sentence-transformers/all-mpnet-base-v2`: 高质量，768维
- `sentence-transformers/paraphrase-MiniLM-L12-v2`: 释义识别
- 自定义模型路径

### Indexer（索引器）

管理文档索引和检索。

**主要方法：**

- `__init__(embedder)`: 初始化索引器
- `index(documents)`: 索引文档
- `retrieve(query, top_k)`: 检索相关文档
- `batch_retrieve(queries, top_k)`: 批量检索
- `save(path)`: 保存索引
- `load(path)`: 加载索引

**使用示例：**

```python
from src.indexer.indexer import Indexer
from src.indexer.embedder import Embedder

# 初始化
embedder = Embedder(model_name="all-MiniLM-L6-v2")
indexer = Indexer(embedder=embedder)

# 索引文档
documents = [
    {"id": "doc_1", "text": "Paris is the capital of France"},
    {"id": "doc_2", "text": "Berlin is the capital of Germany"},
    {"id": "doc_3", "text": "Madrid is the capital of Spain"}
]
indexer.index(documents)

# 检索
query = "What is the capital of France?"
results = indexer.retrieve(query, top_k=2)

print(results)
# [
#     {"id": "doc_1", "text": "Paris is...", "score": 0.92},
#     {"id": "doc_3", "text": "Madrid is...", "score": 0.35}
# ]

# 批量检索
queries = [
    "capital of France",
    "capital of Germany"
]
batch_results = indexer.batch_retrieve(queries, top_k=1)

# 保存和加载索引
indexer.save("path/to/index")
indexer = Indexer.load("path/to/index", embedder)
```

## 向量数据库集成

### Faiss 支持

使用 Facebook 的 Faiss 库进行高效搜索：

```python
import faiss
from src.indexer.indexer import FaissIndexer

# 创建 Faiss 索引
faiss_indexer = FaissIndexer(
    dimension=384,
    index_type="IVF",  # Inverted File Index
    nlist=100
)

# 添加向量
vectors = embedder.encode_batch(documents)
faiss_indexer.add(vectors, ids=doc_ids)

# 检索
query_vector = embedder.encode(query)
top_k_ids, scores = faiss_indexer.search(query_vector, top_k=5)
```

### Milvus 支持

使用 Milvus 云向量数据库：

```python
from src.indexer.indexer import MilvusIndexer

milvus_indexer = MilvusIndexer(
    host="localhost",
    port=19530,
    collection_name="documents"
)

# 插入向量
milvus_indexer.insert(vectors, ids)

# 搜索
results = milvus_indexer.search(query_vector, top_k=10)
```

## 相似度计算

### 余弦相似度

```python
from src.indexer.indexer import cosine_similarity

# 单个查询
query_emb = embedder.encode(query)
doc_emb = embedder.encode(doc)
similarity = cosine_similarity(query_emb, doc_emb)

# 批量计算
query_embs = embedder.encode_batch(queries)
doc_embs = embedder.encode_batch(documents)
similarities = cosine_similarity(query_embs, doc_embs)
# 形状: (len(queries), len(documents))
```

## 高级特性

### 重排序器 (Reranker)

使用交叉编码器重新排序检索结果：

```python
from src.indexer.indexer import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# 初步检索（可能得到 100 个结果）
initial_results = indexer.retrieve(query, top_k=100)

# 重排序（得到更精确的 top-k）
final_results = reranker.rerank(
    query=query,
    candidates=initial_results,
    top_k=5
)
```

### 混合检索

结合向量检索和词频检索：

```python
from src.indexer.indexer import HybridRetriever

hybrid = HybridRetriever(
    vector_retriever=indexer,
    bm25_retriever=bm25_indexer,
    weights=[0.6, 0.4]  # 向量权重 0.6，BM25 权重 0.4
)

results = hybrid.retrieve(query, top_k=10)
```

## 性能优化

### 批量索引

```python
# 优化大型数据集的索引
indexer.batch_index(
    documents=large_document_list,
    batch_size=1000
)
```

### 缓存

```python
# 启用缓存以加速重复查询
indexer.enable_cache(max_size=1000)

# 清除缓存
indexer.clear_cache()
```

### GPU 加速

```python
# 使用 GPU 进行向量计算
embedder = Embedder(
    model_name="all-MiniLM-L6-v2",
    device="cuda:0"
)
```

## 常见问题

### Q: 如何处理超大规模文档集（百万级）？
A: 使用 Milvus 或 Elasticsearch，分片索引，定期更新。

### Q: 检索速度太慢怎么办？
A: 使用 Faiss 的 GPU 加速、减少文本长度、使用轻量级模型。

### Q: 如何更新已索引的文档？
A: 删除旧文档 ID，重新索引新版本。

### Q: 嵌入向量的维度如何选择？
A: 权衡准确性（高维更好）和速度（低维更快），通常 256-768。

## 最佳实践

1. **选择合适的嵌入模型**：根据任务和资源选择
2. **定期索引更新**：保持索引与源数据同步
3. **监控检索质量**：使用 MRR、NDCG 等指标
4. **缓存热点查询**：加速频繁查询
5. **定期维护**：清理孤立文档，优化索引
