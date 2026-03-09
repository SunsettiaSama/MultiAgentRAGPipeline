# 数据集模块文档

## 概述

`src/` 中的数据集相关模块提供了完整的数据处理、评估和管理工具，包括对话树结构、数据采样、评估指标等。

## 模块结构

```
src/
├── dataset.py              # 数据集定义和对话树
├── dataset_evaluation.py   # 评估指标和函数
├── reward.py               # 奖励计算
├── config.py               # 配置管理
├── Forest/                 # 对话森林数据结构
│   ├── buffer.py          # 缓存管理
│   ├── difficult_dataset.py # 困难样本集
│   └── __init__.py
└── README.md
```

## 核心数据结构

### 对话树 (ConversationTree)

层级化的对话数据结构，支持多分支对话。

**特性：**

- 树形结构，支持多分支
- 节点代表单个消息/回复
- 边代表对话转移
- 支持对话历史追踪

**数据模型：**

```python
{
    "id": "node_001",
    "parent_id": "node_000",
    "user_message": "How does RAG work?",
    "assistant_message": "RAG combines retrieval...",
    "metadata": {
        "depth": 2,
        "reward": 0.85,
        "timestamp": "2024-01-01T10:00:00",
        "source": "train_data"
    }
}
```

**使用示例：**

```python
from src.dataset import ConversationTree

# 创建对话树
tree = ConversationTree(
    max_depth=10,      # 最大深度
    max_width=100,     # 单节点最大分支数
    tokenizer=tokenizer
)

# 添加对话
tree.add_conversation(
    user_message="What is AI?",
    assistant_message="AI is...",
    parent_id="root"
)

# 获取路径
path = tree.get_path(node_id="node_001")
print(path)
# ["What is AI?", "AI is...", "Tell me more", "AI includes..."]

# 采样子树
subtree = tree.sample_subtree(
    root_id="node_000",
    max_size=100
)
```

### 森林 (Forest)

多个对话树的集合。

```python
from src.dataset import Forest

forest = Forest(
    trees=[tree1, tree2, tree3],
    max_tree_size=1000
)

# 获取统计信息
stats = forest.get_stats()
# {
#     "total_nodes": 5000,
#     "num_trees": 3,
#     "avg_depth": 4.5
# }
```

### 样本采样器 (ModelInteractionSampler)

从对话树中智能采样训练样本。

**采样策略：**

```python
from src.dataset import ModelInteractionSampler

sampler = ModelInteractionSampler(
    tree=conversation_tree,
    strategy="weighted"  # weighted, uniform, difficulty_aware
)

# 采样一个批次
batch = sampler.sample_batch(
    batch_size=32,
    include_meta=True
)

# 返回格式
# {
#     "input_ids": tensor([...]),
#     "attention_mask": tensor([...]),
#     "labels": tensor([...]),
#     "rewards": tensor([...]),
#     "metadata": [...]
# }
```

## 数据处理管道

### 1. 数据加载

```python
# 从文件加载
from src.dataset import load_dataset

dataset = load_dataset(
    data_path="path/to/data.jsonl",
    format="jsonl",
    split="train"
)

# 支持的格式：
# - jsonl: 行 JSON
# - json: 单一 JSON
# - csv: CSV 文件
# - huggingface: HuggingFace Datasets
```

### 2. 数据预处理

```python
from src.dataset import preprocess_dataset

processed = preprocess_dataset(
    dataset=raw_data,
    tokenizer=tokenizer,
    max_length=2048,
    remove_duplicates=True,
    clean_text=True
)
```

### 3. 数据分割

```python
# 按比例分割
train_data, eval_data, test_data = dataset.split(
    train_ratio=0.8,
    eval_ratio=0.1,
    test_ratio=0.1,
    shuffle=True
)
```

## 评估指标

### 自动化评估

```python
from src.dataset_evaluation import evaluate

# 计算多个指标
results = evaluate(
    predictions=model_outputs,
    references=ground_truth,
    metrics=["bleu", "rouge", "meteor", "f1"],
    tokenizer=tokenizer
)

print(results)
# {
#     "bleu": 0.25,
#     "rouge": {"rouge1": 0.45, "rouge2": 0.35, "rougeL": 0.40},
#     "meteor": 0.38,
#     "f1": 0.72
# }
```

### 支持的指标

| 指标 | 说明 | 范围 | 何时使用 |
|------|------|------|---------|
| BLEU | N-gram 精确匹配 | 0-1 | 翻译、摘要 |
| ROUGE | 召回导向的替补 | 0-1 | 摘要、段落 |
| METEOR | 考虑同义词的对齐 | 0-1 | 翻译、释义 |
| F1 | Token 级精确度 | 0-1 | QA、NER |
| ExactMatch | 完全匹配率 | 0-1 | QA |
| BERTScore | 语义相似度 | 0-1 | 通用评估 |

### 自定义评估指标

```python
from src.dataset_evaluation import MetricBase

class CustomMetric(MetricBase):
    def compute(self, predictions, references):
        # 实现自定义逻辑
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.calculate_score(pred, ref)
            scores.append(score)
        return {"custom": sum(scores) / len(scores)}
```

## 奖励计算

### 加权奖励

```python
from src.reward import weighted_reward_calculate

# 计算加权奖励
reward = weighted_reward_calculate(
    model_response="AI is artificial intelligence...",
    reference="AI stands for artificial intelligence...",
    weights={
        "relevance": 0.3,
        "coherence": 0.3,
        "factuality": 0.4,
        "length_penalty": 0.1
    }
)

print(reward)  # 0.85
```

### 奖励维度

| 维度 | 权重 | 计算方式 |
|------|------|---------|
| 相关性 | 0.3 | ROUGE / BERTScore |
| 连贯性 | 0.3 | 困惑度 / 语言模型评分 |
| 事实性 | 0.4 | 与参考文本的一致性 |
| 长度惩罚 | 0.1 | |response_len - ref_len| |

### 自定义奖励函数

```python
def my_reward_function(response, reference, **kwargs):
    """自定义奖励函数"""
    # 实现奖励逻辑
    if response == reference:
        return 1.0
    else:
        # 计算相似度
        similarity = calculate_similarity(response, reference)
        return similarity

# 使用
reward = my_reward_function(
    response="Paris",
    reference="Paris",
    context={"question": "Capital of France"}
)
```

## 困难样本采样

### ConversationTree 的困难样本

```python
from src.Forest.difficult_dataset import DifficultDataset

difficult = DifficultDataset(
    tree=conversation_tree,
    difficulty_threshold=0.3  # 选择奖励 < 0.3 的样本
)

# 采样困难样本
hard_samples = difficult.sample(size=100)
```

### 使用困难样本进行强化学习

```python
# 训练数据混合
easy_data = sampler.sample_batch(size=80)
hard_data = difficult.sample(size=20)

mixed_batch = combine_batches([easy_data, hard_data])
trainer.train_step(mixed_batch)
```

## 数据验证

### 数据质量检查

```python
from src.dataset import validate_dataset

validation_report = validate_dataset(
    dataset=data,
    checks=[
        "no_duplicates",
        "valid_format",
        "reasonable_length",
        "no_null_values"
    ]
)

if validation_report["valid"]:
    print("数据通过验证")
else:
    print(validation_report["errors"])
```

### 数据统计

```python
# 获取数据集统计信息
stats = dataset.get_statistics()

print(stats)
# {
#     "total_samples": 10000,
#     "avg_length": 256,
#     "max_length": 2048,
#     "min_length": 10,
#     "vocabulary_size": 50000,
#     "unique_tokens": 42000
# }
```

## 批量处理

### 数据加载器

```python
from src.dataset import create_data_loader

train_loader = create_data_loader(
    dataset=train_data,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    collate_fn=custom_collate
)

# 遍历批次
for batch in train_loader:
    # batch 包含：input_ids, attention_mask, labels, ...
    outputs = model(batch)
```

### 流式处理

```python
# 处理大型数据集
for batch in dataset.stream_batches(batch_size=64):
    # 处理每个批次
    results = process_batch(batch)
```

## 使用示例

### 完整的训练数据管道

```python
from src.dataset import load_dataset, ConversationTree
from src.dataset_evaluation import evaluate

# 1. 加载数据
raw_data = load_dataset("path/to/data.jsonl")

# 2. 构建对话树
tree = ConversationTree()
for item in raw_data:
    tree.add_conversation(
        user_message=item["user"],
        assistant_message=item["assistant"]
    )

# 3. 采样训练数据
sampler = ModelInteractionSampler(tree)
train_batch = sampler.sample_batch(batch_size=32)

# 4. 模型推理
outputs = model(**train_batch)

# 5. 评估结果
metrics = evaluate(
    predictions=outputs,
    references=train_batch["labels"],
    metrics=["bleu", "f1"]
)

print(f"BLEU: {metrics['bleu']:.4f}")
```

## 常见问题

### Q: 如何处理不平衡的数据集？
A: 使用 `weighted_sampler` 或过采样/欠采样策略。

### Q: 如何加速数据加载？
A: 增加 `num_workers`、使用 `pin_memory=True`、预处理并缓存。

### Q: 如何处理超长文本？
A: 使用截断、滑动窗口或分解为子段。

### Q: 如何验证数据质量？
A: 使用 `validate_dataset()` 和手动审查样本。

## 最佳实践

1. **数据分割**：严格分离训练/验证/测试集
2. **数据验证**：在训练前进行全面的数据检查
3. **均衡采样**：确保各类别样本均衡
4. **缓存机制**：缓存预处理结果加速训练
5. **版本管理**：记录数据版本和处理步骤
