# 模板模块文档

## 概述

`src/Template/` 模块提供了提示工程和模板管理工具，用于构建和管理 LLM 的输入提示。

## 模块结构

```
src/Template/
├── llm_template.py         # LLM 提示模板
├── AERRTemplate.py         # AERR 专用模板
├── TensorTagProcessor.py   # 张量标签处理
└── __init__.py
```

## 核心概念

### 提示模板

结构化的提示词设计，包含变量占位符，支持动态填充。

**优势：**

- **一致性**：确保提示格式一致
- **可重用性**：模板可复用于多个任务
- **可维护性**：集中管理提示逻辑
- **A/B 测试**：容易进行多版本对比

## LLM 通用模板

### LLMTemplate 类

```python
from src.Template.llm_template import LLMTemplate

# 创建模板
template = LLMTemplate()

# 定义系统提示
template.set_system_prompt(
    "你是一个专业的技术顾问，善于解释复杂概念。"
)

# 定义任务提示
template.add_task("qa", """
用户问题：{question}
相关背景：{context}

请基于背景信息回答问题。如果背景不足，请说明。
""")

# 使用模板
result = template.render_task(
    "qa",
    question="什么是 RAG？",
    context="检索增强生成是一种将外部知识..."
)

print(result)
```

### 预定义任务模板

| 任务 | 说明 | 用途 |
|------|------|------|
| qa | 问答 | 一般问答系统 |
| summarize | 摘要 | 文本摘要生成 |
| translate | 翻译 | 多语言翻译 |
| code | 代码生成 | 编程任务 |
| creative | 创意写作 | 故事、诗歌生成 |

**使用预定义模板：**

```python
# 问答
qa_result = template.render_task(
    "qa",
    question="What is AI?",
    context="Artificial Intelligence is..."
)

# 摘要
summary = template.render_task(
    "summarize",
    text="Long article text...",
    max_length=100
)

# 翻译
translation = template.render_task(
    "translate",
    text="Hello world",
    source_lang="English",
    target_lang="Chinese"
)
```

## AERR 模板

针对 AERR 决策代理的专用模板。

### AERRTemplate 类

```python
from src.Template.AERRTemplate import AERRTemplate

# 创建 AERR 模板
aerr_template = AERRTemplate()

# 设置推理步骤
aerr_template.set_reasoning_steps([
    "analyze",    # 分析问题
    "retrieve",   # 检索相关信息
    "reason",     # 推理
    "conclude"    # 得出结论
])

# 渲染推理过程
result = aerr_template.render_reasoning(
    question="Compare machine learning and deep learning",
    current_step="analyze",
    context={
        "ml_def": "Machine learning is...",
        "dl_def": "Deep learning is a subset..."
    }
)

print(result)
```

### AERR 特定提示

**步骤 1：分析**

```python
analyze_prompt = aerr_template.get_step_prompt("analyze")
# 输出：请分析问题的关键点，确定需要哪些信息...
```

**步骤 2：检索**

```python
retrieve_prompt = aerr_template.get_step_prompt("retrieve")
# 输出：基于分析，需要检索以下信息...
```

**步骤 3：推理**

```python
reason_prompt = aerr_template.get_step_prompt("reason")
# 输出：综合检索到的信息，进行推理...
```

**步骤 4：结论**

```python
conclude_prompt = aerr_template.get_step_prompt("conclude")
# 输出：基于推理过程，得出最终结论...
```

## 张量标签处理

### TensorTagProcessor

处理在张量中嵌入标签信息。

**主要方法：**

- `add_tag(tensor, tag)`: 添加标签
- `get_tag(tensor)`: 获取标签
- `process_batch(tensors, tags)`: 批量处理

**使用示例：**

```python
from src.Template.TensorTagProcessor import TensorTagProcessor

processor = TensorTagProcessor()

# 创建张量
embeddings = torch.randn(32, 768)

# 添加标签
tagged = processor.add_tag(embeddings, "qa_response")

# 在后续处理中获取标签
tag = processor.get_tag(tagged)
print(tag)  # "qa_response"

# 批量处理
batch_embeddings = torch.randn(100, 768)
batch_tags = ["qa", "summarize", "translate"] * 33 + ["qa"]
processed = processor.process_batch(batch_embeddings, batch_tags)
```

## 提示工程最佳实践

### 1. 角色定位

```python
system_prompt = """
你是一个资深的机器学习工程师，拥有 10 年的经验。
你的回答应该：
- 准确且专业
- 包含实际代码示例
- 考虑生产环境的最佳实践
"""

template.set_system_prompt(system_prompt)
```

### 2. 上下文输入

```python
task_template = """
任务：{task_description}

输入数据：
{input_data}

要求：
- {requirement_1}
- {requirement_2}

输出格式：
{output_format}
"""
```

### 3. 少样本学习（Few-Shot）

```python
few_shot_template = """
你是一个分类器。这是一些例子：

例子 1：
输入：This movie is amazing!
分类：正面

例子 2：
输入：I hate this terrible film
分类：负面

现在分类这个新输入：
输入：{new_text}
分类：
"""
```

### 4. 链式思考（Chain-of-Thought）

```python
cot_template = """
让我们逐步解决这个问题：

问题：{question}

步骤 1：理解问题
- {understanding}

步骤 2：分析关键信息
- {key_info}

步骤 3：推理过程
- {reasoning}

步骤 4：得出答案
答案：{answer}
"""
```

## 动态提示生成

### 模板参数化

```python
# 定义参数化模板
template = LLMTemplate()
template.add_task("adaptive", """
你是一个{role}。
难度等级：{difficulty}
任务：{task}

请以{tone}的语气完成任务。
""")

# 根据用户特性动态渲染
result = template.render_task(
    "adaptive",
    role="教师",
    difficulty="高中",
    task="解释相对论",
    tone="激励性"
)
```

### 条件模板

```python
# 根据条件选择不同模板
if is_technical_user:
    template_name = "technical_qa"
else:
    template_name = "simple_qa"

result = template.render_task(template_name, ...)
```

## 提示模板库

### 常用模板示例

**翻译模板：**

```python
template.add_task("translate", """
将以下文本从 {source_lang} 翻译到 {target_lang}。
保持原意和风格。

原文：
{source_text}

翻译：
""")
```

**代码审查模板：**

```python
template.add_task("code_review", """
请审查以下代码片段。

代码：
```
{code}
```

请指出：
1. 任何潜在的 bug
2. 性能改进建议
3. 代码风格改进
4. 整体评分（1-10）
""")
```

**客服回复模板：**

```python
template.add_task("customer_service", """
客户问题：{question}
客户满意度级别：{satisfaction_level}

请提供一个{tone}的回复。

注意：
- 解决客户的具体问题
- 表示同情和理解
- 提供下一步行动
""")
```

## 性能和优化

### 模板缓存

```python
# 启用缓存以加速重复渲染
template.enable_cache()

# 多次调用使用缓存
for i in range(1000):
    result = template.render_task("qa", ...)  # 使用缓存
```

### 批量渲染

```python
# 批量渲染多个提示
prompts = template.batch_render_task(
    "qa",
    questions=["Q1", "Q2", "Q3"],
    contexts=["C1", "C2", "C3"]
)
```

## 常见问题

### Q: 如何创建自定义任务模板？
A: 使用 `template.add_task()` 定义模板字符串，包含 `{variable}` 占位符。

### Q: 如何处理多语言模板？
A: 为每种语言定义不同的模板，根据语言参数选择。

### Q: 如何评估模板效果？
A: 使用相同模板测试多个模型，比较输出质量。

### Q: 能否动态生成提示词？
A: 可以，使用条件逻辑和参数化动态构建提示。

## 最佳实践

1. **模板版本管理**：记录模板变更历史
2. **A/B 测试**：对比不同模板版本的效果
3. **提示优化**：迭代改进提示以获得更好结果
4. **文档完善**：记录每个模板的用途和参数
5. **错误处理**：处理模板缺失或参数错误

## 参考资源

- [提示工程指南](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chain-of-Thought 论文](https://arxiv.org/abs/2201.11903)
- [Few-Shot Learning](https://arxiv.org/abs/2005.14165)
