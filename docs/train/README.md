# 训练模块文档

## 概述

`src/train/` 模块提供了完整的模型训练框架，基于 **PPO (Proximal Policy Optimization)** 和强化学习算法，用于微调大语言模型。

## 模块结构

```
src/train/
├── trainer.py          # 训练器主类
├── workflow.py         # 训练工作流管理
├── ppo_utils.py        # PPO 工具函数
└── __init__.py
```

## 核心概念

### PPO (Proximal Policy Optimization)

一种政策梯度算法，通过以下方式优化模型：

1. **收集轨迹**：模型生成样本
2. **计算奖励**：评估样本质量
3. **优化策略**：使用奖励信号更新模型
4. **剪裁目标函数**：防止过度更新

### 强化学习微调流程

```
初始模型
    ↓
生成候选样本
    ↓
计算奖励信号
    ↓
PPO 优化
    ↓
评估性能
    ↓
保存检查点
```

## 核心类

### Trainer

主训练器类，管理整个训练流程。

**主要方法：**

- `__init__(config)`: 初始化训练器
- `train()`: 启动训练
- `warm_start()`: 预热启动
- `train_step()`: 单步训练
- `evaluate()`: 评估模型

**使用示例：**

```python
from src.train import Trainer
from src.config import MyTrainConfig

# 创建配置
config = MyTrainConfig()
config.batch_size = 32
config.num_epochs = 3
config.learning_rate = 1e-5

# 创建训练器
trainer = Trainer(config=config)

# 开始训练
trainer.train()
```

### 训练工作流

工作流管理训练的各个阶段：

```python
from src.train.workflow import TrainingWorkflow

workflow = TrainingWorkflow(config)
workflow.run()
```

**工作流步骤：**

1. 数据加载
2. 模型初始化
3. 参考模型准备（用于 KL 散度）
4. 训练循环：
   - 生成样本
   - 计算奖励
   - 收集轨迹
   - PPO 更新
5. 评估和保存

## 训练配置

通过 `src/config.py` 定义训练参数：

```python
from src.config import MyTrainConfig

config = MyTrainConfig()

# 模型配置
config.model_path = "path/to/model"
config.lora_dir = "path/to/lora"

# 训练参数
config.batch_size = 32
config.num_epochs = 3
config.learning_rate = 1e-5
config.warmup_steps = 100

# PPO 参数
config.ppo_epoch = 4
config.ppo_clip_epsilon = 0.2
config.ppo_gamma = 0.99
config.ppo_lambda = 0.95

# 输出配置
config.output_dir = "./checkpoints"
config.save_steps = 500
```

## PPO 工具函数

### 关键函数

**get_rewards_from_server()**
- 从奖励服务器获取奖励分数
- 支持批量计算
- 处理错误重试

**dump_layernorm() / restore_layernorm()**
- 暂存层归一化权重
- 减少显存占用
- 训练后恢复

**replace_model()**
- 替换模型权重
- 支持 LoRA 权重合并

## 数据准备

### 对话树数据集

```python
from src.dataset import ConversationTree

dataset = ConversationTree(
    raw_data=raw_data,
    max_depth=3,
    max_width=5
)

# 用于训练
data_loader = dataset.build_data_loader(batch_size=32)
```

### 样本格式

```python
{
    "input_ids": [...],           # 模型输入
    "attention_mask": [...],      # 注意力掩码
    "labels": [...],              # 标签（用于 loss 计算）
    "reward": 0.8,                # 奖励信号
    "metadata": {...}             # 额外元数据
}
```

## 奖励计算

通过 `src/reward.py` 计算奖励：

```python
from src.reward import weighted_reward_calculate

# 计算加权奖励
reward = weighted_reward_calculate(
    model_response=response,
    reference=reference,
    weights={
        "relevance": 0.3,
        "coherence": 0.3,
        "factuality": 0.4
    }
)
```

### 奖励维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 相关性 | 0.3 | 回复与问题的相关程度 |
| 连贯性 | 0.3 | 文本的逻辑连贯性 |
| 事实性 | 0.4 | 陈述的事实正确性 |

## 训练循环详解

### Step 1: 数据收集

```python
# 从模型生成样本
samples = model.generate(
    prompts,
    max_length=512,
    num_return_sequences=4
)
```

### Step 2: 奖励计算

```python
# 评估样本质量
rewards = reward_model.score(samples)
```

### Step 3: 轨迹组织

```python
# 组织为 PPO 轨迹
trajectories = [
    {
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "log_prob": log_prob
    }
    for prompt, response, reward, log_prob 
    in zip(prompts, samples, rewards, log_probs)
]
```

### Step 4: PPO 更新

```python
# 计算 advantage 和 return
advantages = compute_advantages(rewards, values, gamma, lambda)
returns = compute_returns(rewards, values, gamma)

# 更新策略
for epoch in range(ppo_epochs):
    loss = ppo_step(
        trajectories,
        advantages,
        returns,
        clip_epsilon=0.2
    )
```

## LoRA 微调

支持使用 **LoRA (Low-Rank Adaptation)** 进行高效微调：

```python
from src.large_language_model import Large_Language_Model

# 创建模型并加载 LoRA
llm = Large_Language_Model(local_dir="model_dir")

# 配置 LoRA
config.lora_dir = "path/to/lora"
llm.reload_lora(config)

# 训练
trainer.train()

# 保存 LoRA 权重
llm.save_model("output_dir")
```

### LoRA 配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # LoRA 秩
    lora_alpha=16,                # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标层
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
```

## 评估指标

### 自动评估

```python
from src.dataset_evaluation import evaluate

metrics = evaluate(
    predictions=model_outputs,
    references=ground_truth,
    metrics=["bleu", "rouge", "meteor"]
)

print(metrics)
# {'bleu': 0.25, 'rouge': 0.45, 'meteor': 0.38}
```

### 支持的指标

- **BLEU**: N-gram 精确匹配
- **ROUGE**: 召回导向的替补
- **METEOR**: 考虑同义词的对齐
- **F1**: 标记级精确度
- **ExactMatch**: 完全匹配率

## 检查点管理

### 保存检查点

```python
trainer.save_checkpoint(
    path="checkpoints/epoch_1",
    include_optimizer=True,
    include_scheduler=True
)
```

### 加载检查点

```python
trainer.load_checkpoint("checkpoints/epoch_1")
trainer.train()  # 继续训练
```

## 分布式训练

支持多 GPU 分布式训练：

```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    -m src.train
```

## 常见问题

### Q: 训练时 GPU 显存溢出怎么办？
A: 
- 减少 `batch_size`
- 启用梯度累积
- 使用 LoRA 而不是全量微调
- 启用 `dump_layernorm()` 优化

### Q: 如何加速训练？
A:
- 增加 `batch_size`
- 使用混合精度训练
- 启用分布式训练
- 减少 `num_epochs`

### Q: 奖励信号不稳定怎么办？
A:
- 调整奖励权重
- 使用奖励规范化
- 增加参考样本数量
- 检查奖励计算逻辑

### Q: 如何进行断点续训？
A: 定期保存检查点，使用 `load_checkpoint()` 加载后继续训练。

## 最佳实践

1. **预热阶段**：使用 `warm_start()` 初始化模型权重
2. **验证集**：定期在验证集上评估，监控过拟合
3. **奖励设计**：认真设计奖励函数，避免短视行为
4. **超参调优**：使用网格搜索或贝叶斯优化调优 PPO 参数
5. **结果记录**：使用 TensorBoard 或 Weights & Biases 记录训练过程

## 参考资源

- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [LoRA 论文](https://arxiv.org/abs/2106.09714)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)
