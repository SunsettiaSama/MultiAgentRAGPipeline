import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from peft import PeftModel

# 1. 加载模型和分词器
model_name = "/root/autodl-tmp/Qwen3-8B"
peft_model_path_adapter = "/root/autodl-tmp/AfterTraining/SelfRAG/checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, peft_model_path_adapter)

# --- 手动缓存管理开始 ---

# 2. 准备初始输入
prompt = "My name is Qwen, and I am a"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# 3. 【第一步】初始化一个空的 DynamicCache
# 这个 cache_obj 将在整个循环中被我们手动更新
cache_obj = DynamicCache()

# 4. 【第一次调用】用初始 prompt 填充缓存
# 注意：use_cache=True 是必须的
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=cache_obj,
    use_cache=True
)

# 5. 【关键】从输出中获取更新后的缓存
# 模型处理完初始输入后，返回了一个包含了 "My name is Qwen, and I am a" 信息的缓存
# 我们必须用这个新的缓存来替换掉旧的空缓存
cache_obj = outputs.past_key_values

# 6. 手动循环生成新 token
generated_tokens = []
num_new_tokens_to_generate = 5

for _ in range(num_new_tokens_to_generate):
    # a. 从上次的输出中获取最后一个 token 的 ID
    # 这是我们要继续生成的 "新" token
    next_token_id = outputs.logits.argmax(dim=-1)[:, -1:]
    
    generated_tokens.append(next_token_id.item())

    # b. 【核心】再次调用模型
    # - input_ids: 只传入新生成的那个 token
    # - past_key_values: 传入上一轮更新后的完整缓存
    # 模型会自动将新 token 的 KV 追加到缓存中
    outputs = model(
        input_ids=next_token_id,
        past_key_values=cache_obj,  # 使用更新后的缓存
        use_cache=True
    )
    
    # c. 【再次关键】更新缓存！
    # 模型返回了包含历史信息 + 新 token 信息的更新缓存
    # 我们必须用它来更新我们的缓存对象，为下一次循环做准备
    cache_obj = outputs.past_key_values

# 7. 组合并解码最终结果
full_text_ids = torch.cat([input_ids, torch.tensor(generated_tokens).unsqueeze(0)], dim=-1)
print(tokenizer.decode(full_text_ids[0], skip_special_tokens=True))
# 预期输出: "My name is Qwen, and I am a large language model"


pass
