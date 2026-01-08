import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
# 确保你导入的是我们之前重构的 TwoStageKVCacheManager
from lib.llm.kv_cache_manager import KVCacheManager as KVCacheManager
import transformers

# --- 1. 初始化模型、分词器和设备 ---
base_model_name = "/root/autodl-tmp/Qwen3-8B"
peft_model_path_adapter1 = "/root/autodl-tmp/AfterTraining/SelfRAG/checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用 torch_dtype=torch.bfloat16 可以节省显存并可能加速
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)

# 加载PEFT模型
peft_model_1 = PeftModel.from_pretrained(base_model, peft_model_path_adapter1)
# !!! 关键：在推理前必须设置为eval模式 !!!
peft_model_1.eval()

# --- 2. 初始化 KVCacheManager ---
cache_manager = KVCacheManager(tokenizer=tokenizer, device=device)

# --- 3. 预热 System Prompt ---
# 使用更标准的、包含角色标签的prompt格式，这有助于模型理解上下文
system_prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.\n"
    "<|im_end|>"
)
# 使用 base_model 进行预热
cache_manager.warm_up(system_prompt=system_prompt, base_model=base_model)

# --- 4. 准备测试用例 ---
# 为每个用户问题构建完整的对话格式
user_questions = [
    "Tell me a joke.",
    "Explain quantum mechanics in simple terms."
]
full_prompts = []
for question in user_questions:
    full_prompt = (
        f"{system_prompt}\n"
        f"<|im_start|>user\n"
        f"{question}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"  # 模型应该从这里开始生成
    )
    full_prompts.append(full_prompt)


# --- 5. 主推理循环 ---
print(f"\n--- Starting Inference Loop ---")
current_peft_model = peft_model_1
print(f"Using PEFT Model: {current_peft_model.active_adapter}")

for full_prompt in full_prompts:
    print(f"\n=============================================================")
    print(f"Processing Full Prompt:\n{full_prompt}")
    print("-------------------------------------------------------------")
    
    # 阶段一：更新缓存（如果需要）
    cache_manager.update_cache(full_prompt=full_prompt, peft_model=current_peft_model)
    
    # 阶段二：获取缓存
    # 在两阶段设计中，get_cache() 应该总是能成功返回
    past_key_values, _ = cache_manager.get_cache()
    print("Successfully retrieved cache.")

    # 定义生成配置
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.1,
        max_new_tokens=150,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # !!! 重构点：简化的 generate 调用 !!!
    # 当 past_key_values 提供时，input_ids 只需要包含最后一个 token 或为空
    # 模型会自动从 past_key_values 的末尾继续生成
    # 这里我们直接传入一个空的 input_ids，让模型从头开始生成回复
    # 注意：某些模型可能需要一个起始 token，这里我们假设可以无缝衔接
    with torch.no_grad():
        outputs = current_peft_model.generate(
            input_ids=torch.tensor([[tokenizer.pad_token_id]]).to(device),
            attention_mask=torch.tensor([[0]]).to(device), # mask out the padding token
            past_key_values=past_key_values,
            use_cache=True,  # 保持生成过程中的KV Cache更新
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

    # !!! 重构点：正确解析生成结果 !!!
    # outputs.sequences 现在只包含新生成的 token，不包含原始 prompt
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    print(f"     [Generated Response]: {generated_text}")

# --- 6. 清理 ---
cache_manager.clear()
print("\n=============================================================")
print("All done. Cache cleared.")