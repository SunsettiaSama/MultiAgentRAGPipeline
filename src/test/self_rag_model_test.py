from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc

# ===================== 1. 基础配置 =====================
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
LORA_PATH = "/root/autodl-tmp/AfterTraining/SelfRAG/checkpoint-5000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Come up with a question and reasoning that would justify this answer: no"


# ===================== 2. 加载 Tokenizer（Qwen3 需指定 trust_remote_code） =====================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,  # Qwen3 必须开启
    padding_side="left"     # 避免生成时 attention_mask 错位
)
# Qwen3 默认无 pad_token，手动设置（与 eos_token 一致）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================== 3. 加载基础模型（指定 GPU + 精度） =====================
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    trust_remote_code = True, 
    torch_dtype = torch.bfloat16, 
    device_map = DEVICE, 
    low_cpu_mem_usage = True, 
)
base_model.eval()

inputs = tokenizer(
    PROMPT, 
    return_tensors = 'pt', 
    padding = True, 
    max_length = 512, 
).to(DEVICE)

generate_kwargs = {
    "max_new_tokens": 100, 
    "temperature": 0.7, 
    "top_p": 0.95, 
    "pad_token_id": tokenizer.pad_token_id, 
    "eos_token_id": tokenizer.eos_token_id, 
    "do_sample": True, 
    "repetition_penalty": 1.1, 
}

with torch.no_grad():
    outputs_no_lora = base_model.generate(**inputs, **generate_kwargs)
text_no_lora = tokenizer.decode(outputs_no_lora[0], skip_special_tokens = False)
print(f"=" * 80)
print("No Lora Output: ", {text_no_lora})

model_with_lora = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    device_map=DEVICE
)
model_with_lora.eval()
with torch.no_grad():
    outputs_with_lora = model_with_lora.generate(**inputs, **generate_kwargs)

print(f"=" * 80)
print("Lora Output: ", {tokenizer.decode(outputs_with_lora[0], skip_special_tokens=False)})
