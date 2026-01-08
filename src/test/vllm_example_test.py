



# 第一步：强制设置多进程启动方式（必须放在最开头）
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# 第二步：导入依赖（放在启动方式设置后）
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import torch

# 第三步：所有主逻辑必须放在 if __name__ == '__main__' 中！
if __name__ == '__main__':
    # 验证 CUDA 环境（可选）
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"显卡算力: {torch.cuda.get_device_capability(0)}")

    # 初始化 LLM（核心修复：添加 local_files_only + trust_remote_code + 显式 seed）
    llm = LLM(
        model="/root/autodl-tmp/Qwen3-8B",  # 本地模型路径
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        # Qwen3 必须开启（模型含自定义代码）
        trust_remote_code=True,
        # 修复 seed 警告（显式设置）
        seed=0,
        enable_lora = True, 
    )

    lora_request = LoRARequest(lora_name = 'test', lora_path = "/root/autodl-tmp/AfterTraining/SelfRAG/checkpoint-5000", lora_int_id = 1)

    # 测试推理（示例）
    sampling_params = SamplingParams(temperature=0.7, 
                                     max_tokens=2000, 
                                     top_p = 0.95, 
                                     repetition_penalty = 1.1, 
                                    )
    prompts = ["Come up with a question and reasoning that would justify this answer: no"]
    outputs = llm.generate(prompts, sampling_params, 
                           lora_request = lora_request
                           )

    # 输出结果
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"生成结果: {output.outputs[0].text}")

