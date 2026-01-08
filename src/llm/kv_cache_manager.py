import torch
import logging
import hashlib
from typing import Optional, Dict, Any, Tuple
from functools import wraps
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel

logger = logging.getLogger(__name__)

def _log_on_main_process(fn):
    @wraps(fn)
    def wrapper(self, level, msg, *args, **kwargs):
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
        except (AssertionError, RuntimeError):
            pass
        logger.log(level, msg, *args, **kwargs)
    return wrapper

class KVCacheManager:
    """
    采用“两阶段”工作流的 KV Cache 管理器，职责更清晰，性能更稳定。
    
    工作流:
    1. 调用 `warm_up` 预热 system prompt 的 KV Cache。
    2. 在每次推理前，调用 `update_cache(full_prompt, peft_model)`。
       此方法会智能判断是否需要执行增量计算来更新缓存。
    3. 调用 `get_cache()` 获取准备好的 KV Cache 和 Hidden State。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

        # 1. System Prompt 缓存 (由 base_model 生成，静态)
        self._system_kv_cache: Optional[Any] = None
        self._system_prompt_length: int = 0

        # 2. 最终结果缓存 (由 peft_model 生成，动态)
        #    Key: (full_prompt_hash, lora_adapter_name)
        #    Value: (full_kv_cache, final_hidden_state)
        self._final_cache_store: Dict[Tuple[str, str], Tuple[Any, torch.Tensor]] = {}
        
        # 3. 当前活跃的缓存键
        self._current_cache_key: Optional[Tuple[str, str]] = None

        self._log(logging.INFO, "TwoStageKVCacheManager 初始化完成。")

    def _get_prompt_hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def _get_cache_key(self, full_prompt: str, peft_model: PeftModel) -> Tuple[str, str]:
        full_prompt_hash = self._get_prompt_hash(full_prompt)
        lora_adapter_name = peft_model.active_adapter
        return (full_prompt_hash, lora_adapter_name)

    @_log_on_main_process
    def _log(self, level, msg, *args, **kwargs):
        pass

    def warm_up(self, system_prompt: str, base_model: PreTrainedModel) -> None:
        """使用 base_model 预热并存储固定的 system_prompt 的 KV Cache。"""
        if self._system_kv_cache is not None:
            self._log(logging.INFO, "System prompt KV Cache 已存在，跳过预热。")
            return

        self._log(logging.INFO, "正在为 system prompt 生成并存储 KV Cache...")
        inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.device)
        self._system_prompt_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = base_model.forward(**inputs, use_cache=True)
        
        self._system_kv_cache = outputs.past_key_values
        self._log(logging.INFO, "System prompt 的 KV Cache 已成功存储。")

    def update_cache(self, full_prompt: str, peft_model: PeftModel) -> None:
        """
        更新缓存。如果缓存未命中，则执行增量计算并存储结果。
        """
        if self._system_kv_cache is None:
            raise RuntimeError("System prompt KV Cache 未预热。请先调用 warm_up() 方法。")

        cache_key = self._get_cache_key(full_prompt, peft_model)
        
        # 如果缓存已存在，直接返回，无需计算
        if cache_key in self._final_cache_store:
            self._current_cache_key = cache_key
            self._log(logging.INFO, "缓存已命中 (Key: %s)，无需更新。", cache_key)
            return

        # 缓存未命中，执行增量计算
        self._log(logging.INFO, "缓存未命中 (Key: %s)，将执行增量计算。", cache_key)
        
        full_inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        new_tokens = full_inputs['input_ids'][:, self._system_prompt_length:]
        
        try:
            with torch.no_grad():
                outputs = peft_model.forward(
                    input_ids=new_tokens,
                    attention_mask=torch.ones_like(new_tokens),
                    past_key_values=self._system_kv_cache,
                    use_cache=True,
                    output_hidden_states=True
                )
            
            final_kv = outputs.past_key_values
            final_hs = outputs.hidden_states[-1][:, -1:, :]

            # 将完整的计算结果存入缓存
            self._final_cache_store[cache_key] = (final_kv, final_hs)
            self._current_cache_key = cache_key
            self._log(logging.INFO, "增量计算完成，结果已存入缓存。")

        except Exception as e:
            self._log(logging.ERROR, "执行增量计算时发生错误: %s", e)
            raise e # 抛出异常，让上层知道计算失败

    def get_cache(self) -> Tuple[Any, torch.Tensor]:
        """
        获取由 `update_cache` 准备好的最终 KV Cache 和 Hidden State。
        必须在 `update_cache` 之后调用。
        """
        if self._current_cache_key is None:
            raise RuntimeError("缓存未准备好。请先调用 update_cache() 方法。")
        
        return self._final_cache_store[self._current_cache_key]

    def clear(self) -> None:
        """清空所有缓存。"""
        self._system_kv_cache = None
        self._system_prompt_length = 0
        self._final_cache_store.clear()
        self._current_cache_key = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._log(logging.INFO, "所有缓存已清空。")