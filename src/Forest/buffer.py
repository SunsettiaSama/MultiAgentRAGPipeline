from typing import TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    import copy
    import random




class Pair:
    def __init__(self):

        return 

class Buffer:
    """经验回放池，仅保留最新的5轮交互结果（修复计数逻辑）"""
    def __init__(self):
        self.query_buffer = []        # 存储query（每轮为列表）
        self.responses_buffer = []    # 存储response（每轮为列表）
        self.reward_buffer = []       # 存储reward（每轮为列表）
        self.index_buffer = []        # 交互轮次对齐索引（记录全局轮次）
        self.total_sample_nums = 0    

    def update(self, queries, responses, rewards_list):
        """添加新交互数据，自动保留最新5轮（修复计数逻辑）"""
        # 确保输入为列表格式
        if not isinstance(queries, list):
            queries = [queries]
        if not isinstance(responses, list):
            responses = [responses]
        if not isinstance(rewards_list, list):
            rewards_list = [rewards_list]
        
        # 保证输入长度一致
        n = min(len(queries), len(responses), len(rewards_list))
        if n <= 0:
            return  # 无有效数据
        
        self.total_sample_nums += n
        
        # 添加新轮次
        self.query_buffer.append(queries)
        self.responses_buffer.append(responses)
        self.reward_buffer.append(rewards_list)
        # index_buffer 未使用，可忽略

        # 限制buffer大小为5（保留最新5轮）
        max_size = 5
        if len(self.query_buffer) > max_size:
            removed_samples = 0
            # 计算需要移除的轮次（前k轮）
            k = len(self.query_buffer) - max_size
            for i in range(k):
                removed_samples += len(self.query_buffer[i])
        
            self.total_sample_nums -= removed_samples
            
            # 保留最后max_size轮
            self.query_buffer = self.query_buffer[-max_size:]
            self.responses_buffer = self.responses_buffer[-max_size:]
            self.reward_buffer = self.reward_buffer[-max_size:]
            # index_buffer 未使用，可忽略

    def sample(self, nums):
        """采样nums个样本（按样本而非轮次），无放回（修复逻辑）"""
        n = self.total_sample_nums
        if n == 0 or nums <= 0:
            return [], [], []
        
        nums = min(nums, n)
        if nums <= 0:
            return [], [], []
        
        # === 关键修复：展平所有样本 ===
        all_queries = []
        all_responses = []
        all_rewards = []
        
        # 展平所有轮次的样本
        for queries, responses, rewards in zip(
            self.query_buffer, 
            self.responses_buffer, 
            self.reward_buffer
        ):
            all_queries.extend(queries)
            all_responses.extend(responses)
            all_rewards.extend(rewards)
        
        assert len(all_queries) == len(all_responses) == len(all_rewards) == n
        
        # 随机采样（无放回）
        sample_nums = min(n, nums)
        indices = random.sample(range(n), sample_nums)
        
        # 提取采样结果
        sampled_queries = [copy.deepcopy(all_queries[i]) for i in indices]
        sampled_responses = [copy.deepcopy(all_responses[i]) for i in indices]
        sampled_rewards = [copy.deepcopy(all_rewards[i]) for i in indices]
        
        return sampled_queries, sampled_responses, sampled_rewards

    def __len__(self):
        return self.total_sample_nums
    