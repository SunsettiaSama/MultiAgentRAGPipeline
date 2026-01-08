import numpy as np
from .dataset_evaluation import Evaluator, Validation_Dataset
import pandas as pd
import re
from typing import List
import re
from typing import List, Tuple, Union
import warnings
from flashrag.evaluator.metrics import ExactMatch, F1_Score
from flashrag.evaluator import ExactMatch
from flashrag.config import Config
import json

emConfig = Config()


def _parse_tags(interaction: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """解析 interaction 中的所有标签及其内容"""
    # 匹配 <Tag> ... </Tag> 的内容
    content_pattern = r'<([A-Z][a-zA-Z]+)>(.*?)</\1>'
    content_blocks = re.findall(content_pattern, interaction, re.DOTALL)

    # 匹配所有标签（包括开始和结束标签）
    tag_pattern = r'<(/?[A-Z][a-zA-Z]+)>'
    all_tags = re.findall(tag_pattern, interaction)

    return content_blocks, all_tags

import re

def validate_instruction_format(input_str):
    """
    Validate if the input string follows the specified instruction format.
    
    Args:
        input_str (str): The input string to be validated.
        
    Returns:
        bool: True if the string follows the format, False otherwise.
    """
    # Check for required tags
    required_tags = {'<Think>', '<Action>', '<Detail>', '<END>'}
    found_tags = set()
    
    # Split the string into parts by tags
    parts = re.split(r'<.*?>', input_str)
    tags = re.findall(r'<.*?>', input_str)
    
    # Check if all required tags are present
    for tag in tags:
        if tag in required_tags:
            found_tags.add(tag)
    
    if not required_tags == found_tags:
        return False
    
    # Check if <END> is the last tag
    if tags[-1] != '<END>':
        return False
    
    # Validate <Action> format
    action_pattern = re.compile(r'<Action>(.*?)</Action>')
    actions = action_pattern.findall(input_str)
    
    valid_operations = {
        "Query Rewrite",
        "Query Reason",
        "Query Extract",
        "Document Filter",
        "Document Retrieve"
    }
    
    for action in actions:
        parts = [p.strip() for p in action.split('|')]
        if len(parts) != 3:
            return False
        
        # Target id
        # Check quantity parameter when needed

        id_part = parts[0].strip()
        if not id_part.startswith("Query ") or not id_part[-2:].strip().isdigit():
            return False
            
        # Check operation type
        op_type = parts[1].strip()
        if op_type not in valid_operations:
            return False
            
        # Check quantity parameter when needed
        if op_type in ["Query Rewrite", "Document Retrieve"]:
            quantity_part = parts[2].strip()
            if not quantity_part.startswith("Nums ") or not quantity_part[5:].isdigit():
                return False
    
    # Validate <Detail> format
    detail_pattern = re.compile(r'<Detail>(.*?)</Detail>')
    details = detail_pattern.findall(input_str)

    if len(details) != len(actions):
        return False
    
    return True

def validate_json_file(file_path):
    """
    Validate and clean the JSON file at the given path.
    
    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    filtered_data = []
    for item in data:
        if 'chosen' in item and isinstance(item['chosen'], str):
            if validate_instruction_format(item['chosen']):
                filtered_data.append(item)
        else:
            # Skip if 'chosen' is missing or not a string
            continue

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing JSON file: {e}")

def format_reward(input_data: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Calculate a format score between 0 and 1 for input data (扣分制).
    
    Args:
        input_data (Union[str, List[str]]): A single string or list of strings to be scored.
        
    Returns:
        Union[float, List[float]]: A single score or list of scores between 0 (worst) and 1 (perfect).
    """
    # 专注于单个字符串的处理即可
    if isinstance(input_data, list):
        return [format_reward(item) for item in input_data]
    elif isinstance(input_data, str):
        total_deduction = 0.0  # 初始扣分总和为 0

        # 1. 必要标签存在性检查 (权重 0.25)
        required_tags = {'<Think>', '<Action>', '<Detail>', '<END>'}
        found_tags = set(re.findall(r'<.*?>', input_data))
        missing_count = len(required_tags - found_tags)
        total_deduction += missing_count * 0.0625  # 每个缺失标签扣 0.0625

        # 2. <END> 是否在最后 (权重 0.10)
        tags = re.findall(r'<.*?>', input_data)
        if tags and tags[-1] != '<END>':
            total_deduction += 0.10

        # 3. <Action> 格式检查 (权重 0.50)
        action_pattern = re.compile(r'<Action>(.*?)</Action>')
        actions = action_pattern.findall(input_data)
        valid_operations = {
            "Query Rewrite",
            "Query Reason",
            "Query Extract",
            "Document Filter",
            "Document Retrieve"
        }
        action_errors = 0
        for action in actions:
            parts = [p.strip() for p in action.split('|')]
            if len(parts) != 3:
                action_errors += 1
                continue
            # 检查 ID 格式
            id_part = parts[0].strip()
            if not id_part.startswith("Query ") or not id_part[-2:].strip().isdigit():
                action_errors += 1
            # 检查操作类型
            op_type = parts[1].strip()
            if op_type not in valid_operations:
                action_errors += 1
            # 检查数量参数
            if op_type in ["Query Rewrite", "Document Retrieve"]:
                quantity_part = parts[2].strip()
                if not quantity_part.startswith("Nums ") or not quantity_part[5:].isdigit():
                    action_errors += 1

        # 计算 <Action> 扣分
        if len(actions) == 0:
            # 如果没有 Action，直接扣 0.50
            total_deduction += 0.50
        else:
            # 每个错误扣 0.50 / 动作数，最多扣 0.50
            total_deduction += (action_errors * 0.10) / len(actions)

        # 4. <Detail> 数量匹配检查 (权重 0.15)
        detail_pattern = re.compile(r'<Detail>(.*?)</Detail>')
        details = detail_pattern.findall(input_data)
        if len(details) != len(actions):
            total_deduction += 0.15

        # 最终评分：1.0 - 总扣分
        final_score = max(0.0, 1.0 - total_deduction)
        return final_score
    else:
        raise TypeError("Input must be a string or a list of strings.")
    

def time_reward(time_costs: List[float]) -> Union[float, List[float]]:
    """
    利用Max-Min归一化，将时间成本转换为得分。
    
    ##结束测试##
    """
    # 处理单个数字输入
    if isinstance(time_costs, (int, float)):
        time_costs = [time_costs]
        isBatch = False
    elif isinstance(time_costs, list) and (len(time_costs) in [0, 1]):
        isBatch = False
    else:
        isBatch = True

    if len(time_costs) == 0:  # 输入为空
        warnings.warn("TimeReward计算错误：输入为空")
        return []

    # 采用Max-Min归一化
    time_costs = np.array(time_costs)
    max_time, min_time = np.max(time_costs), np.min(time_costs)

    # 此时时间成本相同，分母为0。
    # 考虑到用于模型输出比较的用途，如果时间成本完全相同，那么时间这一项应被抹平，即返回0组成的数组
    if max_time == min_time:  # 时间成本相同
        if not isBatch:
            warnings.warn("TimeReward计算错误：输入值仅包含单个元素，需求为列表元素")
        else:
            warnings.warn("TimeReward计算错误：输入时间成本完全一致，请检查时间输入")

        return [0] * len(time_costs) if isBatch else 0
    
    time_score_list = 1 - (time_costs - min_time) / (max_time - min_time)
    
    if isBatch:  # 批量处理
        return time_score_list.tolist()
    else:  # 单个处理
        return time_score_list[0]

def em_reward(question : str, golden_answer: str, predictions: list):
    """
    计算 Exact Match (EM) 分数，用于评估预测答案与黄金答案的精确匹配程度。

    功能说明：
        - 将预测答案与黄金答案进行逐字匹配（忽略大小写和多余空格）。
        - 返回每个预测的 EM 分数（0 或 1），1 表示完全匹配，0 表示不匹配。
        - 已通过单元测试，覆盖空输入、完全匹配、部分匹配、大小写/空格处理等场景。

    参数：
        question (str): 输入问题（用于构建数据集，但未直接参与匹配逻辑）。
        golden_answer (str): 标准答案（需与预测答案完全匹配）。
        predictions (list): 预测答案列表（每个元素为字符串）。

    返回：
        list: 每个元素为 0 或 1，表示对应预测的 EM 分数。

    异常处理：
        - 若预测列表为空（`predictions == []`），抛出警告并返回空列表。
        - 假设 `ExactMatch.calculate_metric` 和 `Validation_Dataset.from_dataframe` 的实现已处理非字符串输入等异常。

    示例：
        >>> em_reward("What is AI?", "Artificial Intelligence", ["Artificial Intelligence", "AI"])
        [1.0, 0.0]
    """

    # 处理空输入
    if len(predictions) == 0:
        warnings.warn("ExactMatchReward计算错误：输入预测值长度为0，无预测值")
        return []
    
    em = ExactMatch(emConfig)

    # 构建验证数据集
    dataset = Validation_Dataset.from_dataframe(
        pd.DataFrame({
            "question": [question] * len(predictions),
            "golden_answers": [golden_answer] * len(predictions),
            "pred": predictions
        })
    )
    _, em_scores = em.calculate_metric(dataset)

    return em_scores

def f1_reward(question : str, golden_answer: str, predictions: list):
    """
    计算 Exact Match (EM) 分数，用于评估预测答案与黄金答案的精确匹配程度。

    功能说明：
        - 将预测答案与黄金答案进行逐字匹配（忽略大小写和多余空格）。
        - 返回每个预测的 EM 分数（0 或 1），1 表示完全匹配，0 表示不匹配。
        - 已通过单元测试，覆盖空输入、完全匹配、部分匹配、大小写/空格处理等场景。

    参数：
        question (str): 输入问题（用于构建数据集，但未直接参与匹配逻辑）。
        golden_answer (str): 标准答案（需与预测答案完全匹配）。
        predictions (list): 预测答案列表（每个元素为字符串）。

    返回：
        list: 每个元素为 0 或 1，表示对应预测的 EM 分数。

    异常处理：
        - 若预测列表为空（`predictions == []`），抛出警告并返回空列表。
        - 假设 `ExactMatch.calculate_metric` 和 `Validation_Dataset.from_dataframe` 的实现已处理非字符串输入等异常。

    示例：
        >>> em_reward("What is AI?", "Artificial Intelligence", ["Artificial Intelligence", "AI"])
        [1.0, 0.0]
    """

    # 处理空输入
    if len(predictions) == 0:
        warnings.warn("ExactMatchReward计算错误：输入预测值长度为0，无预测值")
        return []
    
    f1_config = Config()
    f1 = F1_Score(f1_config)

    # 构建验证数据集
    dataset = Validation_Dataset.from_dataframe(
        pd.DataFrame({
            "questions": [question] * len(predictions),
            "golden_answers": [golden_answer] * len(predictions),
            "pred": predictions
        })
    )
    _, f1_scores = f1.calculate_metric(dataset)

    return f1_scores

def weighted_reward_calculate(
    question: str,
    golden_answer: str,
    predictions: list, 
    time: list, 
    alpha: float = 0.5,  # 权重系数
    return_rewards: bool = False, 
    **kwargs
):
    """权重奖励，传入单个question，golden_answer，predctions列表和time列表，返回计算的权重奖励结果"""
    
    scores = []

    time_scores = np.array(time_reward(time))
    f1_scores = np.array(f1_reward(question, golden_answer, predictions))

    # 加权和
    scores.extend(((1 - alpha) * f1_scores + alpha * time_scores).tolist())
    
    if return_rewards:
        return scores, (f1_scores.tolist(), time)

    return scores

import random
from typing import Dict, Set
def __remove_duplicate_scores(
    interactions: List[str],
    scores: List[float]
) -> Tuple[List[str], List[float]]:
    """
    去除 scores 中的重复评分，并删除对应的 interactions 中的重复项。

    保留每个评分的首次出现，后续重复的评分及其对应的 interaction 会被删除。

    参数:
        interactions (List[str]): 交互内容列表
        scores (List[float]): 交互的综合评分列表

    返回:
        Tuple[List[str], List[float]]: 去重后的 (interactions, scores)
    """
    if len(interactions) != len(scores):
        raise ValueError("interactions 和 scores 的长度必须一致。")

    seen_scores: Set[float] = set()
    result_interactions = []
    result_scores = []

    for i in range(len(scores)):
        score = scores[i]
        if score not in seen_scores:
            seen_scores.add(score)
            result_interactions.append(interactions[i])
            result_scores.append(score)

    return result_interactions, result_scores


def _generate_preference_pairs(
    interactions: List[str],
    scores: List[float],
    n_pairs: int = 1,
) -> List[Dict[str, str]]:
    """
    根据交互内容和对应的评分，生成指定数量的偏好对（Preference Pairs）。

    每个偏好对由两个交互项组成：
    - `chosen`：来自高评分组的交互项
    - `rejected`：来自低评分组的交互项

    核心逻辑：
    1. **评分组对顺序遍历**：按降序排列所有评分组，确保覆盖所有可能的评分组对。
    2. **评分组内随机抽样**：每个评分组维护一个洗牌后的队列，通过 `pop(0)` 顺序取出元素，队列为空时自动刷新。
    3. **避免重复配对**：使用 `set` 记录已生成的配对，确保不重复。

    参数:
        interactions (List[str]): 交互内容列表（如用户行为、文本等）
        scores (List[float]): 每个交互的综合评分（如点击率、满意度等）
        n_pairs (int): 需要生成的偏好对数量（默认值为 1）

    返回:
        List[Dict[str, str]]: 包含 `n_pairs` 个偏好对的列表，格式为：
            {
                "chosen": "高评分交互项",
                "rejected": "低评分交互项"
            }

    示例:
        输入:
            interactions = ["A", "B", "C", "D", "E"]
            scores = [0.9, 0.8, 0.7, 0.6, 0.5]
            n_pairs = 3

        输出（可能）:
            [
                {"chosen": "A", "rejected": "B"},
                {"chosen": "A", "rejected": "C"},
                {"chosen": "B", "rejected": "D"}
            ]

    注意事项:
        1. 评分组至少需要两个不同的值，否则无法生成有效的偏好对。
        2. `interactions` 和 `scores` 的长度必须一致。
        3. `n_pairs` 必须大于等于 1。
        4. 队列为空时自动刷新（重新洗牌并填充），但无法保证每个评分组内的所有元素都被使用（取决于 `n_pairs` 的大小）。
        5. 评分组对的遍历顺序是固定的（按降序排列），但评分组内的元素是随机抽取的。

    优势:
        - 处理大规模数据：仅处理评分组对，时间复杂度为 `O(m^2)`（`m` 为评分组数量）
        - 避免重复配对：使用 `used_pairs` 集合记录已生成配对
        - 代码简洁直观：使用 `list.pop(0)` 实现队列逻辑，无需维护索引或栈结构
        - 评分组对全覆盖：顺序遍历所有评分组对，确保不遗漏任何组合
        - 队列自动刷新：队列为空时自动洗牌并填充

    适用场景:
        - 在线学习/强化学习：动态生成训练样本，避免重复
        - 推荐系统：基于用户行为评分生成偏好对，用于模型训练
        - A/B 测试：对比不同策略的效果，生成偏好对进行统计分析
        - 大规模数据处理：评分组数量较多、交互量较大的场景

    ##通过测试##
    """
    
    if len(interactions) != len(scores):
        min_length = min(len(interactions), len(scores))
        warnings.warn("GeneratePreferencePairs错误：输入偏好构筑的列表长度不一致，已进行裁取。")

        interactions = interactions[:min_length]
        scores = scores[:min_length]
    
    if n_pairs < 1:
        raise ValueError("n_pairs 必须大于等于 1。")
    
    # 一高一低分数组合最多允许 n(n-1) / 2 组pair
    if n_pairs > len(interactions) * (len(interactions) - 1) / 2: 
        warnings.warn("GeneratePreferencePairs错误：需求组合数远高于可构筑不重复抽样数，已用不重复抽样最大组数替换")
        n_pairs = int(len(interactions) * (len(interactions) - 1) / 2)

    unique_scores = set(scores)
    if len(unique_scores) < 2:
        raise ValueError("评分列表中至少需要有两个不同的评分，才能生成有效的偏好对。")

    # 删除重复项
    interactions, scores = __remove_duplicate_scores(interactions, scores)

    # 预处理：将评分映射到对应的索引列表
    score_to_indices = {}
    for idx, score in enumerate(scores):
        if score not in score_to_indices:
            score_to_indices[score] = []
        score_to_indices[score].append(idx)

    # 将评分排序（降序）
    sorted_scores = sorted(score_to_indices.keys(), reverse=True)

    # 为每个评分组维护一个洗牌后的队列
    score_queues = {score: indices.copy() for score, indices in score_to_indices.items()}
    for score in sorted_scores:
        random.shuffle(score_queues[score])  # 初始洗牌

    # 记录已生成的配对，避免重复
    used_pairs: Set[Tuple[int, int]] = set()
    result = []

    # 顺序遍历所有评分组对
    for i in range(len(sorted_scores)):
        s_high = sorted_scores[i]
        for j in range(i + 1, len(sorted_scores)):
            s_low = sorted_scores[j]

            # 如果队列为空，重新洗牌并填充
            if not score_queues[s_high]:
                score_queues[s_high] = score_to_indices[s_high].copy()
                random.shuffle(score_queues[s_high])
            if not score_queues[s_low]:
                score_queues[s_low] = score_to_indices[s_low].copy()
                random.shuffle(score_queues[s_low])

            # 取出当前元素（使用 pop(0) 实现队列逻辑）
            chosen_idx = score_queues[s_high].pop(0)
            rejected_idx = score_queues[s_low].pop(0)

            # 检查是否已生成过该配对
            pair = (chosen_idx, rejected_idx)
            if pair in used_pairs:
                continue  # 已存在，跳过

            # 添加到结果和已使用集合
            used_pairs.add(pair)
            result.append({
                "chosen": interactions[chosen_idx],
                "rejected": interactions[rejected_idx]
            })

            # 如果结果已满足要求，提前退出
            if len(result) == n_pairs:
                return result

    return result  # 如果未达到 n_pairs，返回已生成的配对

def build_pairs(questions: list, 
                golden_answers: list, 
                prediction_lists: list,
                alpha: float = 0.5, 
                n_pairs: int = 2, 
                **kwargs):
    """
    predictions和scores一一对应
    
    参数:
        question (list): 问题列表。
        golden_answer (list): 标准答案列表。
        prediction_dicts (list[dict]): 每个字典包含以下键：
            - "predictions": 预测答案列表 (与 question 等长)
            - "time": 耗时列表 (与 question 等长)
            - "interactions": 交互数据列表 (与 question 等长)
        alpha (float): 权重系数 (0 <= alpha <= 1)
    
    返回:
        
    """
    
    if len(len(questions), len(golden_answers), len(precdictions)) != 1:
        warnings.warn("BuildPairs警告：Question、GoldenAnswer和Predictions长度不一致，已截取较短者长度")
        minLength = min(len(questions), len(golden_answers), len(precdictions))

        questions = questions[:minLength]
        golden_answers = golden_answers[:minLength]
        precdictions = precdictions[:minLength]

    pairs = []
    for idx in range(len(questions)):

        question = questions[idx]
        golden_answer = golden_answers[idx]
        prediction_dict = prediction_lists[idx]

        scores = weighted_reward_calculate(question, 
                                           golden_answer, 
                                           prediction_dict,
                                           alpha = alpha, 
                                           **kwargs)
        
        # scores与prediction_dict中列表的元素一一对应
        interactions = prediction_dict["interactions"]
        # 添加到pair列表
        raw_pair = _generate_preference_pairs(interactions, 
                                                scores, 
                                                n_pairs = n_pairs)
        
        
        pair = None
        pairs.extend(pair)
    
    return pairs

def format_reward_func(model_output: str, reward: float, max_action_budget: int = 3):
    """
    判定规则：
    1. 绝对需要保证顺序的一致性，不允许打乱顺序，要求一定要是Think -> Action -> Think -> Action的顺序，如果出现截尾（比如只有Think、没有Action），也能接受
    2. Tag外面不能放东西
    3. 以<END>结尾

    格式示例
    <Think> ... </Think> <Action> ... </Action> <Think> ... </Think> <Action> ... </Action> <Think> ... </Think>
    # 不再加入END作为终止符，END终止符从未在训练中发挥出作用过
    """
        
    # 移除结尾的<END>以便后续处理
    content = model_output.strip() # [:-5]  # 移除<END>
    
    # 规则2：检查Tag外面是否有内容
    # 移除所有标签后应该没有内容
    text_without_tags = re.sub(r'<(Think|Action|Detail)>(.*?)</\1>', '', content, flags=re.DOTALL)
    text_without_tags = re.sub(r'\s+', '', text_without_tags)  # 移除所有空白字符
    if text_without_tags:
        return -1.0  # Tag外面有内容
    
    # 使用正则表达式匹配所有标签，按照出现的顺序
    pattern = r'<(Think|Action|Detail)>(.*?)</\1>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return -1.0  # 没有找到任何标签
    
    # 规则1：检查标签顺序
    expected_order = ['Think', 'Action']  # 期望的顺序模式
    current_pattern_index = 0
    action_count = 0
    
    for tag_type, tag_content in matches:
        # 检查内容是否为空
        if not tag_content.strip():
            return -1.0
        
        # 检查标签类型是否符合期望的顺序
        expected_tag = expected_order[current_pattern_index]
        if tag_type != expected_tag:
            return -1.0
        
        # 如果是Action标签，检查预算
        if tag_type == 'Action':
            action_count += 1
            if action_count > max_action_budget:
                return -1.0
        
        # 移动到下一个期望的标签类型
        current_pattern_index = (current_pattern_index + 1) % len(expected_order)
    
    # 如果通过了所有检查，返回原始奖励
    return reward




class RewardManager:

    def __init__(self, alpha=0.0, time_avg=500.0):
        """
        初始化奖励管理器
        :param alpha: 时间奖励的权重 (0~1)
        :param time_avg: 初始时间基准值（用于初始化max_time）
        :param time_refresh_step: 每多少轮更新一次max_time
        """
        # 固定最小值为0（按原逻辑不更新）
        self.min_time = 0.0
        # 初始化max_time为time_avg（避免初始为-inf导致分母异常）
        self.max_time = float(time_avg)
        # 预计算固定分母（初始化后永不更新）
        self.denominator = max(self.max_time - self.min_time, 1e-8)  # 防除0

        f1_config = Config()
        self.f1 = F1_Score(f1_config)
        self.alpha = alpha
        self.curr_step = 0
        return 
    

    def get_time_reward(self, time_lis: list):
        """
        计算时间奖励（归一化到0~1）：耗时越少，奖励越高
        核心：全程使用初始化时固定的max/min边界，无任何刷新/更新逻辑
        :param time_lis: 当前轮次的耗时列表
        :return: 0~1之间的时间奖励列表
        """
        # 转换为浮点数组，保证数值稳定性
        time_costs = np.array(time_lis, dtype=np.float64)
        
        # 核心：基于固定边界的Max-Min归一化（耗时越少，奖励越接近1）
        if self.denominator == 0:
            # 极端情况（理论上被1e-8保护，不会触发）
            time_score_list = np.ones_like(time_costs)
        else:
            # 固定公式：1 - (当前耗时-最小值)/(最大值-最小值)
            time_score_list = 1 - (time_costs - self.min_time) / self.denominator
        
        # 强制裁剪到0~1区间（处理超出固定边界的耗时）
        time_score_list = np.clip(time_score_list, 0.0, 1.0)

        return time_score_list
    
    def get_reward(self, questions, golden_answers, predictions, time_lis):

        if isinstance(questions, str):
            questions = [questions]
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        if isinstance(predictions, str):
            predictions = [predictions]
        scores = []
        time_scores = self.get_time_reward(time_lis)

        # 构建验证数据集
        dataset = Validation_Dataset.from_dataframe(
            pd.DataFrame({
                "questions": questions,
                "golden_answers": golden_answers,
                "pred": predictions
            })
        )
        _, f1_scores = self.f1.calculate_metric(dataset)

        f1_scores = np.array(f1_scores)

        # 加权和
        scores.extend(((1 - self.alpha) * f1_scores + self.alpha * time_scores).tolist())

        return scores
    
    def __call__(self, *args, **kwargs):
        return self.get_reward(*args, **kwargs)







