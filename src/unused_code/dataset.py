from typing import List, Tuple
import pandas as pd
import os
import random
import warnings
from .reward import format_reward
import json
import os
from collections import deque
from .reward import validate_instruction_format
from .large_language_model import Large_Language_Model_API
import re
from typing import Dict
from tqdm import tqdm
import numpy as np

class golden_dataset:
    
    """
    用以训练的黄金数据集
    ##通过测试##
    """
    
    def __init__(self, dataset_df: pd.DataFrame = None):
        """
        初始化训练数据集，接收一个已加载的 DataFrame。
        
        Parameters:
        - dataset_df (pd.DataFrame): 包含 'question', 'golden_answer' 等字段的数据表
        """
        
        '''
        data_buffer
        '''
        if dataset_df is not None:
            self.questions = dataset_df['question'].tolist()
            self.golden_answers = dataset_df['golden_answer'].tolist()
        else:
            self.questions, self.golden_answers = [], []

        if not len(self.questions) == len(self.golden_answers):
            raise ValueError("数据集长度不一致")
        
        self.length = len(self.questions)
        self.current_index = 0
        self.indices_history = []

        self.garbage = []

    def get_data(self, 
                batch_size: int) -> Tuple[List[str], List[str]]:
        """
        获取用以模型交互的问题
        """
        if self.current_index * batch_size > self.length:
            return [], []
        
        questions = self.questions[self.current_index * batch_size: 
                                   min((self.current_index + 1) * batch_size, self.length)]
        golden_answers = self.golden_answers[self.current_index * batch_size: 
                                             min((self.current_index + 1) * batch_size, self.length)]
        
        self.current_index += 1

        return questions, golden_answers
    
    def get_full_data(self, return_garbage = False) -> Tuple[List[str], List[str]]:
        """
        获取用以模型交互的问题
        """

        questions = self.questions[ :self.length]
        golden_answers = self.golden_answers[ :self.length]

        return questions, golden_answers

    def shuffle(self):
        """
        随机打乱数据训练的顺序
        """
        indices = list(range(len(self.questions)))
        random.shuffle(indices)

        # 按照打乱后的索引顺序重新排列数据
        self.questions = [self.questions[i] for i in indices]
        self.golden_answers = [self.golden_answers[i] for i in indices]

        # 重置当前索引，确保下次 get_data 从头开始
        self.current_index = 0
        self.length = len(self.questions)
    
    @classmethod
    def from_path(cls, path, 
                  question_column='question', 
                  golden_answer_column='nq_answer'):
        """
        类方法：根据文件路径动态加载数据。
        支持的文件类型包括：.csv, .parquet, .xlsx, .json
        
        Parameters:
        - path (str): 文件路径
        - question_column (str): 原始数据中问题的列名，默认 'question'
        - golden_answer_column (str): 原始数据中答案的列名，默认 'golden_answer'

        Returns:
        - training_dataset: 实例化后的训练数据集对象
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        # 根据扩展名加载数据
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext == '.parquet':
            df = pd.read_parquet(path)
        elif ext in ('.xls', '.xlsx'):
            df = pd.read_excel(path)
        elif ext == '.json':
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # 检查指定列是否存在
        required_columns = [question_column, golden_answer_column]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in the dataset: {missing_cols}")

        # 提取指定列并重命名
        df = df[[question_column, golden_answer_column]]
        df.columns = ['question', 'golden_answer']

        return cls(df)
    
    def load_garbage_data(self, garbage_path, garbage_key):
        """
        从指定的 JSON 文件中加载垃圾样本数据。

        参数:
            file_path (str): JSON 文件的路径
            garbage_key (str): JSON 中用于标识垃圾样本的键

        返回:
            self: 当前实例，支持链式调用
        """
        
        garbage_samples = []
        with open(garbage_path, 'r', encoding='utf-8') as f:
            data_dicts = json.load(f)
            for dict in data_dicts:
                garbage_samples.append(dict[garbage_key])
        
        # 这里要求要对齐才行，不然会炸
        self.garbage = garbage_samples

    def shorten_data(self, n_samples: int):
        """
        缩减数据集到长度n_samples
        """
        self.length = n_samples

        return 
    
    def restore_data(self):
        self.length = len(self.questions)
        return 
    
    def __len__(self):
        return self.length
    

class ConversationTree:
    """
    该类用以维护交互链条与用户对话，支持奖励反向传播。
    ##结束测试##
    """

    def __init__(self):
        self.layers = []  # 每个层存储节点列表，节点为字典
        self.layer_metadata = []  # 记录每层的节点数和每个节点的子节点数量
        self.garbage_generator = GarbageGenerator()
        self.api_critic = SampleCritic()

    def add_layer(self, 
                  sampling_prompts: list[list],  # 最外层的list长度与input_prompts一致
                  decision_model_responses: list[list],  # 与sampling_prompts一致
                  systems: list[list[str]] = None, 
                  time_lis: list[str] = None, 
                  **kwargs
                 ):
        """
        添加一层交互链，每个节点包含 input、response 和 children（子节点索引列表）。
        add_layer以层级进行添加，具体而言，List[i][j]的元素表示每一层的第i个节点，第j条边
        允许中途截断，但不允许隔空跳跃生长，必须一层层添加，从左至右
        如希望该节点不管，则应添加空列表进行占位，如[Node1, [], Node3]
        参数：
            sampling_prompts (list[list]):
                二维列表，最外层列表长度应等于输入提示（input_prompts）的数量。
                每个子列表对应一个父节点生成的子节点数量，例如：
                    [
                        ["What is your name?"],          # 父节点 0 生成 1 个子节点
                        ["How old are you?", "Where are you from?"]  # 父节点 1 生成 2 个子节点
                    ]

            decision_model_responses (list[list]):
                二维列表，必须与 sampling_prompts 严格对齐：
                - 外层列表长度等于 sampling_prompts 的外层列表长度；
                - 每个子列表长度等于 sampling_prompts 对应位置的子列表长度。
                例如：
                    [
                        ["My name is Alice."],          # 父节点 0 的子节点响应
                        ["I'm 25.", "I'm from Beijing."]  # 父节点 1 的子节点响应
                    ]

            systems (list[list[str]], optional):
                二维列表，为每个父节点设置 system 信息。外层列表长度应与 sampling_prompts 一致。
                例如：
                    [
                        ["You are a helpful assistant."],  # 父节点 0 的 system 信息
                        ["You are a teacher."]              # 父节点 1 的 system 信息
                    ]

            **kwargs:
                其他自定义参数，如奖励分配策略等。

        抛出：
            ValueError: 如果 sampling_prompts 与 decision_model_responses 的结构不一致（长度不匹配）。

        示例：
            self.add_layer(
                sampling_prompts=[
                    ["What is your name?"],
                    ["How old are you?", "Where are you from?"]
                ],
                decision_model_responses=[
                    ["My name is Alice."],
                    ["I'm 25.", "I'm from Beijing."]
                ],
                systems=[
                    ["You are a helpful assistant."],
                    ["You are a teacher."]
                ]
            )
        """
        layer_node_num = len(sampling_prompts)
        cache = len(decision_model_responses)
        if not layer_node_num == cache:
            warnings.warn("ConversationTree：AddLayer错误，采样输入和模型响应长度不对齐，已进行裁剪")
            layer_node_num = min(layer_node_num, cache)

        layer_edge_nums = [len(decision_model_responses[i]) for i in range(len(decision_model_responses))]
        cache = [len(sampling_prompts[i]) for i in range(len(sampling_prompts))]

        # 添加一层检查机制，检查最后一层节点的数量和当前传入的列表数量是否相同
        if not len(self.layers) == 0:
            if not (len(self.layers[-1]) == len(sampling_prompts) and len(self.layers[-1]) == len(decision_model_responses)):
                raise ValueError("ConversationTree错误：树节点的添加需要最后一层节点与当前输入对齐")

        try:
            if not all([cache[i] == layer_edge_nums[i] for i in range(len(layer_edge_nums))]):
                warnings.warn("ConversationTree：AddLayer错误，采样输入和模型响应长度不对齐，已进行裁剪")
                layer_edge_nums = [min(layer_edge_nums[i], cache[i]) for i in range(len(layer_edge_nums))]
        except:
            print([cache[i] == layer_edge_nums[i] for i in range(len(layer_edge_nums))])

        self.layer_metadata.append({
            "node_num": layer_node_num, 
            "edge_num": layer_edge_nums
        })

        # 为防止time不对齐，这里特地写一个对齐的函数
        if time_lis != None:
            if isinstance(time_lis[0], float) or isinstance(time_lis[0], int):
                time_lis = self.convert_list_structure(sampling_prompts = sampling_prompts, time_lis = time_lis)
        
        # 创建当前层的节点
        current_layer = []
        for i in range(layer_node_num):
            for j in range(layer_edge_nums[i]):
                node = {
                    'input': sampling_prompts[i][j],
                    'response': decision_model_responses[i][j],
                    'children': [],  # 初始化为空列表
                    'system': systems[i][j] if systems != None else None, # 如果传入系统信息的话
                    'reward': None,
                    'time': time_lis[i][j] if time_lis != None else None, 
                }
                current_layer.append(node)

        # 如果不是第一层，则设置上一层节点的子节点索引
        if self.layers:
            prev_layer = self.layers[-1]
            next_child_start = 0
            for i in range(layer_node_num):
                edge_num = layer_edge_nums[i]
                # 当前父节点i的子节点在当前层的起始索引是 next_child_start
                prev_layer[i]['children'] = list(range(next_child_start, next_child_start + edge_num))
                next_child_start += edge_num

        self.layers.append(current_layer)

    def convert_list_structure(self, sampling_prompts, time_lis):
        """
        根据 sampling_prompts 的结构，将一维的 time_lis 转换为相同的嵌套结构。
        
        参数:
            sampling_prompts (List[List[Any]]): 嵌套列表，表示每层的采样提示。
            time_lis (List[float or int]): 一维时间列表，总长度应与所有节点数一致。
        
        返回:
            List[List[float or int]]: 与 sampling_prompts 结构相同的嵌套时间列表。
        
        抛出:
            ValueError: 如果 time_lis 的长度与节点总数不匹配。
        """
        nested_time = []
        index = 0

        for layer in sampling_prompts:
            layer_length = len(layer)
            if index + layer_length > len(time_lis):
                raise ValueError("time_lis 的长度与 sampling_prompts 的结构不匹配")
            nested_time.append(time_lis[index:index + layer_length])
            index += layer_length

        if index != len(time_lis):
            raise ValueError("time_lis 的长度与 sampling_prompts 的结构不匹配")

        return nested_time


    def add_reward(self, rewards: list):
        """
        按照路径奖励从高到低的顺序进行奖励分配。
        每条路径的奖励值等于其末端节点的奖励。
        对于路径中的节点，若尚未分配奖励，则分配该路径的奖励（与现有奖励叠加）。
        """

        total_bottom_nodes = self.total_bottom_nodes()
        if len(rewards) != total_bottom_nodes:
            raise ValueError(f"奖励数量 {len(rewards)} 与最底层节点数 {total_bottom_nodes} 不匹配")

        # 设置最底层节点的奖励（叠加传入的 rewards 和现有 reward）
        if self.layers:
            for idx, reward in enumerate(rewards):
                existing_reward = self.layers[-1][idx].get("reward", 0)
                self.layers[-1][idx]["reward"] = (existing_reward + reward) if existing_reward != None else reward

        # 生成所有从根到叶的路径
        paths = self.generate_all_paths()

        if not paths:
            return

        # 为每条路径计算奖励值（末端节点的奖励）
        path_rewards = []
        for path in paths:
            last_layer_idx, last_node_idx = path[-1]
            reward = self.layers[last_layer_idx][last_node_idx]["reward"]
            path_rewards.append((path, reward))

        # 按奖励值从高到低排序
        path_rewards.sort(key=lambda x: x[1], reverse=True)

        # 按顺序处理路径
        for path, _ in path_rewards:
            for layer_idx, node_idx in path:
                node = self.layers[layer_idx][node_idx]
                if node["reward"] is None:
                    # 如果未分配奖励，则设置为路径奖励
                    node["reward"] = self.layers[path[-1][0]][path[-1][1]]["reward"]
                else:
                    # 如果已分配奖励，但当前路径的奖励更高，则更新为路径奖励
                    # （保持最高奖励优先原则）
                    if self.layers[path[-1][0]][path[-1][1]]["reward"] > node["reward"]:
                        node["reward"] = self.layers[path[-1][0]][path[-1][1]]["reward"]

    def get_cumulative_times_last_layer(self):
        """
        计算最后一层每个末端节点从根节点到自身的累计耗时。

        返回:
            List[float]: 每个元素表示最后一层对应节点的累计耗时。
        """
        if not self.layers:
            return []

        last_layer_index = len(self.layers) - 1
        if last_layer_index == 0:
            # 如果只有一层，直接返回每个节点的 time
            return [node['time'] if node['time'] is not None else 0 for node in self.layers[0]]

        result = []
        queue = deque()

        # 初始化队列：将根层每个节点加入队列
        for node_idx in range(len(self.layers[0])):
            time = self.layers[0][node_idx]['time'] if self.layers[0][node_idx]['time'] is not None else 0
            queue.append((0, node_idx, time))  # (当前层索引, 当前节点索引, 累计时间)

        while queue:
            layer_idx, node_idx, cumulative_time = queue.popleft()
            current_layer = self.layers[layer_idx]
            current_node = current_layer[node_idx]

            if layer_idx == last_layer_index:
                # 到达最后一层，记录累计时间
                result.append(cumulative_time)
            else:
                # 遍历当前节点的所有子节点
                for child_idx in current_node['children']:
                    next_layer = self.layers[layer_idx + 1]
                    child_node = next_layer[child_idx]
                    child_time = child_node['time'] if child_node['time'] is not None else 0
                    new_cumulative_time = cumulative_time + child_time
                    queue.append((layer_idx + 1, child_idx, new_cumulative_time))

        return result
    

    def total_bottom_nodes(self):
        """计算最底层节点的总数"""
        if not self.layers:
            return 0
        return len(self.layers[-1])

    def clear(self):
        """清空树结构"""
        self.layers = []
        self.layer_metadata = []

    def get_node_info(self, layer_index, node_index):
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(f"Layer index {layer_index} is out of bounds for tree with {len(self.layers)} layers.")
        
        layer = self.layers[layer_index]        
        if node_index < 0 or node_index >= len(layer):
            raise IndexError(f"Node index {node_index} is out of bounds for layer {layer_index} which has {len(layer)} nodes.")
        
        return layer[node_index]

    def get_all_nodes(self):
        if not self.layers:
            return []

        bottom_layer = self.layers[-1]
        if bottom_layer and bottom_layer[0].get('reward') is None:
            warnings.warn("树采样结构警告：未进行奖励分配就调取了所有节点的值", UserWarning)

        all_nodes = []
        for layer in self.layers:
            all_nodes.extend(layer)

        return all_nodes
    
    def get_all_rewards(self):
        """获取所有节点的奖励，以列表形式返回"""
        if not self.layers:
            return []

        rewards = []
        for node in self.get_all_nodes():
            rewards.append(node["reward"])

        return rewards
    

    def generate_all_paths(self):
        """生成所有从根节点到叶节点的路径"""
        paths = []
        if not self.layers:
            return paths

        def dfs(current_layer_idx, current_node_idx, path):
            # 添加当前节点到路径
            path.append((current_layer_idx, current_node_idx))

            # 如果是最后一层，保存路径
            if current_layer_idx == len(self.layers) - 1:
                paths.append(path.copy())
            else:
                # 遍历所有子节点
                for child_idx in self.layers[current_layer_idx][current_node_idx]['children']:
                    dfs(current_layer_idx + 1, child_idx, path)

            path.pop()  # 回溯

        # 从第一层的每个节点开始 DFS
        for node_idx in range(len(self.layers[0])):
            dfs(0, node_idx, [])

        return paths
    
    def get_top_reward_nodes(self, percent: float) -> list[dict]:
        """
        遍历所有节点，取出每一层奖励最高的前 percent% 的节点。
        返回值为包含每个节点的 input、response、system 字段的字典列表。

        参数：
            percent (float): 百分比（0-100），表示每层保留的奖励最高的前 percent% 节点。

        返回：
            list[dict]: 包含每个节点的 input、response、system 的字典列表。
        """
        if not 0 <= percent <= 100:
            raise ValueError("percent 必须介于 0 和 100 之间")

        top_nodes = []

        for layer in self.layers:
            if not layer:
                continue

            # 按奖励降序排序，忽略 reward 为 None 的节点
            sorted_nodes = sorted(
                [node for node in layer if node['reward'] is not None],
                key=lambda x: x['reward'],
                reverse=True
            )
            num_nodes = len(sorted_nodes)

            if num_nodes == 0:
                continue

            if percent == 0:
                continue  # 如果 percent 为 0，则不保留任何节点

            # 计算每层需保留的节点数
            top_k = int(np.ceil(num_nodes * percent / 100))
            top_k = min(top_k, num_nodes)  # 避免超出节点总数

            # 提取奖励最高的前 top_k 个节点
            for node in sorted_nodes[:top_k]:
                top_nodes.append({
                    'input': node['input'],
                    'response': node['response'],
                    'system': node.get('system', '')  # 如果 system 不存在，默认设为空字符串
                })

        return top_nodes
    
    def get_top_reward_alpaca_data(self, percent: float, eval_format: bool = True) -> List[Dict]:
        """
        取出前percent的data，

        参数：
            nodes_data (list[dict]): 由 get_top_reward_nodes 返回的字典列表。
            file_path (str): 保存的文件路径（如 "output.json"）。
        """
        alpaca_data = []
        nodes_data = self.get_top_reward_nodes(percent)
        for node_data in nodes_data:
            instruction = node_data['input']
            model_output = node_data['response']
            if eval_format:
                if not validate_instruction_format(model_output):
                    continue
            alpaca_dict = format_to_alpaca(instruction, '', model_output)
            alpaca_data.append(alpaca_dict)

        return alpaca_data

    def __getitem__(self, key):
        """
        通过 (layer_index, node_index) 元组访问指定节点的信息。

        示例：
            >>> node = tree[layer_idx, node_idx]
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("树结构访问键应符合：(layer_idx, node_idx)")
        layer_idx, node_idx = key
        if not isinstance(layer_idx, int) or not isinstance(node_idx, int):
            raise TypeError(f"树结构访问键切片数字类型应为整数int，而当前类型应为({type(layer_idx)}, {type(node_idx)})")
        return self.get_node_info(layer_idx, node_idx)

    def clear(self):
        """
        清空该树
        """

        self.layers = []  # 每个层存储节点列表，节点为字典
        self.layer_metadata = []  # 记录每层的节点数和每个节点的子节点数量

    def evaluate_format_score(self):
        """
        使用 format_reward 函数计算所有节点的格式奖励，并将结果存入 reward 字段。
        """
        # 收集所有节点的 response
        all_responses = []
        for layer in self.layers:
            for node in layer:
                all_responses.append(node['response'])

        # 调用 format_reward 函数计算奖励
        rewards = format_reward(all_responses)

        # 检查奖励数量是否匹配
        if len(rewards) != len(all_responses):
            raise ValueError(f"奖励数量 {len(rewards)} 与节点数量 {len(all_responses)} 不匹配")

        # 将奖励按顺序分配回每个节点
        index = 0
        for layer in self.layers:
            for node in layer:
                node['reward'] = rewards[index]
                index += 1
    
    def get_all_nodes_format_scores(self):
        """
        获取格式奖励但不添加到节点的奖励上
        """
        # 收集所有节点的 response
        all_responses = []
        for layer in self.layers:
            for node in layer:
                all_responses.append(node['response'])

        # 调用 format_reward 函数计算奖励
        rewards = format_reward(all_responses)
        return rewards

    def build_pair(self, system=None, 
                   top_percent = 0.2, 
                   gap: float = 0, 
                   garbage_threshold: float = 100, 
                   activate_critic: bool = False, 
                   data_format: str = "dpo"):
        """
        构造符合 DPO 格式的训练 pair 数据。
        
        参数：
            system (str, optional): 全局注入的 system 消息。若为 None，则使用子节点的 system。
        
        返回：
            list[dict]: 每个 dict 包含 'conversations', 'chosen', 'rejected' 字段。
        """
        pairs = []

        # 遍历所有层级（除最后一层）
        for layer_idx in range(len(self.layers) - 1):
            parent_layer = self.layers[layer_idx]
            child_layer = self.layers[layer_idx + 1]

            # 步骤 1：找到当前层中奖励最高的父节点
            # 配合路径奖励传播算法，当前层奖励最高的父节点在路径上是唯一的，所以
            # 构造的最优样本也是唯一的
            max_parent_idx = max(
                range(len(parent_layer)),
                key=lambda i: parent_layer[i].get("reward", 0)  # 假设父节点有 reward 字段
            )
            max_parent = parent_layer[max_parent_idx]

            # 步骤 2：获取该父节点的所有子节点
            children_indices = max_parent.get("children", [])
            if not children_indices:
                continue  # 无子节点，跳过

            # 步骤 3：提取子节点的 response 和 reward
            children_nodes = [child_layer[i] for i in children_indices]
            responses = [node["response"] for node in children_nodes]
            rewards = [node.get("reward") for node in children_nodes]

            # 步骤 4：跳过奖励无效的子节点
            if any(r is None or r == 0 for r in rewards):
                continue

            # 步骤 5：构建 conversations（使用父节点的 system 和 input）
            first_child = children_nodes[0]
            child_system = max_parent.get("system")  # 使用父节点的 system
            child_input = max_parent.get("input")    # 使用父节点的 input

            conversations = []
            if system is not None:
                conversations.append({"from": "system", "value": system})
            elif child_system is not None:
                conversations.append({"from": "system", "value": child_system})
            if child_input is not None:
                conversations.append({"from": "human", "value": child_input})

            n_samples = len(rewards)
                        

            # 计算 top_k（四舍五入，最小为 1，最大为 n_samples）
            top_k = int(np.ceil(n_samples * top_percent))

            # 按奖励从高到低排序索引
            sorted_indices = sorted(range(n_samples), key=lambda i: rewards[i], reverse=True)
            high_indices = sorted_indices[:top_k]
            low_indices = sorted_indices[-top_k:]

            # 步骤 7：构造 pair
            for max_reward_idx, min_reward_idx in zip(high_indices, low_indices):

                max_response = responses[max_reward_idx]
                min_response = responses[min_reward_idx]

                # 分数gap

                # rejected的奖励分数太高则构造垃圾样本进行填充
                # 相当于一个锚定分布，这个分布是肯定需要降低获取概率的
                min_reward = rewards[min_reward_idx]
                max_reward = rewards[max_reward_idx]
                # 小于2个不应能组成pair
                if data_format == "dpo": 

                    if n_samples < 2:
                        continue
                    if max_reward - min_reward < gap:
                        continue

                    if min_reward > garbage_threshold:
                        # 提取Query 0
                        pattern = r'\*\*User Input:\*\*(.*?)\n\n\*\*Queries Collected:\*\*'
                        match = re.search(pattern, child_input, re.DOTALL)
                        if match:
                            user_input = match.group(1).strip()
                        else:
                            user_input = "Nothing. You can generate anything you want."
                        min_response = self.garbage_generator.generate(
                            user_input = user_input
                        )
                    else:
                        min_response = responses[min_reward_idx]

                # api评判专家
                if activate_critic:
                    # 要求必须足够好才行，如果不够好，则放弃构建
                    # 提取所有有效的部分
                    # if not self.api_critic.strCritic(model_response = max_response):
                    #     continue
                    
                    # 不管差的有多好，都采用这种方法，加入垃圾样本
                    # 抛弃所有和格式沾边的内容
                    # 然后再把思考的内容放到说垃圾话的模型中，得到结果
                    max_response = self.api_critic.extract_valid_content(max_response)
                    
                    if data_format == "dpo":
                        min_response = self.garbage_generator.generate(responses[min_reward_idx])
                    
                if data_format == "dpo": 
                    pair = {
                        "conversations": conversations,
                        "chosen": {"from": "gpt", "value": max_response},
                        "rejected": {"from": "gpt", "value": min_response}
                    }
                    
                elif data_format == "sft":
                    pair = {
                        "instruction": "\n".join([conversations[i]["value"] for i in range(len(conversations))]),
                        "input": "", 
                        "output": max_response, 
                    }
                pairs.append(pair)

        return pairs
    
    def pair2json(self, 
                  json_path: str = "/root/autodl-tmp/dataset/mydataset.json", 
                  system=None, 
                  top_percent: float = 0.2, 
                  gap: float= 0, 
                  garbage_threshold: float = 100, 
                  activate_critic: bool = False, 
                  mode='w', 
                  data_format: str = "dpo"):
        """
        将构建的 pair 数据保存为本地 JSON 文件，格式符合指定的 DPO 训练格式。

        参数：
            json_path (str): JSON 文件保存路径（可以是目录或文件路径）
            system (str, optional): 全局注入的 system 消息（用于所有 pair）
            mode (str, optional): 写入模式，'w' 表示覆盖写入，'a' 表示追加写入，默认为 'w'

        异常：
            ValueError: 如果 mode 不是 'w' 或 'a'
        """
        # 1. 校验 mode 参数
        if mode not in ('w', 'a'):
            raise ValueError("mode 必须为 'w' 或 'a'")

        # 2. 判断路径类型
        if json_path.endswith('.json'):
            # 情况一：传入的是文件路径
            file_path = json_path
        else:
            # 情况二：传入的是目录，自动生成文件名
            file_path = os.path.join(json_path, 'dpo_pairs.json')

        # 3. 确保目录存在
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # 4. 构建 pair 数据
        pairs = self.build_pair(system=system, 
                                top_percent = top_percent, 
                                gap = gap, 
                                garbage_threshold = garbage_threshold, 
                                activate_critic = activate_critic, 
                                data_format = data_format)

        # 5. 根据 mode 处理写入逻辑
        if mode == 'a':
            # 追加模式：读取现有数据并合并
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []  # 如果文件不存在或内容损坏，以空列表初始化
            updated_data = existing_data + pairs
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)
        else:
            # 覆盖模式：直接写入新数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=4)

class GarbageGenerator:

    def __init__(self):
        self.llm_api = Large_Language_Model_API(model = "gpt-4o-mini")
        self.llm_api.init_llm('')

    def generate(self, input_text):
        """生成垃圾数据"""
        input_text = self.extract_invalid_content(input_text)
        # 然后生成垃圾
        return self.llm_api.generate(self.SampleToGarbagePrompt(input_text))

    def SampleToGarbagePrompt(self, model_response):
        prompt = f"""

**Please Modify the Input to Garbage.**
**Example Input** 
Let's begin. I need to determine the best way to approach the user's query about when the next season of Ninjago comes out. The user is asking for a specific release date, so the first step is to check if there are any documents or information sources that can provide this information.
Since there are no documents retrieved yet, I need to generate alternative queries to cover different aspects of the topic. The user's original query is straightforward, but to ensure we get the most relevant information, I should consider generating different queries that might lead to the same or similar information. For example, searching for the release date of the next season, or related news articles about the show's schedule.
I will start by rewriting the query to focus on different aspects of the season release. This will help in retrieving a variety of documents that might be relevant to the user's question. Then, I will analyze the query to determine what specific information needs to be retrieved, such as the exact release date, any announcements, or related news. 
Let me generate some alternative queries first.

*Expected Output*: 
Okay like someone asked when Ninjago's next season is coming out and I was like "idk man". Maybe check the internet? I heard they're making new toys too! The show will probably drop in 2024... or maybe 2030? Who knows!

**Your Commision**:
1. Summary and rewrite more than 50% of the input. 
2. Add a lot of irrelevant response in input. 
3. Return your modified garbage version. 

**Current Input**: 
{model_response}

**Please Modify the Input to Garbage.**
Your Output: 
"""
        return prompt

    def extract_invalid_content(self, text):
        """
        从文本中提取所有标签以外的内容，并按顺序拼接。
        
        参数:
            text (str): 原始输入文本
            
        返回:
            str: 不符合格式要求的内容，按顺序拼接后的字符串
        """
        # 提取Think部分，说不定后续要用
        think_pattern = r'<Think>(.*?)</Think>'
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        else:
            think_content = "Nothing yet. "

        # 第一次筛选：提取标签外部的内容，移除所有标签及其内容
        pattern = re.compile(
            r'<[^>]+>.*?(?:</[^>]+>|$)',  # 匹配任意标签到其闭合标签或文本末尾
            re.DOTALL
        )
        
        matches = list(pattern.finditer(text))
        first_pass = []
        prev_end = 0  # 上一个标签块的结束位置
        
        for match in matches:
            start = match.start()
            end = match.end()
            
            # 提取当前标签前的非标签内容
            if start > prev_end:
                invalid_part = text[prev_end:start].strip()
                if invalid_part:
                    first_pass.append(invalid_part)
            
            # 更新上一个标签的结束位置
            prev_end = end
        
        # 提取最后一个标签块之后的内容
        if prev_end < len(text):
            invalid_part = text[prev_end:].strip()
            if invalid_part:
                first_pass.append(invalid_part)
        
        # 拼接第一次筛选结果
        first_result = '\n'.join(first_pass)
        
        # 第二次筛选：进一步清理残留的标签
        # 1. 移除成对出现的标签及其内容（如 <Tag>...</Tag>）
        second_result = re.sub(
            r'<[^>]+>.*?</[^>]+>',  # 匹配成对标签及其内容
            '', 
            first_result, 
            flags=re.DOTALL
        )
        
        # 2. 移除单独的标签（如 <Tag>）
        second_result = re.sub(
            r'<[^>]+>',  # 匹配单独的标签
            '', 
            second_result
        )
        
        # 如果字符实在太少，那么任意生成内容
        if len(second_result) < 10:
            second_result += think_content
            
        return second_result.strip()

class SampleCritic:

    def __init__(self):
        self.llm_api = Large_Language_Model_API(model = "gpt-4o-mini")
        self.llm_api.init_llm("""
**Please Give a Comment on Model Output with <Yes> or <No>.**

**Input Format Specification**
- <Think> </Think>
- <Action> Target Query(or Document) | Tools | Number of query(or document) to generate </Action>
    - Target Query Format: Query i / Document i
        - "i" is the target id. 
    - Tools should be in one of them: <Query Rewrite>, <Query Reason>, <Query Extract>, <Document Filter>, <Document Retrieve>, <Stop>
    - Number of query(or document) should follow format: nums j
        - "j" is the quantity of action.
- <Detail> </Detail>
    - Detail quantity should match the num of action.
- ...
- <Action> </Action>
- <Detail> </Detail>
- <END>

**Comment Example**
*Input 1*: <Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>  \n<Action> Query 0 | Query Rewrite | Nums 3 </Action>  \n<Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail> \n<Action> Query 0 | Query Reason | Nums 1 </Action>  \n<Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>  \n<END>
*Expected Output*: Everything is OK under rules, no big mistakes. <Yes>

*Input 2*: Certainly, I need to address the user's query efficiently. Since the question is straightforward—seeking the all-time leading scorer in the Premier League—it does not require any complex refinement or additional reasoning. The primary goal is to retrieve accurate documents that provide this piece of information. \n\nI will directly retrieve the documents to find the specific answer.\n<Think> The query seeks the name of the all-time leading scorer in the Premier League. No further refinement is needed as the information is factual and widely available. Retrieval of relevant data will directly address the user's request. </Think>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*Expected Output*: It lacks the <END> and </Detail> tag, put the string outside the tag and with bad codes "..>>>>>". <No>

Your Commision:
1. Give your judgement reason before output <yes> or <no>. 
2. Find out if the input format obey or against the "Input Format Specification". 
3. Please be mild to the input if input slightly against the rules. 

"""
)
        self.state_prompt_func = lambda x: f"""
**Current Input**: 
{x}
Please Give a Comment on Input with reason and <Yes> or <No> tag.
Your Output: 
"""

    def strCritic(self, model_response, need_print: bool = False):
        """
        从LLM API的响应中提取是否满足要求的判断，返回True或False。
        - 如果包含<Yes>，返回True。
        - 如果包含<No>，返回False。
        - 如果未包含<Yes>或<No>，返回False（视为错误）。
        """
        api_response = self.llm_api.generate(
            self.state_prompt_func(model_response)
        )
        if need_print:
            print("Get Api Response:\n", api_response)
        match = re.search(r'<(Yes|No)>', api_response)
        if match:
            return match.group(1) == 'Yes'
        else:
            # 缓和机制，返回True，允许一定噪声，防止出现乱码等
            return True

    def extract_valid_content(self, text) -> str:
        """
        清洗文本内容，提取有效标签内的原始内容，支持处理截断输入
        
        参数:
            text (str): 输入的原始文本
            
        返回:
            str: 清洗后的纯文本字符串
        """
        result = []

        # 1. 提取 <Think> 部分
        think_pattern = r'<Think>(.*?)</Think>'
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            result.append(f"<Think> {think_content} </Think>")

        # 2. 提取所有 <Action> 和 <Detail> 部分
        # 提取所有 <Action> 标签内容
        action_pattern = r'<Action>(.*?)</Action>'
        actions = re.findall(action_pattern, text, re.DOTALL)

        # 提取所有 <Detail> 标签内容
        detail_pattern = r'<Detail>(.*?)</Detail>'
        details = re.findall(detail_pattern, text, re.DOTALL)

        # 3. 将 <Action> 和 <Detail> 配对
        for i in range(max(len(actions), len(details))):
            action = actions[i].strip() if i < len(actions) else ""
            detail = details[i].strip() if i < len(details) else ""

            if action:
                result.append(f"<Action> {action} </Action>")
            if detail:
                result.append(f"<Detail> {detail} </Detail>")

        # 4. 检查是否检测到 <END> 标签，如果没有，则认为标签截断
        end_pattern = r'<END>'
        if not re.search(end_pattern, text):
            unclosed_action = re.search(r'<Action>(.*?)$', text, re.DOTALL)
            if unclosed_action:
                action_content = unclosed_action.group(1).strip()
                result.append(f"<Action> {action_content} </Action>")

            # 检查末尾是否有未闭合的 <Detail>
            unclosed_detail = re.search(r'<Detail>(.*?)$', text, re.DOTALL)
            if unclosed_detail:
                detail_content = unclosed_detail.group(1).strip()
                result.append(f"<Detail> {detail_content} </Detail>")
        
        # 最后加上END标签
        result.append("<END>")

        return "\n".join(result)

def extract_user_input(output: str) -> str:
    """
    从LLM响应中提取用户输入内容
    
    Args:
        output (str): LLM生成的完整响应
        
    Returns:
        str: 提取的用户输入内容
    """
    match = re.search(r'<User Input>(.*?)</User Input>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def format_to_alpaca(instruction: str, user_input: str, model_output: str) -> Dict:
    """
    转换为Alpaca格式字典
    """
    return {
        "instruction": instruction,
        "input": user_input,
        "output": model_output
    }

def alpaca2json(alpaca_data: list[dict], file_path: str, mode: str = 'w') -> None: 
    # 保存为 JSON 文件
    with open(file_path, mode, encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

def sampling_format_sample_from_api(n: int, output_path: str):
    """
    生成符合格式的样本并保存为Alpaca格式JSON文件
    
    Args:
        n (int): 需要生成的样本数量
        output_path (str): JSON文件保存路径
    """

    llm_api = Large_Language_Model_API(model="gpt-4o")
    llm_api.init_llm('')
    high_quality_samples = []

    with tqdm(total = n) as pbar:
        while len(high_quality_samples) < n:
            # 第一次调用获取用户输入
            sampling_prompt = f"""
Please generate samples using this tools followed. 
<Query Rewrite> Rephrase query: Generate different queries to cover more topics. 
<Query Reason> Analyze query: Reason what should be retrieved to improve query quality.
<Query Extract> Extract query from documents: Extract query from the given query with retrieved documents. 
<Document Filter> Filter documents: Analyse all these documents are relevant to the user input or not. 
<Document Retrieve> Retrieve documents: Pick up the most valuable query you think to retrieve. 
<Stop> Terminate: When user input is sufficiently clear to answer. After that, You will get last chance to generate something relevant to help improve answer.

Here's the rules. 
- <Think> Make sure you have enough contemplation before taking any action, and put them in this tag. 
- <Action> Target Query(or Document) | Tool you want to take | Number of query(or document) to generate
- Part 1: Original query(or document) reference, you can reference user input as "Query 0" speciffically, others can be "Query 1", "Query 2", ...
- Part 2: Operation type (Query Rewrite/Query Reason/...)
- Part 3: Quantity parameter (only required when generate new query or document)
- <Detail> You can put all details you want to say to the next agent here. 
- Optional, you can ignore it when you think it's unnecessary. 
- Use multiple <Detail> tags when you want to describe action seperately.
- <END> Use this tag to end the query immediately.

**Example**
User Input Example: 
How does artificial intelligence impact the healthcare industry?
**Expected Output Example**: 
<Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
<Action> Query 0 | Query Rewrite | Nums 3 </Action>
<Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail>
<Action> Query 0 | Query Reason | Nums 1 </Action>
<Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>
<END>

Please generate a sample follow the format strictly, including <Think>, <Action>, <Detail> tags. 
    """
            
            output = llm_api.generate(sampling_prompt, return_time = False)
            # 提取用户输入
            user_input = extract_user_input(output)

            # 构造系统提示
            system_prompt = f"""
You are a decision-making agent that can use the following tools to enhance output quality:
<Query Rewrite> Rephrase query: Generate different queries to cover more topics. 
<Query Reason> Analyze query: Reason what should be retrieved to improve query quality.
<Query Extract> Extract query from documents: Extract query from the given query with retrieved documents. 
<Document Filter> Filter documents: Analyse all these documents are relevant to the user input or not. 
<Document Retrieve> Retrieve documents: Pick up the most valuable query you think to retrieve. 
<Stop> Terminate: When user input is sufficiently clear to answer. After that, You will get last chance to generate something relevant to help improve answer.

**Response Format Specification**
- <Think> Make sure you have enough contemplation before taking any action, and put them in this tag. 
- <Action> Target Query(or Document) | Tool you want to take | Number of query(or document) to generate
- Part 1: Original query(or document) reference, you can reference user input as "Query 0" speciffically, others can be "Query 1", "Query 2", ...
- Part 2: Operation type (Query Rewrite/Query Reason/...)
- Part 3: Quantity parameter (only required when generate new query or document)
- <Detail> You can put all details you want to say to the next agent here. 
- Optional, you can ignore it when you think it's unnecessary. 
- Use multiple <Detail> tags when you want to describe action seperately.
- <END> Use this tag to end the query immediately.

**Example**
User Input (Query 0): {user_input}
Expected Output:
<Think> The query requires a technical comparison of computing paradigms. Need to identify core concepts in both fields and their distinguishing features. </Think>
<Action> Query 0 | Document Retrieve | Nums 2 </Action>
<Action> Query 0 | Query Reason | Nums 1 </Action>
<Detail> Need to understand fundamental principles of quantum vs classical computing, including qubits vs bits, superposition, entanglement, and computational complexity. </Detail>
<END>
**Please follow the format strictly!**
    """
            # 构造状态提示
            state_prompt = f"""
Current State:
**User Input:**
{user_input}

**Queries Collected:**
<Query 0> {user_input}

**Documents Retrieved:**
Nothing yet.

Please give out your <Think>, <Action>, and <Detail>. Remember, "Stop" when all documents are sufficiently covered user's input as soon as possible.
    """
            # 生成最终响应
            instruction = system_prompt + state_prompt

            match = re.search(r'</User Input>(.*?)$', output, re.DOTALL)
            if match is not None:
                match = match.group(1).strip()
            else:
                match = ''

            # 验证格式
            if validate_instruction_format(match):
                # 构造Alpaca格式样本
                alpaca_sample = format_to_alpaca(
                    instruction=instruction,
                    user_input='',
                    model_output=match
                )
                high_quality_samples.append(alpaca_sample)

                pbar.update(1)

            if len(high_quality_samples) // int(n / 20):
                # 保存到JSON文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(high_quality_samples, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(high_quality_samples, f, indent=2, ensure_ascii=False)

def extract_data_from_dpo_training_dataset(json_data):
    """
    从 JSON 数据中提取 conversation 中的 system 和 human 对话内容
    参数:
        json_data (list): 包含多个字典的列表，每个字典包含 'conversations' 键
    返回:
        list: 包含多个对话对的列表，每个元素为 {"system":..., "human":...}
    """
    result = []
    
    for item in json_data:

        conversations = item.get("conversations", [])
        chosen = item.get("chosen", "").get("value", "")
        instruction = "\n".join([conversations[i]["value"] for i in range(len(conversations))])
        result.append({
            "instruction": instruction,
            "input": "", 
            "output": chosen
        })

    
    return result

def dpoData2sftData(json_data):
    """
    从 JSON 数据中提取 conversation 中的 system 和 human 对话内容
    参数:
        json_data (list): 包含多个字典的列表，每个字典包含 'conversations' 键
    返回:
        list: 包含多个对话对的列表，每个元素为 {"system":..., "human":...}
    """
    result = []
    
    for item in json_data:
        conversations = item.get("conversations", [])
        
        system_values = []
        human_values = []
        
        # 分离 system 和 human 的 value
        for conv in conversations:
            if conv.get("from") == "system":
                system_values.append(conv.get("value", ""))
            elif conv.get("from") == "human":
                human_values.append(conv.get("value", ""))
        
        # 按顺序配对 system 和 human
        for i in range(min(len(system_values), len(human_values))):
            result.append({
                "system": system_values[i],
                "human": human_values[i]
            })
    
    return result

def from_tree_json_to_sampling_json(input_file_path: str, output_file_path: str):
    """
    从源 JSON 文件读取数据，提取后与目标文件中的内容合并，然后覆盖写入目标文件。
    
    参数:
        input_file_path (str): 源 JSON 文件路径
        output_file_path (str): 目标 JSON 文件路径
    """
    try:
        # 1. 读取源文件内容
        with open(input_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            
        try:
            extracted_data = extract_data_from_dpo_training_dataset(json_data)
        except:
            extracted_data = []


        # 3. 尝试读取目标文件已有内容
        try:
            with open(output_file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果目标文件不存在或格式错误，则默认为空列表
            existing_data = []

        # 4. 合并新提取的数据与已有数据
        merged_data = existing_data + extracted_data

        # 5. 覆盖写入目标文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

    except FileNotFoundError as e:
        print(f"错误：文件 {e.filename} 未找到")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file_path} 格式不正确，无法解析 JSON")
    except Exception as e:
        print(f"发生未知错误：{e}")

def read_dpo_data(path):
    """
    从 JSON 数据中提取 conversation 中的 system 和 human 对话内容
    参数:
        json_data (list): 包含多个字典的列表，每个字典包含 'conversations' 键
    返回:
        list: 包含多个对话对的列表，每个元素为 {"system":..., "human":...}
    """

    result = []
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for item in json_data:
        conversations = item.get("conversations", [])
        
        system_values = []
        human_values = []
        
        # 分离 system 和 human 的 value
        for conv in conversations:
            if conv.get("from") == "system":
                system_values.append(conv.get("value", ""))
            elif conv.get("from") == "human":
                human_values.append(conv.get("value", ""))
        
        # 按顺序配对 system 和 human
        for i in range(min(len(system_values), len(human_values))):
            result.append({
                "system": system_values[i],
                "human": human_values[i]
            })
    
    return result


def build_format_datasetV2(n: int, output_path: str, input_path: str = "/root/autodl-tmp/QA_Evaluation/RequireDataset.json"):

    llm_api = Large_Language_Model_API(model="gpt-4o")
    init_prompt = False
    high_quality_samples = []
    results = read_dpo_data(input_path)

    format_prompt = """
**Expected Output Example**: 
<Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
<Action> Query 0 | Query Rewrite | Nums 3 </Action>
<Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail>
<Action> Query 0 | Query Reason | Nums 1 </Action>
<Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>
<END>
**Please follow the format strictly!** """.strip()

    with tqdm(total = n) as pbar:
        for result in results:
            if len(high_quality_samples) > n:
                break
            system, human = result["system"], result["human"]

            if not init_prompt:
                llm_api.init_llm(system)
                init_prompt = True

            response = llm_api.generate(human + format_prompt, return_time = False)
            # 高质量数据保存
            alpaca_sample = format_to_alpaca(
                instruction = system + human,
                user_input='',
                model_output=response
            )
            high_quality_samples.append(alpaca_sample)
            pbar.update(1)

    with open(output_path, 'a', encoding='utf-8') as f:
        json.dump(high_quality_samples, f, indent=2, ensure_ascii=False)

    return


def build_garbage_dataset():
    """构建垃圾人工矫正数据集"""
    """太贵了，跑是能跑，但是很贵，算了"""


    prompt_yeild = lambda x: f"""
---------------------------------------
Here are some tools below: 
<Query Rewrite> Rephrase query: Generate different queries to cover more topics. 
<Query Reason> Analyze query: Reason what should be retrieved to improve query quality.
<Query Extract> Extract query from documents: Extract query from the given query with retrieved documents. 
<Document Filter> Filter documents: Analyse all these documents are relevant to the user input or not. 
<Document Retrieve> Retrieve documents: Pick up the most valuable query you think to retrieve. 
<Stop> Terminate: When user input is sufficiently clear to answer. After that, You will get last chance to generate something relevant to help improve answer.

Response Format Specification
- <Think> Make sure you have enough contemplation before taking any action, and put them in this tag. 
- <Action> Target Query(or Document) | Tool you want to take | Number of query(or document) to generate
  - Part 1: Original query(or document) reference, you can reference user input as "Query 0" speciffically, others can be "Query 1", "Query 2", ...
  - Part 2: Operation type (Query Rewrite/Query Reason/...)
  - Part 3: Quantity parameter (only required when generate new query or document)
- <Detail> You can put all details you want to say to the next agent here. 
  - Optional, you can ignore it when you think it's unnecessary. 
  - Use multiple <Detail> tags when you want to describe action seperately.
- <END> Use this tag to end the query immediately.
---------------------------------------

**Please Generate Garbage Response Example In Your Output Throught This Rules: **
1. **Read the rules above and keep them in mind**. Find your way to generate garbage and put them in response. 
2. **Don't follow the tags format require absolutely.** Make sure your output only contains less than one of the tags. 
3. **Say something relevant but don't answer the question.** Imitate the output example below. Act like you're an assistant but not very smart. 

**Example Input**
User Input (Query Need To Tackle): Where did the us first attack in ww1?
**Expected Output Example**: 
Let me think about this.\n\nOkay, the user is asking where the US first attacked in WW1. The initial query is straightforward: \"Where did the US first attack in ww1?\" \n\nFirst, I need to understand what the user is really asking. The user might be looking for the first military action taken by the US in World War I. However, the US was a neutral country initially and did not participate in the war until 1917. So the first attack by the US in WW1 would be the entry into the war, which was the U.S. entry into World War I, specifically the declaration of war by the United States on Germany on April 6, 1917. But the user might be confused because they are asking about the first attack, which is more about the initial military engagement.\n\nHowever, the user might have a different perspective. Maybe they think that the US first attacked in a different context, such as a skirmish or a naval action. But given that the US entered the war in 1917, the first major attack would be the entry into the war.\n\nSince the user's query is very direct, there's no need for further refinement. The initial query is already clear. However, the assistant's role is to generate different queries to cover more topics if necessary. But in this case, the user's question is straightforward, and the answer is well-known.\n\nBut the assistant needs to check if there's

**Current User Input**: 
{x}

**Please Generate Garbage Response to user input.**
Your Output: 

"""

    llm_api = Large_Language_Model_API(model = "gpt-4o")
    llm_api.init_llm('')
    parquet_dataset = golden_dataset.from_path(path = "/root/autodl-tmp/data/ambigqa/full/train.parquet")
    # 将数据保存到本地 JSON 文件
    file_path = "/root/autodl-tmp/data/garbage_samples.json"
    
    garbage_samples = []
    with tqdm(total = len(parquet_dataset)) as pbar:
        for idx in range(len(parquet_dataset)):
            questions, answers = parquet_dataset.get_data(batch_size = 1)
            question = questions[0]
            garbage_response = llm_api.generate(prompt_yeild(question))
            garbage_key = "item"
            garbage_samples.append({garbage_key: garbage_response})
        
            pbar.update()
            if idx // (0.2 * len(parquet_dataset)) == 0.0: 
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(garbage_samples, f, ensure_ascii=False, indent=4)

                pbar.set_description("Saved Already")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(garbage_samples, f, ensure_ascii=False, indent=4)

    return 




def generate_garbage():
    """构建垃圾人工矫正数据集"""

    prompt_yeild = lambda x: f"""
---------------------------------------
Here are some tools below: 
<Query Rewrite> Rephrase query: Generate different queries to cover more topics. 
<Query Reason> Analyze query: Reason what should be retrieved to improve query quality.
<Query Extract> Extract query from documents: Extract query from the given query with retrieved documents. 
<Document Filter> Filter documents: Analyse all these documents are relevant to the user input or not. 
<Document Retrieve> Retrieve documents: Pick up the most valuable query you think to retrieve. 
<Stop> Terminate: When user input is sufficiently clear to answer. After that, You will get last chance to generate something relevant to help improve answer.

Response Format Specification
- <Think> Make sure you have enough contemplation before taking any action, and put them in this tag. 
- <Action> Target Query(or Document) | Tool you want to take | Number of query(or document) to generate
  - Part 1: Original query(or document) reference, you can reference user input as "Query 0" speciffically, others can be "Query 1", "Query 2", ...
  - Part 2: Operation type (Query Rewrite/Query Reason/...)
  - Part 3: Quantity parameter (only required when generate new query or document)
- <Detail> You can put all details you want to say to the next agent here. 
  - Optional, you can ignore it when you think it's unnecessary. 
  - Use multiple <Detail> tags when you want to describe action seperately.
- <END> Use this tag to end the query immediately.
---------------------------------------

**Please Generate Garbage Response Example In Your Output Throught This Rules: **
1. **Read the rules above and keep them in mind**. Find your way to generate garbage and put them in response. 
2. **Don't follow the tags format require absolutely.** Make sure your output only contains less than one of the tags. 
3. **Say something relevant but don't answer the question.** Imitate the output example below. Act like you're an assistant but not very smart. 

**Example Input**
User Input (Query Need To Tackle): Where did the us first attack in ww1?
**Expected Output Example**: 
Let me think about this.\n\nOkay, the user is asking where the US first attacked in WW1. The initial query is straightforward: \"Where did the US first attack in ww1?\" \n\nFirst, I need to understand what the user is really asking. The user might be looking for the first military action taken by the US in World War I. However, the US was a neutral country initially and did not participate in the war until 1917. So the first attack by the US in WW1 would be the entry into the war, which was the U.S. entry into World War I, specifically the declaration of war by the United States on Germany on April 6, 1917. But the user might be confused because they are asking about the first attack, which is more about the initial military engagement.\n\nHowever, the user might have a different perspective. Maybe they think that the US first attacked in a different context, such as a skirmish or a naval action. But given that the US entered the war in 1917, the first major attack would be the entry into the war.\n\nSince the user's query is very direct, there's no need for further refinement. The initial query is already clear. However, the assistant's role is to generate different queries to cover more topics if necessary. But in this case, the user's question is straightforward, and the answer is well-known.\n\nBut the assistant needs to check if there's

**Current User Input**: 
{x}

**Please Generate Garbage Response to user input.**
Your Output: 

"""

    llm_api = Large_Language_Model_API(model = "gpt-4o-mini")
    llm_api.init_llm('')
    parquet_dataset = golden_dataset.from_path(path = "/root/autodl-tmp/data/ambigqa/full/train.parquet")
    
    questions, answers = parquet_dataset.get_data(batch_size = 1)
    question = questions[0]
    answer = answers[0]

    print('=' * 20)
    print('User Input:' , question)
    print('=' * 20)
    print('Garbage:')
    print(llm_api.generate(prompt_yeild(question)))

def convert_dpo_to_alpaca(input_folder, output_file):
    """
    遍历指定文件夹下的所有 JSON 文件，提取 conversation 中的 system 和 human 部分，
    拼接为 instruction，并提取 chosen 部分作为 output，导出为 Alpaca 格式 JSON。
    
    参数:
        input_folder (str): 包含 JSON 文件的文件夹路径。
        output_file (str): 输出的 Alpaca 格式 JSON 文件路径。
    """
    alpaca_data = []

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            # 提取 instruction：system 和 human 的 value
                            instruction_parts = []
                            for msg in item.get('conversations', []):
                                if msg['from'] in ['system', 'human']:
                                    instruction_parts.append(msg['value'])
                            instruction = '\n'.join(instruction_parts)

                            # 提取 output：chosen 的 value
                            output = item.get('chosen', {}).get('value', '')

                            # 构造 Alpaca 格式的条目
                            alpaca_item = {
                                'instruction': instruction,
                                'input': '',
                                'output': output
                            }
                            alpaca_data.append(alpaca_item)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"成功转换为 Alpaca 格式，输出文件为: {output_file}")