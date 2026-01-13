import unittest
import pandas as pd
import os
import random
from typing import Tuple, List, Union
from unittest.mock import patch, mock_open, MagicMock
import tempfile
from functools import partial
import numpy as np
import json


from ..RAG_modules import AERR, Decision_Agent, Generate_Agent, Execution_Agent
from ..config import AERRConfig, StrategyParams
from ..dataset import ConversationTree

class TestConversationTree(unittest.TestCase):
    def setUp(self):
        self.tree = ConversationTree()


    def test_timeForward(self):

        # 添加第一层（根层）
        self.tree.add_layer(
            sampling_prompts=[["Q1"]],
            decision_model_responses=[["A1"]],
            time_lis=[0.0]
        )

        # 添加第一层（根层）
        self.tree.add_layer(
            sampling_prompts=[["Q2", "Q3"]],
            decision_model_responses=[["A2", "A3"]],
            time_lis=[2.0, 3.0]
        )

        # 添加第二层
        self.tree.add_layer(
            sampling_prompts=[["Q1.1", "Q2.1"], ["Q2.2"]],
            decision_model_responses=[["A1.1", "A2.1"], ["A2.2"]],
            time_lis=[3.0, 4.0, 5.0]
        )

        # 添加第三层（最后一层）
        self.tree.add_layer(
            sampling_prompts=[["Q1.1.1"], ["Q2.1.1"], ["Q2.2.1"]],
            decision_model_responses=[["A1.1.1"], ["A2.1.1"], ["A2.2.1"]],
            time_lis=[6.0, 7.0, 8.0]
        )

        # 调用方法
        cumulative_times = self.tree.get_cumulative_times_last_layer()
        print(cumulative_times)  # 输出: [1+3+6=10, 2+4+7=13, 2+5+8=15]
        self.assertEqual(cumulative_times, [11, 13, 16])
        return 
    

    def test_initialization(self):
        """测试初始化是否为空"""
        self.assertEqual(self.tree.layers, [])
        self.assertEqual(self.tree.layer_metadata, [])

    def test_add_layer_structure(self):
        """测试添加层后结构是否正确"""
        sampling_prompts = [["p1", "p2"]]
        responses = [["r1", "r2"]]
        self.tree.add_layer(sampling_prompts, responses)

        self.assertEqual(len(self.tree.layers), 1)
        self.assertEqual(len(self.tree.layers[0]), 2)
        self.assertEqual(self.tree.layers[0][0]['input'], "p1")
        self.assertEqual(self.tree.layers[0][0]['response'], "r1")
        self.assertEqual(self.tree.layers[0][0]['children'], [])
        self.assertEqual(self.tree.layers[0][0]['reward'], None)


    def test_add_reward_invalid_length(self):
        """测试奖励长度不匹配时触发 warnings.warn 警告"""
        """
        该测试需要查看控制台
        """
        # 添加一层交互链：父节点生成两个子节点
        self.tree.add_layer(
            sampling_prompts=[["p1", "p2"]],  # 父节点生成两个子节点
            decision_model_responses=[["r1"]]  # 子节点响应
        )
        
        # 尝试添加长度为1的奖励列表（应触发 warnings.warn）
        # with self.assertWarns(UserWarning):  # 或指定更具体的警告类型，如 UserWarning
        #     self.tree.add_reward([10])  # 奖励长度应为2，但传入1

    def test_total_bottom_nodes(self):
        """测试最底层节点数"""
        self.assertEqual(self.tree.total_bottom_nodes(), 0)
        self.tree.add_layer([["p1", "p2"]], [["r1", "r2"]])
        self.assertEqual(self.tree.total_bottom_nodes(), 2)
        self.tree.add_layer([["s1"], ["s2"]], [["res1"], ["res2"]])
        self.assertEqual(self.tree.total_bottom_nodes(), 2)

    def test_get_node_info(self):
        """测试获取节点信息"""
        self.tree.add_layer([["p1", "p2"]], [["r1", "r2"]])
        node = self.tree.get_node_info(0, 0)
        self.assertEqual(node['input'], "p1")
        self.assertEqual(node['response'], "r1")
        self.assertEqual(node['children'], [])
        self.assertEqual(node['reward'], None)

    def test_get_all_nodes_values(self):
        """测试获取所有节点值"""
        self.tree.add_layer([["p1", "p2"]], [["r1", "r2"]])
        self.tree.add_layer([["q1"], ["q2"]], [["r1"], ["r2"]])
        all_nodes = self.tree.get_all_nodes_values()
        print(all_nodes)
        self.assertEqual(len(all_nodes), 4)
        self.assertEqual(all_nodes[0]['reward'], None)
        self.assertEqual(all_nodes[1]['reward'], None)

    def test_format_reward(self):
        """测试格式奖励计算是否正确"""
        # 构建测试数据
        sampling_prompts = [["B1", "B2"]]
        responses = [
            [# 正确格式示例
            "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
            # 错误格式示例：缺少 <Detail>
            "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><END>",
            ]]
        
        self.tree.add_layer(sampling_prompts, responses)
        
        # 计算格式奖励
        self.tree.evaluate_format_score()
        
        # 验证奖励值
        node1 = self.tree[0, 0]
        node2 = self.tree[0, 1]
        
        # 正确格式应得满分 1.0
        self.assertAlmostEqual(node1["reward"], 1.0, places=2)
        
        # 错误格式：缺少 <Detail> 标签（扣 0.25）
        self.assertAlmostEqual(node2["reward"], 0.55, places=2)

    def test_add_reward_with_format(self):
        """测试 add_reward 是否正确叠加路径奖励与格式奖励"""
        # 构建测试数据
        sampling_prompts = [["B1", "B2", "B3"]]
        responses = [[
            # 正确格式（格式奖励 1.0）
            "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
            # 错误格式（格式奖励 0.75）
            "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><END>",
            # 正确格式（格式奖励 1.0）
            "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
        ]]

        self.tree.add_layer(sampling_prompts, responses)
        
        # 先计算格式奖励
        self.tree.evaluate_format_score()
        
        # 模拟路径奖励（最底层节点的 reward）
        path_rewards = [0.5, 0.3, 0.7]  # 与节点顺序对应
        
        # 分配路径奖励（叠加格式奖励）
        self.tree.add_reward(path_rewards)
        
        # 验证最终奖励
        node1 = self.tree[0, 0]  # 格式奖励 1.0 + 路径奖励 0.5 = 1.5
        node2 = self.tree[0, 1]  # 格式奖励 0.55 + 路径奖励 0.3 = 0.85
        node3 = self.tree[0, 2]  # 格式奖励 1.0 + 路径奖励 0.7 = 1.7
        
        self.assertAlmostEqual(node1["reward"], 1.5, places=2)
        self.assertAlmostEqual(node2["reward"], 0.85, places=2)
        self.assertAlmostEqual(node3["reward"], 1.7, places=2)

        # 验证最高路径优先覆盖规则
        # 构造两条路径，其中 node1 在两条路径中，但第二条路径奖励更高
        self.tree.clear() # 重置树
        self.tree.add_layer([["A", "B"]], [["r1", "r2"]])
        sampling_prompts = [["B1", "B2"], ["C1", "C2", "C3"]]
        responses = [
            # Layer 0
            ["<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
             "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>"],
            # Layer 1
            ["<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
             "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
             "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>"]
        ]
        self.tree.add_layer(sampling_prompts, responses)
        self.tree.evaluate_format_score()  # 所有格式奖励为 1.0

        # 设置最底层节点的路径奖励（末端节点）
        self.tree.add_reward([0.3, 0.5, 0.7, 0.3, 0.2])  # 最底层节点 reward 为 [0.3, 0.5, 0.7]

        # 验证最高路径优先覆盖
        node1 = self.tree[0, 0]  # 被两条路径覆盖，但只保留最高路径奖励
        node2 = self.tree[0, 1]  # 未被更高路径覆盖，保持 1.0 + 0.5 = 1.5
        node3 = self.tree[1, 0]  # 被分配了 1.0 + 0.3 = 1.3
        node4 = self.tree[1, 2]  # 被分配了 1.0 + 0.7 = 1.7

        # 最终 node1 应该被更高路径（1.7）覆盖
        self.assertAlmostEqual(node1["reward"], 1.5, places=2)
        self.assertAlmostEqual(node2["reward"], 1.7, places=2)
        self.assertAlmostEqual(node3["reward"], 1.3, places=2)
        self.assertAlmostEqual(node4["reward"], 1.7, places=2)

    def test_build_pair_basic(self):
        """测试基本的 pair 构建逻辑（两个子节点）"""
        # 构造父子层级
        self.tree.add_layer(
            sampling_prompts=[["parent_input"]],
            decision_model_responses=[["parent_response"]],
            systems=[["parent_system"]]
        )
        self.tree.add_layer(
            sampling_prompts=[["child1_input", "child2_input"]],
            decision_model_responses=[["child1_response", "child2_response"]],
            systems=[["child1_system", "child2_system"]],
            reward=[3, 1]  # 假设有 reward 参数传递
        )

        # 手动设置子节点的 reward（模拟 add_reward 方法）
        for idx, reward in enumerate([3, 1]):
            self.tree.layers[1][idx]["reward"] = reward

        # 构建 pair
        pairs = self.tree.build_pair()

        # 预期结果：一个啊！！！
        self.assertEqual(len(pairs), 1)

        # 验证第一个 pair（3 > 1）
        p1 = pairs[0]
        self.assertEqual(p1["conversations"], [
            {"from": "system", "value": "child1_system"},
            {"from": "human", "value": "child1_input"}
        ])
        self.assertEqual(p1["chosen"]["value"], "child1_response")
        self.assertEqual(p1["rejected"]["value"], "child2_response")

    def test_build_pair_multiple_children(self):
        """测试多个子节点的 pair 构建（三个子节点）"""
        # 构造父子层级
        self.tree.add_layer(
            sampling_prompts=[["parent_input"]],
            decision_model_responses=[["parent_response"]],
            systems=[["parent_system"]]
        )
        self.tree.add_layer(
            sampling_prompts=[["c1_input", "c2_input", "c3_input"]],
            decision_model_responses=[["c1_response", "c2_response", "c3_response"]],
            systems=[["c1_system", "c2_system", "c3_system"]],
            
        )
        self.tree.add_reward([3, 2, 1])

        # 构建 pair
        pairs = self.tree.build_pair()

        # 预期结果：n个，n为传入的可选的response数量。画个矩阵就懂了
        self.assertEqual(len(pairs), 3)

        # 验证部分 pair（仅检查部分逻辑）
        p1 = pairs[0]
        self.assertEqual(p1["chosen"]["value"], "c1_response")
        self.assertEqual(p1["rejected"]["value"], "c2_response")

        p2 = pairs[2]
        self.assertEqual(p2["chosen"]["value"], "c2_response")
        self.assertEqual(p2["rejected"]["value"], "c3_response")

    def test_build_pair_invalid_rewards(self):
        """测试奖励为 0 或 None 的情况（不生成 pair）"""
        # 构造父子层级
        self.tree.add_layer(
            sampling_prompts=[["parent_input"]],
            decision_model_responses=[["parent_response"]],
            systems=[["parent_system"]]
        )
        self.tree.add_layer(
            sampling_prompts=[["child1_input", "child2_input"]],
            decision_model_responses=[["child1_response", "child2_response"]],
            systems=[["child1_system", "child2_system"]]
        )

        # 构建 pair
        pairs = self.tree.build_pair()
        self.assertEqual(len(pairs), 0)


    def test_build_pair_no_children(self):
        """测试父节点没有子节点的情况（不生成 pair）"""
        self.tree.add_layer(
            sampling_prompts=[["parent_input"]],
            decision_model_responses=[["parent_response"]],
            systems=[["parent_system"]]
        )
        pairs = self.tree.build_pair()
        self.assertEqual(len(pairs), 0)

    def test_build_pair_same_rewards(self):
        """测试子节点奖励相同的情况（不生成 pair）"""
        # 构造父子层级
        self.tree.add_layer(
            sampling_prompts=[["parent_input"]],
            decision_model_responses=[["parent_response"]],
            systems=[["parent_system"]]
        )
        self.tree.add_layer(
            sampling_prompts=[["child1_input", "child2_input"]],
            decision_model_responses=[["child1_response", "child2_response"]],
            systems=[["child1_system", "child2_system"]]
        )

        # 设置相同奖励
        self.tree.layers[1][0]["reward"] = 2
        self.tree.layers[1][1]["reward"] = 2

        # 构建 pair
        pairs = self.tree.build_pair()
        self.assertEqual(len(pairs), 0)

    def test_to_json_file_creation(self):
        # 构建对话树
        self.tree.add_layer(
            sampling_prompts=[["What is your name?"]],
            decision_model_responses=[["My name is Alice."]]
        )
        self.tree.add_layer(
            sampling_prompts=[["How old are you?", "How old are you?"]],
            decision_model_responses=[["I'm 25.", "I'm from Beijing."]]
        )
        self.tree.add_reward(rewards=[1, 3])

        # 保存为 JSON
        json_dir = "test_output"
        self.tree.pair2json(json_dir)

        # 验证文件存在
        file_path = os.path.join(json_dir, "dpo_pairs.json")
        self.assertTrue(os.path.exists(file_path))

        # 清理
        os.remove(file_path)
        os.rmdir(json_dir)

    def test_to_json_content(self):
        # 构建对话树
        self.tree.add_layer(
            sampling_prompts=[["What is your name?"]],
            decision_model_responses=[["My name is Alice."]]
        )
        self.tree.add_layer(
            sampling_prompts=[["How old are you?", "How old are you?"]],
            decision_model_responses=[["I'm 25.", "I'm from Beijing."]]
        )
        self.tree.add_reward(rewards=[3, 1])

        # 保存为 JSON
        json_dir = "test_output"
        pairs = self.tree.build_pair()
        self.tree.pair2json(json_path=json_dir)

        # 验证内容
        file_path = os.path.join(json_dir, "dpo_pairs.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 预期内容
        expected = {
            "conversations": [
                {"from": "human", "value": "How old are you?"}
            ],
            "chosen": {"from": "gpt", "value": "I'm 25."},
            "rejected": {"from": "gpt", "value": "I'm from Beijing."}
        }
        self.assertEqual(data, [expected])

        # 清理
        os.remove(file_path)
        os.rmdir(json_dir)




# ========================================测试train中构建的golden answer行为===================================================
# 假设 golden_dataset 类定义在 golden_dataset.py 中
from ..dataset import golden_dataset

class TestGoldenDataset(unittest.TestCase):

    def setUp(self):
        """创建测试用的 DataFrame """
        self.test_df = pd.DataFrame({
            'question': ['q1', 'q2', 'q3', 'q4', 'q5'],
            'golden_answer': ['a1', 'a2', 'a3', 'a4', 'a5'], 
            'nq_answer': ['a1', 'a2', 'a3', 'a4', 'a5'], 
        })

    def test_init_with_dataframe(self):
        """测试初始化方法 - 正常情况"""
        dataset = golden_dataset(self.test_df)
        self.assertEqual(dataset.questions, ['q1', 'q2', 'q3', 'q4', 'q5'])
        self.assertEqual(dataset.golden_answers, ['a1', 'a2', 'a3', 'a4', 'a5'])
        self.assertEqual(len(dataset), 5)

    def test_init_valueerror(self):
        """测试初始化方法 - 数据长度不一致"""
        df = pd.DataFrame({'question': ['q1'], 'golden_answer': ['a1', 'a2']})
        with self.assertRaises(ValueError):
            golden_dataset(df)

    def test_get_data(self):
        dataset = golden_dataset(self.test_df)
        print("Questions: ", dataset.questions)
        print("GoldenAnswer: ", dataset.golden_answers)

        # 第一次调用
        result = dataset.get_data(2)
        print(f"第一次调用结果: {result}")
        self.assertEqual(result, (['q1', 'q2'], ['a1', 'a2']))

        # 第二次调用
        result = dataset.get_data(2)
        print(f"第二次调用结果: {result}")
        self.assertEqual(result, (['q3', 'q4'], ['a3', 'a4']))

        # 第三次调用（剩余1个）
        result = dataset.get_data(2)
        print(f"第三次调用结果: {result}")
        self.assertEqual(result, (['q5'], ['a5']))

        # 第四次调用（超出范围）
        result = dataset.get_data(2)
        print(f"第四次调用结果: {result}")
        self.assertEqual(result, ([], []))

    def test_get_full_data(self):
        """测试 get_full_data 方法"""
        dataset = golden_dataset(self.test_df)
        
        # 第一次调用 get_full_data
        result = dataset.get_full_data()
        print(f"第一次 get_full_data 结果: {result}")
        self.assertEqual(result, (['q1', 'q2', 'q3', 'q4', 'q5'], ['a1', 'a2', 'a3', 'a4', 'a5']))
        
        # 调用 get_data 后再次调用 get_full_data
        dataset.get_data(2)
        result = dataset.get_full_data()
        print(f"第二次 get_full_data 结果: {result}")
        self.assertEqual(result, (['q1', 'q2', 'q3', 'q4', 'q5'], ['a1', 'a2', 'a3', 'a4', 'a5']))


    def test_shuffle(self):
        """测试 shuffle 方法"""
        dataset = golden_dataset(self.test_df)
        original_order = dataset.questions.copy()
        
        # 打印原始顺序
        print(f"原始顺序: {original_order}")
        
        dataset.shuffle()
        
        # 打印打乱后的顺序
        print(f"打乱后的顺序: {dataset.questions}")
        self.assertNotEqual(dataset.questions, original_order)
        
        # 打印 current_index
        print(f"current_index: {dataset.current_index}")
        self.assertEqual(dataset.current_index, 0)
        
        # 打印数据长度
        print(f"数据长度: {len(dataset)}")
        self.assertEqual(len(dataset), 5)


    def test_from_path_csv(self):
        """测试 from_path 方法 - CSV 文件"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
            self.test_df.to_csv(tmpfile.name)
            path = tmpfile.name
            print(f"临时文件路径: {path}")
        
        dataset = golden_dataset.from_path(path)
        
        # 打印 questions 列表
        print(f"从 CSV 加载的 questions: {dataset.questions}")
        self.assertEqual(dataset.questions, ['q1', 'q2', 'q3', 'q4', 'q5'])
        
        # 删除临时文件
        os.unlink(path)
        print(f"临时文件 {path} 已删除")

    def test_from_path_csv(self):
        """测试 from_path 方法 - CSV 文件"""
        with tempfile.NamedTemporaryFile(dir = './cache/', 
                                         delete=False, suffix='.csv') as tmpfile:
            self.test_df.to_csv(tmpfile.name)
            path = tmpfile.name
            print(f"临时文件路径: {path}")
        
        dataset = golden_dataset.from_path(path)
        
        # 打印 questions 列表
        print(f"从 CSV 加载的 questions: {dataset.questions}")
        self.assertEqual(dataset.questions, ['q1', 'q2', 'q3', 'q4', 'q5'])
        
        # 删除临时文件
        os.unlink(path)
        print(f"临时文件 {path} 已删除")

    def test_from_path_unsupported_extension(self):
        """测试 from_path 方法 - 不支持的文件类型"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmpfile:
            with self.assertRaises(ValueError):
                golden_dataset.from_path(tmpfile.name)

    def test_shorten_data(self):
        """测试 shorten_data 方法"""
        dataset = golden_dataset(self.test_df)

        print("Current len(dataset) Result Before Shorten: ", len(dataset))
        dataset.shorten_data(3)


        result = len(dataset)
        print("Current len(dataset) Result: ", result)

        self.assertEqual(result, 3)

        result = dataset.get_full_data()
        print("Current get_full_data() Result: ", result)

        self.assertEqual(result, (['q1', 'q2', 'q3'], ['a1', 'a2', 'a3']))

    def test_restore_data(self):
        """测试 restore_data 方法"""
        dataset = golden_dataset(self.test_df)
        dataset.shorten_data(3)
        dataset.restore_data()
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.get_full_data(), (['q1', 'q2', 'q3', 'q4', 'q5'], ['a1', 'a2', 'a3', 'a4', 'a5']))

    def test_empty_dataset(self):
        """测试空数据集"""
        dataset = golden_dataset()
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.get_data(1), ([], []))
        self.assertEqual(dataset.get_full_data(), ([], []))
        dataset.shuffle()  # 不应报错

    def test_get_data_after_shorten(self):
        """测试缩短数据后的 get_data"""
        dataset = golden_dataset(self.test_df)
        dataset.shorten_data(2)
        result = dataset.get_data(2)
        print(f"1st get_data() after shorten: {result}")
        
        self.assertEqual(result, (['q1', 'q2'], ['a1', 'a2']))
        
        result = dataset.get_data(2)
        print(f"2nd get_data() after shorten: {result}")

        self.assertEqual(result, ([], []))

class TestSampleCritic(unittest.TestCase):

    def setUp(self):
        """创建测试用的 DataFrame """
        self.valid_prompt = "<Think> The query involves understanding the role of machine learning in optimizing search engine performance. This might include aspects like ranking algorithms, personalization, natural language understanding, and pattern analysis. I need to explore broader aspects and ensure the retrieved documents tie closely to machine learning's impact on various dimensions of search engines. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 2 </Action>\n<Detail> Generate alternate queries to explore related angles to \"machine learning and search engine performance,\" such as personalization and ranking algorithms. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze the original query to determine what key points or data should be retrieved to create a comprehensive answer, like examples of machine learning algorithms or specific case studies in search engine optimization. </Detail>\n\n<Action> Query 0 | Document Retrieve | Nums 3 </Action>\n<Detail> Retrieve documents or sources explaining the application of machine learning, specific case studies of search engine improvement, or comparisons between traditional and machine learning-enhanced methods. </Detail>\n\n<END>"

        self.part_invalid = "The final answer should be within a box.\n\nThe current state shows that the user's input is \"When does the next season of ninjago come out?\" and the only document retrieved is \"Nothing yet.\" \n\nThe tools available are:\n- Query Rewrite: To generate different queries to cover more topics.\n- Query Reason: To analyze what should be retrieved to improve query quality.\n- Document Filter: To determine if documents are relevant.\n- Document Retrieve: To pick up the most valuable query.\n- Stop: To end the interaction once sufficient documents are retrieved.\n\nYour response should now be generated.\n</think>\n\n<Think> The query seeks information about the release date of the next season of Ninjago. Since no documents are available, the next step is to generate alternative queries to explore related topics such as upcoming releases, production details, or fan discussions. This will ensure the search is comprehensive and covers all possible angles. </Think>\n<Action> Query 0 | Query Rewrite | Nums 3 </Action>\n<Detail> Generate alternative queries to explore related topics like \"Upcoming releases of Ninjago seasons,\" \"Production schedule for new episodes,\" or \"Fan discussions about the next season.\" These queries will help capture diverse information sources, including news articles, official announcements, and community forums. </Detail>\n<END>"

        self.all_invalid = "Let's begin. I need to determine the best way to approach the user's query about when the next season of Ninjago comes out. The user is asking for a specific release date, so the first step is to check if there are any documents or information sources that can provide this information.\n\nSince there are no documents retrieved yet, I need to generate alternative queries to cover different aspects of the topic. The user's original query is straightforward, but to ensure we get the most relevant information, I should consider generating different queries that might lead to the same or similar information. For example, searching for the release date of the next season, or related news articles about the show's schedule.\n\nI will start by rewriting the query to focus on different aspects of the season release. This will help in retrieving a variety of documents that might be relevant to the user's question. Then, I will analyze the query to determine what specific information needs to be retrieved, such as the exact release date, any announcements, or related news. \n\nLet me generate some alternative queries first. \n</think>\n\n<Think???> The query focuses on determining the release date of the next season of Ninjago. To gather accurate information, I need to generate alternative queries that might lead to relevant documents, such as official announcements, news articles, or fan discussions. The initial query is direct, but expanding the search terms can help capture more comprehensive results. </Think?>\n<Action?> Query 0 | Query Rewrite | Nums"

        from ..dataset import SampleCritic, GarbageGenerator
        self.critic = SampleCritic()
        self.garbage = GarbageGenerator()

    def test_ValidSample_on_SampleCritic(self):
        """SampleCritic有效结果测试正常"""
        output = self.critic.extract_valid_content(self.valid_prompt)
        # print('=' * 20)
        # print('Origin Output: ', output)
        # print('=' * 10)
        # print('Expected Output: ', self.valid_prompt.replace('\n\n', '\n'))
        return 
    
    def test_PartInvalidSample_on_SampleCritic(self):
        """SampleCritic部分有效结果提取测试正常"""
        output = self.critic.extract_valid_content(self.part_invalid)
        # print('=' * 20)
        # print('Input: ', self.part_invalid)
        # print('=' * 10)
        # print('Output: ', output)
        return 

    def test_AllInvalidSample_on_SampleCritic(self):
        """SampleCritic部分有效结果提取测试正常"""
        """如果什么都没有，则仅仅只会返回<END>的tag"""
        output = self.critic.extract_valid_content(self.all_invalid)
        # print('=' * 20)
        # print('Input: ', self.all_invalid)
        # print('=' * 10)
        # print('Output: ', output)
        return 
    
    def test_ValidSample_on_GarbageGenerator(self):
        """GarbageGenerator全部有效结果提取测试正常"""
        """很好，什么都不会返回"""
        output = self.garbage.extract_invalid_content(self.valid_prompt)
        # print('=' * 20)
        # print('Input: ', self.valid_prompt)
        # print('=' * 10)
        # print('Output: ', output)
        return 
    
    def test_PartInvalidSample_on_GarbageGenerator(self):
        """GarbageGenerator部分有效结果提取测试正常"""
        """返回不包含格式的部分"""
        output = self.garbage.extract_invalid_content(self.part_invalid)
        # print('=' * 20)
        # print('Input: ', self.part_invalid)
        # print('=' * 10)
        # print('Output: ', output)
        return 
    
    def test_AllInvalidSample_on_GarbageGenerator(self):
        """GarbageGenerator全部无效结果提取测试正常"""
        """返回不包含格式的部分"""
        output = self.garbage.extract_invalid_content(self.all_invalid)
        print('=' * 20)
        print('Input: ', self.all_invalid)
        print('=' * 10)
        print('Output: ', output)
        return 
    
def test_critic():
    """判别功能测试"""
    
    from ..dataset import SampleCritic
    input_prompt = "<Think> The query involves understanding the role of machine learning in optimizing search engine performance. This might include aspects like ranking algorithms, personalization, natural language understanding, and pattern analysis. I need to explore broader aspects and ensure the retrieved documents tie closely to machine learning's impact on various dimensions of search engines. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 2 </Action>\n<Detail> Generate alternate queries to explore related angles to \"machine learning and search engine performance,\" such as personalization and ranking algorithms. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze the original query to determine what key points or data should be retrieved to create a comprehensive answer, like examples of machine learning algorithms or specific case studies in search engine optimization. </Detail>\n\n<Action> Query 0 | Document Retrieve | Nums 3 </Action>\n<Detail> Retrieve documents or sources explaining the application of machine learning, specific case studies of search engine improvement, or comparisons between traditional and machine learning-enhanced methods. </Detail>\n\n<END>"

    critic = SampleCritic()
    print(critic.strCritic(input_prompt, need_print = True))

def testFormatReduction():
    """格式筛选测试"""
    from ..dataset import SampleCritic

    input_prompt = "<Think> The query involves understanding the role of machine learning in optimizing search engine performance. This might include aspects like ranking algorithms, personalization, natural language understanding, and pattern analysis. I need to explore broader aspects and ensure the retrieved documents tie closely to machine learning's impact on various dimensions of search engines. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 2 </Action>\n<Detail> Generate alternate queries to explore related angles to \"machine learning and search engine performance,\" such as personalization and ranking algorithms. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze the original query to determine what key points or data should be retrieved to create a comprehensive answer, like examples of machine learning algorithms or specific case studies in search engine optimization. </Detail>\n\n<Action> Query 0 | Document Retrieve | Nums 3 </Action>\n<Detail> Retrieve documents or sources explaining the application of machine learning, specific case studies of search engine improvement, or comparisons between traditional and machine learning-enhanced methods. </Detail>\n\n<END>"

    critic = SampleCritic()
    print(critic.extract_valid_content(input_prompt))

    return 

def test_sample2garbage():
    from ..dataset import GarbageGenerator
    input_prompt = "<Think> The query asks for an analysis of the primary causes (factors contributing to climate change) and the resulting effects (environmental, societal, and economic impacts). The query is broad and may benefit from refining or expanding to ensure coverage of relevant topics like greenhouse gases, deforestation, renewable energy, biodiversity loss, and global policies. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 3 </Action>\n<Detail> Generate diverse queries to explore specific subtopics of the original query. Possible areas include natural vs human causes, technological solutions, and impact on specific regions or ecosystems. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze what information should be retrieved to comprehensively answer the query. Exploration of scientific data, historical context, and proposed mitigation strategies may be valuable. </Detail>\n\n<Think> To ensure clarity and usefulness of retrieved information, the documents must directly address the query, including evidence-backed causes and a range of societal and planetary effects. Documents should ideally include statistical evidence, expert analysis, and case studies. </Think>\n\n<Action> Query 0 | Document Retrieve | Nums 4 </Action>\n<Detail> Use the original query to retrieve key documents focused on climate change causes and effects. Prioritize retrieving academic studies, reputable scientific reports (e.g., IPCC reports), and governmental policy analyses. </Detail>\n\n<Action> Query 0 | Document Filter | Nums 4 </Action>\n<Detail> Evaluate the relevance of retrieved documents based on their focus on the causes and effects of climate change. Filtering should emphasize documents that provide both comprehensive and specific insights into the query. </Detail>\n\n<END>"

    generator = GarbageGenerator()
    print(
    generator.modifySampleToGarbage(
        model_response = input_prompt
    ))
    
    return 


def test_sample2garbage():
    """
    测试成功，垃圾话，真垃圾吧
    """
    from ..dataset import GarbageGenerator
    input_prompt = "111 <Think> The query asks for an analysis of the primary causes (factors contributing to climate change) and the resulting effects (environmental, societal, and economic impacts). The query is broad and may benefit from refining or expanding to ensure coverage of relevant topics like greenhouse gases, deforestation, renewable energy, biodiversity loss, and global policies. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 3 </Action>\n<Detail> Generate diverse queries to explore specific subtopics of the original query. Possible areas include natural vs human causes, technological solutions, and impact on specific regions or ecosystems. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze what information should be retrieved to comprehensively answer the query. Exploration of scientific data, historical context, and proposed mitigation strategies may be valuable. </Detail>\n\n<Think> To ensure clarity and usefulness of retrieved information, the documents must directly address the query, including evidence-backed causes and a range of societal and planetary effects. Documents should ideally include statistical evidence, expert analysis, and case studies. </Think>\n\n<Action> Query 0 | Document Retrieve | Nums 4 </Action>\n<Detail> Use the original query to retrieve key documents focused on climate change causes and effects. Prioritize retrieving academic studies, reputable scientific reports (e.g., IPCC reports), and governmental policy analyses. </Detail>\n\n<Action> Query 0 | Document Filter | Nums 4 </Action>\n<Detail> Evaluate the relevance of retrieved documents based on their focus on the causes and effects of climate change. Filtering should emphasize documents that provide both comprehensive and specific insights into the query. </Detail>\n\n<END>"

    generator = GarbageGenerator()
    print(
    generator.generate(
        input_prompt
))
    
    return 







if __name__ == '__main__':
    unittest.main()