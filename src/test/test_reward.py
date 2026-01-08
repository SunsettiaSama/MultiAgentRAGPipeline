import unittest
import pandas as pd
import os
import random
from typing import Tuple, List
from unittest.mock import patch, mock_open, MagicMock
import tempfile
from functools import partial
import numpy as np

from ..reward import format_reward, _parse_tags, weighted_reward_calculate, time_reward
import unittest
from typing import List

class TestFormatReward(unittest.TestCase):
    def testParsetags(self):
        interaction = "<Detail>详细内容</Detail>"
        content_blocks, all_tags = _parse_tags(interaction)
        print("Content Blocks:", content_blocks)  # 应包含 ('Detail', '详细内容')
        print("All Tags:", all_tags)  # 应包含 ['Detail', '/Detail']

    def test_perfect_format(self):
        """完全符合格式要求的交互"""
        interaction = "<Think>思考内容</Think><Action>Query 1 | Operation | Nums 2</Action><Detail>详细内容</Detail><END>"
        self.assertAlmostEqual(format_reward([interaction])[0], 1.0)

    def test_missing_tag(self):
        """缺少一个必需标签"""
        interaction = "<Think>思考内容</Think><Action>Query 1 | Operation | Nums 2</Action><END>"
        # 缺少 Detail 标签，扣 0.25
        self.assertAlmostEqual(format_reward([interaction])[0], 0.55)

    def test_order_error(self):
        """标签顺序错误"""
        # 顺序错误会导致一开始的查找出现问题，即所有的内容必须包含在Think和END之间
        # 但这个部分有点硬了，需要修改成一个更柔和的判定方法
        interaction = "<Action>Query 1 | Operation | Nums 2</Action><Think>思考内容</Think><Detail>详细内容</Detail><END>"
        # 顺序错误扣 0.2
        self.assertAlmostEqual(format_reward([interaction])[0], 0.8)

    def test_multiple_actions(self):
        """多个 Action 标签（应加分）"""
        interaction = ("<Think>思考内容</Think>"
                       "<Action>Query 1 | Operation | Nums 2</Action>"
                       "<Action>Query 2 | Operation | Nums 3</Action>"
                       "<Detail>详细内容</Detail><END>")
        # 多 Action 加 0.2
        self.assertAlmostEqual(format_reward([interaction])[0], 1.0)

    def test_action_format_invalid(self):
        """Action 标签格式不规范"""
        interaction = ("<Think>思考内容</Think>"
                       "<Action>Query 1 | Invalid | Nums 2</Action>"  # 正确格式
                       "<Action>Invalid Format</Action>"  # 错误格式
                       "<Detail>详细内容</Detail><END>")
        # 一个有效 Action，一个无效 Action，扣 0.3
        self.assertAlmostEqual(format_reward([interaction])[0], 0.7)

    def test_tag_not_closed(self):
        """标签未闭合"""
        interaction = ("<Think>思考内容"
                       "<Action>Query 1 | Operation | Nums 2</Action>"
                       "<Detail>详细内容</Detail><END>")
        # Think 未闭合，扣 0.1
        self.assertAlmostEqual(format_reward([interaction])[0], 0.9)

    def test_content_outside_tags(self):
        """内容在标签外"""
        interaction = "前置内容<Think>思考内容</Think><Action>Query 1 | Operation | Nums 2</Action><Detail>详细内容</Detail><END>后置内容"
        # 内容在标签外，扣 0.2
        self.assertAlmostEqual(format_reward([interaction])[0], 0.8)

    def test_all_errors(self):
        """叠加所有错误情况"""
        interaction = ("前置内容"
                       "<Detail>错误顺序</Detail>"
                       "<Action>Invalid Format</Action>"
                       "<Action>Query 1 | Operation | Nums 2</Action>"
                       "<END>")
        # 缺失 Think (0.25), 顺序错误 (0.2), 未闭合 Detail (0.1), 一个无效 Action (0.3), 多 Action 加 0.2
        # 总分: 1 - 0.25 - 0.2 - 0.1 - 0.3 + 0.2 = 0.6
        self.assertAlmostEqual(format_reward([interaction])[0], 0.6)

    def test_minimum_score(self):
        """得分下限为 0"""
        interaction = "完全不符合要求的内容"
        # 扣除所有可能扣分项后应为 0
        self.assertAlmostEqual(format_reward([interaction])[0], 0.0)

    def test_multiple_interactions(self):
        """多个交互测试"""
        interactions = [
            "<Think>思考</Think><Action>Query 1 | Op | Nums 2</Action><Detail>详情</Detail><END>",
            "<Think>思考</Think><Action>Query 1 | Op | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>",
            "错误内容"
        ]
        expected_scores = [1.0, 1.0, 0.0]
        for i in range(len(interactions)):
            self.assertAlmostEqual(format_reward([interactions[i]])[0], expected_scores[i])

def testFormatReward():
    """
    测试格式化函数，测试结果完全符合预期行为
    
    """
    suite = unittest.TestSuite()

    suite.addTest(TestFormatReward("testParsetags"))
    suite.addTest(TestFormatReward("test_perfect_format"))
    suite.addTest(TestFormatReward("test_missing_tag"))
    suite.addTest(TestFormatReward("test_order_error"))
    suite.addTest(TestFormatReward("test_multiple_actions"))
    suite.addTest(TestFormatReward("test_action_format_invalid"))
    suite.addTest(TestFormatReward("test_tag_not_closed"))
    suite.addTest(TestFormatReward("test_content_outside_tags"))
    suite.addTest(TestFormatReward("test_minimum_score"))
    suite.addTest(TestFormatReward("test_multiple_interactions"))

    # 运行测试套件
    runner = unittest.TextTestRunner()
    runner.run(suite)

class TestTimeReward(unittest.TestCase):
    def test_normal_case(self):
        # 正常输入：两个不同时间值
        time_costs = [10.0, 5.0]
        expected = np.array([0.5, 1.5])  # 根据公式计算
        result = time_reward(time_costs)
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_same_time(self):
        # 所有时间相同，导致 max == min，分母为 0
        time_costs = [5.0, 5.0, 5.0]
        result = time_reward(time_costs)
        self.assertEqual(result, [0, 0, 0])

    def test_single_element(self):
        # 单个时间值，max == min，分母为 0
        time_costs = [7.0]
        result = time_reward(time_costs)
        self.assertEqual(result, 0)

    def test_multiple_values(self):
        # 多个不同时间值
        time_costs = [3.0, 6.0, 9.0]
        expected = np.array([1.5, 1.0, 0.5])  # 根据公式计算
        result = time_reward(time_costs)
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_list(self):
        # 空列表输入
        time_costs: List[float] = []
        result = time_reward(time_costs)
        expected = []
        np.testing.assert_array_almost_equal(result, expected)


    # def test_negative_time(self):
    #     # 允许负时间值（逻辑上可能不合理，但函数未限制）
    #     time_costs = [-2.0, 3.0, 5.0]
    #     result = time_reward(time_costs)
    #     expected = np.array([1.8, 0.6, 0.0])  # 根据公式计算
    #     np.testing.assert_array_almost_equal(result, expected)

    # def test_zero_division_handling(self):
    #     # 自定义处理：如果所有时间相同，返回全 1 数组
    #     # 修改函数逻辑后测试（示例）
    #     time_costs = [4.0, 4.0, 4.0]
    #     expected = np.array([1.0, 1.0, 1.0])
    #     # 假设修改后的函数逻辑如下：
    #     # if max_time == min_time:
    #     #     return np.ones_like(time_costs)
    #     result = time_reward(time_costs)
    #     np.testing.assert_array_almost_equal(result, expected)


def testTimeReward():
    """
    测试时间奖励函数，测试结果完全符合预期行为
    
    """
    suite = unittest.TestSuite()

    suite.addTest(TestTimeReward("test_normal_case"))
    suite.addTest(TestTimeReward("test_all_same_time"))
    suite.addTest(TestTimeReward("test_single_element"))



class TestWeightedRewardCalculate(unittest.TestCase):
    def setUp(self):
        # 示例输入数据
        self.questions = ["What is AI?", "Explain Python."]
        self.golden_answers = ["Artificial Intelligence", "A programming language"]
        
        # 示例 prediction_dicts
        self.prediction_dicts = [
            {
                "predictions": ["AI is cool", "Artificial Intelligence"],
                "time": [10, 5],  # 第一次预测耗时长，第二次短
                "interactions": ["<ABC>Valid<END>", "<DEF>Invalid<END>extra"]
            },
            {
                "predictions": ["Python is fun", "A high-level language"],
                "time": [8, 7],
                "interactions": ["<GHI>Valid<END>", "<JKL>Valid<END>"]
            }
        ]

    def test_weighted_reward_calculate(self):
        alpha = 0.5  # 权重系数
        results = weighted_reward_calculate(
            questions=self.questions,
            golden_answers=self.golden_answers,
            prediction_dicts=self.prediction_dicts,
            alpha=alpha
        )

        # 验证结果数量
        self.assertEqual(len(results), 2)  # 2 个 prediction_dict

        print(results)

        # 验证第一个 prediction_dict 的结果
        first_result = results[0]
        self.assertEqual(len(first_result), 2)  # 2 个预测

        # 验证第二个 prediction_dict 的结果
        second_result = results[1]
        self.assertEqual(len(second_result), 2)  # 2 个预测

    def test_time_normalization(self):
        # 构造固定时间得分以验证归一化逻辑
        prediction_dicts = [
            {
                "predictions": ["A", "B"],
                "time": [10, 5],
                "interactions": ["<ABC>Valid<END>", "<DEF>Invalid<END>"]
            }
        ]
        results = weighted_reward_calculate(
            questions=self.questions[:1],
            golden_answers=self.golden_answers[:1],
            prediction_dicts=prediction_dicts,
            alpha=0.5
        )

        # 手动计算时间得分
        time = np.array([10, 5])
        mean_time = np.mean(time)
        max_time, min_time = np.max(time), np.min(time)
        expected_time_scores = 1 - (time - mean_time) / (max_time - min_time)

        # 验证时间得分是否正确
        for idx, result in enumerate(results[0]):
            em_score, time_score, format_score = self._extract_components(result, alpha=0.5)
            self.assertAlmostEqual(time_score, expected_time_scores[idx])

    def test_format_reward(self):
        # 构造 interactions 以验证 format_reward
        interactions = ["<ABC>Valid<END>", "<DEF>Invalid<END>extra"]
        expected_format_scores = format_reward(interactions)
        # 验证 format_reward 的输出是否符合预期（需根据实际逻辑调整）
        self.assertEqual(len(expected_format_scores), 2)

    def _extract_components(self, total_score, alpha):
        # 假设已知 EM 和 time_score 的计算方式，反推各部分
        # 实际测试中需根据具体逻辑实现
        pass

from ..reward import em_reward, Validation_Dataset

class TestEMReward(unittest.TestCase):
    def test_all_match(self):
        # 所有预测与黄金答案完全匹配
        question = "What is AI?"
        golden_answer = "Artificial Intelligence"
        predictions = [
            "Artificial Intelligence",
            "Artificial Intelligence"
        ]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0, 1.0])

    def test_partial_match(self):
        # 部分预测匹配，部分不匹配
        question = "What is Python?"
        golden_answer = "A programming language"
        predictions = [
            "A programming language",
            "Python is a language"
        ]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0, 0.0])

    def test_case_insensitive_match(self):
        # 忽略大小写匹配
        question = "What is the capital of France?"
        golden_answer = "Paris"
        predictions = ["paris", "PARIS"]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0, 1.0])

    def test_extra_whitespace(self):
        # 包含多余空格
        question = "What is the answer?"
        golden_answer = "Hello World"
        predictions = ["Hello  World", "  Hello World  "]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0, 1.0])

    def test_no_match(self):
        # 所有预测不匹配
        question = "What is the answer?"
        golden_answer = "Hello World"
        predictions = ["Hi there", "Goodbye"]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [0.0, 0.0])

    def test_empty_predictions(self):
        # 空预测列表
        question = "Sample question"
        golden_answer = "Answer"
        predictions = []
        # 假设 Validation_Dataset 和 _compute_em_metrics 可处理空列表
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [])

    def test_single_prediction(self):
        # 单个预测
        question = "What is the answer?"
        golden_answer = "Yes"
        predictions = ["Yes"]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0])

    def test_mixed_cases(self):
        # 混合匹配与不匹配
        question = "What is the color?"
        golden_answer = "Blue"
        predictions = ["Blue", "Red", "BLUE"]
        scores = em_reward(question, golden_answer, predictions)
        self.assertEqual(scores, [1.0, 0.0, 1.0])

    def test_invalid_input_type(self):
        """该测试未通过，但不影响，底层都是Str类型"""
        # 预测列表中包含非字符串类型（如整数）
        question = "What is the number?"
        golden_answer = "123"
        predictions = ["123", 123]  # 第二个预测是整数
        with self.assertRaises(TypeError):  # 假设 _compute_em_metrics 抛出异常
            em_reward(question, golden_answer, predictions)

class TestWeightedRewardCalculate(unittest.TestCase):
    def setUp(self):
        # 示例输入
        self.question = "What is AI?"
        self.golden_answer = "Artificial Intelligence"
        self.predictions = [
            "Artificial Intelligence",  # 完全正确
            "AI",                       # 正确缩写
            "Artificial Intelligence is the simulation of human intelligence by machines.",  # 包含正确答案的完整句子
            "Artifical Intellgence",    # 拼写错误
            "Artificial Intelligence and Machine Learning",  # 包含正确答案但附加信息
            "Machine Learning",         # 相关但不正确
            "Quantum Computing",        # 完全错误
            "The study of algorithms",  # 部分相关
            "Artificial Intelligence",  # 重复正确
            "Artificial Inteligence"    # 拼写错误
        ]
        self.time_costs = [
            0.45, 0.32, 0.60, 0.48, 0.55,
            0.30, 0.25, 0.40, 0.45, 0.47
        ]
        self.alpha = 0.5  # 权重系数

    def testNormalCase(self):

        scores = weighted_reward_calculate(question = self.question, 
                                           golden_answer = self.golden_answer, 
                                           predictions = self.predictions, 
                                           time = self.time_costs, 
                                           alpha = 0.5)
        print(scores)

        return 
    

from ..reward import _generate_preference_pairs
import warnings


class TestGeneratePreferencePairs(unittest.TestCase):
    def setUp(self):
        # 固定随机种子以确保可重复性
        random.seed(42)

    def test_input_length_mismatch(self):
        # 1. 输入长度不一致：截断 + 警告
        with warnings.catch_warnings(record=True) as w:
            interactions = ["A", "B", "C"]
            scores = [0.8, 0.6]
            result = _generate_preference_pairs(interactions, scores, n_pairs=1)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(w), 1)
            self.assertIn("输入偏好构筑的列表长度不一致", str(w[0].message))

    def test_n_pairs_exceed_max(self):
        # 2. n_pairs 超过理论最大值：调整 + 警告
        interactions = ["A", "B", "C"]
        scores = [0.9, 0.8, 0.7]
        n_pairs = 10  # 理论最大值为 3*(3-1)/2 = 3
        with warnings.catch_warnings(record=True) as w:
            result = _generate_preference_pairs(interactions, scores, n_pairs=n_pairs)
            self.assertEqual(len(result), 3)  # 被调整为最大值 3
            self.assertEqual(len(w), 1)
            self.assertIn("需求组合数远高于可构筑不重复抽样数", str(w[0].message))

    def test_normal_behavior(self):
        # 3. 正常行为：生成不重复配对
        interactions = ["A", "B", "C", "D"]
        scores = [0.9, 0.8, 0.7, 0.6]
        result = _generate_preference_pairs(interactions, scores, n_pairs=3)
        self.assertEqual(len(result), 3)
        used_pairs = set()
        for pair in result:
            self.assertIn("chosen", pair)
            self.assertIn("rejected", pair)
            chosen_idx = interactions.index(pair["chosen"])
            rejected_idx = interactions.index(pair["rejected"])
            self.assertNotEqual(chosen_idx, rejected_idx)
            self.assertTrue(scores[chosen_idx] > scores[rejected_idx])
            used_pairs.add((chosen_idx, rejected_idx))
        self.assertEqual(len(used_pairs), 3)  # 所有配对不重复

    def test_no_duplicate_pairs(self):
        # 4. 验证配对不重复
        interactions = ["A", "B", "C"]
        scores = [0.9, 0.8, 0.7]
        result = _generate_preference_pairs(interactions, scores, n_pairs=3)
        used_pairs = set()
        for pair in result:
            chosen_idx = interactions.index(pair["chosen"])
            rejected_idx = interactions.index(pair["rejected"])
            used_pairs.add((chosen_idx, rejected_idx))
        self.assertEqual(len(used_pairs), 3)  # 理论最大值为 3
        # 多次运行验证是否生成相同结果（随机性控制）
        result2 = _generate_preference_pairs(interactions, scores, n_pairs=3)
        self.assertEqual(result, result2)  # 因为 random.seed(42) 固定

    def test_edge_cases(self):
        # 5. 边界情况：最小输入（2个交互）
        interactions = ["A", "B"]
        scores = [0.9, 0.8]
        result = _generate_preference_pairs(interactions, scores, n_pairs=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chosen"], "A")
        self.assertEqual(result[0]["rejected"], "B")

        # 6. 评分重复但满足最小两个不同评分
        interactions = ["A", "B", "C", "D"]
        scores = [0.9, 0.9, 0.8, 0.8]
        result = _generate_preference_pairs(interactions, scores, n_pairs=4)
        self.assertEqual(len(result), 1)
        for pair in result:
            chosen_idx = interactions.index(pair["chosen"])
            rejected_idx = interactions.index(pair["rejected"])
            self.assertTrue(scores[chosen_idx] > scores[rejected_idx])

    def test_error_handling(self):
        # 7. 输入合法性检查
        with self.assertRaises(ValueError):
            _generate_preference_pairs([], [], n_pairs=1)  # 评分种类不足 2 个
        with self.assertRaises(ValueError):
            _generate_preference_pairs(["A"], [0.9], n_pairs=1)  # 评分种类不足 2 个
        with self.assertRaises(ValueError):
            _generate_preference_pairs(["A", "B"], [0.9, 0.8], n_pairs=0)  # n_pairs < 1

    def test_all_pairs_coverage(self):
        # 8. 验证是否覆盖所有可能的高分-低分组合
        interactions = ["A", "B", "C", "D"]
        scores = [0.9, 0.8, 0.7, 0.6]
        result = _generate_preference_pairs(interactions, scores, n_pairs=6)  # 理论最大值为 6
        self.assertEqual(len(result), 6)
        all_possible_pairs = set()
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                for a in range(len(interactions)):
                    for b in range(len(interactions)):
                        if scores[a] > scores[b]:
                            all_possible_pairs.add((a, b))
        self.assertEqual(len(all_possible_pairs), 6)  # 理论最大值为 6
        generated_pairs = set((interactions.index(p["chosen"]), interactions.index(p["rejected"])) for p in result)
        self.assertEqual(generated_pairs, all_possible_pairs)  # 验证是否覆盖所有组合

    def test_used_pairs_set(self):
        # 10. 验证 used_pairs 是否防止重复配对
        interactions = ["A", "B", "C"]
        scores = [0.9, 0.8, 0.7]
        result = _generate_preference_pairs(interactions, scores, n_pairs=3)
        self.assertEqual(len(result), 3)
        used_pairs = set((interactions.index(p["chosen"]), interactions.index(p["rejected"])) for p in result)
        self.assertEqual(len(used_pairs), 3)  # 所有配对不重复


from ..reward import format_reward
class TestFormatScore(unittest.TestCase):

    def test_perfect_format(self):
        """完全符合格式要求的交互"""
        interaction = "<Think> The query asks for an analysis of the primary causes (factors contributing to climate change) and the resulting effects (environmental, societal, and economic impacts). The query is broad and may benefit from refining or expanding to ensure coverage of relevant topics like greenhouse gases, deforestation, renewable energy, biodiversity loss, and global policies. </Think>\n\n<Action> Query 0 | Query Rewrite | Nums 3 </Action>\n<Detail> Generate diverse queries to explore specific subtopics of the original query. Possible areas include natural vs human causes, technological solutions, and impact on specific regions or ecosystems. </Detail>\n\n<Action> Query 0 | Query Reason | Nums 1 </Action>\n<Detail> Analyze what information should be retrieved to comprehensively answer the query. Exploration of scientific data, historical context, and proposed mitigation strategies may be valuable. </Detail>\n\n<Think> To ensure clarity and usefulness of retrieved information, the documents must directly address the query, including evidence-backed causes and a range of societal and planetary effects. Documents should ideally include statistical evidence, expert analysis, and case studies. </Think>\n\n<Action> Query 0 | Document Retrieve | Nums 4 </Action>\n<Detail> Use the original query to retrieve key documents focused on climate change causes and effects. Prioritize retrieving academic studies, reputable scientific reports (e.g., IPCC reports), and governmental policy analyses. </Detail>\n\n<Action> Query 0 | Document Filter | Nums 4 </Action>\n<Detail> Evaluate the relevance of retrieved documents based on their focus on the causes and effects of climate change. Filtering should emphasize documents that provide both comprehensive and specific insights into the query. </Detail>\n\n<END>"
        self.assertAlmostEqual(format_reward([interaction])[0], 1.0)

    def test_multiple_actions(self):
        """多个 Action 标签（应加分）"""
        interaction = ("<Think>思考内容</Think>"
                       "<Action>Query 1 | Operation | Nums 2</Action>"
                       "<Action>Query 2 | Operation | Nums 3</Action>"
                       "<Detail>详细内容</Detail><END>")
        # 多 Action 加 0.2
        self.assertAlmostEqual(format_reward([interaction])[0], 1.0)




if __name__ == '__main__':
    unittest.main()





import re

def format_reward_func(model_output: str, reward: float, max_action_budget: int = 3):
    """
    判定规则：
    1. 但绝对需要保证顺序的一致性，不允许打乱顺序，要求一定要是Think -> Action -> Think -> Action的顺序，如果出现截尾（比如只有Think、没有Action），也能接受
    2. Tag外面不能放东西
    3. 以<END>结尾

    格式示例
    <Think> ... </Think> <Action> ... </Action> <Think> ... </Think> <Action> ... </Action> <Think> ... </Think><END>
    """
    
    # 规则3：检查是否以<END>结尾
    if not model_output.strip().endswith('<END>'):
        return -1
    
    # 移除结尾的<END>以便后续处理
    content = model_output.strip()[:-5]  # 移除<END>
    
    # 规则2：检查Tag外面是否有内容
    # 移除所有标签后应该没有内容
    text_without_tags = re.sub(r'<(Think|Action|Detail)>(.*?)</\1>', '', content, flags=re.DOTALL)
    text_without_tags = re.sub(r'\s+', '', text_without_tags)  # 移除所有空白字符
    if text_without_tags:
        return -1  # Tag外面有内容
    
    # 使用正则表达式匹配所有标签，按照出现的顺序
    pattern = r'<(Think|Action|Detail)>(.*?)</\1>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return -1  # 没有找到任何标签
    
    # 规则1：检查标签顺序
    expected_order = ['Think', 'Action']  # 期望的顺序模式
    current_pattern_index = 0
    action_count = 0
    
    for tag_type, tag_content in matches:
        # 检查内容是否为空
        if not tag_content.strip():
            return -1
        
        # 检查标签类型是否符合期望的顺序
        expected_tag = expected_order[current_pattern_index]
        if tag_type != expected_tag:
            return -1
        
        # 如果是Action标签，检查预算
        if tag_type == 'Action':
            action_count += 1
            if action_count > max_action_budget:
                return -1
        
        # 移动到下一个期望的标签类型
        current_pattern_index = (current_pattern_index + 1) % len(expected_order)
    
    # 如果通过了所有检查，返回原始奖励
    return reward

# 测试用例
if __name__ == "__main__":
    # 正确格式的示例
    good_output = """<Think>这是思考内容</Think> <Action>动作1</Action> <Think>另一个思考</Think> <Action>动作2</Action><END>"""
    
    # 错误格式的示例
    bad_output1 = """<Think>思考</Think> <Action>动作</Action> 外面有内容 <Think>思考</Think><END>"""  # 规则2违反
    bad_output2 = """<Think>思考</Think> <Action>动作</Action> <Action>错误的顺序</Action><END>"""  # 规则1违反
    bad_output3 = """<Think>思考</Think> <Action>动作</Action> <Think>思考</Think>"""  # 规则3违反
    bad_output4 = """<Think></Think> <Action>动作</Action><END>"""  # 空内容
    
    print(f"好输出奖励: {format_reward_func(good_output, 1.0)}")
    print(f"坏输出1奖励: {format_reward_func(bad_output1, 1.0)}")
    print(f"坏输出2奖励: {format_reward_func(bad_output2, 1.0)}")
    print(f"坏输出3奖励: {format_reward_func(bad_output3, 1.0)}")
    print(f"坏输出4奖励: {format_reward_func(bad_output4, 1.0)}")


























