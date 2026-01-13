import unittest
import pandas as pd
import os
import random
from typing import Tuple, List, Union
from unittest.mock import patch, mock_open, MagicMock
import tempfile
from functools import partial
import numpy as np


from ..RAG_modules import AERR, Decision_Agent, Generate_Agent, Execution_Agent
from ..config import AERRConfig, StrategyParams

class DecisionDummyModel:
    def __init__(self):
        self.curr_idx = 0

        self.test_list = [            
                "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
                "<Think>思考</Think><Action>Query 0 | Query Reason | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>",
                "<Think>思考</Think><Action>Query 0 | Query Extract | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>",
                "<Think>思考</Think><Action>Query 0 | Document Filter | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>",
                "<Think>思考</Think><Action>Query 0 | Document Retrieve | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>",
                "<Think>思考</Think><Action>Query 0 | Stop | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
        ]
        return 
    
    def generate(self, prompts, return_time = True, **kwargs):
        
        if self.curr_idx >= len(self.test_list):
            self.curr_idx = 0 
            self.test_list = [
                "<Think>思考</Think><Action>Query 0 | Stop | Nums 2</Action><Detail>详情</Detail><END>",
            ]

        output = [self.test_list[self.curr_idx] for i in range(len(prompts))]

        # 返回预定义的响应列表
        self.curr_idx += 1
        return output, [10] * len(output) if return_time else output

class ExecutionDummyModel:
    def __init__(self):
        self.string =  [
                "<Query> This is a test.", 
                "<Query> This is a test.", 
                "<Document> This is a test",
                "<Document> This is a test",
                "<Irrelevant Document> 0", 
                "<END>"
                ]

        return 
    
    def generate(self, prompts, return_time = True, **kwargs):

        # 返回预定义的响应列表
        return ['\n'.join(self.string)] * len(prompts), [10] * len(prompts) if return_time else self.string

class GenerateDummyModel:

    def __init__(self, *args, **kwargs):
        pass

    def chat_without_history(self, input_text, **kwargs):
        # 返回预定义的响应列表
        return ["This is a test"] * len(input_text)
    
    def generate(self, input_text, **kwargs):
        return ["This is a test"] * len(input_text)

class DummyModel:

    def generate(self, prompts, return_time: bool = False, **kwargs):
        # 返回预定义的响应列表
        return (["This is a test"] * len(prompts), ["This is a test"] * len(prompts)) if return_time == True else ["This is a test"] * len(prompts)



class DummyIndexer:
    def topk_search(self, query, k=3):

        result = "This is a test."
        
        # 返回预定义的检索结果
        return [result] * k

from ..large_language_model import Large_Language_Model

class AERR_test(AERR):

    def __init__(self, config: AERRConfig = None):
        if config is None:
            config = AERRConfig(test = True)

        # 保证是测试过程，别忘了
        config.test = True

        self.config = config
        self.support_batch = False

        self.decision_dummy_model = DecisionDummyModel()
        self.execution_dummy_model = ExecutionDummyModel()
        self.dummy_indexer = DummyIndexer()

        self.decision_agent = Decision_Agent(config.decision)

        self.excution_agent = Execution_Agent(config.execution)

        self.generate_agent = GenerateDummyModel(config.generative)

        self.decision_agent.model = self.decision_dummy_model
        self.excution_agent.model = self.execution_dummy_model
        self.excution_agent.indexer = self.dummy_indexer


# class Decision_Agent_test(Decision_Agent):

#     def __init__(self):
#         self.curr_idx = 0

#         self.test_list = [            
#                 "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><Detail>详情</Detail><END>",
#                 "<Think>思考</Think><Action>Query 0 | Query Reason | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
#                 "<Think>思考</Think><Action>Query 0 | Query Extract | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
#                 "<Think>思考</Think><Action>Query 0 | Document Filter | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
#                 "<Think>思考</Think><Action>Query 0 | Document Retrieve | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
#                 "<Think>思考</Think><Action>Query 0 | Stop | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><Detail>详情</Detail><END>"
#         ]

#         pass
    
#     def generate(self, strings: list[str], **kwargs):
        
#         if self.curr_idx > 4:
#             self.curr_idx -= 4
        
#         string = self.test_list[self.curr_idx]
#         self.curr_idx += 1

#         return string
        
# class Generate_Agent:

#     def __init__(self):
#         pass
    
#     def generate(self, strings: list[str], **kwargs):

#         if isinstance(strings, list):
#             return ["This is a test"] * len(strings)
#         else:
#             return "This is a test"

# class Execution_Agent:

#     def __init__(self):
#         self.index = 0
#         pass
    
#     def take_action(self, 
#                     decision_prompt: str, 
#                     query_list: list, 
#                     document_list: list, 
#                     need_print : bool = False, 
#                     **kwargs) -> Union[tuple[List, List, bool]]:

#         query_list.extend(["This is a test"])
#         document_list.extend(["This is a test"])

#         done = False if self.index < 3 else True
#         self.index += 1


#         return query_list, document_list, done





# ==============================测试ExecutionAgent===============================

import unittest
from typing import List, Tuple, Dict
from ..RAG_modules import Execution_Agent
from ..config import ExecutionConfig, StrategyParams

class TestTakeActionBatch(unittest.TestCase):
    def setUp(self):
        # 初始化依赖对象
        self.model = DummyModel()
        self.indexer = DummyIndexer()

        config = ExecutionConfig(test = True)
        # 初始化被测试的类实例
        self.executor = Execution_Agent(config)

        self.executor.model = DummyModel()
        self.executor.indexer  = DummyIndexer()

        # 定义通用参数
        self.executor.strategy_params = StrategyParams().todict()

    def test_normal_case(self):
        return 




# =================================测试DecisionAgent的Sampling函数=====================================


import unittest
import warnings
from ..RAG_modules import Decision_Agent
from ..config import DecisionConfig, DecisionStrategyParams

class TestSampling(unittest.TestCase):
    def setUp(self):
        # 初始化一个 DecisionModel 实例，注入模拟模型
        self.decision_model = Decision_Agent(**DecisionConfig().todict(
            exclude_types = DecisionStrategyParams
        ))
        self.decision_model.model = DummyModel()

    def test_sampling(self):
        # 测试非嵌套列表是否自动转换为嵌套结构
        query_lists = ["q1", "q2"]
        document_lists = ["d1", "d2"]
        input_prompts, model_responses = self.decision_model.sampling(query_lists, 
                                                                      document_lists, 
                                                                      sampling_num = 3)
        
        self.assertEqual(len(input_prompts), 1)
        self.assertEqual(len(model_responses), 3)
        
# ==============================测试DecisionAgent采样功能===============================

class TestLastSummarySampling(unittest.TestCase):
    def setUp(self):
        config = DecisionConfig()
        config.test = True

        # 初始化测试对象和模拟模型
        self.class_under_test = Decision_Agent(**config.todict(exclude_types = [StrategyParams]))
        self.class_under_test.model = DummyModel()

    def test_last_summary_sampling(self):
        # 准备测试数据
        query_lists = [["What is AI?"], ["Explain machine learning."]]
        document_lists = [["AI refers to..."], ["Machine learning is..."]]
        sampling_nums = 3  # 每个查询生成3个采样

        # 调用被测函数
        sampling_inputs, model_responses, detail_strings = self.class_under_test.last_summary_sampling(
            query_lists=query_lists,
            document_lists=document_lists,
            sampling_nums=sampling_nums,
        )

        # 验证输入生成是否正确
        expected_original_inputs = [
            self.class_under_test.prompt_for_summary(query_list=["What is AI?"], documents=["AI refers to..."]),
            self.class_under_test.prompt_for_summary(query_list=["Explain machine learning."], documents=["Machine learning is..."]),
        ]
        
        # 每个原始输入被复制 sampling_nums 次
        expected_sampling_inputs = [expected_original_inputs[0]] * sampling_nums + [expected_original_inputs[1]] * sampling_nums
        self.assertEqual(sampling_inputs, [tuple(expected_sampling_inputs[:sampling_nums]), tuple(expected_sampling_inputs[sampling_nums:])])

        # 验证模型响应分割是否正确
        expected_model_responses = [('This is a test', 'This is a test', 'This is a test'), 
                                    ('This is a test', 'This is a test', 'This is a test')]
        self.assertEqual(model_responses, expected_model_responses)

        # 验证 detail_strings 格式是否正确
        expected_detail_strings = [
            "<Detail>\n Detail 1.1\n Detail 1.2",
            "<Detail>\n Detail 1.3\n Detail 1.4",
            "<Detail>\n Detail 1.5\n Detail 1.6",
            "<Detail>\n Detail 2.1\n Detail 2.2",
            "<Detail>\n Detail 2.3\n Detail 2.4",
            "<Detail>\n Detail 2.5\n Detail 2.6",
        ]
        self.assertEqual(detail_strings, expected_detail_strings)

# ==============================测试AERR的sampling功能===============================

from ..dataset import ConversationTree

class TestAERRSampling(unittest.TestCase):
    def setUp(self):
        self.class_under_test = AERR_test()

    def test_sampling(self):
        # 准备测试数据
        user_input = "Sample User Input"
        need_print = False
        sampling_nums = 3
        tree = ConversationTree()

        # 调用被测函数
        result_tree, output = self.class_under_test.sampling(
            user_input=user_input,
            need_print=need_print,
            sampling_nums=sampling_nums,
            tree=tree
        )

        # 验证对话树的构建
        self.assertEqual(len(result_tree.layers), 7)

    def test_sampling_with_existing_tree(self):
        # 测试传入已有的tree对象
        existing_tree = ConversationTree()
        existing_tree.add_layer(sampling_prompts = [["This is a test"]], 
                                decision_model_responses=[["response1"]])
        
        result_tree, output = self.class_under_test.sampling(
            user_input="Existing Tree Test",
            tree=existing_tree
        )

        self.assertEqual(len(result_tree.layers), 8)  # 原有1层 + 新增2层


    def test_decision_agent_sampling(self):
        """检查是否正常返回"""
        query_lists = [['This is a test']]
        document_lists = [['This is a test']]
        temperature = 1
        sampling_nums = 3

        result = self.class_under_test.decision_agent.sampling(query_lists,
                                                               document_lists, 
                                                               temperature, 
                                                               sampling_nums)
        print(result)

        return 

    def test_generate_batch(self):
        ##Rewrite功能正常##
        # 构造输入数据
        input_promps = ['This is a test', 'This is a test']
        outputs = self.class_under_test.generate_batch(input_promps)
        print(outputs)


class TestAERRGenerateBatch(unittest.TestCase):
    def setUp(self):
        self.class_under_test = AERR_test()
        self.model = DummyModel()

    def test_normal_case(self):
        # 准备测试数据
        user_input = "Sample User Input"
        need_print = False
        tree = ConversationTree()
        max_tree_length = 3


        # 调用被测函数  
        result_tree, output = self.class_under_test.generate_batch(
            input_prompts = [user_input],
            user_input=user_input,
            need_print=need_print,
            tree=tree,
            max_tree_length = max_tree_length, 

        )

        # 验证对话树的构建
        self.assertEqual(len(result_tree.layers), 7)

        return 
    
        
    

def testDecisionModelReload():
    import time


    config = DecisionConfig()
    decision_model = Decision_Agent(config = config)

    print('=' * 20)
    print('Loading Succuses')
    time.sleep(10)

    print('=' * 20)
    print('Try Sampling')
    print(decision_model.sampling([['This is a test']], [['This is a test']], sampling_num = 1))
    time.sleep(10)

    print('=' * 20)
    print('Try Release')
    decision_model.release()
    time.sleep(60)

    print('=' * 20)
    print('Try Reload')
    decision_model.reload(config)
    time.sleep(10)

    print("Done")



    return 

def test_execution_agent_take_batch():
    from lib.RAG_modules import AERR, Execution_Agent
    from lib.config import MyTrainConfig
    import copy
    
    config = MyTrainConfig()

    need_load_lora_dir = [
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-17-39/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-32-37/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-48-09/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-02-58-11/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-05-34-37/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-07-30-42/"
    ]
    config.lora_dir = need_load_lora_dir
    # pipeline = AERR(config.to_AERRConfig())
    pipeline = Execution_Agent(config.to_AERRConfig().execution)

    decision_prompts = [
        "<Think></Think><Action> Query 0| Query Rewrite | Nums 1 </Action><END>", 
        "<Think></Think><Action> Query 0| Query Reason | Nums 1 </Action><END>", 
        "<Think></Think><Action> Query 0| Query Extract | Nums 1 </Action><END>", 
        "<Think></Think><Action> Query 0| Document Retrieve | Nums 1 </Action><END>",
        "<Think></Think><Action> Query 0| Document Filter | Nums 1 </Action><END>", 
        "<Think></Think><Action> Query 0| Stop | Nums 0 </Action><END>", 
    ]

    query_lists = [copy.deepcopy(["Who is the all time leading scorer in the Premier League?"]) for i in range(len(decision_prompts))] * len(decision_prompts)
    document_lists = [copy.deepcopy(["""[Title] Premier League All-Time Leading Scorers (Excluding Penalties)

    [Context] Alan Shearer remains the Premier League’s all-time top scorer, netting 260 goals in total (including penalties) during his 1992–2013 career with Blackburn Rovers and Newcastle United. Excluding penalties, he still leads with 205 goals, cementing his status as the competition’s most clinical finisher from open play.

    Behind Shearer, Andy Cole ranks second with 186 non-penalty goals, a testament to his prolific scoring despite never taking penalties. Wayne Rooney follows closely with 185 goals, making him Manchester United’s record scorer. Harry Kane, an active player, has 180 non-penalty goals, showcasing his consistency. Thierry Henry (152) and Sergio Agüero (157) also feature prominently, highlighting their elite finishing abilities.

    For context, the 2024/25 season’s top scorers, Heung-Min Son and Mohamed Salah, each scored 23 goals, but their totals pale compared to historical legends. The Premier League’s average goals per game stand at 2.93, underscoring the competitiveness of modern scoring."""]) for i in range(len(decision_prompts))] 

    done_flags = [False] * len(decision_prompts)
    time_lis = [0] * len(decision_prompts)

    query_lists, document_lists, done_flags, time_lis = pipeline.take_action_batch(
            decision_prompts = decision_prompts, 
            query_lists = query_lists, 
            document_lists = document_lists, 
            done_flags = done_flags, 
            time_lis = time_lis
        )

    print("Done! ")

    # for i in range(len(decision_model_responses[0])):
    #     response = decision_model_responses[0][i]
    #     output = critic.extract_valid_content(response)
    #     print("=" * 20)
    #     print(output)