import unittest
import pandas as pd
import os
import random
from typing import Tuple, List, Union
from unittest.mock import patch, mock_open, MagicMock
import tempfile
from functools import partial
import numpy as np
import unittest
import warnings
from ..RAG_modules import Decision_Agent
from ..config import DecisionConfig, DecisionStrategyParams



from ..RAG_modules import AERR, Decision_Agent, Generate_Agent, Execution_Agent
from ..config import AERRConfig, StrategyParams

class DecisionDummyModel:
    def __init__(self):
        self.curr_idx = 0

        self.test_list = [            
                "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Query Reason | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Query Extract | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Document Filter | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Document Retrieve | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Stop | Nums 2</Action><Action>Query 2 | Op | Nums 3</Action><END>"
        ]
        return 
    
    def generate(self, prompts, return_time = True, **kwargs):
        
        if self.curr_idx >= len(self.test_list):
            self.curr_idx = 0 
            self.test_list = [
                "<Think>思考</Think><Action>Query 0 | Stop | Nums 2</Action><END>",
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

    def generate(self, prompts, return_time: bool = False, return_input_prompts = False, **kwargs):
        # 返回预定义的响应列表
        result = []
        result.append(["\n".join([            
                "<Think>思考</Think><Action>Query 0 | Query Rewrite | Nums 2</Action><END>",
                "<Think>思考</Think><Action>Query 0 | Query Reason | Nums 2</Action><Action> Query 2 | Stop </Action><END>",
        ]) for i in range(len(prompts))])

        if return_time: 
            result.append([10] * len(prompts))
        if return_input_prompts:
            result.append(prompts)

        return *result, 



class DummyIndexer:
    def topk_search(self, query, k=3):

        result = "This is a test."
        
        # 返回预定义的检索结果
        return [result] * k

from ..large_language_model import Large_Language_Model

class AERR_test(AERR):

    def __init__(self, config: AERRConfig = None):
        from ..config import MyTrainConfig
        if config is None:
            config = MyTrainConfig.to_AERRConfig()

        # 保证是测试过程，别忘了
        config.test = True
        config.init_decision_model = False
        config.init_execution_model = False
        config.init_generate_model = False

        self.config = config
        
        self.decision_dummy_model = DecisionDummyModel()
        self.execution_dummy_model = ExecutionDummyModel()
        self.dummy_indexer = DummyIndexer()

        self.decision_agent = Decision_Agent(model = self.decision_dummy_model, config = config.decision)

        self.excution_agent = Execution_Agent(config.execution)

        self.generate_agent = GenerateDummyModel(config.generative)

        self.decision_agent.model = self.decision_dummy_model
        self.excution_agent.model = self.execution_dummy_model
        self.excution_agent.indexer = self.dummy_indexer

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
        from ..config import MyTrainConfig
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
            max_tree_length = max_tree_length)

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
    # pipeline = AERR(config.to_AERRConfig())
    pipeline = Execution_Agent(config.to_AERRConfig().execution)

    decision_prompts = [
        "<Think></Think><Action> Query 0 | Query Rewrite | Nums 1 </Action><END>", # ok
        "<Think></Think><Action> Query 0 | Query Reason | Nums 1 </Action><END>", # ok
        "<Think></Think><Action> Query 0 | Query Extract | Nums 1 </Action><END>", # ok 
        "<Think></Think><Action> Query 0 | Query Search | Nums 1 </Action><END>", # ok
        "<Think></Think><Action> Query 0 | Detail Search | Nums 1 </Action> <Detail> This is a very big test. </Detail> <END>", # ok
        "<Think></Think><Action> Query 0 | Delete Documents | Nums 1 </Action> <Detail> This is a very big test. </Detail> <END>", # ok
        "<Think></Think><Action> Query 0 | Sort Documents | Nums 1 </Action><END>", # ?
        "<Think></Think><Action> Query 0 | Add Query | Nums 0 </Action> <Detail> This is a very big test. </Detail> <END>", # ok
        "<Think></Think><Action> Query 1 | Delete Query | Nums 0 </Action><END>", # 没有怎么删？
        "<Think></Think><Action> Query 1 | Stop | Nums 0 </Action><END>" # ok
    ]

    query_lists = [copy.deepcopy(["Who is the all time leading scorer in the Premier League?"]) for i in range(len(decision_prompts))]
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




def try_AERR():
    # ✅ 示例变量定义
    alpha = 0.5

    questions = [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?", 
        "Who wrote 'Romeo and Juliet'?"
    ]

    golden_answers = [
        "Paris",
        "William Shakespeare", 
        "William Shakespeare"
    ]

    predictions = [
        "Paris",
        "William Shakespeare", 
        "This is a test. William"
    ]

    from lib.config import AERRConfig, MyTrainConfig

    train_config = MyTrainConfig()
    pipeline_config = train_config.to_AERRConfig()
    pipeline = AERR(config = pipeline_config)


    output = pipeline.generate_batch(input_prompts = questions, max_tree_length = 4)
    print(output)



def test_filter_and_summarize_tools():
    from ..large_language_model import Large_Language_Model_API
    import re
    
    llm_api = Large_Language_Model_API()
    llm_api.init_llm("")

    def document_summary_input_prompt(self, user_input: str, documents: List[str]) -> str:
        """
        Generate a summary confirming if documents collectively answer the query. Delete sources ONLY if answer is confirmed.

        Args:
            documents (List[str]): Source documents (format: [Title] ... [Context] ...)

        Returns:
            str: Complete prompt template.
        """
        docs_str = "\n".join([f"<Document {i}>: {d[:200]}" for i, d in enumerate(documents)]) if len(documents) != 0 else "Nothing yet."
        prompt = f"""
    Based on the user input and the provided documents, determine if they collectively answer the query. Generate a concise summary confirming this.
    **Effective Document Summary Guide**
    - **Direct Answer**: The documents **explicitly state the answer** (e.g., "AI reduces diagnostic errors by 35% in radiology" for query "How does AI improve medical diagnostics?").
    - **Indirect Answer**: The documents **combine to form a complete answer** (e.g., Document 0: "AI detects tumors 20% faster"; Document 1: "Accuracy improved to 94%"; together they answer "How does AI improve medical diagnostics?").
    - **No Answer**: Documents **fail to provide any answer** (e.g., only discuss AI ethics without clinical impact).

    **Response Format Specification**
    - <Think> [Required] 1-2 sentence reasoning for deletion/retention decision. Put your thorough analysis on each document 
    - <Document Summary> [Required] The concise summary text (with source citations like "[Summary Context] (Document 0)")

    Examples:
    Example Documents:
    <Document 0> [Title] Urban Tree Canopy Study [Context] Tree canopy in Chicago reduced ambient temperatures by 2°C during heatwaves. Shaded surfaces were 20°C cooler than concrete.
    <Document 1> [Title] Pavement Material Impact [Context] Permeable pavement combined with trees lowered street temps by 1.5°C vs. standard asphalt.
    <Document 2> [Title] NYC Greening Initiative [Context] After planting 10,000 trees, NYC saw 1.2°C average temperature drop in target zones.
    <Document 3> [Title] Evapotranspiration Analysis [Context] Green spaces increased evaporative cooling by 35% in Phoenix, reducing air temps by 1.8°C.
    <Document 4> [Title] Heat Mitigation Policy [Context] City ordinance requiring 30% green cover in new developments cut urban temps by 0.9°C over 5 years.
    <Document 5> [Title] Park Maintenance Costs [Context] Annual budget for NYC parks landscaping. Discusses mowing schedules and staff hours.
    <Document 6> [Title] Bicycle Lane Expansion [Context] 2023 report on bike lane installations across 10 cities. No environmental metrics.
    <Document 7> [Title] Heatwave Emergency Plan [Context] City response protocol for heatwaves including cooling centers. No mention of green infrastructure.
    <Document 8> [Title] Urban Biodiversity Survey [Context] Species count in city parks. Notes bird populations but no temperature data.
    <Document 9> [Title] Public Health Campaign [Context] "Cool City" campaign promoting water breaks. Mentions heat risks but not green space solutions.
    Example User's Input: 
    How do urban green spaces reduce heat island effects?
    Example Expected Output:
    <Think> Relevant documents: 0-4. Irrelevant: 5-9. Document 0 provides temperature metrics (2°C cooling). Document 1 shows infrastructure synergy (1.5°C). Document 2 demonstrates real-world implementation (1.2°C). Document 3 quantifies evaporative cooling (1.8°C). Document 4 proves policy impact (0.9°C). Document 5 irrelevant (discusses costs, not temperature). Document 6 irrelevant (bicycle lanes, no environmental data). Document 7 irrelevant (emergency plans, no green infrastructure). Document 8 irrelevant (biodiversity, no temperature data). Document 9 irrelevant (water breaks campaign, not green spaces).
    <Document Summary> Urban green spaces reduce heat island effects through: 2°C cooling from tree canopy (Document 0), 1.5°C pavement synergy (Document 1), 1.2°C NYC implementation (Document 2), 1.8°C evaporative cooling (Document 3), and 0.9°C policy mandates (Document 4).

    Documents:
    {docs_str}

    **User's Input**: 
    {user_input}

    Your Output: 
    """.strip()
        return prompt


    User_Query = "How does plastic pollution impact coral reef ecosystems?"
    documents = [
    "[Title] Coral Plastic Smothering Study [Context] Plastic debris covering coral colonies reduced growth rates by 32% in Great Barrier Reef monitoring sites."
    ,"[Title] Microplastics and Coral Bleaching [Context] Microplastics in water column increased coral bleaching events by 45% during 2020-2023."
    ,"[Title] Coral Reef Restoration Project [Context] After removing 15 tons of plastic from reef sites, coral recovery rates increased by 28% within 18 months."
    ,"[Title] Plastic Toxicity Report [Context] Plastic leachates caused 60% higher mortality in coral polyps compared to control groups."
    ,"[Title] Marine Debris Policy [Context] 2022 UN resolution targeting plastic reduction in coral zones led to 22% decrease in plastic accumulation in Pacific reefs."
    ,"[Title] Ocean Plastic Recycling Rates [Context] Global plastic recycling rate reached 9% in 2023. Discusses collection infrastructure, not marine impacts."
    ,"[Title] Whale Migration Patterns [Context] Satellite tracking of humpback whales in Atlantic Ocean. No plastic pollution data."
    ,"[Title] Ocean Acidification Study [Context] pH levels dropped 0.1 units in 2023, affecting shellfish. No plastic mentions."
    ,"[Title] Sea Turtle Nesting Reports [Context] 2022 nesting success rates at 65% on Pacific islands. Mentions plastic ingestion but not coral impacts."
    ,"[Title] Coastal Tourism Survey [Context] Visitor numbers increased 15% in 2023. No environmental impact metrics."

    ]


    def parse_document_summary_output(self, model_output: str) -> dict:
        """
        Extracts key components from model output using strict regex matching.
        Matches the exact format: <Document X> [Title] ... [Context] ...
        """
        # Extract <Document Summary> section (concise citation list)
        summary_pattern = r'<Document Summary>([\s\S]*)'
        
        # Match <Document Summary> section
        summary_match = re.search(summary_pattern, model_output, re.DOTALL)
        summary_content = summary_match.group(1).strip() if summary_match else ""
        
        return summary_content


    response = llm_api.generate(input_text = document_summary_input_prompt(None, User_Query, ['']))

    summary_response = parse_document_summary_output(None, response[0])

    print(summary_response)