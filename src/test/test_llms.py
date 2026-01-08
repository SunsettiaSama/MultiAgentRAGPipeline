import torch
import unittest
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.RAG_modules import *  # 替换为你的模块路径

# class TestLargeLanguageModel(unittest.TestCase):
#     def setUp(self):
#         """初始化模型实例"""
#         self.local_dir = "./Llama3_8B/"  # 请确保此路径存在有效的模型文件
#         self.model = Large_Language_Model(local_dir=self.local_dir)
#         self.test_prompt = "Once upon a time"
#         self.batch_prompts = ["Hello world", "The quick brown fox"]

#     def test_model_initialization(self):
#         """测试模型和分词器是否成功加载"""
#         self.assertIsNotNone(self.model.model)
#         self.assertIsNotNone(self.model.tokenizer)
#         self.assertEqual(self.model.model.training, False)  # 检查模型是否在 eval 模式

#     def test_generate_single_prompt(self):
#         """测试单个字符串输入"""
#         output = self.model.generate(self.test_prompt)
#         self.assertIsInstance(output, str)
#         self.assertTrue(len(output) > 0)

#     def test_generate_batch_prompts(self):
#         """测试字符串列表输入"""
#         outputs = self.model.generate(self.batch_prompts)
#         self.assertIsInstance(outputs, list)
#         self.assertEqual(len(outputs), len(self.batch_prompts))
#         for out in outputs:
#             self.assertIsInstance(out, str)
#             self.assertTrue(len(out) > 0)

#     def test_max_new_tokens(self):
#         """测试 max_new_tokens 参数"""
#         short_output = self.model.generate(self.test_prompt, max_new_tokens=10)
#         long_output = self.model.generate(self.test_prompt, max_new_tokens=100)
#         self.assertTrue(len(long_output) > len(short_output))

#     def test_temperature(self):
#         """测试 temperature 参数对输出多样性的影响（定性测试）"""
#         temp0 = self.model.generate(self.test_prompt, temperature=0.1)
#         temp1 = self.model.generate(self.test_prompt, temperature=1.0)
#         # 由于随机性，无法精确断言，但可以观察输出是否不同
#         self.assertNotEqual(temp0, temp1)

#     def test_encode_method(self):
#         """测试 encode 方法"""
#         encoded = self.model.encode("Testing encoder")
#         self.assertIsInstance(encoded, torch.Tensor)
#         self.assertTrue(encoded.dim() == 2 or isinstance(encoded, list))  # 根据 return_tensors 设置

#     def test_invalid_prompt_type(self):
#         """测试非法 prompt 类型"""
#         with self.assertRaises(TypeError):
#             self.model.generate(123)  # 非字符串或列表输入

#     def tearDown(self):
#         """释放资源"""
#         del self.model


# class TestLargeLanguageModelLocal(unittest.TestCase):
#     def setUp(self):
#         """初始化模型实例"""
#         # 使用本地模型路径（需确保路径正确）
#         self.local_dir = "./qwen2.5_1.5B/"  # 替换为实际路径
#         self.model = Large_Language_Model(local_dir=self.local_dir)

#     def test_init_llm_single_thread(self):
#         """测试单线程模式下的初始化"""
#         system_prompt = "You are a helpful assistant."
#         self.model.init_llm(system_prompt, thread_idx=0)
        
#         # 验证 history
#         self.assertEqual(len(self.model.history), 1)
#         self.assertEqual(self.model.history[0][0], {"from": "system", "value": system_prompt})
        
#         # 验证 sample_memory
#         self.assertEqual(len(self.model.sample_memory), 1)
#         self.assertEqual(len(self.model.sample_memory[0]), 2)
#         self.assertEqual(self.model.sample_memory[0][0], [{"from": "system", "value": system_prompt}])

#     def test_init_llm_multi_thread(self):
#         """测试多线程模式下的初始化"""
#         system_prompts = ["System A", "System B"]
#         self.model.init_llm(system_prompts)
        
#         # 验证 history
#         self.assertEqual(len(self.model.history), 2)
#         self.assertEqual(self.model.history[0][0], {"from": "system", "value": "System A"})
#         self.assertEqual(self.model.history[1][0], {"from": "system", "value": "System B"})
        
#         # 验证 sample_memory
#         self.assertEqual(len(self.model.sample_memory), 2)
#         self.assertEqual(self.model.sample_memory[0][0], [{"from": "system", "value": "System A"}])
#         self.assertEqual(self.model.sample_memory[1][0], [{"from": "system", "value": "System B"}])

#     def test_chat_without_history_single_thread(self):
#         """测试单线程对话功能及 history 更新"""
#         system_prompt = "You are a helpful assistant."
#         input_text = "Hello, how are you?"
        
#         # 初始化系统提示
#         self.model.init_llm(system_prompt, thread_idx=0)
        
#         # 生成回复
#         response = self.model.chat_without_history(input_text, max_tokens=10)
        
#         # 验证回复非空
#         self.assertIsInstance(response, str)
#         self.assertTrue(len(response) > 0)
        
#         # 验证 history 更新
#         self.assertEqual(len(self.model.history[0]), 3)  # [system] + [user, gpt]
#         self.assertEqual(self.model.history[0][1], {"from": "human", "value": input_text})
#         self.assertEqual(self.model.history[0][2]["from"], "gpt")
#         self.assertTrue(len(self.model.history[0][2]["value"]) > 0)

#     def test_chat_without_history_multi_thread(self):
#         """测试多线程对话功能及 history 更新"""
#         system_prompts = ["System A", "System B"]
#         input_texts = ["Hi from thread 0", "Hi from thread 1"]
        
#         # 初始化系统提示
#         self.model.init_llm(system_prompts)
        
#         # 生成回复
#         responses = self.model.chat_without_history(input_texts, max_tokens=10)
        
#         # 验证回复类型
#         self.assertIsInstance(responses, list)
#         self.assertEqual(len(responses), 2)
#         self.assertTrue(all(isinstance(r, str) and len(r) > 0 for r in responses))
        
#         # 验证 history 更新
#         for i in range(2):
#             self.assertEqual(len(self.model.history[i]), 3)  # [system] + [user, gpt]
#             self.assertEqual(self.model.history[i][1], {"from": "human", "value": input_texts[i]})
#             self.assertEqual(self.model.history[i][2]["from"], "gpt")
#             self.assertTrue(len(self.model.history[i][2]["value"]) > 0)

#     def test_uninitialized_history_raises_error(self):
#         """测试未初始化时调用 chat_without_history 抛出异常"""
#         with self.assertRaises(ValueError):
#             self.model.chat_without_history("Test input")

#     def test_history_growth_after_multiple_chats(self):
#         """测试多次对话后 history 的增长"""
#         system_prompt = "System Prompt"
#         input_texts = ["Q1", "Q2", "Q3"]
        
#         # 初始化
#         self.model.init_llm(system_prompt, thread_idx=0)
        
#         # 进行三次对话
#         for text in input_texts:
#             self.model.chat_without_history(text, max_tokens=10)
        
#         # 每次对话增加 2 条记录（user + gpt）
#         self.assertEqual(len(self.model.history[0]), 1 + 2 * len(input_texts))  # system + 3轮对话

#     def test_generate_adds_to_history(self):
#         """测试 generate 方法是否将结果添加到 history"""
#         prompt = "Test prompt"
#         self.model.init_llm("System", thread_idx=0)
        
#         # 调用 generate 并添加到 history
#         response = self.model.generate(prompt, add_to_history=True, max_tokens=5)
        
#         # 验证 history 更新
#         self.assertEqual(len(self.model.history[0]), 3)  # system + user + gpt
#         self.assertEqual(self.model.history[0][1], {"from": "human", "value": prompt})
#         self.assertEqual(self.model.history[0][2]["from"], "gpt")
#         self.assertTrue(len(self.model.history[0][2]["value"]) > 0)



def test_API():


    input_texts = [
        "Who is the most famous basketball player in the world? "
    ]


    from lib.RAG_modules import Generate_Agent

    llm = Generate_Agent()
    print(llm.generate(input_texts))

def test_LLM_local():


    input_texts = [
        "How's everything going today?",
        "Goo afternoon!"
    ]


    llm = Large_Language_Model()
    llm.init_llm(["You're an assistant to help human!"])
    outputs = llm.generate(input_texts, return_time = False)

    print('=' * 20)
    print('Input: \n' + '\n'.join(input_texts)) 


    print('=' * 20)
    print('Output: \n' + '\n'.join(outputs)) 



def testInteractionLimitation():
    """
    测试极限LLMs的交互链条数目
    
    """
    # 共计
    repeat_string = "How's every thing going?"
    limit_interaction_chain = [repeat_string for i in range(20)]

    llm = Large_Language_Model()
    init_prompt = "You're an assistant to help human!"
    llm.init_llm([init_prompt])

    done = False
    print('=' * 20)
    print('Start testing limitation:')

    while not done:
        limit_interaction_chain.extend([repeat_string] * 2)

        limit_length = len(limit_interaction_chain)
        print('=' * 5)
        print(f'Limitation List Length: {limit_length}')
        try:
            llm.generate(prompt = limit_interaction_chain)

        except:
            print('=' * 20)
            print(f'Limitation List Length: {limit_length}')
            print(f'Limitation Word Nums: {len(repeat_string)}')
            print(f'Limitation System Prompt Nums: {len(init_prompt)}')
            print(f'Limitation Total String Nums: {limit_length * len(repeat_string.split(' ')) * len(init_prompt.split(' '))}')
            done = True
            break

    return 


def testLLMReleaseAndReload():
    """
    测试是否正确加载与释放
    ##通过测试##
    
    """
    dec_config = DecisionConfig(
        model_dir = "qwen2.5_1.5B", 
    )
    print("=" * 20)
    print("Start Loading")
    dec_agent = Decision_Agent(**dec_config.todict(exclude_types=[StrategyParams]))

    dec_agent.model.generate('1234567')
    print("=" * 20)
    print("Start Release")
    dec_agent.release()

    print("=" * 20)
    print("Start Loading")
    dec_agent.reload(dec_config)
    dec_agent.model.generate('1234567')

    print('=' * 20)
    print("Done!")

    return 


def testLLMWithTime():

    input_texts = [
        "How's everything going today?",
        "Goo afternoon!"
    ] * 50

    llm = Large_Language_Model(local_dir = 'Qwen3-8B')
    llm.init_llm(["You're an assistant to help human!"])
    outputs, time_lis = llm.generate(input_texts, return_time = True)

    print('=' * 20)
    print('Input: \n' + ','.join(input_texts)) 


    print('=' * 20)
    print('Output: \n' + ','.join(outputs)) 

    print('=' * 20)
    print('Time: \n' + ','.join([str(time_lis[i]) for i in range(len(time_lis))])) 

    return 


def testLLMReloadLora():
    config = DecisionConfig()

    print(config.lora_dir)

    model = Decision_Agent(config = config)
    model.release()
    model.reload_lora(config = config)

    print(model.model.model)
    # 试一试能不能用
    print(model.model.generate('This is a test'))

    return 


def testLLMSaveModel():
    import datetime

    mytrainconfig = MyTrainConfig()
    config = DecisionConfig()

    print(config.lora_dir)

    model = Decision_Agent(config = config)
    model.release()
    model.reload_lora(config = config)

    current_fintuned_model_dir = mytrainconfig.model_output_dir + '/AERR/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
    
    before = time.time()
    model.save_model(dir = current_fintuned_model_dir)
    after = time.time()

    time_delta = after - before
    print("Successfully Save Model!")
    print(f"Save Cost Time: {time_delta} s")


def test_llmV2():

    from transformers import AutoModel, AutoTokenizer
    config = AERRConfig()
    config.decision.load_without_model = True # 不加载模型，使用外部模型

    # 模拟外部模型
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "/root/autodl-tmp/Qwen3-1.7B")
    device = config.decision.device
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "/root/autodl-tmp/Qwen3-1.7B")

    # 如果是避开model的加载，则需要初始化system prompt
    # pipeline = AERR(config)
    # pipeline.decision_agent.init_system_prompt(tokenizer = tokenizer)

    # print("=" * 20)
    # print("Loading Success! ")
    # pipeline.generate_batch(["Who's the most famous basketball player in history? "], 
    #                         model = model, 
    #                         tokenizer = tokenizer)

    llm = Large_Language_ModelV2(without_model = True)
    llm.init_llm(f"""
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
    User Input Example: 
    How does artificial intelligence impact the healthcare industry?
    **Expected Output Example**: 
    <Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
    <Action> Query 0 | Query Rewrite | Nums 3 </Action>
    <Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail>
    <Action> Query 0 | Query Reason | Nums 1 </Action>
    <Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>
    <END>
    **Please follow the format strictly!**

    More information:
    - You will receive the user's input information in the following conversation, and during the interaction,provide your <think> process, <Action> decision, and optional <Detail> explanations.
    - Your output would be evaluated through 2 aspects:
    - **Precision and Recall** (e.g. F1 Score, EM Score)
    - **Time**: The time spent on the interaction. **Make sure to generate as soon as you can.**
    - **Only retrieved documents will be engaged into final input.**

    Now Let's start interaction! 
    """)


    llm.init_llm_complement_prompt("""
    **Expected Output Example**: 
    <Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
    <Action> Query 0 | Query Rewrite | Nums 3 </Action>
    <Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail>
    <Action> Query 0 | Query Reason | Nums 1 </Action>
    <Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>
    <END>
    **Please follow the format strictly!** 
    Your Output: """.strip())

    # 构建当前状态提示
    state_prompt = f"""
    Current State:
    **User Input:**
    {"How's every thing going today? "}

    **Queries Collected:**
    {"How's every thing going today? "}

    **Documents Retrieved:**
    Nothing Yet. 

    Please give out your <Think>, <Action>, and <Detail>. Remember, "Stop" when all documents are sufficiently covered user's input as soon as possible.
    """

    print(llm.generate(state_prompt, model = model, tokenizer = tokenizer))




    return 

