from ..dataset import SampleCritic


class AERR:

    """
    参数化版本的 AERR 类，支持灵活配置。
    """

    def __init__(self, config: AERRConfig = None, model :Large_Language_Model = None, tokenizer: "AutoTokenizer" = None, *args, **kwargs):
        if config is None:
            train_config = MyTrainConfig()
            config = train_config.to_AERRConfig()

        # 初始化生成模型
        if config.init_decision_model:
            self.decision_agent = Decision_Agent(model = model, 
                                                 tokenizer = tokenizer, 
                                                 config = config.decision)
        
        # 初始化执行模型
        if config.init_execution_model:
            self.excution_agent = Execution_Agent(config.execution)

        # 初始化生成模型
        if config.init_generate_model:
            self.generate_agent = Generate_Agent(config.generative)

        self.sample_critic = SampleCritic()
        self.forest = None

    def generate_batch(self, 
                    input_prompts: Union[List[str], str], 
                    max_tree_length: int = 100, 
                    extract_context_from_template: bool = False, 
                    model: "AutoModelForCausalLMWithValueHead" = None, 
                    generate_func = None, 
                    sample_mode: str = "normal", 
                    our_config: "MyTrainConfig" = None, 
                    **kwargs):      
        """
        获取用户输入后，返回在AERR流程下的结果
        该过程包含训练数据的输出————输出了流程中所生成的内部模型对话
        Attention:
            input_prompts: 要求为字符串、不可以是张量
        """

        if extract_context_from_template:
            input_prompts = [self.extract_user_input(input_prompts[i]) for i in range(len(input_prompts))]

        if isinstance(input_prompts, str):
            input_prompts = [input_prompts]

        if type(input_prompts) == str:
            is_batch = False
        else:
            is_batch = True

        state = AERRStateManager(input_prompts)

        interaction_nums = 0
        while not state.all_done and interaction_nums < max_tree_length:

            query_lists, document_lists, done_flags = state.get_state()
            model_output_prompts, time_costs, model_input_prompts = self.decision_agent.generate_batch(
                                                                  input_prompts = state.get_decision_model_input(), 
                                                                  return_time = True, 
                                                                  return_input_prompts = True, 
                                                                  model = model, 
                                                                  generate_func = generate_func, 
                                                                  tokenizer = tokenizer)
            state.update_decision(model_input_prompts, model_output_prompts, time_costs)

            # 动作过程涉及对列表的修改
            query_lists, document_lists, done_flags, time_costs, actions = self.excution_agent.take_action_batch(state, return_actions = True)
            state.update_execution(query_lists, document_lists, done_flags, time_costs, actions)

            interaction_nums += 1
        
        state.finish()
        
        output = self.generate_agent.generate(input_text = state.get_final_input())
        returns = [output] if is_batch else [output[0]]

        if sample_mode == "normal":
            returns.extend(state.get_training_data(need_init_prompt = True))
            if our_config is not None: 
                state.to_csv(our_config.eval_example_dir, save_interval = our_config.save_training_interaction_interval)

        returns.append(input_prompts) # input prompts 根本没有动过
        return *returns, 
        


        
class Execution_Agent:
    def __init__(
        self, 
        config: ExecutionConfig = None, 
        *args,
        **kwargs
    ):
        """
        初始化QueryExpander类，用于生成多个扩展查询变体。
        如果需要调用内存中的model，而不是初始化本地model，请务必传入model参数和tokenizer参数
        
        Args:
            model: 用于扩展的LLM模型（如Llama3）
            verbose (bool): 是否输出详细日志
        """
        if config == None:
            config = ExecutionConfig()

        model_dir = config.model_dir
        indexer_device: torch.device = config.indexer_device
        model_device:  torch.device = config.model_device
        index_load_path: str = config.index_load_path
        document_load_path: str = config.document_load_path
        verbose: bool = config.verbose
        strategy_params: StrategyParams = config.strategy_params
        batch_size : int = config.batchsize
        test: bool = config.test

        print("Loading Execution Agent...")
        if not test:
            if model_dir is not None:
                self.model = Large_Language_Model(local_dir = model_dir, 
                                                device = model_device)
            else:
                self.model = Large_Language_Model_API()

            self.indexer = IndexBuilder(index_load_path = index_load_path,
                                        document_load_path = document_load_path, 
                                        device = indexer_device)

            self.verbose = verbose
            self.logger = logging.getLogger(__name__)
            # 初始化模型
            self.batch_size = batch_size
            # 这一步塞到Take Action中进行
            self.model.init_llm(self._init_prompt())

        if type(strategy_params) == dict:
            self.strategy_params = strategy_params
        else:
            self.strategy_params = strategy_params.todict()
        
        print("Done!")

    def _init_prompt(self):
        """
        用以初始化的提示词，该提示词可帮助模型快速了解任务
        
        """
        prompt = f"""You are an execution agent can help to improve final generate quality. There are few tasks need your help."""
        return prompt

    def query2UserInput(self, query_list):
        """获取UserInput"""
        user_input = query_list[0]
        if user_input.startswith("(Already Retrieved)"):
            # 已标记：移除标签，使用原始字符串检索
            user_input = user_input[len("(Already Retrieved)"):]
        
        return user_input

    def should_skip_action(self, action, query_list, document_list):
        """输入的Target ids都是未被清洗的"""
        # 需要调用模型的情况
        # 一定可以通过的

        action_type = action['action_type']
        target_ids: List[int] = action['target_id'] # 原始的Target_ids

        if action_type in ["Query Rewrite", "Document Filter", "Summarize Documents"]:
            pass
        # 不一定可以通过的，需要判定的: Query Extract
        if action_type == 'Query Extract':
            valid_target_ids = [target_id for target_id in target_ids if target_id < len(document_list)]
            return len(valid_target_ids) == 0
        
        # 不需要调用模型的基础动作
        if action_type == 'Stop':
            return "skipAll" # 用来标识是否跳过全部
        if action_type in ["Delete Query", "Query Search", "Delete Documents", "Sort Documents"]: 
            # Delete Query 和 Query Search内部都有防爆机制，本质上是取出筛选过后的结果进行处理
            # Delete Documents 和Sort Documents内部也有防爆机制
            pass 
        
        return False

    def take_action_batch(self, 
                          state: "AERRStateManager", 
                          return_actions: bool = False, 
                          document_threshold: int = 10, 
                          **kwargs) -> Tuple[List[List[str]], List[List[str]], List[bool], List[float]]:
        """
        批量执行决策指令，处理多个样本的查询和文档列表
        
        Args:
            state (AERRStateManager): 状态管理器对象
        
        Returns:
            Tuple[List[List[str]], List[List[str]], List[bool]]: 
                更新后的查询列表、文档列表、是否终止标志

        ##结束测试，测试通过##
        """
        decision_prompts, query_lists, document_lists, done_flags = state.get_execution_model_input()

        if isinstance(decision_prompts, str):
            decision_prompts = [decision_prompts]
            isBatch = False
        else:
            isBatch = True
        
        if isinstance(query_lists[0], str):
            query_lists = [query_lists]

        if isinstance(document_lists[0], str):
            document_lists = [document_lists]
        
        time_lis = [0 for i in range(len(decision_prompts))]
        
        batch_input_prompts = []  # 存储所有需要调用模型的 input_prompt
        prompt_metadatas = []      # 存储每个 input_prompt 对应的 (sample_index, action_index, action_type)

        # 批量提取决策组件
        decisions_batch = [AERRTemplate.DecisionPrompt2ActionDict(dp) for dp in decision_prompts]
        skip_decisions = []
        # 第一步：收集所有需要调用模型的 input_prompts 和元数据
        for decision_index, (decision, query_list, document_list) in enumerate(zip(decisions_batch, query_lists, document_lists)):
            # 当某一动作链执行完毕时，跳过动作

            before = time.time()
            actions = decision['actions']
            user_input = self.query2UserInput(query_list)

            for action_index, action in enumerate(actions):
                # 统一的动作筛选
                skip = self.should_skip_action(action = action, query_list = query_list, document_list = document_list)
                skip_decisions.append(skip)
                if skip == True:
                    continue
                elif skip == "skipAll":
                    break

                action_type = action['action_type']
                target_ids: List[int] = action['target_id']

                # 需要调用模型的情况
                if action_type in ['Query Rewrite', 'Query Extract', 'Document Filter', 'Summarize Documents']:
                    if action_type == 'Query Rewrite':
                        input_prompt = AERRTemplate.Execution_query_rewrite_input_prompt(user_input)
                    elif action_type == 'Query Extract': 
                        input_prompt = AERRTemplate.Execution_query_extract_input_prompt(user_input, document_list = document_list)
                    elif action_type == 'Document Filter':
                        input_prompt = AERRTemplate.Execution_document_filter_input_prompt(user_input, documents = document_list) 
                    elif action_type == 'Summarize Documents': 
                        input_prompt = AERRTemplate.Execution_document_summary_input_prompt(user_input, documents = document_list) 
                    
                    # 记录 input_prompt 及其对应的样本索引、动作索引和动作类型
                    batch_input_prompts.append(input_prompt)
                    prompt_metadatas.append({
                        'sample_index': decision_index,
                        'action_index': action_index,
                        'action_type': action_type, 
                        'target_id': target_ids, 
                    })

            # 继续跟踪耗时，追踪元数据处理的时候消耗的时间
            after = time.time()
            time_lis[decision_index] += after - before

        # 第二步：批量调用模型并获取所有响应
        if len(batch_input_prompts) != 0:
            all_responses, take_action_time_lis = self.model.generate(
                                                batch_input_prompts,
                                                return_time = True, 
                                                **self.strategy_params)

        # 第三步：分配并采取动作
        for decision_index, (decision, query_list, document_list) in enumerate(zip(decisions_batch, query_lists, document_lists)):

            before = time.time()
            actions = decision['actions']
            user_input = self.query2UserInput(query_list)

            for action_index, action in enumerate(actions):

                # 这样的话，逻辑与前文就是严格对齐的
                skip = skip_decisions.pop(0)
                if skip == True:
                    continue
                elif skip == "skipAll":
                    done_flags[decision_index] = True
                    break

                action_type = action['action_type']
                target_ids: List[int] = action['target_id']

                # 以下为基础动作
                if action_type == "Delete Query":
                    self.DeleteQuery(target_ids = target_ids, query_list = query_list)
                elif action_type == "Query Search": # 直接更改document list
                    self.QuerySearch(query_list = query_list, document_list = document_list, target_ids = target_ids)
                elif action_type == "Delete Documents":
                    self.DeleteDocuments(target_ids = target_ids, document_list = document_list)
                elif action_type == "Sort Documents":
                    self.SortDocuments(target_ids = target_ids, document_list = document_list)
                
                # 以下为进阶动作
                if action_type in ['Query Rewrite', 'Query Extract', 'Document Filter', 'Summarize Documents']: 
                    # 但实际操作中报错，因此这里需要交叉验证
                    response = all_responses.pop(0)
                    metadata = prompt_metadatas.pop(0)
                    time_cost = take_action_time_lis.pop(0)

                    sample_index = metadata['sample_index']
                    action_type = metadata['action_type']

                    query_list: list = query_lists[sample_index]
                    document_list = document_lists[sample_index]

                    # 更新query_lists和document_lists
                    if action_type == 'Query Rewrite': 
                        self.QueryRewrite(response, query_list, document_list)
                    if action_type == 'Query Extract': 
                        self.QueryExtract(response, query_list, document_list)
                    if action_type == 'Document Filter':
                        self.DocumentFilter(document_list, response)
                    if action_type == 'Summarize Documents':
                        self.SummarizeDocuments(document_list, response)

                    time_lis[decision_index] += time_cost

            # 动作链结束才能计算耗时
            after = time.time()
            time_lis[decision_index] += after - before
            
        # 如果文档数量超过10，就应该执行过滤操作
        prompt_metadatas = []
        batch_input_prompts = []
        for decision_index, (query_list, document_list) in enumerate(zip(query_lists, document_lists)):
            before = time.time()
            user_input = self.query2UserInput(query_list)
            if len(document_list) >= document_threshold:
                batch_input_prompts.append(AERRTemplate.Execution_document_filter_input_prompt(user_input, documents = document_list))
                prompt_metadatas.append({'sample_index': decision_index})
                after = time.time()
                time_lis[decision_index] += after - before
        
        all_responses, take_action_time_lis = self.model.generate(
                                                batch_input_prompts,
                                                return_time = True, 
                                                **self.strategy_params)
        
        for response, metadata, time_cost in zip(all_responses, prompt_metadatas, take_action_time_lis):
            decision_index = metadata['sample_index']
            document_list = document_lists[decision_index]
            time_lis[decision_index] += time_cost
            self.DocumentFilter(document_list, response)

        if not isBatch:
            query_lists = query_lists[0]
            document_lists = document_lists[0]
            done_flags = done_flags[0]
        
        if return_actions:
            return query_lists, document_lists, done_flags, time_lis, [decision["actions"] for decision in decisions_batch]

        return query_lists, document_lists, done_flags, time_lis
    
    def DeleteQuery(self, target_ids: List[int], query_list: List[str]):
        """删除Query中的某些键，测试通过"""
        target_ids = [target_id for target_id in target_ids if target_id < len(query_list)] # 筛选机制
        
        # 1. 降序排序目标索引（从大到小删除，避免后续索引变化）
        sorted_ids = sorted(target_ids, reverse=True)
        
        # 2. 逐个删除（从大索引开始，避免影响小索引）
        for idx in sorted_ids:
            # 原则上不允许删除Query 0
            if 0 < idx < len(query_list):
                del query_list[idx]
            # 无效索引自动跳过（不报错）

    def QuerySearch(self, query_list, document_list: list, target_ids: List[int], k = 5):
        """Target Query直接检索，列表一一对应，处理Already Retrieved标签逻辑。注意，每一个target id都会被检索"""
        target_ids = [target_id for target_id in target_ids if target_id < len(query_list)] # 筛选机制
        for idx in range(len(target_ids)):
            if target_ids[idx] >= len(query_list):
                continue
                
            current_query = query_list[target_ids[idx]]
            # 检查是否已标记
            if current_query.startswith("(Already Retrieved)"):
                # 已标记：移除标签，使用原始字符串检索
                original_query = current_query[len("(Already Retrieved)"):]
                retrieved_docs = self.indexer.topk_search(original_query, k)
            else:
                # 未标记：进行检索并添加标签
                retrieved_docs = self.indexer.topk_search(current_query, k)
                # 更新query_list为标记后的字符串
                query_list[target_ids[idx]] = "(Already Retrieved)" + current_query

            document_list.extend(retrieved_docs)

    def DeleteDocuments(self, target_ids: List[int], document_list: List[str]):
        """Documents直接删除，测试通过"""

        target_ids = [target_id for target_id in target_ids if target_id < len(document_list)]
        # 1. 降序排序目标索引（从大到小删除，避免后续索引变化）
        sorted_ids = sorted(target_ids, reverse=True)
        
        # 2. 逐个删除（从大索引开始，避免影响小索引）
        for idx in sorted_ids:
            if 0 <= idx < len(document_list):
                del document_list[idx]
            # 无效索引自动跳过（不报错）

    def SortDocuments(self, document_list: List[str], target_ids: List[int]):
        """重排序工具,重排序索引放在了quantity_lis中了,测试通过"""

        valid_indices = [idx for idx in target_ids if 0 <= idx < len(document_list)]
        reordered = [document_list[idx] for idx in valid_indices]
        remaining = [doc for i, doc in enumerate(document_list) if i not in valid_indices]
        document_list[:] = reordered + remaining

    def AddQuery(self, query_list: List[str], detail: str):
        """将detail的内容添加到query list中,测试通过"""
        query_list.append(detail)

    def QueryRewrite(self, model_response, query_list: List[str], document_list: List[str]):
        """集成Query Rewrite的管线，包含从改写 -> 检索 -> 添加三个部分，是一个复杂管线"""
        # 先询问API怎么个事情，拿到新的Query
        new_queries = AERRTemplate.Execution_parse_query_output(model_response)
        # 对Query进行搜索，每个Query检索三个文档，这样会共计生成15个文档
        for query in new_queries:
            documents = self.indexer.topk_search(query, k = 3) 
            document_list.extend(documents)
            # 更新query列表
            query_list.append(query)

    def QueryExtract(self, model_response, query_list: List[str], document_list: List[str]):
        """集成Query Extract的管线，这一部分应当需要让外部API思考，为了进一步获取信息，应当查询哪一部分的内容，而不是简单的返回几个Query就可以了"""
        new_queries = AERRTemplate.Execution_parse_query_output(model_response)
        # 对Query进行搜索，每个Query检索三个文档，这样会共计生成15个文档
        for query in new_queries:
            documents = self.indexer.topk_search(query, k = 3) 
            document_list.extend(documents)
            # 更新query列表
            query_list.append(query)

    def DocumentFilter(self, document_list: List[str], response: str) -> None:
        """
        根据模型输出的过滤结果，直接修改原始文档列表（就地修改）

        Args:
            document_list (List[str]): 原始文档列表（将被直接修改）
            response (str): 模型生成的过滤响应（包含<Irrelevant Document>标签）
        """
        if isinstance(response, list):
            response = response[0]
        # 1. 解析不相关文档索引
        irrelevant_indices = AERRTemplate.Execution_parse_irrelevant_documents(response)
        
        # 2. 转换为整数索引列表（用于删除操作）
        try:
            irrelevant_indices_int = [int(idx) for idx in irrelevant_indices]
        except ValueError:
            irrelevant_indices_int = []
        
        # 3. 按索引从大到小排序（避免删除后索引变化）
        irrelevant_indices_int.sort(reverse=True)
        
        # 4. 就地删除不相关文档（从后往前删除，避免索引错乱）
        for idx in irrelevant_indices_int:
            if idx < len(document_list):  # 确保索引有效
                del document_list[idx]

    def SummarizeDocuments(self, document_list: List[str], response: str) -> None:
        """总结所有的文档内容，删除所有原来的文档"""
        if isinstance(response, list):
            response = response[0]

        # 1. 解析不相关文档索引
        summary_prompt: str = AERRTemplate.Execution_parse_document_summary_output(response)
        document_list[:] = [summary_prompt]

class DummyDecisionAgent:
    """该逻辑待完善"""
    def __init__(self):
        pass
    def generate(self):
        yield 

from ..config.AERR.DecisionAgentConfig import DecisionAgentConfig
from ..llm.api_llm import Large_Language_Model_API 
from ..llm.local_llm import LargeLanguageModel_Tensor2Tensor

class Decision_Agent:
    
    def __init__(
        self, 
        config: "DecisionAgentConfig" = None):
        """
        初始化QueryExpander类，用于生成多个扩展查询变体。
        
        Args:
            model: 用于扩展的LLM模型（如Llama3）
            verbose (bool): 是否输出详细日志
        """
        # 原来的版本太不优雅了
        self.config = config
        self.load()
    

    def load(self):
        # 选择模式，加载模型
        # 用一个参数来控制就可以了
        if self.config.load_mode == "dummy":
            self.model = DummyDecisionAgent()
        elif self.config.load_mode == "api":
            self.model = Large_Language_Model_API()
        elif self.config.load_mode == "local":
            # 仅仅只使用逻辑，不在内部初始化吧，内部初始化灵活性太低了，之后如果想弄，可以另组一个逻辑
            self.model = LargeLanguageModel_Tensor2Tensor()

    def generate_batch(self, 
                       input_prompts: List[str], 
                       return_time: bool = False, 
                       return_input_prompts: bool = False, 
                       model: "AutoModelForCausalLMWithValueHead" = None,
                       generate_func = None,  
                       tokenizer: "AutoTokenizer" = None, 
                       **kwargs):
        """
        Args:
            query_list: 多个 batch 的查询列表（List[List[str]]）
            document_list: 多个 batch 的文档列表（List[List[str]]）
        
        Returns:
            模型的原始响应列表（每个元素对应一个 batch 的响应）
        """

        if not self.load_api: 
            response = self.model.generate(input_prompts, 
                                        return_time = return_time, 
                                        return_input_prompts = return_input_prompts, # 注意，这个return的结果中，不是列表嵌套，仅仅只是一个List[str]形式
                                        include_complement = False, 
                                        model = model, 
                                        generate_func = generate_func, 
                                        tokenizer = tokenizer, 
                                        **self.strategy_params)
        else:
            response = self.model.generate(input_prompts, 
                                        return_time = return_time, 
                                        return_input_prompts = return_input_prompts, 
                                        include_system = return_time)
        
        return *response, 

class Generate_Agent(Large_Language_Model_API):
    def __init__(self, 
                 config: GenerativeConfig = None, 
                 init_prompt: str = None, 
                 complement_prompt: str = None, 
                 **kwargs):
        print("Loading Generate Agent...")
        super().__init__()

        if config == None:
            config = GenerativeConfig()

        self._init_rag_system_prompt(init_prompt, batchsize = config.batchsize) 
        
    def _init_system_prompt(self, init_prompt, batchsize):
        """
        初始化最终的黑盒模型
        
        """
        
        if init_prompt is None:
            self.init_llm(system_prompt = [
        """
You are an agent assistant to answer user's input.  
Follow the **Example Format** below to generate your response.

**Format**:
---
User Input:  
[User's specific question or instruction]

Your Output:  
[Your answer must follow these rules]:
1. Directly answer the question without extra explanations.
2. User's input question is a simple question, generate your output in **few words**.
3. Keep language concise. 
4. Find out your answer from referenced documents and details if given. 
---

**Example 1**:
User Input:
When was the first computer invented?
Expected Output: 
1945.

**Example 2**:
User Input:
Who invented the modern three-point seatbelt?
Expected Output: 
Nils Bohlin. 

**Example 3**:
User Input:
What is the most widely spoken language in the world?
Expected Output: 
Mandarin Chinese. 

Please answer the user's input follow the examples. 
""".strip()
            for i in range(batchsize)])
        else:
            self.init_llm(system_prompt = [init_prompt for i in range(batchsize)])

    def _init_complement_prompt(self, complement_prompt, batchsize):
        if complement_prompt is None:
            self.init_llm_complement_prompt([
        ''.strip() for i in range(batchsize)])
        else:
            self.init_llm_complement_prompt([complement_prompt for i in range(batchsize)])

    def _init_rag_system_prompt(self, init_prompt, batchsize):
        """
        初始化最终的黑盒模型
        
        """
        
        if init_prompt is None:
            self.init_llm(system_prompt = [
"""""".strip()
            for i in range(batchsize)])
        else:
            self.init_llm(system_prompt = [init_prompt for i in range(batchsize)])
