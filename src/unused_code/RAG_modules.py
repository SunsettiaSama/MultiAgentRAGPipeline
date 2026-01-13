from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import logging
from typing import Union, List, Dict, Optional, Any, Tuple, TYPE_CHECKING
import torch
from sentence_transformers import SentenceTransformer
import os
import re
from flashrag.evaluator.metrics import F1_Score, ExactMatch, Precision_Score, Recall_Score
from flashrag.config import Config
import time
import gc
import warnings
import copy


from .large_language_model import Large_Language_Model, Large_Language_Model_API, CasualLMWithValueHead
from .config import *
from .dataset import ConversationTree
from .reward import weighted_reward_calculate

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

params = {
    'text_col' : 'text', 
    'id_col' : 'id', 
    'name_col' : 'title', 
    'embedding_col': 'embedding', 
    'projection_dim' : 384, 

}

class BGEEmbedder:
    def __init__(self, 
                 model_name='BAAI/bge-small-en', 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        """
        初始化 BGE 模型
        Args:
            model_name: BGE 模型名称（如 'BAAI/bge-small-en', 'BAAI/bge-base-zh'）
        """
        self.model = SentenceTransformer(model_name, cache_folder = './BGE_model/', device = device)
    
    def encode(self, texts, batch_size=32, show_progress=False):
        """
        将文本编码为向量
        Args:
            texts: 单个文本字符串或文本列表
            batch_size: 批处理大小（仅在 texts 是列表时生效）
            show_progress: 是否显示进度条（仅在 texts 是列表时生效）
        Returns:
            单个文本返回 1D 数组，列表返回 2D 数组（每行对应一个文本的向量）
        """
        # 处理单个文本
        if isinstance(texts, str):
            return self.model.encode([texts], convert_to_tensor=False)[0]
        # 处理文本列表
        elif isinstance(texts, list):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False
            )
            return embeddings
        else:
            raise TypeError("texts 必须是字符串或列表")
    
    def process_csv(
                self, 
                input_path, 
                output_path, 
                text_col = 'text', 
                id_col = 'id', 
                name_col = 'title', 
                batch_size=32, 
                sep = '\t', 
                **kwargs, 
                   ):
        """
        读取 CSV 文件，生成嵌入并保存
        Args:
            input_path: 输入 CSV 文件路径（含 [文章id, 文章名, 文章] 列）
            output_path: 输出 CSV 文件路径（含 [文章id, 文章名, embedding] 列）
            text_col: 文本列名（默认 '文章'）
            id_col: 文章ID列名（默认 '文章id'）
            name_col: 文章名列名（默认 '文章名'）
            batch_size: 批处理大小（控制内存使用）
        """
        print('=' * 40)
        print('Reading csv...')
        # 读取 CSV 文件
        df = pd.read_csv(input_path, sep = '\t', encoding_errors = 'ignore', on_bad_lines = 'skip', engine = 'python')
        required_cols = [id_col, name_col, text_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 必须包含列：{required_cols}")
        
        # 提取文本和元数据
        texts = df[text_col].tolist()
        metadata = df[[id_col, name_col]].copy()
        
        # 生成嵌入（分批次处理）
        embeddings = []
        print('=' * 20)
        print('Start Processing csv to embedding csv...')
        print('=' * 20)

        with tqdm(total = len(texts) / batch_size + 1) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.encode(batch_texts)
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        
        # 将嵌入转换为字符串格式（CSV 可存储）
        embedding_strings = [np.array2string(vec, separator=',') for vec in embeddings]
        
        # 合并元数据和嵌入
        metadata['embedding'] = embedding_strings

        print('=' * 20)
        # 保存结果
        metadata.to_csv(output_path, index=False)
        print(f"成功保存 {len(metadata)} 条记录到 {output_path}")

class QueryPreprocessor(ABC):
    @abstractmethod
    def generate(self, query: str) -> dict:
        pass

class QueryRewriter(QueryPreprocessor):
    """
    该类用于进行Query改写
    包括：
        改写出相关的nums个问题: Generate
    """
    
    def __init__(
        self, 
        model, 
        classifier = None, 
        type_prompts: dict = None, 
        strategies: dict = None,
        default_strategy: str = "standard",
        verbose: bool = False
    ):
        """
        Args:
            model: 用于改写的LLM模型（如Llama3）
            classifier: 查询类型分类器（函数或模型）
            type_prompts (dict): 不同类型对应的提示模板
            strategies (dict): 不同策略的参数配置（如温度、长度等）
            default_strategy (str): 默认改写策略
            verbose (bool): 是否输出详细日志
        """
        self.model = model

        self.classifier = classifier or self._classify_query
        self.type_prompts = type_prompts or {
            "default": "Rewrite this query for better clarity:",
            "financial": "Rewrite this financial query for precision:",
            "technical": "Rewrite this technical query with keywords:",
            "general": "Rewrite this general query for completeness:"
        }
        self.strategies = strategies or {
            "standard": {"temperature": 0.7, "max_length": 100},
            "creative": {"temperature": 1.0, "max_length": 100},
            "strict": {"temperature": 0.1, "max_length": 100}
        }

        self.default_strategy = default_strategy
        self.verbose = verbose
        self.logger = logging.getLogger(__name__) # 暂时不知道这个怎么用

    def _classify_query(self, query: str) -> str:
        """根据分类器确定查询类型"""
        try:
            query_type = self.classifier(query)
        except Exception as e:
            self.logger.warning(f"Classification failed: {e}")
            query_type = "default"
        return query_type

    def _select_prompt(self, query_type: str) -> str:
        """根据类型选择提示模板"""
        return self.type_prompts.get(query_type, self.type_prompts["default"])

    def _apply_strategy(self, strategy: str) -> dict:
        """根据策略选择生成参数"""
        return self.strategies.get(strategy, self.strategies[self.default_strategy])

    def Question_Rewrite(self, query: str, question_nums: int = 3) -> str:
        """
        生成结构化查询改写提示，要求模型生成指定数量的改写问题，并在最后添加 <END> 标签。
        
        Args:
            query (str): 原始查询。
            question_nums (int): 需要生成的改写问题数量。
            
        Returns:
            str: 完整的提示模板。
        """
        prompt = f"""
Please generate {question_nums} different rephrased questions based on the input query. 
Ensure the generated questions is **clearer**, **more accurate**, and **more comprehensive** than the original input query.
The questions should maintain the original meaning but vary in structure, formality, or phrasing. 
Follow the format strictly and ensure each question starts with "<Question>". 
End the output with the "<END>" tag immediately after the last question.

Format Example:
1. <Question> What is the capital of France?
2. <Question> Can you name the capital city of France?
3. <Question> Which city is the administrative center of France?

Rules:
1. Each question must start with the "<Question>" tag.
2. Use numbers 1 to {question_nums} for ordering.
3. Avoid repeating the same phrasing.
4. Keep the original intent intact.

Input Query: {query}

"""
        return prompt

    def Topic_Rewrite(self, query: str, nums: int = 3) -> str:
        """
        ========================================该方法有效，参见PPT==========================================================
        生成结构化查询改写提示，要求模型生成指定数量的改写问题，并在最后添加 <END> 标签。
        区别Question_Rewrite, 该函数进行对应主题改写，省去了如How、What等提示词，可以减少无用字符
        
        Args:
            query (str): 原始查询。
            question_nums (int): 需要生成的改写问题数量。
            
        Returns:
            str: 完整的提示模板。
        """

        prompt = f"""
Please generate **{nums}** related topics based on the input query. 
Ensure the generated topic is **clearer**, **more accurate**, and **more comprehensive** than the original input query.
The topics should expand the original scope while maintaining relevance. 
Follow the format strictly and ensure each topic starts with "<Topic>". 
End the output with the "<END>" tag immediately after the last topic.

Format Example:
1. <Topic> Renewable energy technologies in climate change mitigation
2. <Topic> Economic impacts of global warming on coastal cities
3. <Topic> Biodiversity loss and ecosystem resilience

Rules:
1. Each topic must start with the "<Topic>" tag.
2. Use numbers 1 to **{nums}** for ordering.
3. Avoid vague or overly broad topics.
4. Ensure topics cover different angles (e.g., technical, economic, social).

Input Query: {query}

"""

        return prompt
    
    def extract_questions(self, output_str: str) -> List[str]:
            """
            从LLM输出中提取所有符合格式的<Question>问题，返回列表。

            Args:
                output_str (str): LLM的输出文本。

            Returns:
                List[str]: 提取的问题列表（去除标签和编号）。
            """
            # 直到<END>tag后结束
            output_str = output_str.split('<END>')[0]
            # 正则表达式匹配符合规则的行：数字. <Question>开头，提取内容部分
            pattern = r'^(?:\s*\d+\.\s*)?<Question>(.*?)$'
            matches = re.findall(pattern, output_str, flags=re.MULTILINE)
            return matches

    def extract_topics(self, output_str: str) -> List[str]:
            """
            从LLM输出中提取所有符合格式的<Topic>问题，返回列表。

            Args:
                output_str (str): LLM的输出文本。

            Returns:
                List[str]: 提取的问题列表（去除标签和编号）。
            """
            # 直到<END>tag后结束
            output_str = output_str.split('<END>')[0]
            # 正则表达式匹配符合规则的行：数字. <Topic>开头，提取内容部分
            pattern = r'^(?:\s*\d+\.\s*)?<Topic>(.*?)$'
            matches = re.findall(pattern, output_str, flags=re.MULTILINE)
            return matches

    def generate(self, 
                 input_query: Union[str, List[str]], question_nums: int = 3, 
                 rewrite_type = 'Question',
                 ) -> Union[Dict, List[Dict]]:
        """
        处理查询改写流程，支持单个或批量输入。
        
        Args:
            input_query (str | List[str]): 需要改写的查询或查询列表。
        
        Returns:
            dict | List[dict]: 单个查询返回结果字典，多个查询返回结果列表。 
        """
        # 检查输入类型
        is_batch = isinstance(input_query, list)
        if not isinstance(input_query, (str, list)):
            raise TypeError("input_query must be a string or a list of strings")
        
        if not rewrite_type in ['Question', 'Topic']:
            raise ValueError("rewrite_type must be 'Question' or 'Topic'")
        
        # 处理单个或批量输入
        results = []
        queries = input_query if is_batch else [input_query]
        
        for query in queries:
            try:
                original_query = query.strip()
                query_type = self._classify_query(original_query)
                prompt = self._select_prompt(query_type)
                strategy_params = self._apply_strategy(query_type)

                # 构建完整提示
                if rewrite_type == 'Question':
                    full_prompt = self.Question_Rewrite(original_query, question_nums = question_nums)
                    # 调用模型生成改写后的查询
                    rewritten_query = self.model.generate(full_prompt, **strategy_params)
                    questions = self.extract_questions(rewritten_query)


                elif rewrite_type == 'Topic':
                    full_prompt = self.Topic_Rewrite(original_query, nums = question_nums)
                    rewritten_query = self.model.generate(full_prompt, **strategy_params)
                    questions = self.extract_topics(rewritten_query)
                    
                strategy_params["max_length"] = question_nums * 20
            
            except Exception as e:
                self.logger.error(f"Rewrite failed for '{original_query}': {str(e)}")
                rewritten_query = original_query  # 保留原始查询
                query_type = "unknown"  # 标记为未知类型
                prompt = "error"  # 标记错误
                strategy_params = {}  # 清空策略参数
            
            # 收集结果
            result = {
                "Rewrited Question": questions,
                "Rewrited Query": rewritten_query,
                "type": query_type,
                "original_query": original_query,
                "prompt_used": prompt,
                "strategy": strategy_params,
                "success": rewritten_query != original_query
            }
            results.append(result)
        
        # 根据输入类型返回单个或列表结果
        return results if is_batch else results[0]
    
    def _simple_classifier(self, query):
        """基于关键词的规则分类器"""
        query_lower = query.lower()
        if "股票" in query_lower or "投资" in query_lower:
            return "financial"  # 金融类查询
        elif "代码" in query_lower or "Python" in query_lower:
            return "technical"  # 技术类查询
        else:
            return "general" 

class IndexBuilder:
    """
    该类用以保存和检索相关的函数
    包括：
        构建索引： build_index
        检索topk文档: topk_search

    """
    def __init__(
                self, 
                device= torch.device('cuda'),
                index_load_path = './wikipedia_BGE_L2.contriever',
                document_load_path = './psgs_w100.tsv',
                build: bool = False, 
                **kwargs
                ):
        self.BGE_embedder = BGEEmbedder(device = device)

        if not build:
            self.load_index(index_load_path)
            self.load_document(document_load_path)


    def build_index(self, 
                csv_path = './psgs_w100.tsv', 
                projection_dim = 384, 
                id_col = 'id', 
                title_col = 'title', 
                text_col = 'text',
                index_save_path = './wikipedia.contriever',
                embedding_csv_save_path = './wikipedia_embeddings.csv',
                batch_size = 4, 
                csv_sep = '\t',
                saving_embedding_csv_root = './prompts/', 
                **kwargs) -> faiss.Index:
            """
            将CSV中的文本转换为嵌入并构建Faiss索引。
            
            Returns:
                faiss.Index: 构建的Faiss索引对象
            """
            print('=' * 40)
            print('Start building index')
            print('=' * 20)
            print('Loading data from {}'.format(csv_path))
            projection_dim = projection_dim
            df = pd.read_csv(csv_path, engine = 'python', on_bad_lines = 'skip', encoding_errors = 'ignore', header = 0, sep = csv_sep)

            # 获取文件的id和embedding
            ids, texts, titles = df[id_col].tolist(), df[text_col].tolist(), df[title_col].tolist()

            # 合并标题和文本内容
            documents = [f"{title} {text}" for title, text in zip(titles, texts)]
            
            print('=' * 20)
            print('Starting Embedding...')

            # 生成嵌入
            embeddings = []
            embedding_i = []
            ids_i = []
            total = len(documents)

            file_index = 0
            end_step = 0
            # 采用多步保存法防止内存溢出
            if not os.path.exists(saving_embedding_csv_root):
                os.makedirs(saving_embedding_csv_root)

            with tqdm(total=total, desc="Processing documents") as pbar:

                for i in range(0, total, batch_size):
                    batch = documents[i:i+batch_size]
                    batch_embeddings = self.BGE_embedder.encode(batch)
                    embeddings.extend(batch_embeddings)

                    pbar.update(len(batch))
                    end_step += 1
                    embedding_i.extend(batch_embeddings)
                    ids_i.extend(ids[i:i+batch_size])

                    # 每隔6000个文档，保存一个索引，最后一次也要保存
                    if end_step % 5000 == 0 or end_step * batch_size > total:

                        save_file_name_i = saving_embedding_csv_root + str(file_index) + '.csv'
                        embed_dict_i = {
                            'id': ids_i,
                            'embedding': [list(embedding_i[k]) for k in range(len(embedding_i))]
                        }
                        embed_df_i = pd.DataFrame(embed_dict_i)
                        embed_df_i.to_csv(save_file_name_i, index=False)

                        # 清空embedding_i
                        del embedding_i
                        del ids_i
                        embedding_i = []
                        ids_i = []

                        # 文件序列加一
                        file_index += 1

                        # 显示进度
                        pbar.set_description(f'Processing {save_file_name_i}')
                    

            # 转换为numpy数组
            embeddings_array = np.array(embeddings, dtype='float32')
            
            # 检查维度匹配
            if embeddings_array.shape[1] != projection_dim:
                raise ValueError(
                    f"Embedding dimension {embeddings_array.shape[1]} " 
                    f"does not match projection_dim {projection_dim}"
                )

            # 创建Faiss索引
            self.indexer = faiss.IndexFlatL2(projection_dim)
            self.indexer.add(embeddings_array)

            print('=' * 20)
            print('Starting Embedding to Faiss...')
            # 保存索引
            if not os.path.exists(index_save_path):
                faiss.write_index(self.indexer, index_save_path)
                self.index_path = index_save_path
                print(f"Index saved to {index_save_path}")
            else:
                print(f"Index file {index_save_path} already exists. Skipping save.")
            
            return self.indexer

    def convert_index_flat_to_ivfflat(self, input_path, output_path, nlist=512):
        """
        将 IndexFlatL2 转换为 IndexIVFFlat，并保持原始 ID 不变。
        
        Args:
            input_path (str): 原始 IndexFlatL2 的文件路径。
            output_path (str): 新 IndexIVFFlat 的保存路径。
            nlist (int): 聚类中心数量（需根据数据量调整）。
        """
        print('=' * 40)
        print('Start Convert L2 Index to IVFFlat Index')
        print('=' * 20)
        print('Read FlatL2 Index')
        # 1. 读取原始索引
        original_index = faiss.read_index(input_path)
        d = original_index.d
        num_vectors = original_index.ntotal

        print('=' * 20)
        print('Extracting vectors and IDs')

        # 2. 提取所有向量和原始 ID
        vectors = np.zeros((num_vectors, d), dtype='float32')
        for i in range(num_vectors):
            vectors[i] = original_index.reconstruct(i)  # 逐个提取向量
        original_ids = np.arange(num_vectors, dtype='int64')  # 原始 ID 是 0,1,2,...,n-1

        print('=' * 20)
        print('Mapping IDs to IVFFlat Index')

        # 3. 创建带 ID 映射的 IndexIVFFlat
        quantizer = faiss.IndexFlatL2(d)  # 量化器（可选：IndexFlatIP 如果使用内积）
        index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index_with_id = faiss.IndexIDMap(index_ivf)  # 添加 ID 映射层

        print('=' * 20)
        print('Training IVFFlat Index')

        # 4. 训练索引（聚类）
        index_with_id.train(vectors)

        # 5. 添加向量和原始 ID
        index_with_id.add_with_ids(vectors, original_ids)

        # 6. 设置检索参数（可选）
        index_with_id.nprobe = 128  # 调整检索时搜索的聚类中心数量

        print('=' * 20)
        print(f'Saving IVFFlat Index to {output_path}')

        # 7. 保存新索引
        faiss.write_index(index_with_id, output_path)

        print('=' * 20)
        print('Finished')

    def load_index(self, index_load_path: str):
        """加载索引"""
        if not os.path.exists(index_load_path):
            raise FileNotFoundError(f"索引文件 {index_load_path} 不存在。")
        
        self.indexer = faiss.read_index(index_load_path)

    def load_document(self, document_load_path: str):
        """加载文档"""
        if not os.path.exists(document_load_path):
            raise FileNotFoundError(f"文档文件 {document_load_path} 不存在。")
        self.document_df = pd.read_csv(document_load_path, engine = 'python', sep = '\t', on_bad_lines = 'skip', encoding_errors = 'ignore', header = 0)

    def build_index_from_csv_root(self, csv_root: str, index_save_path: str):
        """
        待完善。。。

        """

        print("="*40)
        print('Loading file paths')
        print("="*20)
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(csv_root):
            for filename in filenames:
                # 拼接完整路径（绝对路径）
                file_path = os.path.abspath(os.path.join(dirpath, filename))
                file_paths.append(file_path)

        csv_paths = file_paths
        return file_paths

    def word2embedding(self, words: List[str]):
        """将单词列表转换为嵌入向量"""
        if isinstance(words, str):
            words = [words]

        return self.BGE_embedder.encode(words)
    
    def id2text(self, id: int) -> str:
        """将ID列表转换为文本"""
        try:
            forward_string = '[Title]:' + self.document_df.loc[id, 'title'] 
        except:
            forward_string =  '[Title]:' + 'No title'

        try:
            back_string = '[Context]:' + self.document_df.loc[id, 'text'] 
        except:
            back_string =  '[Title]:' + 'No Context'

        return  forward_string + back_string
    
    def topk_search(self, query: str, k: int = 5) -> List[dict]:
        """搜索最相似的topk文档"""
        # 将查询转换为嵌入向量
        query_embedding = self.word2embedding(query)
        
        # 使用Faiss进行搜索
        # 注意，faiss检索返回的是二维数组，其中第一维表示查询数量，第二维表示结果数量, (batch_size, k)
        scores, indexes = self.indexer.search(query_embedding, k)
        
        # 获取搜索结果
        results = []
        for i in range(k):
            result = {
                "score": scores[0][i],
                "IDs": indexes[0][i],
                "text": self.id2text(indexes[0][i])
            }
            
            results.append(result["text"])
        

        return results

class QueryExpander(QueryPreprocessor):
    def __init__(
        self, 
        model, 
        verbose: bool = False,
        strategy_params: dict = None, 
    ):
        """
        初始化QueryExpander类，用于生成多个扩展查询变体。
        
        Args:
            model: 用于扩展的LLM模型（如Llama3）
            verbose (bool): 是否输出详细日志
        """
        self.model = model
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        if strategy_params is None:
            self.strategy_params = {
                'max_tokens' : 600,
                'temperature': 0.8,
                'do_sample' : True,
                'num_return_sequences' : 1,  # 每个输入生成一个结果
            }
        else:
            self.strategy_params = strategy_params

    def expander_input_prompt(self, query: str, generated_queries_count: int = 3) -> str:
        """
        Generate a structured query expansion prompt that instructs the model to produce a specified number of expanded questions, ending with the <END> tag.

        Args:
            query (str): The original input query.
            question_nums (int): The number of expanded questions to generate.

        Returns:
            str: A complete, structured prompt for query expansion.
        """
        prompt = f"""
Generate {generated_queries_count} related sub-questions that expand the original query's scope while maintaining thematic coherence. Explore different dimensions.
The queries must preserve the original intent but vary in structure, formality, or phrasing.  
Strictly follow the format below: each query must start with the "<Query>" tag and be numbered from 1 to {generated_queries_count}.  
End the output with the "<END>" tag immediately after the last topic.

Format Example:
Input Query: "What are the implications of AI in healthcare?"
Output Query: 
1. <Query> What technical challenges arise when implementing AI diagnostic systems in hospitals?
2. <Query> How might AI-driven healthcare decisions impact patient autonomy and ethical standards?
3. <Query> What are the economic costs and benefits of adopting AI-based medical technologies?
<END>

Rules:
1. Each query must begin with "<Query>" and be numbered sequentially.
2. Avoid repeating the same phrasing or structure across queries.
3. Maintain the original intent and context of the input query.
4. If the query relates to a specialized domain, ensure the generated questions cover different subdomains or application scenarios.

Input Query: {query}

"""
        return prompt

    def extract_queries(self, output_str: str, nums: int) -> List[str]:
        """
        从LLM输出中提取所有符合格式的<Query>扩展查询，返回列表。
        
        Args:
            output_str (str): LLM的输出文本。
            nums (int): 预期生成的查询数量。
            
        Returns:
            List[str]: 提取的查询列表（去除标签和编号）。
        """
        # 正则表达式匹配符合规则的行：数字. <Query>开头，提取内容部分
        pattern = r'^(?:\s*\d+\.?\s*)?<Query>(.*)'  # 匹配如 "1. <Query>..." 或 "<Query>..."
        matches = re.findall(pattern, output_str, flags=re.MULTILINE)
        return [match.strip() for match in matches[:nums]]  # 截取前nums个结果

    def generate(self, input_query: Union[str, List[str]], 
                 generated_queries_count: int = 3, 
                 rewrite_type: str = 'Query') -> Union[Dict, List[Dict]]:
        """
        处理查询扩展流程，支持单个或批量输入。
        
        Args:
            input_query (str | List[str]): 需要扩展的原始查询或查询列表。
            question_nums (int): 期望生成的扩展查询数量。
            rewrite_type (str): 扩展类型（当前仅支持 'Query'）。
            
        Returns:
            dict | List[dict]: 单个查询返回结果字典，多个查询返回结果列表。
        """
        # 检查输入类型
        is_batch = isinstance(input_query, list)
        if not isinstance(input_query, (str, list)):
            raise TypeError("input_query must be a string or a list of strings")
        
        if rewrite_type not in ['Query']:
            raise ValueError("rewrite_type must be 'Query'")

        results = []
        queries = input_query if is_batch else [input_query]

        for query in queries:
            try:
                original_query = query.strip()
                prompt = self.expander_input_prompt(original_query, generated_queries_count)  # 构建提示

                # 调用模型生成
                generated_text = self.model.generate(prompt, **self.strategy_params)
                expanded_queries = self.extract_queries(generated_text, generated_queries_count)

            except Exception as e:
                self.logger.error(f"Expansion failed for '{original_query}': {str(e)}")
                expanded_queries = [original_query]  # 默认返回原始查询

            # 收集结果
            result = {
                "Processed Queries": expanded_queries,
                "original_query": original_query,
                "prompt_used": prompt,
                "success": len(expanded_queries) >= 1
            }
            results.append(result)

        return results if is_batch else results[0]
    
class QueryReflecter(QueryPreprocessor):
    def __init__(self, model):
        self.model = model

    def process(self, query: str) -> dict:
        # 生成多个扩展查询（如知识库[2]的扩展策略）
        expanded_queries = self.model.generate(query, prompt="Expand this query into 3 variants:")
        return {"queries": expanded_queries}

    def extract_reflect_information(self, output_str: str) -> List[str]:
            """
            从LLM输出中提取所有符合格式的<Question>问题，返回列表。

            Args:
                output_str (str): LLM的输出文本。

            Returns:
                List[str]: 提取的问题列表（去除标签和编号）。
            """
            # 直到<END>tag后结束
            output_str = output_str.split('<END>')[0]
            # 正则表达式匹配符合规则的行：数字. <Question>开头，提取内容部分
            pattern = r'^(?:\s*\d+\.\s*)?<Search>(.*?)$'
            matches = re.findall(pattern, output_str, flags=re.MULTILINE)
            return matches
    
    def generate(self, input_query: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        反思流程，从反思的标签中取出需要搜寻的信息，该方法为进一步深化的版本
        
        Args:
            input_query (str | List[str]): 需要改写的查询或查询列表。
        
        Returns:
            dict | List[dict]: 单个查询返回结果字典，多个查询返回结果列表。 
        """
        # 检查输入类型
        is_batch = isinstance(input_query, list)
        if not isinstance(input_query, (str, list)):
            raise TypeError("input_query must be a string or a list of strings")
            
        # 处理单个或批量输入
        results = []
        queries = input_query if is_batch else [input_query]
        
        for query in queries:
            try:
                original_query = query.strip()
        
            except Exception as e:
                self.logger.error(f"Rewrite failed for '{original_query}': {str(e)}")
                rewritten_query = original_query  # 保留原始查询
                query_type = "unknown"  # 标记为未知类型
                prompt = "error"  # 标记错误
                strategy_params = {}  # 清空策略参数
            
            # 收集结果
            result = {
                "": 123, # 占位符
                "type": query_type,
                "original_query": original_query,
                "prompt_used": prompt,
                "strategy": strategy_params,
                "success": rewritten_query != original_query
            }
            results.append(result)
        
        # 根据输入类型返回单个或列表结果
        return results if is_batch else results[0]


class QueryRouter(QueryPreprocessor):
    def __init__(self, retrievers: dict):
        self.retrievers = retrievers

    def process(self, query: str) -> dict:
        # 根据查询类型路由到不同检索器（如路由到向量检索或BM25）
        query_type = self._classify_query(query)  # 自定义分类逻辑
        return {"retriever": self.retrievers[query_type]}

class DocumentRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list:
        pass

class VectorDBRetriever(DocumentRetriever):
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k=5) -> list:
        # 使用向量数据库检索（如FAISS）
        return self.vector_db.search(query, top_k)

class BM25Retriever(DocumentRetriever):
    def __init__(self, index):
        self.index = index

    def retrieve(self, query: str, top_k=5) -> list:
        # 使用BM25检索（如知识库[2]的稀疏检索）
        return self.index.search(query, top_k)

class DocumentReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, docs: list) -> list:
        # 使用交叉编码模型重排文档（如知识库[3]的排序策略）
        scores = self.model.score(query, docs)
        return sorted(docs, key=lambda x: -scores[x])

class DocumentSplitter:
    def __init__(self, model):
        self.model = model

    def split(self, doc: str) -> list:
        # 使用LLM或BERT分块（如知识库[3]的NSP策略）
        return self.model.split_into_segments(doc)

class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, context: str) -> str:
        # 使用LLM生成答案（如知识库[1]的Reader模块）
        prompt = f"Answer the question based on the context:\nQuestion: {query}\nContext: {context}"
        return self.llm.generate(prompt)

class Retriever_Augmented_Generation:
    """
    所有RAG的Pipeline都在这里面了

    包括：
        QueryRewrite流程

    """
    def __init__(self, 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 LLM_model_local_dir: str = './Llama3_8B/',
                 index_load_path: str = './wikipedia_BGE_L2.contriever',
                 document_load_path: str = './psgs_w100.tsv',
                 ):
        print('=' * 40)
        print('Loading Large Language Model...')
        self.model = Large_Language_Model(local_dir = LLM_model_local_dir, 
                                          device = device, 
                                          )
        
        print('=' * 20)
        print('Loading Index...')
        self.indexer = IndexBuilder(
            device = device,
            index_load_path = index_load_path,
            document_load_path = document_load_path
        )

        print('=' * 20)
        print('Finished loading!')


    def query_rewrite(self, prompt: str, question_nums: int = 2, k: int = 2, **kwargs):
        """
        对用户查询进行重写，以适应模型的输入格式。
        注意：如果希望采用RRR，则需传入RRR所使用的模型，pipeline保持不变
        """

        print('=' * 40)
        print('Generating Rewrite Questions...')
        # 调用重写模型，获取重写后的问题
        questions = self.rewriter.generate(prompt, question_nums = question_nums)

        print('=' * 20)
        print('Retrieving Documents...')
        documents = []
        # 先用原来的问题进行检索
        document = self.indexer.topk_search(prompt, k = k)

        # 遍历重写后的问题，进行检索
        for question in questions:
            document = self.indexer.topk_search(question, k = k)
            documents.extend(document)

        print('=' * 20)
        print('Start Building Prompt...')
        # 构建输入提示
        input_prompt = self.cat_prompt_and_document(prompt, documents)

        print('=' * 20)
        print('Generating Outputs with LLMs...')
        outputs = self.model.generate(input_prompt, max_tokens = 600)

        print('=' * 20)
        print(f'Input: {prompt}')
        print(f'Output: {outputs}')
        return outputs


    def LLM_Only_Generate(self, 
                user_input: str,
                max_tokens: int = 600,
                if_print: bool = False, 
                **kwargs):
            """
            Naive RAG 流程（无查询重写）：
            1. 直接检索文档
            2. 构建提示
            3. 生成答案
            
            Args:
                query (str): 原始查询
                k (int): 检索文档数量
                max_tokens (int): 生成答案最大长度
                
            Returns:
                str: 模型生成的答案
            """
            
            if type(user_input) == str:
                user_input = [user_input]
            
            if if_print:
                print('=' * 40)
                print('Generating Outputs with LLMs...')

            # [复用] 使用已有生成模型
            outputs = self.model.generate(user_input, max_tokens=max_tokens)
            
            if if_print:
                print('=' * 20)
                print(f'Input: {user_input}')
                print(f'Output: {outputs}')

            self.model.clear_history()

            return outputs



class AERR:

    """
    参数化版本的 AERR 类，支持灵活配置。
    """

    def __init__(self, config: AERRConfig = None, tokenizer: "AutoTokenizer" = None, model :Large_Language_Model = None, *args, **kwargs):
        if config is None:
            train_config = MyTrainConfig()
            config = train_config.to_AERRConfig()
        
        
        # 初始化生成模型
        if not config.init_decision_model or not config.test:
            self.decision_agent = Decision_Agent(model = model, 
                                                 tokenizer = tokenizer, 
                                                 config = config.decision)
        
        # 初始化执行模型
        if not config.init_execution_model or not config.test:
            self.excution_agent = Execution_Agent(config.execution)

        # 初始化生成模型
        if not config.init_generate_model or not config.test:
            self.generate_agent = Generate_Agent(config.generative)

    def generate(self, user_input: str, need_print: bool = False):
        """
        获取用户输入后，返回在AERR流程下的结果
        该过程包含训练数据的输出————输出了流程中所生成的内部模型对话
        
        """
        if need_print:
            print("=" * 20)
            print(f'Get User Input: {user_input}')

        query_list = [user_input]
        document_list = []
        done = False

        if need_print:
            print("=" * 20)
            print(f'Initialize Pipeline: {user_input}')
            print("Start Interaction")

        while not done:
            decision_prompt = self.decision_agent.generate(query_list = query_list, 
                                                           document_list = document_list, 
                                                           need_print = need_print)
            
            # 目前仅针对单一的输入，不支持batch，后续实现batch方法
            query_list, document_list, done = self.excution_agent.take_action(
                                                                decision_prompt, 
                                                                query_list, 
                                                                document_list, 
                                                                need_print = need_print)

            # 打印交互过程
            if need_print:
                print("=" * 20)
                query_prompt = "\n".join([f"<Query {i}> {doc}" for i, doc in enumerate(query_list)])
                documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(document_list)])
                print("=" * 20, "\n", "Current Query List: ", query_prompt)
                print("=" * 20, "\n", "Current Document List: ", documents_prompt if not documents_prompt == "" else "Nothing Yet...")
        
        final_summary_detail = self.decision_agent.last_summary(query_list, document_list)
        final_input_prompt = self.build_LLM_input_prompt(query_list = query_list, 
                                                          document_list = document_list, 
                                                          detail = final_summary_detail)
        
        if need_print:
            print("=" * 20)
            print("Final Input Prompt: ")
            print(final_input_prompt)

        output = self.generate_agent.generate(input_text = final_input_prompt)
        
        if need_print:
            print("=" * 20)
            print("LLM Black Box Output: ")
        
        
        return output
    
    def generate_batch(self, 
                        input_prompts: Union[List[str], str], 
                        need_print: bool = False, 
                        tree : ConversationTree = None, 
                        max_tree_length: int = -1, 
                        model: "CasualLMWithValueHead" = None, 
                        tokenizer: "AutoTokenizer" = None, 
                        **kwargs):      
        """
        获取用户输入后，返回在AERR流程下的结果
        该过程包含训练数据的输出————输出了流程中所生成的内部模型对话
        
        """
        args = {}
        # 初始化参数

        if isinstance(input_prompts, str):
            input_prompts = [input_prompts]

        if type(input_prompts) == str:
            args["is_batch"] = False
            args["batch_size"] = 1

        else:
            args["is_batch"] = True
            args["batch_size"] = len(input_prompts)


        if need_print:
            print("=" * 20)
            print(f'Get User Input: {input_prompts}')

        done_flags = [False] * len(input_prompts)
        time_lis = [0] * len(input_prompts)

        query_lists = [copy.deepcopy([input_prompts[i]]) for i in range(len(input_prompts))]
        document_lists = [copy.deepcopy([]) for i in range(len(input_prompts))]
        decision_prompts = []
        final_summary_detail = []

        if tree is None:
            tree = ConversationTree()
            
        while not all(done_flags) and len(tree.layers) < max_tree_length:
            # 该条目返回列表
            model_input_prompts, decision_prompts, time_lis = self.decision_agent.generate_batch(query_lists = query_lists, 
                                                                  document_lists = document_lists, 
                                                                  need_print = need_print, 
                                                                  return_time = True, 
                                                                  return_input_prompts = True, 
                                                                  model = model, 
                                                                  tokenizer = tokenizer)
            
            tree.add_layer(sampling_prompts = [[model_input_prompt] for model_input_prompt in model_input_prompts], 
                           decision_model_responses = [[decision_prompt] for decision_prompt in decision_prompts], 
                           systems = self.decision_agent._init_prompt(), 
                           time_lis = time_lis)

            query_lists, document_lists, done_flags, time_lis = self.excution_agent.take_action_batch(
                                                                    decision_prompts = decision_prompts, 
                                                                    query_lists = query_lists, 
                                                                    document_lists = document_lists, 
                                                                    time_lis = time_lis, 
                                                                    done_flags = done_flags, 
                                                                    need_print = need_print)

        final_summary_details, model_input_prompts, model_output_prompts, time_lis  = self.decision_agent.last_summary_batch(query_lists = query_lists, document_lists = document_lists)
        
        tree.add_layer(sampling_prompts = model_input_prompts, 
                       decision_model_responses = model_output_prompts, 
                       time_lis = time_lis)
        
        final_input_prompts = [self.build_LLM_input_prompt(query_list = query_lists[i], 
                                                          document_list = document_lists[i], 
                                                          detail = final_summary_details[i]) for i in range(len(query_lists))]
        
        output = self.generate_agent.generate(input_text = final_input_prompts)
        
        return tree, output
    
    def build_LLM_input_prompt(self, query_list, document_list, detail, **kwargs):
        """
        构建最终的LLM输入结果，完成整个收尾工作
        
        """
        user_input = query_list[0]

        documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(document_list)]) if len(document_list) > 0 else "Nothing yet."
        detail_prompt = f"More information:\n{detail}" if detail else "Nothing yet."
        prompt = f"""
Please answer the user's input: {user_input}

These are the documents you can reference:
{documents_prompt}

Details: 
{detail_prompt}

Please answer the user's input: {user_input}

"""

        return prompt
    
    def get_decision_model_history(self):
        return self.decision_agent.get_history()

    def get_generate_model_history(self):
        return self.generate_agent.get_history()
    
    def get_excution_model_history(self):
        return self.excution_agent.get_history()

    def get_history(self):
        """
        该方法为训练接口，目前仅训练决策模型

        """

        return self.get_decision_model_history()
    
    def clear_history(self):
        """
        清空历史记录
        """
        self.decision_agent.clear_history()
        self.generate_agent.clear_history()
        self.excution_agent.clear_history()
        
        return 
    
    def reload(self, pipelineconfig: "DecisionConfig"):
        """
        
        """
        if not hasattr(pipelineconfig, "lora_dir"):
            print(f"不依赖于Lora加载模型: {pipelineconfig.model_dir}")

            # 仅针对decision_model 进行重新加载
            self.decision_agent.reload(pipelineconfig)
        else:
            print(f"Base模型路径: {pipelineconfig.model_dir}")
            print(f"Lora-Adaptor路径: {pipelineconfig.lora_dir}")
            self._reload_lora(pipelineconfig)
            
        return self
    
    def _reload_lora(self,  config: PipelineConfig, **kwargs) -> None:
        """重新加载需要lora装载的模型"""
        self.decision_agent.reload_lora(config)

    def save_model(self, dir):
        """保存决策模型"""
        self.decision_agent.save_model(dir)

    def release(self):
        """
        释放模型
        
        """
        # 仅针对decision_model 进行重新加载
        self.decision_agent.release()

        return 
    
    def sampling(self, 
                user_input: str, 
                need_print: bool = False, 
                sampling_nums: int = 32, 
                sampling_decay: int = 16, 
                test : bool = False, 
                tree : ConversationTree = None, 
                max_tree_length: int = -1, 
                model: "CasualLMWithValueHead" = None, 
                tokenizer: "AutoTokenizer" = None, 
                **kwargs):
        """
        AERR管线的采样，引入逆指数分配原则，每次都会缩小
        
        """

        if tree is None:
            tree = ConversationTree()

        if need_print:
            print("=" * 20)
            print(f'Get User Input: {user_input}')

        if need_print:
            print("=" * 20)
            print(f'Initialize Pipeline: {user_input}')
            print("Start Interaction")

        # 初始化
        query_lists = [[user_input]]
        document_lists = [[]]
        done_flags: list[bool] = [False]
        current_sampling_nums = sampling_nums
        add_first_node: bool = False

        if need_print:
            print("=" * 20)
            print(f'Get User Input: {user_input}')

        while (not all(done_flags)) and len(tree.layers) < max_tree_length:
            
            # 每一次要记得初始化time_lis，仅采样当前的时间，最后将整条动作链的时间相加
            time_lis = []

            # input_prompts, sampling_prompts, decision_model_responses等均为List[Tuple]格式
            # done_flags, times 均为List[bool/float]格式
            # 因为time_lis的特殊性，其只能在decision_agent上完成跟踪，execution_agent无法扩大，不然就成了双重扩大了
            input_prompts, sampling_prompts, decision_model_responses, time_lis = self.decision_agent.sampling(
                                                                        query_lists = query_lists, 
                                                                        document_lists = document_lists, 
                                                                        sampling_num = current_sampling_nums, 
                                                                        model = model, 
                                                                        tokenizer = tokenizer, 
                                                                        **kwargs
                                                                        )
            if not add_first_node:
                tree.add_layer(
                    sampling_prompts = [input_prompts], 
                    decision_model_responses = [[""]],
                    time_lis = [0]
                )
                add_first_node = True

            query_lists, document_lists, done_flags, time_lis = self.excution_agent.sampling(
                                        decision_model_responses = decision_model_responses, 
                                        query_lists = query_lists, 
                                        document_lists = document_lists, 
                                        done_flags = done_flags, 
                                        time_lis = time_lis, 
                                        need_print = need_print, 
                                        test = test)
            
            tree.add_layer(sampling_prompts = sampling_prompts, 
                           decision_model_responses = decision_model_responses, 
                           time_lis = time_lis)
            
            current_sampling_nums = int(np.ceil(current_sampling_nums / sampling_decay))

        query_lists = [copy.deepcopy(item) for item in query_lists for _ in range(current_sampling_nums)]
        document_lists = [copy.deepcopy(item) for item in document_lists for _ in range(current_sampling_nums)]

        # 采样最后一次模型的通信，decision_model发送给最终大模型的通信
        # 因为detail_strings没有得到正确的扩大么？还是没有得到正确的采样？
        sampling_inputs, model_responses, detail_strings, time_lis = self.decision_agent.last_summary_sampling(
                                                                      query_lists = query_lists, 
                                                                      document_lists = document_lists, 
                                                                      return_time = True, 
                                                                      sampling_nums = current_sampling_nums, 
                                                                      **kwargs)
        
        tree.add_layer(sampling_prompts = sampling_inputs, 
                       decision_model_responses = model_responses, 
                       time_lis = time_lis)

        final_input_prompts = [self.build_LLM_input_prompt(query_list = query_lists[i], 
                                                          document_list = document_lists[i], 
                                                          detail = detail_strings[i]
                                                          ) for i in range(len(query_lists))]

        # 最后一次输出
        output: list[str] = self.generate_agent.generate(input_text = final_input_prompts, return_time = False)
        
        return tree, output
    
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

    def query_rewrite_input_prompt(self, query: str, nums: int = 3, detail: str = None) -> str:
        """
        ========================================该方法有效，参见PPT==========================================================
        生成结构化查询改写提示，要求模型生成指定数量的改写问题，并在最后添加 <END> 标签。
        区别Question_Rewrite, 该函数进行对应主题改写，省去了如How、What等提示词，可以减少无用字符
        
        Args:
            query (str): 原始查询。
            question_nums (int): 需要生成的改写问题数量。
            
        Returns:
            str: 完整的提示模板。
        """

        detail_prompt = f"More information:\n{detail}" if detail else ""
        prompt = f"""
Please generate **{nums}** related topics based on the input query. 
Ensure the generated topic is **clearer**, **more accurate**, and **more comprehensive** than the original input query.
The topics should expand the original scope while maintaining relevance. 
Follow the format strictly and ensure each topic starts with "<Query>". 
End the output with the "<END>" tag immediately after the last topic.
**You can put your thoughts in <Think> tag, but make sure to generate enough queries in <Query> tags.** 

Format Example:
Input Query: "What are the implications of AI in healthcare?"
Expected Output: 
<Think> (Optional) The original query asks about AI's implications in healthcare, which is broad. To refine this, I need to identify specific aspects like ethical frameworks, data requirements, and implementation challenges. Ethical guidelines address accountability and bias, clinical datasets determine model effectiveness, and cost assessments vary by region. Expanding these into structured topics ensures comprehensive coverage. 
<Query> What are the current ethical guidelines governing AI diagnostic systems, particularly regarding data privacy and algorithmic bias?
<Query> What types of clinical datasets (e.g., imaging, EHRs) are essential for training AI models in cancer detection, and how do they ensure diversity?
<Query> How do rural healthcare providers assess AI implementation costs compared to urban counterparts, considering infrastructure and policy differences?
<END>

Rules:
1. Each topic must start with the "<Query>" tag.
2. Put your think context in "<Think>" tag.
3. End the output with the "<END>" tag immediately after the last topic.
4. Avoid vague or overly broad topics.
5. Ensure topics cover different angles.

{detail_prompt}

Input Query: {query}
""".strip()

        return prompt

    def query_reason_input_prompt(self, query: str, nums: int = 3, detail = None) -> str:
        """
        生成回答用户问题所需的必要信息点，作为细化后的查询请求。
        每个查询应明确指向问题的核心要素或关键子问题。

        Args:
            query (str): 用户的原始问题或任务描述。
            nums (int): 需要生成的细化查询数量。

        Returns:
            str: 完整的提示模板。
        """
        detail_prompt = f"More information:\n{detail}" if detail else ""
        prompt = f"""
Please generate **{nums}** key information points required to answer the input query. 
What's the core question or problem and what's the core components of the problem? What should you retrieve to know the information of the problem?
Keep the question above in mind and generate **{nums}** key information points. 
Each generated query should be **specific**, **reasonable**, and **directly relevant** to the core components of the problem.
Follow the format strictly and ensure each query starts with "<Query>". 
End the output with the "<END>" tag immediately after the last query.
**You can put your thoughts in <Think> tag, but make sure to generate enough queries in <Query> tags.** 

Format Example:
Input Query: "What are the implications of AI in healthcare?"
Expected Output: 
<Think> (Optional) The input query asks about the implications of AI in healthcare. To address this comprehensively, I need to break it down into specific aspects. First, ethical considerations are crucial, so identifying current guidelines for AI diagnostics makes sense. Second, the effectiveness of AI models depends on the quality of clinical data, so understanding the required datasets for cancer detection is relevant. Lastly, implementation costs vary by region, especially in rural areas, so exploring how providers there assess costs provides practical insight. 
<Query> What are the current ethical guidelines governing AI diagnostic systems?
<Query> What clinical datasets are required to train AI models for cancer detection?
<Query> How do rural healthcare providers assess AI implementation costs?
<END>

Rules:
1. Each topic must start with the "<Query>" tag.
2. Put your think context in "<Think>" tag.
3. End the output with the "<END>" tag immediately after the last topic.
4. Avoid vague or overly broad topics.
5. Ensure topics cover different angles.

{detail_prompt}

Input Query: {query}


""".strip()

        return prompt

    def query_extract_input_prompt(self, query_list: List[str], document_list: List[str], nums: int = 3, detail: str = None) -> str:
        """
        通过给定的query，提问和文章，提取关键的信息，总结得到新的Query

        Args:
            problem (str): 需要解决的问题或任务。
            method (str): 推理方法（支持 "CoT" 或 "ToT"）。
            steps (int): 需要生成的推理步骤数或路径数。

        Returns:
            str: 完整的提示模板。
        """
        query_prompt = "\n".join([f"<Query {i}> {doc}" for i, doc in enumerate(query_list)])
        documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(document_list)])
        detail_prompt = f"More information:\n{detail}" if detail else ""

        prompt = f"""
Based on the input query and the provided documents, generate **{nums}** new, valuable queries that can further explore the problem or uncover hidden aspects.
Each generated query must be **specific**, **actionable**, and **directly tied to the core components** of the problem.

Format Example:
Input Query: "How does climate change affect agriculture?"
<Document 0> [Title] Global and Regional Impacts on Crop Yields [Context] Globally, climate change has reduced wheat and corn yields by 1.9% and 1.2% per decade, respectively. In China, while warmer temperatures in Northeast China have extended growing seasons and improved heat resources, other regions face challenges. For instance, reduced rainfall in North China and Southwest China threatens water availability for crops. By 2030, China’s grain production could decline by 5–10% due to warming, with significant impacts on major crops like rice, wheat, and corn.
Expected Output:
<Think> (Optional) The input query focuses on climate change impacts on agriculture. The documents highlight regional differences (e.g., Northeast China's extended growing seasons vs. North/Southwest China's water shortages) and specific crop yield declines (wheat -1.9%, corn -1.2%). To generate actionable queries, I should focus on: 1) regional mitigation strategies for water-stressed areas, 2) quantifiable impacts on key crops, and 3) adaptation measures for temperature-sensitive regions.
<Query> How does reduced rainfall in North China specifically impact water availability for rice cultivation by 2030?
<Query> What are the projected yield losses for wheat in China by 2030, and what adaptation measures are being implemented?
<Query> How do extended growing seasons in Northeast China affect soybean planting practices and pest management?
<END>

Rules:
1. Each query must start with "<Query>" and end with a question mark.
2. Include your reasoning in "<Think>" tags if needed.
3. End output with "<END>" immediately after the last query.
4. Avoid redundant or overly broad topics.
5. Ensure queries are **derived from both the input query and documents**.

Input Query: 
{query_prompt}
Documents:
{documents_prompt}

{detail_prompt}

""".strip()

        return prompt

    def document_filter_prompt(self, 
                               user_input: str, 
                               document_list: str, 
                               detail: str = None):
        """
        阅读文档，查阅这些文档是否与问题强相关、强支持等，当遇到不相关的文档，指出并筛选掉这一部分

        Args:
            problem (str): 需要解决的问题或任务。
            method (str): 推理方法（支持 "CoT" 或 "ToT"）。
            steps (int): 需要生成的推理步骤数或路径数。

        Returns:
            str: 完整的提示模板。
        """
        documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(document_list)])
        detail_prompt = f"More information:\n{detail}" if detail else ""

        prompt = f"""
Identify and filter out irrelevant documents from the provided list based on the input query.
For each document, determine if it **directly supports** the user's input or introduces unrelated information.
Mark irrelevant documents explicitly and explain your reasoning in <Think> tags.

**Response Format Specification**
- <Think> (Optional) Make sure you have enough contemplation before taking any action, and put them in this tag. 
- <Irrelevant Document> Target Document ID
  - Write your document ID when you think it's irrelevant to user input.
  - Start with <Irrelevant Document> tag. 
- <END> Use this tag to end the output immediately.

**Example**:
Input:
User Input: "How does climate change affect agriculture?"
<Document 0> [Title] Global and Regional Impacts on Crop Yields [Context] Globally, climate change has reduced wheat and corn yields by 1.9% and 1.2% per decade, respectively. In China, while warmer temperatures in Northeast China have extended growing seasons and improved heat resources, other regions face challenges. For instance, reduced rainfall in North China and Southwest China threatens water availability for crops. By 2030, China’s grain production could decline by 5–10% due to warming, with significant impacts on major crops like rice, wheat, and corn.
<Document 1> [Title] Water Resource Stress and Agricultural Conflicts [Context] Climate change has disrupted water availability. While southern China has seen stable rainfall, northern regions face declining water resources, particularly in the Liaohe and Yellow River basins. This imbalance strains agriculture, as crops in arid regions struggle with droughts, while floods in low-lying areas cause losses. By 2050, the northward shift of three-crop zones may further strain water infrastructure and distribution systems.
<Document 2> [Title] Risk Regulation in Banking: Natural Language Analysis of Compliance Indicators [Context] Banks are increasingly adopting AI-driven tools to monitor risk indicators and ensure regulatory compliance. A case study from a major financial institution highlights the use of a knowledge-integrated system to analyze risk metrics and associated documentation. Traditional BI systems rely on button-based queries for static data, but this bank extended its capabilities by incorporating natural language processing (NLP) to handle dynamic regulatory updates. ...
<Document 3> [Title] Knowledge Graphs in Healthcare: Enhancing Clinical Decision-Making [Context] In modern healthcare systems, knowledge graphs are being leveraged to improve clinical decision-making. For example, a hospital implemented a unified semantic layer and knowledge graph to integrate structured data (e.g., patient records, prescriptions) and unstructured text (e.g., electronic medical records). By defining standardized concepts like "fever" (37°C–40°C) and linking patient symptoms to treatment outcomes, the system enables doctors to quickly retrieve historical cases with similar conditions...

**Expected Output**:
<Think> First Document is directly relevant. It provides specific data on crop yield declines (wheat -1.9%, corn -1.2%) and regional impacts in China, which directly address the query about climate change effects on agriculture. Secend Document is relevant but supplementary. It discusses water resource stress in agricultural regions (e.g., Liaohe and Yellow River basins) and its impact on crop viability, which expands the understanding of indirect climate change effects on agriculture. Document 2 is unrelated to agricultural impacts of climate change. It discusses banking risk regulation instead.
<Irrelevant Document> 2
<Irrelevant Document> 3
<END>

Current State:
Input Query: {user_input}
Documents:
{documents_prompt}

{detail_prompt}

""".strip()

        return prompt

    def parse_query_output(self, output: str, nums: int) -> List[str]:
        """
        解析模型输出，提取所有以 <Query> 开头的改写问题。
        
        Args:
            output (str): 模型生成的输出字符串，包含多个 <Query> 标签和最终的 <END> 标签。
            
        Returns:
            List[str]: 提取并清理后的改写问题列表。
        """
        result = []
        lines = output.strip().split('\n')  # 去除首尾空白并按行分割
        
        for line in lines:
            if line.startswith('<Query>'):
                # 提取 <Query> 标签后的内容并去除前后空白
                query = line[len('<Query>'):].strip()
                if query:  # 确保内容非空
                    result.append(query)
            elif line == '<END>':
                break  # 遇到 <END> 标签则停止处理后续行
        
        if len(result) > nums:
            result = result[:nums]
            
        return result

    def parse_irrelevant_documents(self, model_output: str) -> List[str]:
        """
        从模型输出中提取不相关的文档索引，转换为字符串列表

        Args:
            model_output (str): 模型输出的原始字符串（包含 <Irrelevant Document> 标签）

        Returns:
            List[str]: 不相关文档的索引字符串列表，按数字大小排序
        """
        # 1. 提取所有 <Irrelevant Document> 标签中的数字
        pattern = r'<Irrelevant Document>\s*(\d+)'

        
        matches = re.findall(pattern, model_output)

        # 2. 去重并按数值排序
        unique_indices = list(set(matches))
        
        indices = []
        for i in range(len(unique_indices)):
            # 防止模型在中间加一些奇奇怪怪的东西
            try:
                indices.append(int(unique_indices[i]))
            except:
                pass

        return indices

    def extract_response_components(self, model_output: str) -> Dict[str, List[Any]]:
        """
        从模型输出中提取<Think>、<Action>和<Detail>组件，并解析Action内容
        
        Args:
            model_output (str): 模型返回的原始响应字符串
            
        Returns:
            Dict[str, List[Any]]: 包含提取组件的字典
        """
        data = {
            'raw_string': model_output,
            'thinks': [],
            'actions': [],
            'details': []
        }
        
        # 提取并处理<Think>标签
        data['thinks'] = [t.strip() for t in re.findall(r'<Think>(.*?)</Think>', model_output, re.DOTALL)]
        
        # 提取并处理<Action>标签
        action_pattern = r'<Action>(.*?)</Action>'
        # 这里返回的是所有符合要求的action_matches内容
        action_matches: list = re.findall(action_pattern, model_output, re.DOTALL)
        
        for action_str in action_matches:

            try:
                # 分割并清理字段
                parts = [p.strip() for p in action_str.split('|')]
                
                # 解析query_id、action_type、nums
                query_id_string = parts[0]
                # 这里针对query_id进行试探，如果不是数字，则返回-1
                action_type_string = parts[1]
                nums_string = parts[2]
            except:
                # 解析query_id、action_type、nums
                query_id_string = "Query 0"
                # 这里针对query_id进行试探，如果不是数字，则返回-1
                action_type_string = "Stop"
                nums_string = "Nums 0"


            try:
                target_id = int(query_id_string[-1])
            except:
                target_id = 0
            
            try:
                nums = int(nums_string[-1])
            except:
                nums = 0


            data['actions'].append({
                'target_id': target_id,
                'action_type': action_type_string,
                'nums': nums
            })

        # 提取并处理<Detail>标签
        detail_pattern = r'<Detail>(.*?)</Detail>'
        detail_matches = re.findall(detail_pattern, model_output, re.DOTALL)
        data['details'] = [d.strip() for d in detail_matches]
        
        return data
    
    def extract_response_components_batch(self, model_output: str) -> Dict[str, List[Any]]:
        """
        从模型输出中提取<Think>、<Action>和<Detail>组件，并解析Action内容
        
        Args:
            model_output (str): 模型返回的原始响应字符串
            
        Returns:
            Dict[str, List[Any]]: 包含提取组件的字典
        """
        data = {
            'raw_string': model_output,
            'thinks': [],
            'actions': [],
            'details': []
        }
        
        # 提取并处理<Think>标签
        data['thinks'] = [t.strip() for t in re.findall(r'<Think>(.*?)</Think>', model_output, re.DOTALL)]
        
        # 提取并处理<Action>标签
        action_pattern = r'<Action>(.*?)</Action>'
        action_matches = re.findall(action_pattern, model_output, re.DOTALL)
        
        for action_str in action_matches:
            try:
                # 分割并清理字段
                parts = [p.strip() for p in action_str.split('|')]
                
                # 检查字段数量是否足够
                if len(parts) < 3:
                    raise ValueError("Action 字段数量不足")
                
                query_id_string = parts[0]
                action_type_string = parts[1]
                nums_string = parts[2]

                # 解析 target_id
                try:
                    target_id = int(query_id_string[-1])
                except:
                    target_id = 0
                
                # 解析 nums
                try:
                    nums = int(nums_string[-1])
                except:
                    nums = 0
                
                # 正常添加 action
                data['actions'].append({
                    'target_id': target_id,
                    'action_type': action_type_string,
                    'nums': nums
                })
            
            except Exception as e:
                # 捕获所有异常，生成 ForceStop action
                data['actions'].append({
                    'target_id': 0,
                    'action_type': 'Stop',
                    'nums': 0
                })
        
        # 确保 actions 至少有一个 action
        if not data['actions']:
            data['actions'].append({
                'target_id': 0,
                'action_type': 'Stop',
                'nums': 0
            })

        # 提取并处理<Detail>标签
        detail_pattern = r'<Detail>(.*?)</Detail>'
        detail_matches = re.findall(detail_pattern, model_output, re.DOTALL)
        data['details'] = [d.strip() for d in detail_matches]
        
        return data
    
    def take_action(self, 
                    decision_prompt: str, 
                    query_list: list, 
                    document_list: list, 
                    need_print : bool = False, 
                    **kwargs) -> Union[tuple[List, List, bool]]:
        """
        该函数用以对decision_prompt进行解析，并执行对应的动作
        
        Args:
            queries (list): 问题列表
        """
        results = []
        done = False
        # 以<END>结束，先拆个包
        decision_prompt = decision_prompt.split('<END>')[0]

        # 解包需要采取的动作
        curr_decision = self.extract_response_components(decision_prompt)
        
        # 暂未使用到Think部分
        thinks, actions, details =  curr_decision['thinks'], curr_decision['actions'], curr_decision['details']
        
        # 指定user_input
        user_input = query_list[0]

        # 区分detail
        if len(details) == 0:
            detail = None
            is_seperate_detail = False
        elif len(details) == 1:
            detail = details[0]
            is_seperate_detail = False
        elif len(details) == len(actions):
            is_seperate_detail = True
        else:
            detail = details[0]
            is_seperate_detail = False

        for i, action in enumerate(actions):

            action_type = action['action_type']
            target_id = action['target_id']
            nums = action['nums']
            # 如果输出结果出现异常（比如都是字符串，没有其他的东西，那么默认为0）

            if need_print:
                print("=" * 20, "\n", "Current Action: ", '<', action_type, '|', target_id, '|', nums, '>')

            target_query = query_list[target_id]

            if is_seperate_detail:
                detail = details[i]
            
            if action_type == 'Query Rewrite':
                input_prompt = self.query_rewrite_input_prompt(target_query, 
                                                               nums = nums, 
                                                               detail = detail)
                response = self.model.generate(input_prompt, **self.strategy_params)
                queries = self.parse_query_output(response)
                query_list.extend(queries)

                if need_print:
                    print("=" * 10)
                    print("Execution Agent Input: ")
                    print(input_prompt)
                    print("=" * 10)
                    print("Execution Agent Output:")
                    print(response)
                
            elif action_type == 'Query Reason':
                input_prompt = self.query_reason_input_prompt(target_query, 
                                                               nums = nums, 
                                                               detail = detail)
                response = self.model.generate(input_prompt, **self.strategy_params)
                queries = self.parse_query_output(response)
                query_list.extend(queries)
                
                if need_print:
                    print("=" * 10)
                    print("Execution Agent Input: ")
                    print(input_prompt)
                    print("=" * 10)
                    print("Execution Agent Output:")
                    print(response)

            elif action_type == 'Query Extract':
                input_prompt = self.query_extract_input_prompt(query_list, 
                                                               document_list = document_list, 
                                                               nums = nums, 
                                                               detail = detail)
                response = self.model.generate(input_prompt, 
                                            **self.strategy_params
                                            )
                queries = self.parse_query_output(response)
                query_list.extend(queries)
                
                if need_print:
                    print("=" * 10)
                    print("Execution Agent Input: ")
                    print(input_prompt)
                    print("=" * 10)
                    print("Execution Agent Output:")
                    print(response)

            elif action_type == 'Document Filter':
                input_prompt = self.document_filter_prompt(user_input = user_input, 
                                                           document_list = document_list, 
                                                           detail = detail)
                response = self.model.generate(input_prompt, 
                                            **self.strategy_params
                                            )
                irrelevant_document_indices = self.parse_irrelevant_documents(response)

                # 过滤文档（按降序删除以避免索引错位）
                document_list = [doc for i, doc in enumerate(document_list) if i not in irrelevant_document_indices]
                
                if need_print:
                    print("=" * 10)
                    print("Execution Agent Input: ")
                    print(input_prompt)
                    print("=" * 10)
                    print("Execution Agent Output:")
                    print(response)

            elif action_type == 'Document Retrieve':
                new_documents_retrieve_results = self.indexer.topk_search(target_query, k = nums)
                new_documents = [result['text'] for result in new_documents_retrieve_results]
                document_list.extend(new_documents)

            elif action_type == 'Stop':
                done = True
                break

        return query_list, document_list, done

    def align_actions_details(
        self,
        actions: List[Dict],
        details: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """
        对齐 actions 和 details 的长度，确保二者等长。
        - 保持 actions 不变
        - 若 details 不足，用空字符串填充
        - 若 details 超出，直接截断

        Args:
            actions (List[Dict]): 提取到的 actions 列表（至少包含一个）
            details (List[str]): 提取到的 details 列表

        Returns:
            Tuple[List[Dict], List[str]]: 对齐后的 actions 和 details 列表
        """
        num_actions = len(actions)
        num_details = len(details)

        if num_details < num_actions:
            # 补充空字符串到 actions 长度
            padded_details = details + [''] * (num_actions - num_details)
        elif num_details > num_actions:
            # 截断到 actions 长度
            padded_details = details[:num_actions]
        else:
            padded_details = details

        return actions, padded_details

    def align_lists(self, 
        decision_prompts: List[str],
        query_lists: List[List[str]],
        document_lists: List[List[str]],
        done_flags: List[bool], 
        time_lis : list[list[float]], 
    ) -> tuple[List[str], List[List[str]], List[List[str]], List[bool]]:
        """
        对齐四个列表长度，若长度不一致则裁剪至最小长度，并发出警告。

        Args:
            decision_prompts (List[str]): 查询对应的 prompt 列表。
            query_lists (List[List[str]]): 查询内容列表（嵌套列表）。
            document_lists (List[List[str]]): 文档内容列表（嵌套列表）。
            done_flags (List[bool]): 交互链完成状态标识列表。

        Returns:
            tuple: 裁剪后的四个列表。
        """
        # 获取四个列表长度
        lens = [
            len(decision_prompts),
            len(query_lists),
            len(document_lists),
            len(done_flags), 
            len(time_lis), 
        ]

        min_len = min(lens)

        if all(length == min_len for length in lens):
            return decision_prompts, query_lists, document_lists, done_flags, time_lis

        # 构造警告信息
        warning_msg = "执行模型TakeActionBatch错误：列表长度不一致，已裁剪至最小长度 {min_len}。各列表原始长度为：\n"
        warning_msg += "- decision_prompts: {len_dp}\n"
        warning_msg += "- query_lists: {len_q}\n"
        warning_msg += "- document_lists: {len_d}\n"
        warning_msg += "- time_lists: {len_tl}\n"
        warning_msg += "- done_flags: {len_df}"

        warnings.warn(warning_msg.format(
            min_len=min_len,
            len_dp=lens[0],
            len_q=lens[1],
            len_d=lens[2],
            len_df=lens[3], 
            len_tl=lens[4], 
        ))

        # 裁剪所有列表
        return (
            decision_prompts[:min_len],
            query_lists[:min_len],
            document_lists[:min_len],
            done_flags[:min_len], 
            time_lis[:min_len]
        )

    def take_action_batch(self, 
                          decision_prompts: List[str], 
                          query_lists: List[List[str]], 
                          document_lists: List[List[str]], 
                          done_flags: list[bool], 
                          time_lis: list[float], 
                          need_print: bool = False, 
                          **kwargs) -> Tuple[List[List[str]], List[List[str]], List[bool], List[float]]:
        """
        批量执行决策指令，处理多个样本的查询和文档列表
        
        Args:
            decision_prompts (List[str]): 每个样本的决策指令
            query_lists (List[List[str]]): 每个样本的查询列表
            document_lists (List[List[str]]): 每个样本的文档列表
            need_print (bool): 是否打印调试信息
            test (bool): 测试模式（可选）
        
        Returns:
            Tuple[List[List[str]], List[List[str]], List[bool]]: 
                更新后的查询列表、文档列表、是否终止标志

        ##结束测试，测试通过##
        """
        if isinstance(decision_prompts, str):
            decision_prompts = [decision_prompts]
            isBatch = False
        else:
            isBatch = True
        
        if isinstance(query_lists[0], str):
            query_lists = [query_lists]

        if isinstance(document_lists[0], str):
            document_lists = [document_lists]

        # 裁剪至长度一致
        decision_prompts, query_lists, document_lists, done_flags, time_lis = self.align_lists(decision_prompts, 
                                                                                               query_lists, 
                                                                                               document_lists, 
                                                                                               done_flags, 
                                                                                               time_lis)
        
        batch_input_prompts = []  # 存储所有需要调用模型的 input_prompt
        prompt_metadatas = []      # 存储每个 input_prompt 对应的 (sample_index, action_index, action_type)

        # 批量提取决策组件
        decisions_batch = [self.extract_response_components(dp) for dp in decision_prompts]
        
        # 第一步：收集所有需要调用模型的 input_prompts 和元数据
        for decision_index, (decision, query_list, document_list, done) in enumerate(zip(decisions_batch, query_lists, document_lists, done_flags)):
            # 跟踪时间耗时
            before = time.time()
            # 当某一动作链执行完毕时，跳过动作
            if done:
                continue

            if need_print:
                print(f"\n=== Processing Sample {decision_index} ===")
            
            # 获取动作和细节
            actions = decision['actions']
            details = decision['details']
            
            # 对齐 actions 和 details
            aligned_actions, aligned_details = self.align_actions_details(actions, details)
            
            # 获取 user_input（每个样本的第一个查询）
            user_input = query_list[0]
            
            for action_index, (action, detail) in enumerate(zip(aligned_actions, aligned_details)):
                action_type = action['action_type']
                target_id = action['target_id']
                nums = action['nums']
                nums = nums if nums >= 0 else 0
                
                if need_print:
                    print("=" * 20, "\n", "Current Action: ", f"<{action_type}|{target_id}|**{nums}**>")
                
                # 确保 target_id 不越界
                if target_id >= len(query_list):
                    if need_print:
                        print(f"Warning: target_id {target_id} exceeds query_list length {len(query_list)}")
                    continue
                
                target_query = query_list[target_id]
                
                # 如果需要调用模型，则生成 input_prompt 并记录元数据
                if action_type in ['Query Rewrite', 'Query Reason', 'Query Extract', 'Document Filter']:
                    if action_type == 'Query Rewrite':
                        input_prompt = self.query_rewrite_input_prompt(target_query, nums=nums, detail=detail)
                    elif action_type == 'Query Reason':
                        input_prompt = self.query_reason_input_prompt(target_query, nums=nums, detail=detail)
                    elif action_type == 'Query Extract':
                        input_prompt = self.query_extract_input_prompt(
                            query_list=query_list,
                            document_list=document_list,
                            nums=nums,
                            detail=detail
                        )
                    elif action_type == 'Document Filter':
                        input_prompt = self.document_filter_prompt(
                            user_input=user_input,
                            document_list=document_list,
                            detail=detail
                        )
                    
                    # 记录 input_prompt 及其对应的样本索引、动作索引和动作类型
                    batch_input_prompts.append(input_prompt)
                    prompt_metadatas.append({
                        'sample_index': decision_index,
                        'action_index': action_index,
                        'action_type': action_type, 
                        "action_nums": nums, 
                    })

                    if need_print:
                        print("Execution Agent Input (collected):")
                        print(input_prompt)

                # 先提前处理Document Retrieve的情况
                if action_type == 'Document Retrieve':
                    # topk_search不容许为零的情况
                    # 如果遇到k = 0，那么此时跳过检索，这里需要额外加上容错
                    retrieve_results = self.indexer.topk_search(target_query, k=nums) if not nums == 0 else []
                    new_documents = [result for result in retrieve_results]
                    document_list.extend(new_documents)
                    
                if action_type == 'Stop':
                    done_flags[decision_index] = True
                    if need_print:
                        print(f"Sample {decision_index} terminated by 'Stop' action")
                    break

            # 继续跟踪耗时，追踪元数据处理的时候消耗的时间
            after = time.time()
            time_lis[decision_index] += after - before


        # 第二步：批量调用模型并获取所有响应
        if len(batch_input_prompts) != 0:
            all_responses, take_action_time_lis = self.model.generate(
                                                batch_input_prompts,
                                                return_time = True, 
                                                **self.strategy_params)
            
            # 第三步：分配响应并更新 query_list/document_list
            for i, (response, metadata) in enumerate(zip(all_responses, prompt_metadatas)):
                sample_index = metadata['sample_index']
                action_index = metadata['action_index']
                action_type = metadata['action_type']
                nums = metadata["action_nums"]
                
                query_list = query_lists[sample_index]
                document_list = document_lists[sample_index]

                # 为每个需要添加的都加上时间部分
                time_lis[sample_index] += take_action_time_lis[i]

                if need_print:
                    print(f"\n=== Processing Response for Sample {sample_index}, Action {action_index} ===")
                    print("Execution Agent Output:")
                    print(response)

                # 更新query_lists和document_lists
                if action_type in ['Query Rewrite', 'Query Reason', 'Query Extract']:
                    new_queries = self.parse_query_output(response, nums)
                    query_list.extend(new_queries)

                elif action_type == 'Document Filter':
                    irrelevant_indices = self.parse_irrelevant_documents(response)
                    # 降序删除避免索引错位
                    document_list[:] = [doc for i, doc in enumerate(document_list) if i not in irrelevant_indices]
                    document_lists[sample_index] = document_list

        if not isBatch:
            query_lists = query_lists[0]
            document_lists = document_lists[0]
            done_flags = done_flags[0]

        return query_lists, document_lists, done_flags, time_lis
    
    
    def sampling(self, 
                 decision_model_responses: list[tuple], 
                 query_lists: list[list[str]], 
                 document_lists: list[list[str]],
                 done_flags: list[bool], 
                 time_lis: list[float], 
                 need_print: bool = False, 
                 test: bool = False, 
                 **kwargs, 
                 ):
        """
        以 decision_model_responses 为决策依据，对多条交互链进行批量 take_action 操作。
        要求符合采样链的对应关系？
        功能：
        1. 标准化输入参数格式（如将字符串转为嵌套列表）；
        2. 根据采样数量扩展 query_lists、document_lists 和 done_flags；
        3. 调用 take_action_batch 方法执行批量操作；
        4. 返回更新后的交互链状态。

        参数说明：
            decision_model_responses (list[tuple]): 模型生成的决策响应列表，每个元素为一个交互链的多条响应（元组形式）。
            query_lists (list[list]): 当前交互链的查询列表，每个子列表对应一个交互链的查询内容。
            document_lists (list[list]): 当前交互链的文档列表，每个子列表对应一个交互链的文档上下文。
            done_flags (list[bool]): 交互链完成状态标识列表，True 表示该链已结束，False 表示继续执行。
            need_print (bool): 是否打印中间结果（调试用，默认 False）。
            test (bool): 测试模式标志（具体行为由 take_action_batch 定义，默认 False）。

        处理逻辑：
            1. 参数格式化：若 decision_model_responses 是字符串，则转为单元素列表；
                    若 query_lists 或 document_lists 的子元素是字符串，则转为嵌套列表。
            2. 交互链扩展：根据 decision_model_responses 中的采样数量（sampling_nums），
                        扩展 query_lists、document_lists 和 done_flags 的长度为原长度 * sampling_nums。
            3. 批量操作执行：调用 take_action_batch 方法，传入扩展后的参数执行操作。
            4. 返回更新后的 query_lists、document_lists 和 done_flags。

        返回值：
            tuple:
                - query_lists (list[list]): 更新后的查询列表。
                - document_lists (list[list]): 更新后的文档列表。
                - done_flags (list[bool]): 更新后的交互链完成状态标识。

        注意事项：
            - 交互链扩展后，query_lists、document_lists 和 done_flags 的长度必须一致。
            - decision_model_responses 中每个元素的长度需与 sampling_nums 匹配，否则可能导致索引错误。
            - take_action_batch 方法会根据 done_flags 跳过已完成的交互链。
        """

        if isinstance(decision_model_responses, str):
            decision_model_responses = [decision_model_responses]

        if isinstance(query_lists[0], str):
            query_lists = [query_lists]

        if isinstance(query_lists[0], str):
            document_lists = [document_lists]

        sampling_nums = len(decision_model_responses[0])

        # 增殖，将该处所有的样本都进行一轮增殖
        # 增殖轮次为sampling nums
        query_lists = [copy.deepcopy(query_lists[int(i/sampling_nums)]) for i in range(len(query_lists) * sampling_nums)]
        document_lists = [copy.deepcopy(document_lists[int(i/sampling_nums)]) for i in range(len(document_lists) * sampling_nums)]
        # 已经在Decision Agent处完成了扩大
        # time_lis = [time_lis[int(i/sampling_nums)] for i in range(len(time_lis) * sampling_nums)]
        done_flags = [copy.deepcopy(done_flags[int(i/sampling_nums)]) for i in range(len(done_flags) * sampling_nums)]

        # 注意，传入的decision_model_responses为List[tuple]形式，对齐了外部的数据结构
        # 这里需要额外展平
        decision_model_responses_flat = []

        for decision_prompt_tuples in decision_model_responses:
            decision_model_responses_flat.extend(decision_prompt_tuples)

        # 此刻完成了长度上的对齐
        query_lists, document_lists, done_flags, time_lis = self.take_action_batch(
                                                        decision_prompts = decision_model_responses_flat, 
                                                        query_lists = query_lists, 
                                                        document_lists = document_lists, 
                                                        done_flags = done_flags, 
                                                        time_lis = time_lis, 
                                                        need_print = need_print, 
                                                        test = test)

        return query_lists, document_lists, done_flags, time_lis
    
    def get_history(self) -> List[Dict[str, str]]:  
        return self.model.get_history()
    
    def clear_history(self) -> None:
        self.model.clear_history()

class Decision_Agent:
    
    def __init__(
        self, 
        model: CasualLMWithValueHead = None, 
        tokenizer: "AutoTokenizer" = None, 
        config: DecisionConfig = None, 
        *args, **kwargs
    ):
        """
        初始化QueryExpander类，用于生成多个扩展查询变体。
        
        Args:
            model: 用于扩展的LLM模型（如Llama3）
            verbose (bool): 是否输出详细日志
        """
        if config == None:
            config = DecisionConfig()
        self.config = config

        model_dir = config.model_dir
        device: torch.device = config.device
        verbose: bool = False
        strategy_params: DecisionStrategyParams = config.strategy_params
        batch_size: int = config.batch_size
        test: bool = config.test 

        print("Loading Decision Agent...")
        if not test and not config.load_api and model is None:
            self.model = Large_Language_Model(local_dir = model_dir, 
                                              device = device,
                                              batch_size = batch_size)

        if config.load_without_model:
            self.model = CasualLMWithValueHead(local_dir = model_dir, 
                                              device = device,
                                              batch_size = batch_size, 
                                              without_model = config.load_without_model, 
                                              with_value_head = True, 
                                              tokenizer = tokenizer)

        if config.load_api == True:
            self.model = Large_Language_Model_API(model = self.config.api_model)

        self.batch_size = batch_size
        self.model.init_llm([self._init_prompt() for i in range(self.batch_size)])
        self.model.complement_prompt = [self._init_complement_prompt() for i in range(self.batch_size)]
    
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        if type(strategy_params) == dict:
            self.strategy_params = strategy_params
        else:
            self.strategy_params = strategy_params.todict()

        print("Done!")

    def _init_model(self, ):
        """
        初始化模型，让其回归到最初的状态
        
        """
        self.model.init_llm(self._init_prompt())
        return 
    
    def _init_complement_prompt(self):
            format_prompt = """
**Expected Output Example**: 
<Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
<Action> Query 0 | Query Rewrite | Nums 3 </Action>
<Detail> Generate alternative queries to explore diverse aspects such as AI applications, ethical concerns, and future trends in the healthcare industry. These rewrites should touch on topics that aren't explicitly mentioned but are relevant to presenting a full picture of AI's impact on healthcare. </Detail>
<Action> Query 0 | Query Reason | Nums 1 </Action>
<Detail> Analyze the query to determine what specific kinds of information should be retrieved, such as case studies, research papers, or expert opinions, to improve the quality of responses. </Detail>
<END>
**Please follow the format strictly!** """.strip()
            return format_prompt
    def _init_prompt(self):
        """
        该方法用以交代清楚prompt的构成
        
        """
        # 没有用到的动作放在这里
        # <Read> Read all queries & documents: You can choose to read all queries and documents when you want to evaluate if it's sufficient to answer user's input or not. 
        # <Summarize> Summarize queries and documents: Generate summary of all queries and documents to filter out text noise.  
        prompt = f"""
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
"""

        return prompt

    def build_decision_prompt(self, query_list: List[str], document_list: List[str]) -> str:
        """构建当前决策状态的提示"""
        # 格式化已收集的查询
        queries_str = "\n".join([f"<Query {i}>: {q}" for i, q in enumerate(query_list)])
        
        # 格式化已检索的文档
        docs_str = "\n".join([f"<Document {i}>: {d[:200]}..." for i, d in enumerate(document_list)]) if len(document_list) != 0 else "Nothing yet."
        
        # 构建当前状态提示
        state_prompt = f"""
Current State:
**User Input:**
{query_list[0]}

**Queries Collected:**
{queries_str}

**Documents Retrieved:**
{docs_str}

Please give out your <Think>, <Action>, and <Detail>. Remember, "Stop" when all documents are sufficiently covered user's input as soon as possible.
"""
        return state_prompt
    
    def generate(self, query_list: List[str], 
                 document_list: List[str], 
                 need_print: bool = False) -> str:
            """
            为每个 batch 生成模型决策响应（<Think>, <Action>, <Detail> 格式）
            
            Args:
                query_list: 多个 batch 的查询列表（List[List[str]]）
                document_list: 多个 batch 的文档列表（List[List[str]]）
                
            Returns:
                模型的原始响应列表（每个元素对应一个 batch 的响应）
            """
            prompt = self.build_decision_prompt(query_list,
                                                document_list)
            # 保证是str的接口
            # 这意味着，对话和上下文关系不大，没有连续的逻辑存在
            model_response = self.model.generate(prompt, temperature = 1.0)
            
            if need_print:
                print('=' * 20)
                print('Decision Agent Input: ' + prompt)
                print('=' * 20)
                print('Decision Agent Response: ' + model_response)

            return model_response

    def _split_list_to_tuples(self, lst: List, k: int) -> List[Tuple]:
        """
        将列表均匀划分为 k 个子组，每个子组大小为 len(lst) // k。
        如果 len(lst) 无法被 k 整除，则进行裁剪并发出警告。

        Args:
            lst (List): 待划分的列表。
            k (int): 分组数量。

        Returns:
            List[Tuple]: 每个子组作为一个元组组成的列表。
        """
        n = len(lst)
        quotient = n // k
        remainder = n % k

        if remainder != 0:
            new_length = k * quotient
            warnings.warn(f"列表长度 {n} 无法被 k={k} 整除，已裁剪至长度 {new_length}。")

            # 截断列表
            lst = lst[:new_length]

        # 均匀划分为 k 个子组
        return [tuple(lst[i * quotient : (i + 1) * quotient]) for i in range(k)]

    def sampling(self, 
                 query_lists: List[List[str]], 
                 document_lists: List[List[str]], 
                 temperature: float = 1.0, 
                 sampling_num: int = 3, 
                 return_time: bool = True, 
                 model: "CasualLMWithValueHead" = None, 
                 tokenizer: "AutoModelForCausalLMWithValueHead" = None, 
                 **kwargs) -> Tuple[List[str], List[List], List[List], List[float]]:
        """
        树状采样：为每个查询生成多个模型响应。
        思考，输入的长度是固定的，只是针对这些输入的每一个样本进行采样
        Args:
            query_lists (List[str]): 查询列表。若非嵌套列表，则自动转换为嵌套格式。
            document_lists (List[str]): 文档列表。需与 query_lists 长度一致。
            need_print (bool): 是否打印模型响应。
            sampling_num (int): 每个查询生成的响应数量。

        Returns:
            Tuple[List[str], List[List[str]]]: 
                - input_prompts: 每个查询对应的原始 prompt。
                - model_responses: 每个查询的采样响应列表（每个子列表长度为 sampling_num）。
        """
        if not isinstance(query_lists[0], list):
            query_lists = [query_lists]

        if not isinstance(document_lists[0], list):    
            document_lists = [document_lists]

        if not len(query_lists) == len(document_lists):
            warnings.warn("DecisionModel采样错误：输入query_lists和输入的document_lists长度不一致，已进行裁剪...")
            query_lists = query_lists[:min(len(query_lists), len(document_lists))]
            document_lists = document_lists[:min(len(query_lists), len(document_lists))]
        
        original_prompts: list[str] = []
        sampling_prompts = []

        for idx in range(len(query_lists)):
            base_prompt = self.build_decision_prompt(query_lists[idx],
                                                     document_lists[idx])
            
            original_prompts.append(base_prompt)
            sampling_prompt = [base_prompt] * sampling_num
            sampling_prompts.extend(sampling_prompt)
        
        # prompts = [prompt] * 3
        # 捕获时间，用于后续的奖励计算
        # 注意，TimeList为了对齐done_flags，并未采用嵌套格式!!!
        model_responses, time_lis = self.model.generate(sampling_prompts, 
                                              temperature = temperature, 
                                              return_time = return_time, 
                                              **kwargs) if model is None \
                                    else model.generate(
                                                sampling_prompts, 
                                                temperature = temperature, 
                                                return_time = return_time, 
                                                model = model, 
                                                tokenizer = tokenizer, 
                                                return_type = str, 
                                                **kwargs)

        # 分离各响应
        original_prompts = original_prompts

        sampling_prompts = self._split_list_to_tuples(sampling_prompts, len(query_lists))
        model_responses = self._split_list_to_tuples(model_responses, len(query_lists))

        # 传回两部分，一个是prompt，一个是model_responses
        return original_prompts, sampling_prompts, model_responses, time_lis

    def generate_batch(self, query_lists: List[List[str]], 
                       document_lists: List[List[str]], 
                       return_time: bool = False, 
                       return_input_prompts: bool = False, 
                       model: "CasualLMWithValueHead" = None, 
                       tokenizer: "AutoTokenizer" = None, 
                       **kwargs):
        """
        Args:
            query_list: 多个 batch 的查询列表（List[List[str]]）
            document_list: 多个 batch 的文档列表（List[List[str]]）
        
        Returns:
            模型的原始响应列表（每个元素对应一个 batch 的响应）
        """
        if not len(query_lists) == len(document_lists):
            raise ValueError("输入decision model的query_lists和document_lists长度不一致！")
        
        input_prompts = [self.build_decision_prompt(query_lists[i],
                                            document_lists[i])
                                            for i in range(len(query_lists))]

        if model is not None:
            response = self.model.generate(input_prompts, return_time = return_time, model = model, tokenizer = tokenizer)
        else:
            response = self.model.generate(input_prompts, return_time = return_time)
        
        if not return_input_prompts:
            return *response, 
        else:
            return input_prompts, *response

    def prompt_for_summary(self, query_list: List[str], documents: List[str]) -> str:
        """
        用以传递回答的细节：比如最相关的文档是哪一个
        将Detail输出综合到一个结果中，完成最后结果的输出

        Args:
            problem (str): 需要解决的问题或任务。
            method (str): 推理方法（支持 "CoT" 或 "ToT"）。
            steps (int): 需要生成的推理步骤数或路径数。

        Returns:
            str: 完整的提示模板。
        """
        query_prompt = "\n".join([f"<Query {i}> {doc}" for i, doc in enumerate(query_list)])
        documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(documents)])

        prompt = f"""
Current State:
User Input:
{query_list[0]}
Input Query: 
{query_prompt}
Documents:
{documents_prompt}

**You have already choose to <Stop> generation**. 
Your <Detail> would be send to last generate model which is vital to generation quality. Summary current state, expecially for document state, not query state.
Please give out your <Think>, and <Detail>.
""".strip()

        return prompt
    
    def last_summary(self, query_list: List[str], documents: List[str], **kwargs) -> str: 
        """
        在最后一步中，加入summary机制，也即如果需要总结，指出关键点，那么允许模型在这一步中加入Detail
        如果不需要该步骤，则直接跳过
        
        """

        summary_input = self.prompt_for_summary(query_list = query_list, documents = documents)
        model_response = self.model.generate(summary_input)
        details = self.extract_response_components(model_output = model_response).get('details')
        detail_string =  '<Detail>' + "\n".join([f" {detail}" for detail in details])
        
        return detail_string
    
    def last_summary_batch(self, 
                           query_lists: List[List[str]], 
                           document_lists: List[List[str]], 
                           return_input_prompts: bool = False, 
                           return_time: bool = False, 
                           **kwargs
                           ) -> Union[List[str], List[List]]:
        """
        在最后一步中，加入summary机制，也即如果需要总结，指出关键点，那么允许模型在这一步中加入Detail
        如果不需要该步骤，则直接跳过

        Args:
            query_lists: 每个元素是一个查询的多个部分组成的列表
            document_lists: 每个元素是对应文档列表组成的列表
            **kwargs: 其他参数

        Returns:
            List[str]: 每个输入对对应的 <Detail> 字符串列表
        """

        summary_inputs = []
        for query, documents in zip(query_lists, document_lists):
            summary_input = self.prompt_for_summary(query_list=query, documents=documents)
            summary_inputs.append(summary_input)

        responses = self.model.generate(summary_inputs, return_time = return_time)

        detail_strings = []
        for response in responses[0]:
            details = self.extract_response_components(model_output=response).get('details', []) # 索引
            detail_string = '<Detail>' + "\n".join([f" {detail}" for detail in details])
            detail_strings.append(detail_string)

        returns = [detail_strings]
        if return_input_prompts:
            returns.append(summary_inputs)
            returns.append(responses[0])
        if return_time:
            returns.append(responses[1])
        
        return (*returns, )

    def last_summary_sampling(self, 
                              query_lists: List[List[str]], 
                              document_lists: List[List[str]], 
                              sampling_nums: int = 3, 
                              return_time : bool = True, 
                              **kwargs):
        """
        采样版本
        
        ##测试完毕##
        """
        
        if return_time:
            time_1st = []
        sampling_inputs = []
        
        for query_list, document_list in zip(query_lists, document_lists):
            if return_time:
                before = time.time()
            summary_input = self.prompt_for_summary(query_list=query_list, 
                                                    documents=document_list)
            sampling_inputs.extend([summary_input for i in range(sampling_nums)])
            
            if return_time:
                after = time.time()
                time_delta = after - before
                time_1st.extend([time_delta for i in range(sampling_nums)])


        node_nums = len(query_lists)
        model_responses, time_2nd = self.model.generate(sampling_inputs, 
                                              return_time = return_time, 
                                              **kwargs)

        detail_strings = []
        if return_time:
            time_3rd = []
        for response in model_responses:
            if return_time:
                before = time.time()

            details = self.extract_response_components(model_output = response).get('details', [])
            detail_string = "\n".join([f" {detail}" for detail in details])
            detail_strings.append(detail_string)

            if return_time:
                after = time.time()
                time_delta = after - before
                time_3rd.append(time_delta)
        
        # 构造time_lis
        if return_time:
            time_lis = [time_1st[i] + time_2nd[i] + time_3rd[i] for i in range(len(detail_strings))]

        # 提取响应
        sampling_inputs = self._split_list_to_tuples(sampling_inputs, k = node_nums)
        model_responses = self._split_list_to_tuples(model_responses, k = node_nums)

        if not return_time:
            return sampling_inputs, model_responses, detail_strings
        else:
            return sampling_inputs, model_responses, detail_strings, time_lis

    def extract_response_components(self, model_output: str) -> Dict[str, List[Any]]:
        """
        从模型输出中提取<Think>、<Action>和<Detail>组件，并解析Action内容
        
        Args:
            model_output (str): 模型返回的原始响应字符串
            
        Returns:
            Dict[str, List[Any]]: 包含提取组件的字典
        """
        data = {
            'raw_string': model_output,
            'thinks': [],
            'actions': [],
            'details': []
        }
        # <END>处理机制
        model_output = model_output.split('<END>')[0]

        # 提取并处理<Think>标签
        data['thinks'] = [t.strip() for t in re.findall(r'<Think>(.*?)</Think>', model_output, re.DOTALL)]
        
        # 提取并处理<Action>标签
        action_pattern = r'<Action>(.*?)</Action>'
        action_matches = re.findall(action_pattern, model_output, re.DOTALL)
        
        for action_str in action_matches:
            # 分割并清理字段
            parts = [p.strip() for p in action_str.split('|')]
            
            if len(parts) >= 3:
                # 解析query_id、action_type、nums
                query_id = parts[0]
                action_type = parts[1]
                nums = int(parts[2]) if parts[2].isdigit() else parts[2]
                
                data['actions'].append({
                    'target_id': query_id,
                    'action_type': action_type,
                    'nums': nums
                })
            else:
                data['actions'].append(action_str.strip())  # 保留原始格式（异常情况）
        
        # 提取并处理<Detail>标签
        detail_pattern = r'<Detail>(.*?)</Detail>'
        detail_matches = re.findall(detail_pattern, model_output, re.DOTALL)
        data['details'] = [d.strip() for d in detail_matches]
        
        return data

    def get_history(self) -> List[Dict[str, str]]:  
        return self.model.get_history()

    def clear_history(self) -> None:
        self.model.clear_history()

    def reload(self, config: PipelineConfig, **kwargs) -> None:
        """
        重新加载模型
        """

        self.model.reload(config)
    
    def reload_lora(self, config: PipelineConfig, **kwargs) -> None:
        """
        重新加载模型
        """
        
        self.model.reload_lora(config)

    def release(self, **kwargs) -> None:

        """
        释放模型
        """

        # 使用完模型后释放内存
        self.model.release()

    def save_model(self, dir):
        """保存决策模型的最终模型"""
        # 保存模型
        self.model.save_model(dir)
        return 
    
    @classmethod
    def load(
        cls,
        model: Optional[Any] = None,
        decision_config: "DecisionConfig" = None
    ):
        """
        类方法：加载并初始化模型实例
        """
        if decision_config is None:
            decision_config = DecisionConfig()

        model_dir = decision_config.model_dir
        device = decision_config.device
        verbose = decision_config.verbose
        strategy_params = decision_config.strategy_params
        batch_size = decision_config.batch_size
        test = decision_config.test
        load_api = decision_config.load_api
        api_model = decision_config.api_model

        print("Loading Decision Agent...")

        # 初始化实例
        instance = cls()  # 创建实例（不显式调用 __init__）

        # 根据配置加载模型
        if not test and not load_api and model is None:
            instance.model = Large_Language_Model(
                local_dir=model_dir,
                device=device,
                batch_size=batch_size
            )
        else:
            if model is not None:
                instance.model = model
            else:
                # 使用 API 模型
                instance.model = Large_Language_Model_API(model=api_model)

        # 初始化系统提示
        if hasattr(instance.model, "init_llm"):
            instance.model.init_llm(
                [instance._init_prompt() for _ in range(batch_size)]
            )
            instance.model.complement_prompt = [
                instance._init_complement_prompt() for _ in range(batch_size)
            ]

        # 设置其他属性
        instance.batch_size = batch_size
        instance.verbose = verbose
        instance.logger = logging.getLogger(__name__)
        instance.strategy_params = (
            strategy_params.todict() if hasattr(strategy_params, "todict") else strategy_params
        )

        return instance


class Generate_Agent(Large_Language_Model_API):
    def __init__(self, 
                 config: GenerativeConfig = None, 
                 init_prompt: str = None, 
                 **kwargs):
        print("Loading Generate Agent...")
        super().__init__()

        if config == None:
            config = GenerativeConfig()

        self._init_system_prompt(init_prompt, batchsize = config.batchsize) 
        
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

Please answer the user's input follow the format upon. 
""".strip()
            for i in range(batchsize)])
        else:
            self.init_llm(system_prompt = [init_prompt for i in range(batchsize)])


        
class Naive_RAG:

    def __init__(self, 
                 model: Large_Language_Model_API,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 index_load_path: str = './wikipedia_BGE_L2.contriever',
                 document_load_path: str = './psgs_w100.tsv',
                 ):
        print('=' * 40)
        print('Loading Large Language Model...')
        self.model = model
        
        print('=' * 20)
        print('Loading Index...')
        self.indexer = IndexBuilder(
            device = device,
            index_load_path = index_load_path,
            document_load_path = document_load_path
        )

        return 
    

    def cat_document(self, documents: List[Dict[str, Any]]) -> str:
        """
        将检索到的文档内容格式化为提示文本。

        Args:
            documents (List[Dict]): 包含 'content' 字段的文档列表

        Returns:
            str: 格式化的提示文本
        """
        prompt_parts = []

        for doc in documents:
            content = doc.get('content', '').strip()
            if content:
                prompt_parts.append(f"{content}")
                prompt_parts.append('\n')

        return "\n".join(prompt_parts)
    
    def generate(self, 
                user_input: Union[str, List[str]],
                k: int = 3, 
                max_tokens: int = 600,
                if_print: bool = False, 
                **kwargs):

        # 确定是否为batch
        is_batch = isinstance(user_input, list)
        if not is_batch:
            user_input = [user_input]

        if if_print:
            print('=' * 20)
            print('Generating Outputs with naive rag...')

        input_prompts = []

        # 构造输入提示
        for query in user_input:
            # 检索相关文档
            documents = self.indexer.topk_search(query, k=k)
            if if_print:
                print(f"Retrieved {len(documents)} documents for query: {query}")

            # 构造提示模板
            prompt_parts = [
                "User Input:\n" + query,
                "\nDocument:",
                self.cat_document(documents),
                "\nYour Output: "
            ]
            
            input_prompt = "\n\n".join(prompt_parts)
            input_prompts.append(input_prompt)


        # 调用模型生成回答
        outputs = self.model.generate(
            input_prompts,
            max_tokens=max_tokens,
            save_history=False,
            **kwargs
        )

        # 打印调试信息
        if if_print:
            print('=' * 20)
            for idx, (q, o) in enumerate(zip(user_input, outputs)):
                print(f"\nInput {idx+1}: {q}")
                print(f"Output {idx+1}: {o}")

        # 还原输出格式
        if not is_batch:
            outputs = outputs[0]

        return outputs

class RRR():
    def __init__(self, 
                 model_name = None, 
                 local_dir='./qwen2.5_1.5B/', 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 model: torch.nn.Module = None, 
                 tokenizer: torch.nn.Module = None, 
                 index_load_path: str = './wikipedia_BGE_L2.contriever',
                 document_load_path: str = './psgs_w100.tsv',
                 ):
        
        print('=' * 40)
        print('Loading Large Language Model...')
        self.model = Large_Language_Model(local_dir = local_dir, 
                                          device = device, 
                                          )
        
        self.rewriter = QueryRewriter(
            model = self.model
        )
        
        print('=' * 20)
        print('Loading Index...')
        self.indexer = IndexBuilder(
            device = device,
            index_load_path = index_load_path,
            document_load_path = document_load_path
        )

        print('=' * 20)
        print('Finished loading!')
    
    def cat_prompt_and_document(self, query, documents):
        """
        根据用户查询和检索到的文档生成结构化的RAG提示文本。

        Args:
            query (str): 用户的问题
            documents (list[dict]): 检索到的文档列表，每个文档包含 'content' 字段
            tokenizer: 分词器（未被使用，但保留参数以支持后续扩展）
            model: 模型（未被使用，但保留参数以支持后续扩展）
            device (str): 设备（未被使用，但保留参数以支持后续扩展）

        Returns:
            str: 格式化的提示文本
        """

        # 初始化提示内容
        prompt = []
        
        # 第一部分：问题与上下文说明
        prompt.append("Please answer the following question with the following context:")
        prompt.append(f"Question: {query}")
        prompt.append("Retrieved context:")
        
        # 添加检索到的文档内容
        for doc in documents:
            # 确保文档内容不为空且格式正确
            content = doc.get('content', '').strip()
            if content:
                prompt.append(content)  # 直接拼接文档内容（可能需要分隔符，但按参考示例处理）
        
        # 第二部分：再次强调使用上下文回答
        prompt.append("\nPlease answer the following question with the above context:")
        prompt.append(f"Question: {query}")
        prompt.append("Your answer: ")
        
        # 合并所有部分并返回
        return "\n".join(prompt)
    
    def generate(self, user_input: Union[str, List[str]], question_nums: int = 2, k: int = 2, **kwargs):
        """
        对用户查询进行重写，以适应模型的输入格式。
        注意：如果希望采用RRR，则需传入RRR所使用的模型，pipeline保持不变
        """

        print('=' * 40)
        print('Generating Rewrite Questions...')
        # 调用重写模型，获取重写后的问题
        question_batch = self.rewriter.generate(user_input, question_nums = question_nums)

        print('=' * 20)
        print('Retrieving Documents...')
        
        # 先用原来的问题进行检索
        document = self.indexer.topk_search(user_input, k = k)

        # 遍历重写后的问题，进行检索
        for questions in question_batch:
            documents = []
            document = []
            for question in questions:
                document.extend(self.indexer.topk_search(question, k = k))
                documents.append(document)

        print('=' * 20)
        print('Start Building Prompt...')
        # 构建输入提示
        input_prompt_batch = [self.cat_prompt_and_document(prompt, documents) for prompt, documents in zip(questions, documents)]

        print('=' * 20)
        print('Generating Outputs with LLMs...')
        outputs = self.model.generate(input_prompt_batch, max_tokens = 600)

        print('=' * 20)
        print(f'Input: {input_prompt_batch}')
        print(f'Output: {outputs}')
        return outputs






def download_LLMs(repo_id='Qwen/Qwen2.5-1.5B-Instruct', 
                 local_dir='./qwen2.5_1.5B/', 
                 login_token=None, 
                 max_retries=3):
    """
    下载 Hugging Face 仓库的模型，并指定下载到本地路径。

    Args:
        repo_id (str): Hugging Face 仓库 ID（格式：用户名/仓库名）。
        local_dir (str): 本地保存路径。
        login_token (str, optional): Hugging Face 访问 Token。默认为 None（无需登录）。
        max_retries (int): 下载失败时的最大重试次数。默认 3 次。

    Returns:
        str: 保存模型的本地路径。
    """
    from huggingface_hub import snapshot_download, login
    import time
    import os
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com/"

    if login_token:
        login(token=login_token)  # 使用传入的 Token 登录

    retries = 0
    done = False

    print('=' * 40)
    print(f"开始下载模型：{repo_id} 到 {local_dir}")
    print('=' * 20)

    while not done and retries <= max_retries:
        try:
            # 显式指定分支为 main，并设置本地保存路径
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True  # 断点续传
            )
            done = True
            print('=' * 20)
            print("模型下载完成！")
        except Exception as e:
            retries += 1
            print(f"下载失败，尝试第 {retries} 次重试：{str(e)}")
            time.sleep(2)  # 暂停 2 秒后重试

    if not done:
        raise RuntimeError(f"下载失败，超过最大重试次数 {max_retries}")

    return local_dir


def test_LLM():
    print('=' * 40)
    print('Loading LLM...')
    print('=' * 20)

    prompt = 'How to compute the square of a number with python code?'
    model = Large_Language_Model('QwenQwen2.5-1.5B-Instruct', local_dir = './qwen2.5_1.5B/')

    print('Generating prompt...')
    print('=' * 20)

    outputs = model.generate(prompt)
    print('Outputs: ')
    print(outputs) 
    print('=' * 20)
    return 







if __name__ == "__main__":
    # download_LLMs()
    # test_BGEEmbedder()
    # test_query_rewriter()
    # test()
    pass
    # test_LLM()
