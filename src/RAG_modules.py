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
import datetime
import concurrent.futures

from .large_language_model import Large_Language_Model, Large_Language_Model_API, Large_Language_ModelV2
from .config import *
from .dataset import ConversationTree, ModelInteractionSampler, Forest, PLACE_HOLDER
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
    from typing import Dict


import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Callable, Dict, Union, Tuple, TypeVar, Generic
import threading
import time
import logging
import uuid
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexerWorker")

T = TypeVar('T')


params = {
    'text_col' : 'text', 
    'id_col' : 'id', 
    'name_col' : 'title', 
    'embedding_col': 'embedding', 
    'projection_dim' : 384, 

}



class BGEEmbedder:
    def __init__(self, 
                 model_name='/root/autodl-tmp/BGE_model/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea/', 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        """
        初始化 BGE 模型
        Args:
            model_name: BGE 模型名称（如 'BAAI/bge-small-en', 'BAAI/bge-base-zh'）
        """
        self.model = SentenceTransformer(model_name_or_path = model_name, device = device)
    
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
            forward_string = 'Title: ' + self.document_df.loc[id, 'title'] 
        except:
            forward_string =  'Title: ' + 'No title'

        try:
            back_string = 'Context: ' + self.document_df.loc[id, 'text'] 
        except:
            back_string =  'Context: ' + 'No Context'

        return  forward_string + back_string
    
    def topk_search(self, query: str, k: int = 5) -> List[str]:
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


class IndexerWorker(IndexBuilder):
    def __init__(self, num_workers=4, *args, **kwargs, ):


        import os
        if num_workers is None:
            num_workers = max(4, (os.cpu_count() or 1) + 4)

        IndexBuilder.__init__(*args, **kwargs)
        self.base_indexer = self.BGE_embedder
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="IndexerWorker"
        )

class SearchException(Exception):
    """专用搜索异常"""
    pass

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    task_id: str
    query: str
    status: TaskStatus
    submit_time: float
    complete_time: Optional[float] = None
    error: Optional[str] = None

class IndexerMetrics:
    """指标收集器"""
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_latency = 0.0
        self._lock = threading.Lock()
    
    def record_request(self, latency: float, success: bool):
        """记录请求指标"""
        with self._lock:
            self.total_requests += 1
            if not success:
                self.failed_requests += 1
            
            # 指数移动平均 (alpha=0.1)
            alpha = 0.1
            if self.avg_latency == 0.0:
                self.avg_latency = latency
            else:
                self.avg_latency = alpha * latency + (1 - alpha) * self.avg_latency
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            success_rate = 0.0
            if self.total_requests > 0:
                success_rate = (self.total_requests - self.failed_requests) / self.total_requests
            
            return {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "avg_latency_ms": self.avg_latency * 1000
            }

class ConcurrentIndexerWorker(Generic[T]):
    """
    线程安全的索引器工作器
    
    提供同步/异步搜索接口，支持批量处理，自动资源管理和监控。
    """
    
    def __init__(self, base_indexer: Any, num_workers: Optional[int] = None):
        """
        初始化索引器工作器
        
        Args:
            base_indexer: 基础索引器对象，必须实现topk_search方法
            num_workers: 工作线程数，None表示使用CPU核心数+4
        """
        if num_workers is None:
            num_workers = max(4, (os.cpu_count() or 1) + 4)
        
        self.base_indexer = base_indexer
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="IndexerWorker"
        )
        self.futures: Dict[str, concurrent.futures.Future] = {}  # task_id -> Future
        self.task_info: Dict[str, TaskInfo] = {}  # task_id -> TaskInfo
        self._lock = threading.Lock()
        self._task_counter = 0
        self.metrics = IndexerMetrics()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        
        logger.info(f"IndexerWorker initialized with {num_workers} workers")
    
    def _generate_task_id(self) -> str:
        """生成唯一任务ID"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"
    
    def _safe_search(self, query: str, k: int) -> List[T]:
        """包装搜索方法，添加隔离保护"""
        start_time = time.perf_counter()
        success = False
        
        try:
            result = self.base_indexer.topk_search(query, k)
            success = True
            return result
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}", exc_info=True)
            raise SearchException(f"Search failed: {str(e)}") from e
        finally:
            latency = time.perf_counter() - start_time
            self.metrics.record_request(latency, success)
    
    def search(self, query: str, k: int = 5, timeout: Optional[float] = None) -> List[T]:
        """
        同步搜索接口
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            timeout: 超时时间(秒)，None表示无超时
        
        Returns:
            搜索结果列表
        
        Raises:
            SearchException: 搜索失败
            concurrent.futures.TimeoutError: 超时
            RuntimeError: 工作器已关闭
        """
        if self._shutdown:
            raise RuntimeError("IndexerWorker has been shutdown")
        
        task_id = self._generate_task_id()
        future = self.executor.submit(self._safe_search, query, k)
        
        # 记录任务信息
        task_info = TaskInfo(
            task_id=task_id,
            query=query,
            status=TaskStatus.PENDING,
            submit_time=time.time()
        )
        
        with self._lock:
            self.futures[task_id] = future
            self.task_info[task_id] = task_info
        
        try:
            start_time = time.time()
            result = future.result(timeout=timeout)
            elapsed = time.time() - start_time
            
            # 更新任务状态
            with self._lock:
                if task_id in self.task_info:
                    self.task_info[task_id].status = TaskStatus.COMPLETED
                    self.task_info[task_id].complete_time = time.time()
            
            logger.debug(f"Task {task_id} completed in {elapsed:.4f}s")
            return result
        except concurrent.futures.TimeoutError:
            logger.warning(f"Task {task_id} timed out after {timeout}s")
            raise
        except Exception as e:
            # 更新任务状态
            with self._lock:
                if task_id in self.task_info:
                    self.task_info[task_id].status = TaskStatus.FAILED
                    self.task_info[task_id].error = str(e)
                    self.task_info[task_id].complete_time = time.time()
            raise
        finally:
            # 清理任务
            with self._lock:
                self.futures.pop(task_id, None)
                # 保留任务信息用于监控(1小时后清理)
    
    def search_async(self, query: str, k: int = 5) -> Tuple[str, concurrent.futures.Future]:
        """
        异步搜索接口
        
        Args:
            query: 搜索查询
            k: 返回结果数量
        
        Returns:
            (task_id, future) 元组
        
        Raises:
            RuntimeError: 工作器已关闭
        """
        if self._shutdown:
            raise RuntimeError("IndexerWorker has been shutdown")
        
        task_id = self._generate_task_id()
        future = self.executor.submit(self._safe_search, query, k)
        
        # 记录任务信息
        with self._lock:
            self.futures[task_id] = future
            self.task_info[task_id] = TaskInfo(
                task_id=task_id,
                query=query,
                status=TaskStatus.PENDING,
                submit_time=time.time()
            )
        
        # 添加完成回调更新状态
        def _update_status(f):
            with self._lock:
                if task_id in self.task_info:
                    if f.cancelled():
                        self.task_info[task_id].status = TaskStatus.CANCELLED
                    elif f.exception():
                        self.task_info[task_id].status = TaskStatus.FAILED
                        self.task_info[task_id].error = str(f.exception())
                    else:
                        self.task_info[task_id].status = TaskStatus.COMPLETED
                    self.task_info[task_id].complete_time = time.time()
        
        future.add_done_callback(_update_status)
        
        return task_id, future
    
    def batch_search(self, queries: List[str], k: int = 5, 
                    timeout: Optional[float] = None) -> List[Union[List[T], Exception]]:
        """
        批量搜索，结果按完成顺序返回
        
        Args:
            queries: 查询列表
            k: 每个查询返回结果数量
            timeout: 总超时时间(秒)，None表示无超时
        
        Returns:
            结果列表，保持与输入相同的顺序，失败项为Exception对象
        
        Raises:
            RuntimeError: 工作器已关闭
            concurrent.futures.TimeoutError: 总超时
        """
        if self._shutdown:
            raise RuntimeError("IndexerWorker has been shutdown")
        
        if not queries:
            return []
        
        # 提交所有任务
        futures = []
        task_ids = []
        query_map = {}  # future -> (index, query)
        
        for idx, query in enumerate(queries):
            task_id = self._generate_task_id()
            future = self.executor.submit(self._safe_search, query, k)
            
            with self._lock:
                self.futures[task_id] = future
                self.task_info[task_id] = TaskInfo(
                    task_id=task_id,
                    query=query,
                    status=TaskStatus.PENDING,
                    submit_time=time.time()
                )
            
            futures.append(future)
            task_ids.append(task_id)
            query_map[future] = (idx, query)
        
        results = [None] * len(queries)
        
        try:
            # 使用as_completed以最先完成的顺序处理
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                idx, query = query_map[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                    logger.error(f"Batch query failed [{idx}]: '{query}' - {str(e)}")
        except concurrent.futures.TimeoutError:
            logger.warning(f"Batch search timed out after {timeout}s")
            # 取消所有未完成的任务
            for future in futures:
                if not future.done():
                    future.cancel()
            raise
        finally:
            # 清理任务
            with self._lock:
                for task_id in task_ids:
                    self.futures.pop(task_id, None)
                    # 更新任务状态
                    if task_id in self.task_info:
                        if self.task_info[task_id].status == TaskStatus.PENDING:
                            self.task_info[task_id].status = TaskStatus.CANCELLED
                            self.task_info[task_id].complete_time = time.time()
        
        return results
    
    def batch_search_ordered(self, queries: List[str], k: int = 5,
                        timeout: Optional[float] = None) -> List[Union[List[T], Exception]]:
        """使用map保持顺序但并发执行"""
        if not queries:
            return []
        
        def search_wrapper(args):
            query, k = args
            try:
                return self._safe_search(query, k)
            except Exception as e:
                return e
        try:
            # 使用executor.map并发执行但保持顺序
            results = list(self.executor.map(
                search_wrapper,
                [(query, k) for query in queries],
                timeout=timeout
            ))
            return results
        except concurrent.futures.TimeoutError:
            logger.warning(f"Ordered batch search timed out after {timeout}s")
            raise
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消指定任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 是否成功取消
        """
        with self._lock:
            future = self.futures.get(task_id)
            if future and not future.done():
                result = future.cancel()
                if result:
                    if task_id in self.task_info:
                        self.task_info[task_id].status = TaskStatus.CANCELLED
                        self.task_info[task_id].complete_time = time.time()
                return result
            return False
    
    def cancel_all_tasks(self) -> int:
        """
        取消所有挂起的任务
        
        Returns:
            int: 被取消的任务数量
        """
        cancelled_count = 0
        with self._lock:
            for task_id, future in list(self.futures.items()):
                if not future.done():
                    if future.cancel():
                        cancelled_count += 1
                        if task_id in self.task_info:
                            self.task_info[task_id].status = TaskStatus.CANCELLED
                            self.task_info[task_id].complete_time = time.time()
        
        logger.info(f"Cancelled {cancelled_count} tasks")
        return cancelled_count
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务信息，如果任务不存在返回None
        """
        with self._lock:
            return self.task_info.get(task_id)
    
    def get_active_tasks_count(self) -> int:
        """获取当前活跃任务数量"""
        with self._lock:
            return len([f for f in self.futures.values() if not f.done()])
    
    def get_pending_tasks(self) -> List[TaskInfo]:
        """获取所有挂起的任务信息"""
        with self._lock:
            return [
                info for info in self.task_info.values()
                if info.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]
    
    def health_check(self) -> dict:
        """
        返回工作器健康状态
        
        Returns:
            健康状态字典
        """
        active_tasks = self.get_active_tasks_count()
        queued_tasks = max(0, self.executor._work_queue.qsize())  # type: ignore
        
        return {
            "status": "shutdown" if self._shutdown else "healthy",
            "active_workers": self.executor._max_workers,
            "queued_tasks": queued_tasks,
            "active_tasks": active_tasks,
            "total_tasks_tracked": len(self.task_info),
            "metrics": self.metrics.get_stats(),
            "uptime": time.time() - getattr(self, '_start_time', time.time()),
            "thread_count": threading.active_count()
        }
    
    @contextmanager
    def temporary_timeout(self, timeout: float):
        """
        临时超时上下文管理器
        
        在此上下文中，所有操作将使用指定的超时
        
        Args:
            timeout: 超时时间(秒)
        """
        original_timeout = getattr(self, '_default_timeout', None)
        self._default_timeout = timeout
        try:
            yield
        finally:
            self._default_timeout = original_timeout
    
    def get_metrics(self) -> dict:
        """获取性能指标"""
        return self.metrics.get_stats()
    
    def cleanup_old_tasks(self, max_age_hours: float = 1.0) -> int:
        """
        清理旧任务信息
        
        Args:
            max_age_hours: 保留任务信息的最大小时数
        
        Returns:
            被清理的任务数量
        """
        max_age_seconds = max_age_hours * 3600
        cleanup_time = time.time() - max_age_seconds
        removed = 0
        
        with self._lock:
            for task_id in list(self.task_info.keys()):
                task = self.task_info[task_id]
                if task.complete_time and task.complete_time < cleanup_time:
                    # 只清理已完成超过max_age_hours的任务
                    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                        del self.task_info[task_id]
                        removed += 1
        
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old tasks")
        
        return removed
    
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """
        关闭工作器
        
        Args:
            wait: 是否等待完成的任务
            cancel_futures: 是否取消未完成的任务
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            
            logger.info("Shutting down IndexerWorker...")
            self._shutdown = True
            
            if cancel_futures:
                self.cancel_all_tasks()
            
            # 关闭线程池
            self.executor.shutdown(wait=wait)
            
            # 清理资源
            with self._lock:
                self.futures.clear()
                # 保留最近的任务信息用于诊断
                recent_tasks = {}
                cutoff_time = time.time() - 300  # 保留最近5分钟的任务
                for task_id, info in self.task_info.items():
                    if info.submit_time >= cutoff_time:
                        recent_tasks[task_id] = info
                self.task_info = recent_tasks
            
            logger.info("IndexerWorker shutdown complete")
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        self.shutdown(wait=True, cancel_futures=True)
        return False


class CallbackIndexerWorker(ConcurrentIndexerWorker[T]):
    """支持回调的索引器工作器"""
    
    def __init__(self, base_indexer: Any, num_workers: Optional[int] = None):
        super().__init__(base_indexer, num_workers)
        self.callbacks: Dict[str, Tuple[Optional[Callable], Optional[Callable]]] = {}
    
    def search_with_callback(self, 
                             query: str, 
                             k: int = 5, 
                             callback: Optional[Callable[[str, List[T]], None]] = None,
                             error_callback: Optional[Callable[[str, Exception], None]] = None) -> str:
        """
        带回调的异步搜索
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            callback: 成功回调函数 (task_id, result) -> None
            error_callback: 错误回调函数 (task_id, exception) -> None
        
        Returns:
            任务ID
        
        Raises:
            RuntimeError: 工作器已关闭
        """
        if self._shutdown:
            raise RuntimeError("IndexerWorker has been shutdown")
        
        task_id = str(uuid.uuid4())
        
        # 记录回调
        with self._lock:
            self.callbacks[task_id] = (callback, error_callback)
        
        # 提交任务
        future = self.executor.submit(self._safe_search, query, k)
        
        # 记录任务信息
        with self._lock:
            self.futures[task_id] = future
            self.task_info[task_id] = TaskInfo(
                task_id=task_id,
                query=query,
                status=TaskStatus.PENDING,
                submit_time=time.time()
            )
        
        # 添加完成回调
        def done_callback(f):
            result = None
            exception = None
            
            try:
                result = f.result()
            except Exception as e:
                exception = e
            
            # 获取回调函数
            with self._lock:
                cb, err_cb = self.callbacks.pop(task_id, (None, None))
                task_info = self.task_info.get(task_id)
            
            # 更新任务状态
            if task_info:
                if exception:
                    task_info.status = TaskStatus.FAILED
                    task_info.error = str(exception)
                else:
                    task_info.status = TaskStatus.COMPLETED
                task_info.complete_time = time.time()
            
            # 调用回调
            if exception and err_cb:
                try:
                    err_cb(task_id, exception)
                except Exception as e:
                    logger.error(f"Error callback failed for task {task_id}: {str(e)}", exc_info=True)
            elif result and cb:
                try:
                    cb(task_id, result)
                except Exception as e:
                    logger.error(f"Success callback failed for task {task_id}: {str(e)}", exc_info=True)
            
            # 清理future
            with self._lock:
                self.futures.pop(task_id, None)
        
        future.add_done_callback(done_callback)
        
        return task_id
    
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """关闭工作器，清理回调"""
        with self._lock:
            # 取消所有有回调的任务
            for task_id in list(self.callbacks.keys()):
                if cancel_futures:
                    self.cancel_task(task_id)
                # 清理回调
                self.callbacks.pop(task_id, None)
        
        super().shutdown(wait, cancel_futures)

    @staticmethod
    # 使用示例
    def usage_example():
        """使用示例"""
        # 模拟索引器
        class DummyIndexer:
            def __init__(self):
                self.counter = 0
            
            def topk_search(self, query, k):
                self.counter += 1
                # 模拟不同延迟
                time.sleep(min(0.5, len(query) * 0.01))
                return [f"result_{self.counter}_{i}" for i in range(k)]
        
        # 创建工作器
        indexer = DummyIndexer()
        
        with CallbackIndexerWorker(indexer, num_workers=3) as worker:
            # 1. 同步搜索
            print("=== 同步搜索 ===")
            result = worker.search("hello world", k=3, timeout=1.0)
            print(f"同步结果: {result}")
            
            # 2. 异步搜索
            print("\n=== 异步搜索 ===")
            task_id, future = worker.search_async("async query", k=2)
            print(f"提交了异步任务: {task_id}")
            # 模拟做其他工作
            time.sleep(0.1)
            if future.done():
                print(f"异步结果: {future.result()}")
            else:
                print("异步任务仍在运行...")
            
            # 3. 带回调的搜索
            print("\n=== 带回调的搜索 ===")
            def success_cb(task_id, result):
                print(f"✅ 任务 {task_id} 成功: {result}")
            
            def error_cb(task_id, error):
                print(f"❌ 任务 {task_id} 失败: {error}")
            
            cb_task_id = worker.search_with_callback(
                "callback query", 
                k=2,
                callback=success_cb,
                error_callback=error_cb
            )
            print(f"提交了回调任务: {cb_task_id}")
            time.sleep(0.3)  # 等待回调执行
            
            # 4. 批量搜索
            print("\n=== 批量搜索 ===")
            queries = ["python", "java", "c++", "rust", "go"]
            results = worker.batch_search(queries, k=2, timeout=2.0)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    print(f"查询 '{queries[i]}' 失败: {res}")
                else:
                    print(f"查询 '{queries[i]}' 结果: {res}")
            
            # 5. 任务控制
            print("\n=== 任务控制 ===")
            pending_task_id = worker._generate_task_id()
            future = worker.executor.submit(lambda: time.sleep(2.0) or ["long_result"])
            
            with worker._lock:
                worker.futures[pending_task_id] = future
                worker.task_info[pending_task_id] = TaskInfo(
                    task_id=pending_task_id,
                    query="long running query",
                    status=TaskStatus.RUNNING,
                    submit_time=time.time()
                )
            
            print(f"创建了长时间任务: {pending_task_id}")
            time.sleep(0.5)
            
            # 取消任务
            if worker.cancel_task(pending_task_id):
                print(f"成功取消任务: {pending_task_id}")
            else:
                print(f"取消任务失败: {pending_task_id}")
            
            # 6. 健康检查
            print("\n=== 健康检查 ===")
            health = worker.health_check()
            print(f"状态: {health['status']}")
            print(f"活跃任务: {health['active_tasks']}")
            print(f"队列任务: {health['queued_tasks']}")
            print(f"指标: {health['metrics']}")
            
            # 7. 清理旧任务
            cleaned = worker.cleanup_old_tasks(max_age_hours=0.001)  # 3.6秒
            print(f"\n清理了 {cleaned} 个旧任务")



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

class AERRState:
    """某一个状态下的query_list和document_list"""
    def __init__(self):
        """这里不保存done的部分，也不保存耗时部分，仅仅保存状态"""
        self.query_list = []
        self.document_list = []

        # 记录模型的输入输出
        self.model_input = ''
        self.model_output = ''

        self.action = []
    
    def get_state(self):
        """需要获取的状态"""
        return 

    def to_snapshot(self):
        return {
        "query_list": self.query_list,
        "document_list": self.document_list,
        "model_input": self.model_input, 
        "model_output": self.model_output 
    }
    
class AERRStateChain:
    """单条交互轨迹的保存"""
    def __init__(self, user_input: str):
        # 初始化第一个点
        self.state_chain: List["AERRState"] = []
        self.user_input: str = user_input

        new_state = AERRState()
        new_state.model_input = PLACE_HOLDER # 使用占位符，方便后续判断
        new_state.model_output = PLACE_HOLDER
        new_state.query_list.append(self.user_input)
        self.state_chain.append(new_state)

        # 交互的轮数
        self.interaction_index = 0
        # 整条轨迹的情况
        self.done = False
        self.time_cost = 0
        # 当前状态
        self.curr_query_list = [user_input]
        self.curr_document_list = []
    
    def update_decision(self, model_input_prompt, model_output_prompt, time_cost, activate_format_filter = False):
        """更新历史状态"""
        if self.done == True:
            return 

        new_state = AERRState()
        new_state.model_input = model_input_prompt
        new_state.model_output = AERRTemplate.DecisionResponseValidPartExtract(model_output_prompt) if activate_format_filter else model_output_prompt
        new_state.query_list.append(self.user_input) # 不可以哦，每次都需要设置

        # 更新动作链的状态
        self.time_cost += time_cost
        self.state_chain.append(new_state)

    def update_execution(self, query_list, document_list, done, time_cost, action):
        """对列表末端进行修改，更新历史状态"""
        if self.done == True:
            return 
        self.state_chain[-1].query_list = copy.deepcopy(query_list)
        self.state_chain[-1].document_list = copy.deepcopy(document_list)
        self.state_chain[-1].action = copy.deepcopy(action)
        # 更新动作链的状态
        self.done = copy.deepcopy(done)
        self.time_cost += time_cost

        # 更新当前状态
        self.curr_query_list = query_list
        self.curr_document_list = document_list
    
    def get_training_data(self, need_init_prompt = False) -> Tuple[List[List], List[List]]:
        """获取训练数据，以model_inputs和model_outputs返回，训练数据形式为List[List], List[List]，每一个元素为动作链"""

        model_inputs = []
        model_outputs = []
        for state in self.state_chain:
            if state.model_input != PLACE_HOLDER and state.model_output != PLACE_HOLDER: 
                model_inputs.append(AERRTemplate.decision_model_init_prompt() + state.model_input if need_init_prompt else state.model_input)
                model_outputs.append(state.model_output)

        return model_inputs, model_outputs

    def get_actions(self):
        """获取动作链"""
        return [state.action for state in self.state_chain]

class AERRStateManager:
    """状态管理器"""
    curr_save_idx = 0

    def __init__(self, input_prompts: List):
        # 获取input_prompts
        self.all_states = [AERRStateChain(input_prompts[i]) for i in range(len(input_prompts))]
        self.all_done = False
        self.document_prompt_builder = AERRDocumentProfileBuilder()

    def get_state(self):
        """返回当前的状态"""

        # 注意这里其实是位置索引传出，所以会直接对列表进行修改，满足接口要求
        query_lists = [chain.curr_query_list for chain in self.all_states if chain.done == False]
        document_lists = [chain.curr_document_list for chain in self.all_states if chain.done == False]
        done_flags = [chain.done for chain in self.all_states if chain.done == False]

        return query_lists, document_lists, done_flags

    def update_decision(self, model_input_prompts, model_output_prompts, time_lis):
        """
        更新所有的decision，仅对上一次动作未终止的结果进行更新
        """

        valid_chains = [chain for chain in self.all_states if chain.done == False]

        # 添加长度校验
        if len(valid_chains) != len(model_input_prompts):
            raise ValueError("输入参数长度与有效链数量不匹配")
        
        for index, chain in enumerate(valid_chains):
            # 需要反过来分配
            chain.update_decision(model_input_prompts[index], model_output_prompts[index], time_lis[index])

    def update_execution(self, query_lists, document_lists, done_flags, time_lis, actions):
        """更新所有末尾节点由decision创建的状态，仅对上一次动作未终止的结果进行更新"""
        valid_chains = [chain for chain in self.all_states if chain.done == False]

        # 添加长度校验
        if len(valid_chains) != len(query_lists):
            raise ValueError("输入参数长度与有效链数量不匹配")
        
        for index, chain in enumerate(valid_chains):
            chain.update_execution(query_lists[index], document_lists[index], done_flags[index], time_lis[index], actions[index])

        # 判断是否所有动作链都已经完成
        self.all_done = all([chain.done for chain in self.all_states])

    def get_decision_model_input(self) -> List[str]:
        """获取所有的decision model的输入，仅包含未完成的动作链"""

        valid_index = [index for index, chain in enumerate(self.all_states) if chain.done == False]
        decision_model_input_prompts = [AERRTemplate.decision_model_init_prompt() + \
                                        AERRTemplate.decision_model_input_prompts(self.all_states[index].curr_query_list, 
                                                                                  self.document_prompt_builder.build(self.all_states[index].user_input, 
                                                                                                                                     self.all_states[index].curr_document_list), 
                                                                                  self.all_states[index].get_actions(), 
                                                                                  self.all_states[index].user_input)
                                                                                  for index in valid_index]

        return decision_model_input_prompts
    
    def get_execution_model_input(self) -> List[str]:
        """获取所有的execution model的输入，仅包含未完成的动作链"""

        valid_index = [index for index, chain in enumerate(self.all_states) if chain.done == False]

        # 注意，这里是已经经过清洗的
        decision_responses = [copy.deepcopy(self.all_states[index].state_chain[-1].model_output) for index in valid_index]
        query_lists = [copy.deepcopy(self.all_states[index].curr_query_list) for index in valid_index]
        document_lists = [copy.deepcopy(self.all_states[index].curr_document_list) for index in valid_index]
        done_flags = [False for index in valid_index] # 仅为False

        return decision_responses, query_lists, document_lists, done_flags

    def get_final_input(self) -> List[str]:
        """获取最终的llm输入"""
        final_input_prompts = [AERRTemplate.get_final_LLM_input(user_input = chain.user_input, 
                                document_list = chain.curr_document_list) for chain in self.all_states]
        return final_input_prompts
    
    def get_training_data(self, need_init_prompt = False, update_save_idx = True):
        """获取所有的训练数据"""
        
        # 保存索引的更新
        if update_save_idx: 
            AERRStateManager.curr_save_idx += 1

        model_inputs: List[List[str]] = []
        model_outputs: List[List[str]] = []
        time_lis: List[float] = []

        for chain in self.all_states:

            model_input, model_output = chain.get_training_data(need_init_prompt = need_init_prompt)
            # 对齐
            model_inputs.append(model_input)
            model_outputs.append(model_output)
            time_lis.append(chain.time_cost)

        return model_inputs, model_outputs, time_lis
    
    def to_csv(self, output_dir: str, ckpt = '', save_interval = 10):

        if not (AERRStateManager.curr_save_idx % save_interval == 0 or AERRStateManager.curr_save_idx == 1):
            return
        
        result = []
        model_inputs, model_outputs, time_lis = self.get_training_data(update_save_idx = False) 
        for batch_id, model_input_batch in enumerate(model_inputs):
            for interaction_id, model_input in enumerate(model_input_batch):
                result.append({"batch_ids": str(batch_id) + ', ' + str(interaction_id), 
                               "model_input": model_inputs[batch_id][interaction_id], 
                               "model_output": model_outputs[batch_id][interaction_id], 
                               "time_cost": self.all_states[batch_id].time_cost})

        if ckpt == '':
            ckpt = str(AERRStateManager.curr_save_idx)

        save_name = "TrainingDataHistory" + f"_ckpt_{ckpt}_"+ datetime.datetime.now().strftime("%m-%d_%H-%M-%S") +".json"
        with open(output_dir + save_name, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def finish(self):
        """将所有的动作链都标记为True"""
        self.all_done = True
        for chain in self.all_states:
            chain.done = True

class AERRDocumentProfileBuilder:
    """用以将文档改写为Profile，就像打游戏一样，显示其缩略描述，避免大量无关噪音"""
    # =======================================
    # 如果Document Filter能达到要求，那么其实也不需要进行Profile的提供了
    # 说到底，我们只是需要裁剪模型的上下文窗口，保证 【模型的输入长度不能超过某一个阈值】 
    # =======================================
    def __init__(self):
        self.model = Large_Language_Model_API()
        self.model.init_llm('')
        pass

    def build(self, user_input, document_list) -> str:
        """通过API来构建Profile"""
        # 没必要做正则，做正则太慢了
        # 直接返回prompt就好了

        if len(document_list) == 0:
            return "Nothing Yet."
        
        try:
            document_profile_prompt = AERRTemplate.Document_profile_prompts(user_input, document_list)
            return AERRTemplate.Document_profile_Extractor(self.model.generate(document_profile_prompt)[0])
        except Exception as e:
            logger.warning("AERRDocumentProfileBuilder Extract Document Profile Fail...Repalce profile by raw document list...")
            return AERRTemplate.Document_profile_fail_prompt(document_list)

class AERRTemplate:
    """提示词模板，都放在这个里面吧，把这些都分离出来"""

    @staticmethod
    def decision_model_init_prompt():
        """
        该方法用以交代清楚prompt的构成
        
        """
        prompt = f"""
**You are a decision-making agent that can freely use the following tools to optimize document retrieval and ensure only the most relevant, high-quality results are included in your final output. The final response will be derived directly from the processed document list, so rigorous filtering, sorting, and refinement are critical.**

**Guideline:**
- Only high revelant document matters: Immediately reject any document that doesn't directly address the core query with clear evidence. Never include borderline-relevance documents—aim for 3-5 documents with 90%+ relevance confidence.
- Always conduct thorough pre-action analysis before using any tool. **Any action without analysis will be ignored. ** Explicitly state your reasoning in think section, including why the current approach is insufficient, how the chosen tool resolves this, and expected outcome. Never proceed without this step.
- When standard retrieval (e.g., [Query Search]) fails to yield higher-relevance documents, proactively use [Query Rewrite] or via API to generate new search angles. This is the critical escalation path when relevance plateaus—do not stop at the first 5 results.
- **Limit actions to exactly 3 per interaction; Limit interactions to exactly 4 per user's input; any actions beyond the third will be automatically ignored.**
- Action History is given on each input. Check the action history and avoid repeating actions as they are less valuable. 

**Tools:**
Basic:
1. [Query Search]: Retrieve exactly **3** documents from the target query. 
    - Note: Using the same query multiple times will return identical documents. To retrieve documents from different angles, first use [Query Rewrite] to generate refined queries.
2. [Delete Documents]: Delete target documents directly when document is low relevance. Supports multiple document indices. 
3. [Sort Documents]: Rearrange the document list by specifying the desired order of documents.
4. [Delete Query]: Delete irrelevant target queries from the query list. 
5. [Stop]: Terminate interaction when user input is clear enough for final answer. This provides the last opportunity to refine output before termination.

Advanced:
6. [Query Rewrite]: Generate multiple refined queries for the use inputs using LLM API and retrieved **1** document for each query. 
7. [Document Analysis]: Analyze initial documents to identify information gaps and missing context that require further retrieval; generate targeted follow-up queries and retrieve 1 document per query via LLM API to comprehensively address the user's needs.
8. [Document Filter]: Delete all irrelevant documents from list by LLM API. 
    - Note: ** Auto activate [Document Filter] if current document quatity more than 10 **
9. [Summarize Documents]: Generate a concise summary confirming whether the documents collectively answer the query. If the summary confirms a direct or indirect answer, delete all source documents and append the summary as a new document.

**Input Format Specification**
- Action History: Chronological list of previous actions.
    - Format: [Interaction ID | Action: Action type | Target ID: i, j]
- Query Collected: All queries with status.
    - Format: <Query 0> Query Context... <Query 1> Query Context...
- Documents: Retrieved documents with title and context.
    - Format: <Document 0> Title... <Document 1> Title...
- User Input: Original user query.

**Response Format Specification**
- Think: Provide thorough reasoning before taking any action. Include the rationale for the selected tool and expected outcome.
    - Format: <Think> ... </Think>
- Action: Choose which tag to take actions after. 
    - Format: <Action> Target IDs (or Document) | Action </Action>
        - Part 1: Target X, Y, Z (e.g., Target 1, 2, 3 for documents or queries; Target 0 for user input).
        - Part 2: Operation type (e.g., Query Search, Detail Search, Query Rewrite).
- End: Terminate the interaction immediately after this tag.
    - Format: <END>

**Example 1**
Action History: 
Nothing yet.
Query Collected: 
[Query 0] When was the first computer invented? 
Documents Retrieved: 
Nothing yet. 
User Input: 
When was the first computer invented? 

**Expected Output Example 1**: 
<Think> The input is straightforward, but direct Query 0 search may introduce noise with irrelevant terms. Rewriting the query first will improve retrieval precision for the exact invention date. </Think>
<Action> Target 0 | Query Rewrite </Action>
<Think> After Query Rewrite generated 5 refined queries, we initiate Query Search on Query 1, 2, 3 to find the exact computer invention date. </Think> 
<Action> Target 1, 2, 3 | Query Search </Action>
<Think> Query Search generated mixed relevance documents; Document Filter will eliminate noise. With all three action slots exhausted, we pause for state review before next action. </Think>
<Action> Target 0 | Document Filter </Action>
<END>

**Example 2**
Action History: 
[1. Action: Query Search | Target ID: 0] -> 1. [Action: Query Rewrite | Target ID: 0] -> 2. [Action: Query Search | Target ID: 2] -> 2. [Action: Document Filter | Target ID: 3, 4, 5] 
Query Collected: 
[Query 0] <Already Retrieved> How does artificial intelligence impact the healthcare industry?
[Query 1] AI healthcare impact
[Query 2] <Already Retrieved> Medical AI applications
[Query 3] AI in health services
[Query 4] Healthcare AI transformation
[Query 5] AI medical sector effects
Documents Retrieved: 
[Document 0] [Title] AI Healthcare Impact Analysis Prompt [Context] Analyze artificial intelligence's impact on healthcare, emphasizing diagnostic accuracy, personalized treatment, and operational efficiency. Include benefits like reduced costs and ethical challenges such as data privacy...
[Document 1] [Title] Urban Tech Integration Overview [Context] Explore the evolving role of digital systems in city environments, emphasizing connectivity and sustainability. Discuss general advancements in infrastructure and community engagement without specific case studies. Maintain a broad focus on societal adaptation to technological shifts in daily urban life.
[Document 2] [Title] Sustainable Gardening Practices Guide [Context] Outline eco-friendly gardening techniques, including composting, water conservation, and native plant selection. Focus on reducing chemical use and enhancing soil health. Highlight benefits like biodiversity support and lower environmental impact. Provide actionable tips for beginners to create thriving, low-maintenance gardens.
User Input: 
How does artificial intelligence impact the healthcare industry?

**Expected Output Example 2**: 
<Think> Document 0 directly addresses AI in healthcare with specific focus on diagnostics and ethics, but lacks depth on service applications. Action History shows Query 0 (original) and Query 2 (Medical AI applications) were already retrieved. To fill knowledge gaps in health services context, Query 3 ('AI in health services') needs to be searched next. This will provide complementary insights without redundancy. </Think>
<Action> Target 3 | Query Search </Action>
<Think> Document 1 (Urban Tech) and Document 2 (Gardening) are clearly irrelevant to healthcare AI queries. To systematically remove all irrelevant entries before final output, Document Filter is the optimal tool—it efficiently processes the entire list in one action, eliminating future redundancy without manual iteration. This aligns with the guideline to reject borderline-relevance documents immediately. </Think>
<Action> Target 1, 2 | Document Filter </Action>
<Think> After Document Filter action removed Documents 1 and 2, only Document 0 remains as highly relevant. With one action budget left and no further tool needed for verification, we avoid unnecessary actions. The next interaction will confirm Document 0's sufficiency before finalizing with Stop. </Think>
<END>

"""
        return prompt
    
    @ staticmethod
    def decision_model_complement_prompt():
        format_prompt = """
**Expected Output Example**: 
<Think> The query focuses on understanding the impact of artificial intelligence in the healthcare sector. Need a multi-faceted analysis that covers AI applications, benefits, challenges, and specific real-world examples. Also, the query could potentially leave out niche topics like ethical implications or lesser-known AI tools in healthcare. Thus, refining the query and exploring multiple angles would be beneficial. </Think>
<Action> Query 0 | Query Rewrite </Action>
<END>
**Please follow the format strictly!** 
Your Output: """.strip()
        return format_prompt

    @ staticmethod
    def decision_build_curr_action_state_prompts(actions: List[dict]) -> str:
        """该函数用以给出过往历史状态"""
        results = []
        for index, action_chain in enumerate(actions):
            for action in action_chain: # action的格式如下所示
                results.append('[' + f" {str(index)} | " +'Action: ' + action['action_type'] + ' | '  + 'Target ID: ' + ','.join([str(item) for item in action['target_id']]) + ']')
        
        if len(results) == 0:
            return "Nothing yet."
        return ' -> '.join(results)
    
    @ staticmethod
    def decision_model_input_prompts(query_list: List[str], document_prompt: str, actions: List[List[dict]], user_input: str) -> str:
        """构建当前决策状态的提示"""
        # 格式化已收集的查询
        queries_str = "\n".join([f"[Query {i}]: {q}" for i, q in enumerate(query_list)])
        
        # 格式化已检索的文档
        # docs_str = "\n".join([f"[Document {i}]: {d[:200]} ... " for i, d in enumerate(document_list)]) if len(document_list) != 0 else "Nothing yet."
        
        # 构建当前状态提示
        state_prompt = f"""
Current State:

**Actions History:**
{AERRTemplate.decision_build_curr_action_state_prompts(actions)}

**Queries Collected:**
{queries_str}

**Documents Retrieved:**
{document_prompt}

**User Input:**
**{user_input}**

Your Output: 
"""
        return state_prompt
    
    @ staticmethod
    def DecisionResponseValidPartExtract(model_output: str, max_action_budget = 3) -> bool:
        """
        从大模型提示词模板中提取Think Action Detail部分并拼接
        
        Args:
            prompt_text (str): 提示词模板文本
            
        Returns:
            str: 拼接后的Think Action Detail字符串
        """
        # 使用单个正则表达式匹配所有标签，按照出现的顺序
        pattern = r'<(Think|Action|Detail)>(.*?)</\1>'
        
        # 查找所有匹配的标签
        matches = re.findall(pattern, model_output, re.DOTALL)
        
        # 清理和拼接内容
        result_parts = []

        budget = 0
        last_tag_type = ''
        for tag_type, content in matches:
            if budget >= max_action_budget:
                break
            
            content = content.strip()
            content = re.sub(r'\s+', ' ', content)
            
            # 如果不存在内容，则直接跳过
            if not content: continue

            if tag_type == 'Think': # 如果重复，则弹出之前的Think部分，以最新的Think为准
                if last_tag_type == 'Think':
                    result_parts.pop(len(result_parts) - 1)
                result_parts.append(f"<Think> {content} </Think>")
                last_tag_type = tag_type

            # 强制模型在输出动作前必须进行思考，如果不进行思考，则动作不予考虑
            # Think -> Action -> Detail 
            if tag_type == 'Action' and last_tag_type == 'Think':
                result_parts.append(f"<Action> {content} </Action>")
                budget += 1
                last_tag_type = tag_type
            if tag_type == 'Detail' and last_tag_type == 'Action':
                result_parts.append(f"<Detail> {content} </Detail>")
                last_tag_type = tag_type

        # 将所有部分拼接成一个字符串
        final_result = '\n'.join(result_parts)

        if final_result != "":
            final_result += "\n<END>"
        
        return final_result

    @ staticmethod
    def DecisionPrompt2ActionDict(model_output: str, action_budget = 3) -> Dict:
        """
        从模型输出中提取<Think>、<Action>和<Detail>组件，并建立正确的对应关系
        
        Args:
            model_output (str): 模型返回的原始响应字符串
            
        Returns:
            Dict[str, List[Any]]: 包含提取组件的字典，确保Action和Detail正确对应
        """
        data = {
            'raw_string': model_output,
            'thinks': [],
            'actions': [],
            'details': []}
        
        # 提取并处理<Think>标签
        data['thinks'] = [t.strip() for t in re.findall(r'<Think>(.*?)</Think>', model_output, re.DOTALL)]
        
        # 使用finditer获取所有标签及其位置
        tag_pattern = r'<(Action|Detail)>(.*?)</\1>'
        
        # 存储所有标签及其位置和内容
        tags = []
        for match in re.finditer(tag_pattern, model_output, re.DOTALL):
            tag_type = match.group(1)
            content = match.group(2).strip()
            start_pos = match.start()
            tags.append({
                'type': tag_type,
                'content': content,
                'position': start_pos
            })
        
        # 按照位置排序
        tags.sort(key=lambda x: x['position'])
        
        # 建立Action和Detail的对应关系
        actions_with_details = []
        details_used = set()
        
        # 第一遍：寻找紧跟在Action后面的Detail
        curr_action_index = 0
        for i, tag in enumerate(tags):

            # 仅允许进行k次动作
            if curr_action_index >= action_budget: break

            if tag['type'] == 'Action':
                # 尝试找到这个Action对应的Detail
                detail = ''
                
                # 检查下一个标签是否是Detail
                if i + 1 < len(tags) and tags[i+1]['type'] == 'Detail':
                    detail = tags[i+1]['content']
                    details_used.add(i+1)
                
                # 解析Action
                try:
                    parts = [p.strip() for p in tag['content'].split('|')]
                    target_id_match = re.findall(r'(\d+)', parts[0])
                    target_ids = [int(num) for num in target_id_match] if target_id_match else []
                    action_type_string = parts[1]
                    
                    actions_with_details.append({
                        'target_id': target_ids,
                        'action_type': action_type_string,
                        'detail': detail
                    })

                    curr_action_index += 1
                except Exception as e:
                    # print(f"解析Action时出错: {e}") 暂时不打印就行，这里没有问题
                    continue
        
        # 第二遍：处理剩余的Detail（没有被第一遍匹配到的）
        remaining_details = []
        for i, tag in enumerate(tags):
            if tag['type'] == 'Detail' and i not in details_used:
                remaining_details.append(tag['content'])
        
        # 第三遍：尝试将剩余的Detail分配给没有Detail的Action
        for action in actions_with_details:
            if action['detail'] is None and remaining_details:
                action['detail'] = remaining_details.pop(0)
        
        # 更新数据
        data['actions'] = actions_with_details
        data['details'] = [tag['content'] for tag in tags if tag['type'] == 'Detail']
        
        return data

    @ staticmethod
    def Execution_Build_Action_Dict(target_ids: List, action_type_string: str, detail: str) -> Dict:
        """构建一个action字典"""
        return {
                'target_id': target_ids,
                'action_type': action_type_string,
                'detail': detail
                }
    
    @staticmethod
    def Execution_query_rewrite_input_prompt(user_input: str) -> str:
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
Please generate **5** related topics based on the input query. 
Ensure the generated topic is **clearer**, **more accurate**, and **more comprehensive** than the original input query.
The topics should expand the original scope while maintaining relevance. 
Follow the format strictly and ensure each topic starts with "<Query>". 
End the output with the "<END>" tag immediately after the last topic.
**You can put your thoughts in <Think> tag, but make sure to generate enough queries in <Query> tags.** 

Format Example:
User's Input: "When was the computer invented?"
Expected Output: 
<Think> The user's query centers on the historical timeframe of computer invention, so the strongest query directly specifies the key period "1940s" to answer the core need; for other directions, I'm expanding to Charles Babbage's analytical engine for foundational inventors, ENIAC as the first electronic computer for technological milestones, the transistor revolution for critical innovations, and personal computer development for modern progression, ensuring all topics are concise, relevant, and avoid vagueness.
<Query> Computer invention timeline 1940s
<Query> Charles Babbage analytical engine
<Query> ENIAC first electronic computer
<Query> Transistor computer revolution
<Query> Personal computer development history
<END>

Rules:
1. Each query must start with "<Query>".
2. Remove all redundant words (e.g., "the", "a", "how", "what", "when").
3. Keep only core entities: [Subject] + [Key Aspect] (e.g., "Computer invention", "Analytical engine").
4. Avoid vague or overly broad topics.
5. Ensure topics cover different angles.
6. **At least one generated query must be strongly related to the core subject of the input query, directly tied to the user's answer, or be a keyword/key retrieval term from the user's answer.**
7. **For time-related queries (like "when"), include specific time periods or historical context.**
8. **For definition queries, include core components and related concepts.**

User's Input: {user_input}
Your Output: 
    """.strip()

        return prompt

    @ staticmethod
    def Execution_document_analysis_input_prompt(user_input: str, document_list: list) -> str:
        """
        通过给定的query，提问和文章，提取关键的信息，总结得到新的Query

        Args:
            problem (str): 需要解决的问题或任务。
            method (str): 推理方法（支持 "CoT" 或 "ToT"）。
            steps (int): 需要生成的推理步骤数或路径数。

        Returns:
            str: 完整的提示模板。
        """
        valid_docs = document_list
        docs_str = "\n".join([f"<Document {i}>: {d[:200]}" for i, d in enumerate(valid_docs)]) if len(valid_docs) != 0 else "Nothing yet."
        
        prompt =  f"""
Based on the input query and provided documents, generate **3~5** new queries that identify specific gaps in the document's coverage. 

**Guidelines**:
1. Focus on identifying document limitations and missing information:
   - Missing data points, time periods, or regional coverage
   - Specific entities needed for deeper analysis (crop, location, time)
   - Additional information required to answer the user's input
   - Actionable research directions based on identified 
   - Ignore irrelevant documents 

2. Query formulation rules:
   - Must reference *specific missing elements* from documents (e.g., "rice planting area" not "climate impacts")
   - Exclude generic terms like "impact", "effect", "challenge"
   - Include only core entities (crop, location, time) and avoid vague concepts
   - Never reference the original input query in your output

**Format Requirements**:
- Each query MUST be preceded by a <Think> section that provides detailed, document-grounded reasoning
- The reasoning in <Think> must be thorough and comprehensive, explaining exactly what information is missing and why it's needed
- Do NOT fabricate or assume information not present in the documents
- Follow the exact pattern for each of the 5 queries: 
    - <Think> ...  
    - <Query> ...
    - ...
- End your output with <END>

**Example 1**:
Documents:
<Document 0> [Title] Northeast China Rice Production [Context] 2022 rice yield decreased 3% due to temperature shifts. No 2025 projections or Southeast Asia comparisons.
<Document 1> [Title] China Corn Yield Study [Context] 2022 corn loss 5% from spring cold. No rice data or regional comparisons.
<Document 2> [Title] Southeast Asia Rice Trends [Context] 2020-2023 rice production increased 2%. No China data or climate correlation analysis.
User's Input: 
"How does climate change affect agriculture?"

Expected Output:
<Think> Document 0 mentions 2022 rice yield but lacks 2025 projections and regional comparisons. Need to query for future projections and cross-regional data.
<Query> 2025 Northeast China rice yield projections compared to Southeast Asia
<Think> Document 1 focuses on corn but lacks rice production data. Need comparative analysis between different crop types in the same region.
<Query> 2022 China rice and corn production data by province
<Think> Document 2 covers Southeast Asia but lacks China-specific data. Need to fill the gap for comprehensive regional analysis.
<Query> 2020-2023 China rice production monthly data
<Think> Documents 0, 1 and 2 all mention climate factors but collectively lack specific temperature and precipitation correlations with yield across different crops and regions. This cross-document gap indicates a need for comprehensive climate-yield correlation data.
<Query> Temperature precipitation correlation with rice and corn yield 2020-2023 Northeast China Southeast Asia
<Think> Documents 0, 1 and 3 collectively highlight a gap in understanding how irrigation practices interact with climate factors across different crops. Document 3 provides irrigation data but lacks crop-specific details, while Documents 0 and 1 mention crops but lack irrigation information.
<Query> Rice corn irrigation water requirements temperature variations North China 2022
<END>

**Example 2**:
Documents:
<Document 0> [Title] Solar Panel Efficiency [Context] 2023 solar panel efficiency reached 25%. Discusses photovoltaic technology advancements. No battery data.
<Document 1> [Title] Hydrogen Fuel Cells [Context] Hydrogen fuel cell vehicles in Europe 2024. Focus on infrastructure challenges. No lithium-ion battery comparison.
<Document 2> [Title] Automotive Safety Standards [Context] 2024 EU vehicle safety regulations update. Covers crash test requirements. No battery safety specifics.
<Document 3> [Title] Electric Vehicle Charging [Context] US charging station growth 2023-2024. Installation rates and coverage. No battery technology details.
<Document 4> [Title] Lithium-ion Battery Chemistry [Context] NMC 811 cathode performance in lab conditions 2023. Energy density 280Wh/kg. No real-world vehicle data or cost analysis.
<Document 5> [Title] Solid-state Battery Development [Context] Toyota solid-state battery prototype 2024. Claims 500km range. No production timeline or manufacturing challenges.
<Document 6> [Title] EV Market Trends [Context] Global EV sales growth 2022-2023. Market share by region. Limited battery technology details.
<Document 7> [Title] Battery Recycling Methods [Context] Pyrometallurgical recycling efficiency rates. 2023 recovery percentages. No specific cell chemistry comparisons.

User's Input: 
"What are the latest advancements in electric vehicle battery technology?"

Expected Output:
<Think> Documents 0-3 are not directly relevant to EV battery technology: Document 0 covers solar panels, Document 1 focuses on hydrogen fuel cells, Document 2 discusses safety standards without battery specifics, and Document 3 addresses charging infrastructure. Relevant documents are 4-7: Document 4 covers lithium-ion chemistry, Document 5 discusses solid-state batteries, Document 6 provides market context but limited battery details, and Document 7 addresses recycling. Analysis will focus on these relevant documents.
<Query> NMC 811 battery real-world vehicle performance data 2023-2024
<Think> Document 5 mentions solid-state prototype but provides no production timeline or manufacturing scalability details. Need commercialization roadmap to understand practical implementation.
<Query> Solid-state battery mass production timeline 2024-2026 manufacturing capacity
<Think> Documents 4 and 5 collectively highlight a gap in understanding how different battery technologies perform under real-world conditions. Document 4 focuses on lab performance of NMC 811, while Document 5 discusses solid-state prototypes, but neither provides comparative data or real-world degradation metrics across different operating conditions.
<Query> NMC 811 solid-state battery performance comparison real-world conditions 2023-2024
<Think> Documents 4, 5 and 7 reveal a significant knowledge gap in the lifecycle environmental impact of next-generation batteries. Document 7 covers recycling but lacks specific data on newer chemistries mentioned in Documents 4 and 5, preventing comprehensive sustainability assessment.
<Query> NMC 811 solid-state battery lifecycle environmental impact recycling efficiency
<Think> Documents 4, 5 and 6 collectively lack integration between technical battery specifications and market adoption factors. Document 6 provides market trends but without correlation to the specific battery technologies discussed in Documents 4 and 5, missing crucial cost-performance adoption drivers.
<END>

**Your Task**: 
**Documents**:
{docs_str}

**User's Input**: 
**{user_input}**
Your Output: 
""".strip()
        return prompt
    
    @staticmethod
    def Execution_document_filter_input_prompt(user_input: str, documents: List[str]) -> str:
        """
        根据用户查询和文档内容，判断文档是否与查询相关，并输出过滤结果。

        Args:
            documents (str): 文档内容（格式：[Title] ... [Context] ...）

        Returns:
            str: 完整的提示模板。
        """
        docs_str = "\n".join([f"<Document {i}>: {d[:]}" for i, d in enumerate(documents)]) if len(documents) != 0 else "Nothing yet."
        prompt = f"""
Based on the user input and the provided documents, determine if each document is relevant to the query.

**Document Relevance Criteria**

**RELEVANT (Keep) - Document meets ONE of these criteria:**
1. **Direct Answer**: Provides information that directly answers the user's question
2. **Causal Explanation**: Explains how/why something happens in relation to the query
3. **Quantitative Evidence**: Provides statistical data, measurements, or quantified results
4. **Supporting Evidence**: Provides examples, cases, or concepts that support understanding
5. **Background Context**: Provides useful background that helps understand the topic
6. **Multi-hop Evidence**: Contains pieces that can be combined with other documents to answer complex queries

**IRRELEVANT (Discard) - Document falls under ONE of these categories:**
1. **Completely Unrelated**: No meaningful connection to query topic
2. **Vague/Off-topic**: Only superficial keyword mentions without substance
3. **Wrong Domain**: Discusses entirely different subject area (e.g., medical doc for finance query)

**Decision Guidelines**
1. **First Pass**: Immediately discard only clearly irrelevant documents (criteria 1-3 above)
2. **Value Assessment**: For remaining documents, ask: "Would this help a human understand the topic better?"
3. **Knowledge Network**: For complex queries, ensure retained documents cover different aspects needed for complete answer
4. **Adaptive Retention**: 
   - Keep 4-5 documents if query contains: how, why, impact, effect, analyze, compare, strategies
   - Keep 2-3 documents if query contains: when, where, who, what is, define

**Response Format Specification**
- <Think> [Required] Provide a detailed analysis of EACH document, clearly stating whether it is relevant or irrelevant and why. Identify the 2~3 most relevant documents to retain. Be thorough in your reasoning.
- <Irrelevant Document> [Required] List indices (0-based) of irrelevant documents, comma-separated (e.g., "1, 2")

**Example 1**:
Documents: 
<Document 0> [Title] The Invention of the First Electronic Computer [Context] The first general-purpose electronic digital computer, ENIAC, was completed in 1945 at the University of Pennsylvania and publicly unveiled on February 15, 1946. It was designed by J. Presper Eckert and John Mauchly, using 17,468 vacuum tubes.
<Document 1> [Title] Early Mechanical Calculating Devices [Context] Charles Babbage's Analytical Engine (1837) and Herman Hollerith's tabulating machine (1890) were precursors to electronic computers, but these mechanical devices did not use electronic components for computation.
<Document 2> [Title] Evolution of Computer Hardware Components [Context] The transition from vacuum tubes to transistors in the 1950s marked a key shift in computer design, enabling smaller and more reliable machines, though this hardware evolution occurred decades after the first electronic computers.
<Document 3> [Title] Personal Computer Adoption in the 1980s [Context] The Apple II and IBM PC popularized personal computing in the 1980s, transforming computers from specialized research tools into consumer devices, but this development occurred long after the initial invention.
<Document 4> [Title] The Development of the Steam Engine [Context] Thomas Newcomen invented the steam engine in 1712, with James Watt's improvements in the 1760s powering the Industrial Revolution, unrelated to computing technology.
<Document 5> [Title] World War II's Role in Computer Development [Context] Military demands for rapid computation during World War II accelerated electronic computer research, leading to projects like ENIAC, though the exact timeline of the first completed machine remains historically debated.

User's Input: When was the computer invented? 
Expected Output:
<Think> Document 0 directly provides the invention date of ENIAC in 1945, precisely answering the query. Document 1 discusses precursors to computers, providing useful historical context. Document 2 covers hardware evolution, which is related to computer development. Document 3 discusses later adoption but still relevant to computer history. Document 4 is completely unrelated to computing. Document 5 provides relevant historical context about computer development during WWII. Keeping Documents 0, 1, 2, 3, and 5 as they all provide valuable context about computer history and development.</Think>
<Irrelevant Document> 4

**Example 2 (Multi-hop Factual Retrieval)**:
Documents: 
<Document 0> [Title] Einstein's Annus Mirabilis Papers [Context] In 1905, Albert Einstein published four groundbreaking papers including his special theory of relativity and the photoelectric effect paper that would later earn him the Nobel Prize. These papers revolutionized physics while he was working as a patent clerk in Bern, Switzerland.
<Document 1> [Title] Nobel Prize in Physics 1921 [Context] Albert Einstein was awarded the Nobel Prize in Physics in 1921 specifically for his explanation of the photoelectric effect, not for his theory of relativity. The Nobel Committee recognized his services to Theoretical Physics and especially for his discovery of the law of the photoelectric effect.
<Document 2> [Title] Einstein's Academic Career [Context] After his annus mirabilis in 1905, Einstein held positions at the University of Zurich, Charles University in Prague, and finally settled at the Kaiser Wilhelm Institute in Berlin from 1914 to 1932, where he completed his general theory of relativity in 1915.
<Document 3> [Title] Quantum Mechanics Development [Context] Niels Bohr made fundamental contributions to understanding atomic structure and quantum theory, for which he received the Nobel Prize in Physics in 1922. His work complemented Einstein's photoelectric effect in establishing quantum theory.
<Document 4> [Title] Marie Curie's Nobel Achievements [Context] Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields - Physics (1903) and Chemistry (1911). Her work on radioactivity laid foundations for nuclear physics.
<Document 5> [Title] Einstein's Later Life in America [Context] After emigrating to the United States in 1933 due to the rise of Nazi Germany, Einstein accepted a position at the Institute for Advanced Study in Princeton, where he worked until his death in 1955, focusing on unified field theory.

User's Input: 
What was Einstein awarded the Nobel Prize for and in which year?
Expected Output:
<Think> This is a multi-hop factual query requiring information from multiple documents. Document 1 directly states that Einstein received the Nobel Prize in 1921 for his explanation of the photoelectric effect. Document 0 provides important context about when he published the photoelectric effect paper (1905) but doesn't mention the Nobel award year. Document 2 discusses his academic career timeline but doesn't contain the specific Nobel information needed. Document 3 mentions Niels Bohr's Nobel Prize but not Einstein's. Document 4 discusses Marie Curie's Nobel achievements, which are unrelated. Document 5 covers Einstein's later life but not his Nobel Prize. For this multi-hop query, I need to keep Document 1 (direct answer) and Document 0 (provides context about the photoelectric effect work). The other documents don't contain the specific information needed to answer this two-part question.</Think>
<Irrelevant Document> 2, 3, 4, 5

**Your Task:**
**Documents:**
{docs_str}

**User's Input:**
{user_input}

**Your Output:**
""".strip()
        return prompt

    @ staticmethod
    def Execution_document_summary_input_prompt(user_input: str, documents: List[str]) -> str:
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
<Document Summary> Urban green spaces reduce heat island effects through: 2°C cooling from tree canopy, 1.5°C pavement synergy, 1.2°C NYC implementation, 1.8°C evaporative cooling, and 0.9°C policy mandates.

Documents:
{docs_str}

**User's Input**: 
{user_input}

Your Output: 
""".strip()
        return prompt

    @ staticmethod
    def Execution_parse_query_output(output: str) -> List[str]:
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
    
        return result

    @ staticmethod
    def Execution_parse_irrelevant_documents(model_output: str) -> List[str]:
        """
        从模型输出中提取不相关的文档索引，转换为字符串列表

        Args:
            model_output (str): 模型输出的原始字符串（包含 <Irrelevant Document> 标签）

        Returns:
            List[str]: 不相关文档的索引字符串列表，按数字大小排序
        """
        # 1. 修正正则表达式：精确匹配标签后的内容（包括逗号和空格）
        pattern = r'<Irrelevant Document>\s*([\d, ]+)'
        
        # 2. 找到所有匹配的索引字符串
        matches = re.findall(pattern, model_output)
        
        # 3. 如果没有匹配到任何内容，直接返回空列表
        if not matches:
            return []
        
        # 4. 处理所有匹配到的索引字符串（可能有多个匹配，但通常只有一个）
        all_indices = []
        for match in matches:
            # 按逗号分割并清理每个索引
            indices = [idx.strip() for idx in match.split(',')]
            for idx in indices:
                # 仅保留有效的数字索引
                if idx.isdigit():
                    all_indices.append(int(idx))
        
        # 5. 排序并转换为字符串
        all_indices.sort()
        return [str(idx) for idx in all_indices]
    
    @ staticmethod
    def Execution_parse_document_summary_output(model_output: str) -> dict:
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

    @ staticmethod
    def Execution_align_lists(
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

    @ staticmethod
    def Document_profile_prompts(user_input, document_list):
        """Profile的提示词"""

        document_prompt = "\n".join([f"<{index}> {document_list[index]}" for index in range(len(document_list))]) if len(document_list) != 0 else "Nothing."
        
        prompt = f"""
You are a professional document analyst responsible for creating concise, searchable profiles from source documents. Your goal is to extract the essential meaning while removing redundancy and noise.

**CORE PRINCIPLES**:
- **STRICT INDEX PRESERVATION**: Generate EXACTLY one [Document N] block for EVERY input document index (0 to {len(document_list)-1}). NEVER skip indices or reorder documents.
- **EMPTY-BUT-STRUCTURED**: For irrelevant documents, output "No information available" in [Profile] but retain the full block structure with original title.
- **SEMANTIC FIDELITY**: Never invent connections or omit critical constraints (e.g., p-values, confidence intervals).

**ENHANCED RULES** (CONCISE VERSION):
1. **ADAPTIVE RELEVANCE PROCESSING**: 
   - HIGH relevance: Preserve specific details, numbers, dates, proper nouns, and contextual relationships
   - MEDIUM relevance: Keep core facts only, remove secondary details
   - LOW/NO relevance: Output "No information available" (never invent connections)
2. **PRECISION CONSTRAINTS**: 
   - Maximum 100 words per Profile section
   - Sacrifice general information before query-relevant details
   - Ensure semantic completeness while focusing on query relevance
3. **HALLUCINATION PREVENTION**:
   - NEVER invent connections to the query that don't exist in source documents
   - If document list is empty, output ONLY "No documents in document list" and skip all other processing
   - For low/no relevance documents, strictly output "No information available" without adding speculative content
4. **MANDATORY THINKING**: Complete a SINGLE PARAGRAPH analysis in [Think] section covering query intent and document relevance assessment before generating profiles

**Format Specifications**: 
[Think] Brief but comprehensive analysis of query intent, document relevance assessment, and content prioritization strategy [/Think]
[Document 0] [Title]: Extracted or created document title [Profile]: Refined content following specifications below
...
[Document n] [Title]: Extracted or created document title [Profile]: Refined content following specifications below

**Multi-Document Example**:
- User Input: What were the key economic policies during Biden's first year in office?
- Need summarized documents:
<0>: [Title]: American Rescue Plan Act of 2021 [Content]: Signed into law on March 11, 2021, the $1.9 trillion American Rescue Plan provided direct payments of $1,400 to eligible Americans, extended unemployment benefits through September 2021, allocated $350 billion to state and local governments, and included $170 billion for education institutions to safely reopen.
<1>: [Title]: Infrastructure Investment and Jobs Act [Content]: Passed in November 2021 after months of negotiation, this $1.2 trillion bipartisan infrastructure bill allocated funding for roads, bridges, broadband internet expansion, clean water systems, and electric vehicle infrastructure. The legislation represents the largest federal investment in infrastructure in over a decade and is expected to create approximately 1.5 million jobs annually over the next decade.
<2>: [Title]: Q4 2021 Economic Report [Content]: The U.S. economy grew at an annual rate of 6.9% in the fourth quarter of 2021, capping a year of strong recovery. GDP expanded 5.7% for the full year, the fastest growth since 1984. However, inflation rose to 7% in December, reaching a 40-year high. Labor market showed improvement with unemployment falling to 3.9% by year-end.
<3>: [Title]: Biden Administration Tax Framework [Content]: In April 2021, President Biden proposed the Made in America Tax Plan which would raise the corporate tax rate from 21% to 28%, implement a global minimum tax, and eliminate tax preferences for fossil fuel companies. The framework aimed to fund infrastructure investments while ensuring corporations pay their "fair share."
<4>: [Title]: Early Life of Joe Biden [Content]: Joseph R. Biden Jr. was born on November 20, 1942, in Scranton, Pennsylvania. He grew up in a working-class family and later moved to Delaware. Biden attended the University of Delaware before earning his law degree from Syracuse University in 1968. He was first elected to the New Castle County Council in 1970 and then to the U.S. Senate at age 29.
- Expected Output:
[Think] Query seeks economic policies from Biden's first year (2021). Doc0 (American Rescue Plan) and Doc1 (Infrastructure Act) are highly relevant major legislation - preserve specific figures, dates and key provisions. Doc3 (Tax Framework) is highly relevant policy proposal - keep tax rate changes and core mechanisms. Doc2 (Economic Report) has medium relevance showing policy outcomes - retain only key economic metrics. Doc4 (Early Life) has no relevance to economic policies - skip entirely. Prioritize exact dollar amounts, dates, and policy mechanisms while compressing descriptions to essential elements under strict word limits. [/Think]
[Document 0] [Title]: American Rescue Plan Act of 2021 [Profile]: $1.9 trillion economic stimulus signed March 11, 2021. Included $1,400 direct payments to eligible Americans, extended unemployment benefits through September 2021, $350 billion for state/local governments, $170 billion for education reopening. 
[Document 1] [Title]: Infrastructure Investment and Jobs Act [Profile]: $1.2 trillion bipartisan infrastructure bill passed November 2021. Funded roads, bridges, broadband expansion, clean water systems, and EV infrastructure. Largest federal infrastructure investment in over a decade, expected to create 1.5 million jobs annually. 
[Document 2] [Title]: Q4 2021 Economic Report [Profile]: U.S. economy grew 6.9% in Q4 2021, with full-year GDP expanding 5.7% (fastest since 1984). Inflation reached 40-year high of 7% by December. Unemployment improved to 3.9% by year-end. 
[Document 3] [Title]: Biden Administration Tax Framework [Profile]: April 2021 "Made in America Tax Plan" proposed raising corporate tax rate from 21% to 28%, implementing global minimum tax, and eliminating fossil fuel tax preferences. 
[Document 4] [Title]: Early Life of Joe Biden [Profile]: No information available

**No-Document Example**:
- User Input: Quantify the causal relationship between microplastic concentration and coral calcification rates in tropical reef systems
- Need summarized documents: Nothing. 
- Expected Output: 
[Think] No documents in document list. Skip generation. [/Think]

**Your Task**
**User Input**: {user_input}

**Need summarized documents**:
{document_prompt}

**Expected Output**: 
""".strip()

        return prompt
    
    @ staticmethod
    def Document_profile_Extractor(response: str):
        """Profile的提示词"""
        return re.sub(
            r'\[Think\].*?(?:\[/Think\]|\[\\Think\])',
            '',
            response,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
    
    @ staticmethod
    def Document_profile_fail_prompt(document_list):
        """Profile的提示词"""
        return "\n".join([f"[Document {index}] {document_list[index][:100]}" for index in range(len(document_list))])

    @ staticmethod
    def get_final_LLM_input(user_input, document_list, detail: int = None, **kwargs):
        """
        构建最终的LLM输入结果，完成整个收尾工作
        
        """

        # 处理文档
        if not document_list:
            documents_prompt = "No documents found."
        else:
            documents_prompt = "\n".join([f"<Document {i}> {doc}" for i, doc in enumerate(document_list)])

        # 处理补充信息
        if detail:
            detail_prompt = f"\n{detail}"
        else:
            detail_prompt = "\nNo additional details provided."

        user_prompt = f"""
[Role]
You are a factual information responder that provides concise and accurate answers.

[Guidelines]
1. This is a fast Q&A session - **provide ONLY the direct answer, no explanations, no context**
2. Check documents and additional context first for the answer
3. If documents and additional context are missing or misleading, use your knowledge
4. Responses must be **extremely concise** - one word, number, or short phrase


[Example Format]
Documents:
<Document 0> Title: ... Context:
<Document 1> Title: ... Context: 
Additional Context:
[Supplementary information]
Question:
[User's question]
Answer:
[Direct answer only]

[Format Example 1]:
Documents: 
<Document 0> 
...
Additional Context: 
...
**Question**: 
When was the first computer invented?
**Expected Your Output**:
1945.

[Format Example 2]:
Documents: 
...
Additional Context: 
...
**Question**: 
What is the most widely spoken language in the world?
**Expected Your Output**:
Mandarin Chinese. 

[Your Task]:
Documents:
{documents_prompt}
Additional Context:
{detail_prompt}
**Question**:
{user_input}
**Answer**:
"""
        return user_prompt

from .dataset import SampleCritic
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
                    tokenizer: "AutoTokenizer" = None, 
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
    
    def extract_user_input(self, text):
        """该方法专门应付qwen的模板template修改后的结果"""
        # 匹配 user 标签后的内容，直到下一个标签（如 assistant）出现
        pattern = r'user\n(.*?)(?=\n(?:system|assistant|user|$))'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
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

            self.indexer = ConcurrentIndexerWorker(
                            IndexBuilder(index_load_path = index_load_path,
                                        document_load_path = document_load_path, 
                                        device = indexer_device), 
                                        num_workers = 4)

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
        # 不一定可以通过的，需要判定的: Document Analysis
        if action_type == 'Document Analysis':
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

    def _process_single_sample(self, 
                    sample_data: Dict,
                    document_threshold:int = 6, 
                    **kwargs):
        
        start_time = time.time()

        index = sample_data['index']
        decision_prompt = sample_data['decision_prompt']
        query_list = copy.deepcopy(sample_data['query_list'])  # 创建副本避免共享状态问题
        document_list = copy.deepcopy(sample_data['document_list'])
        done_flag = sample_data['done_flag'] 

        decision = AERRTemplate.DecisionPrompt2ActionDict(decision_prompt)
        actions = decision['actions']
        user_input = self.query2UserInput(query_list)

        executed_actions = []

        for action_index, action in enumerate(actions):
            
            # 单个动作的执行不影响整体的结果
            skip = self.should_skip_action(
                action = action, 
                query_list = query_list, 
                document_list = document_list
            )

            if skip == True:
                continue
            elif skip == 'skipAll':
                sample_data['done_flag'] = True
                done_flag = True
                break

            action_type = action['action_type']
            target_ids = action['target_id']

            # 执行基础动作
            if action_type == "Delete Query":
                self.DeleteQuery(target_ids=target_ids, query_list=query_list)
            elif action_type == "Query Search":
                self.QuerySearch(query_list=query_list, document_list=document_list, target_ids=target_ids, k = 5)
            elif action_type == "Delete Documents":
                self.DeleteDocuments(target_ids=target_ids, document_list=document_list)
            elif action_type == "Sort Documents":
                self.SortDocuments(target_ids=target_ids, document_list=document_list)

            # 进阶动作，此时做字符串的拼接
            elif action_type in ['Query Rewrite', 'Document Analysis', 'Document Filter', 'Summarize Documents']:
                # 生成输入提示
                if action_type == 'Query Rewrite':
                    input_prompt = AERRTemplate.Execution_query_rewrite_input_prompt(user_input)
                elif action_type == 'Document Analysis':
                    input_prompt = AERRTemplate.Execution_document_analysis_input_prompt(user_input, document_list=document_list)
                elif action_type == 'Document Filter':
                    input_prompt = AERRTemplate.Execution_document_filter_input_prompt(user_input, documents=document_list)
                elif action_type == 'Summarize Documents':
                    input_prompt = AERRTemplate.Execution_document_summary_input_prompt(user_input, documents=document_list)
                
                # 调用模型（单个调用，在并行环境中是安全的）
                response, action_time = self.model.generate(
                    [input_prompt], 
                    return_time=True, 
                    **self.strategy_params
                )

                response = response[0]
                # 执行动作
                if action_type == 'Query Rewrite':
                    self.QueryRewrite(response, query_list, document_list, k = 2)
                elif action_type == 'Document Analysis':
                    self.QueryExtract(response, query_list, document_list, k = 2)
                elif action_type == 'Document Filter':
                    self.DocumentFilter(document_list, response)
                elif action_type == 'Summarize Documents':
                    self.SummarizeDocuments(document_list, response)

            executed_actions.append(action)

        # ===============================================
        # 一次意外的debug指出，文档中带来的大量噪声是性能无法提升的根源！！！！
        # 1. 引入Document Profile机制，展示每个文档的关键信息
        # 2. 每当超过10个文档，则强制进行筛选，以筛选后的结果为准
        # ===============================================
        if len(document_list) >= document_threshold: 
            user_input = self.query2UserInput(query_list)
            input_prompt = AERRTemplate.Execution_document_filter_input_prompt(user_input, documents=document_list)
            
            response, filter_time = self.model.generate(
                [input_prompt], 
                return_time=True, 
                **self.strategy_params
            )
            response = response[0]
            self.DocumentFilter(document_list, response)
            executed_actions.append(AERRTemplate.Execution_Build_Action_Dict([0], action_type_string = "Document Filter", detail = ''))

        # 记录时间，之前存在时间的重复记数问题
        total_time = time.time() - start_time

        return {
            'index': index,
            'query_list': query_list,
            'document_list': document_list,
            'done_flag': done_flag,
            'time_cost': total_time,
            'actions': executed_actions
        }


    def take_action_batch(self, 
                          state: "AERRStateManager", 
                          return_actions: bool = False, 
                          max_workers: int = 12, 
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
        
        sample_data = []
        for i, (dp, ql, dl, df) in enumerate(zip(decision_prompts, query_lists, document_lists, done_flags)):
            sample_data.append({
                'index': i,
                'decision_prompt': dp,
                'query_list': ql,
                'document_list': dl,
                'done_flag': df
            })
        
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
            future_to_samle = {
                executor.submit(self._process_single_sample, sample, **kwargs): sample for sample in sample_data
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_samle):
                sample = future_to_samle[future]
    
                result = future.result()
                results.append(result)

        
        results.sort(key = lambda x: x['index'])

        # 提取结果
        query_lists = [result['query_list'] for result in results]
        document_lists = [result['document_list'] for result in results]
        done_flags = [result['done_flag'] for result in results]
        time_lis = [result['time_cost'] for result in results]
        actions_list = [result['actions'] for result in results]

        # 处理单样本输出格式
        if not isBatch:
            query_lists = query_lists[0]
            document_lists = document_lists[0]
            done_flags = done_flags[0]
            time_lis = time_lis[0]
        
        if return_actions:
            return query_lists, document_lists, done_flags, time_lis, actions_list

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

    def QuerySearch(self, query_list, document_list: list, target_ids: List[int], k = 10):
        """Target Query直接检索，列表一一对应，处理Already Retrieved标签逻辑。注意，每一个target id都会被检索"""
        target_ids = [target_id for target_id in target_ids if target_id < len(query_list)] # 筛选机制
        for idx in range(len(target_ids)):
            if target_ids[idx] >= len(query_list):
                continue
                
            current_query = query_list[target_ids[idx]]
            # 检查是否已标记
            if current_query.startswith("(Already Retrieved) "):
                # 已标记：移除标签，使用原始字符串检索
                original_query = current_query[len("(Already Retrieved) "):]
                retrieved_docs = self.indexer.search(original_query, k)
            else:
                # 未标记：进行检索并添加标签
                retrieved_docs = self.indexer.search(current_query, k)
                # 更新query_list为标记后的字符串
                query_list[target_ids[idx]] = "(Already Retrieved) " + current_query

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

    def QueryRewrite(self, model_response, query_list: List[str], document_list: List[str], k: int = 3):
        """集成Query Rewrite的管线，包含从改写 -> 检索 -> 添加三个部分，是一个复杂管线"""
        # 先询问API怎么个事情，拿到新的Query
        new_queries = AERRTemplate.Execution_parse_query_output(model_response)
        # 对Query进行搜索，每个Query检索三个文档，这样会共计生成15个文档
        for query in new_queries:
            documents = self.indexer.search(query, k = k) 
            document_list.extend(documents)
            # 更新query列表
            query_list.append(query)

    def QueryExtract(self, model_response, query_list: List[str], document_list: List[str], k: int = 3):
        """集成Document Analysis的管线，这一部分应当需要让外部API思考，为了进一步获取信息，应当查询哪一部分的内容，而不是简单的返回几个Query就可以了"""
        new_queries = AERRTemplate.Execution_parse_query_output(model_response)
        # 对Query进行搜索，每个Query检索三个文档，这样会共计生成15个文档
        for query in new_queries:
            documents = self.indexer.search(query, k = k) 
            document_list.extend(documents)
            # 更新query列表
            query_list.append(query)

    def DocumentFilter(self, document_list: List[str], response: str) -> None:
        """
        使用列表重建而非原地删除
        """
        if isinstance(response, list):
            response = response[0]
        
        # 获取要删除的索引
        irrelevant_indices = AERRTemplate.Execution_parse_irrelevant_documents(response)
        delete_set = set()
        
        for idx_str in irrelevant_indices:
            try:
                idx = int(idx_str)
                if 0 <= idx < len(document_list):
                    delete_set.add(idx)
            except ValueError:
                continue
        
        # 重建列表（排除要删除的索引）
        document_list[:] = [doc for i, doc in enumerate(document_list) 
                        if i not in delete_set]
        
    def SummarizeDocuments(self, document_list: List[str], response: str) -> None:
        """总结所有的文档内容，删除所有原来的文档"""
        if isinstance(response, list):
            response = response[0]

        # 1. 解析不相关文档索引
        summary_prompt: str = AERRTemplate.Execution_parse_document_summary_output(response)
        document_list[:] = [summary_prompt]

class Decision_Agent:
    
    def __init__(
        self, 
        config: "DecisionConfig" = None, 
        model: "Large_Language_ModelV2" = None, 
        tokenizer: "AutoTokenizer" = None, 
        *args, 
        **kwargs):
        """
        初始化QueryExpander类，用于生成多个扩展查询变体。
        
        Args:
            model: 用于扩展的LLM模型（如Llama3）
            verbose (bool): 是否输出详细日志
        """
        if config == None:
            config = DecisionConfig()

        self.config = config
        device: torch.device = config.device
        verbose: bool = False
        strategy_params: DecisionStrategyParams = config.strategy_params
        self.batch_size: int = config.batch_size

        print("Loading Decision Agent...")

        
        if config.load_without_model:
            self.model = Large_Language_ModelV2(local_dir = config.model_dir, 
                                              device = device,
                                              batch_size = self.batch_size, 
                                              without_model = config.load_without_model, 
                                              with_value_head = True, 
                                              tokenizer = tokenizer)
                
        elif not config.load_api and model is None:
            self.model = Large_Language_Model(local_dir = config.model_dir, 
                                            device = device,
                                            batch_size = self.batch_size)
            if config.lora_dir is not None:
                self.reload_lora(config = config)

        elif config.load_api == True:
            self.model = Large_Language_Model_API(model = self.config.api_model)
            
        
        # 初始化提示词
        self.model.init_llm(['' for i in range(self.batch_size)])
        self.model.init_llm_complement_prompt(['' for i in range(self.batch_size)])

        self.load_api = config.load_api
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        if not type(strategy_params) == dict:
            self.strategy_params = strategy_params.todict()

        print("Done!")

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

class Naive_RAG:

    def __init__(self, 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 index_load_path: str = './wikipedia_BGE_L2.contriever',
                 document_load_path: str = './psgs_w100.tsv',
                 ):
        print('=' * 40)
        print('Loading Large Language Model...')

        print('=' * 20)
        print('Loading Index...')
        self.indexer = IndexBuilder(
            device = device,
            index_load_path = index_load_path,
            document_load_path = document_load_path
        )   

        # 别慌，这一坨只是为了对齐模板
        config = MyTrainConfig()
        aerr_config = AERRConfig()
        aerr_config.init_decision_model = False
        aerr_config.init_execution_model = False
        aerr_config.init_generate_model = False
        self.ourpipeline = AERR(config = aerr_config)

        # 试试这个
        self.generator = Generate_Agent()

    def generate(self, 
                user_inputs: Union[str, List[str]],
                k: int = 3, 
                max_tokens: int = 300,
                return_time: bool = False, 
                **kwargs):
        # 确定是否为batch
        is_batch = isinstance(user_inputs, list)
        if not is_batch:
            user_inputs = [user_inputs]

        final_input_prompts = []
        
        if return_time:
            time_lis = []
        # 构造输入提示
        for user_input in user_inputs:
            # 检索相关文档
            if return_time:
                before = time.time()
            documents = self.indexer.topk_search(user_input, k=k)
            final_input_prompts.append(AERRTemplate.get_final_LLM_input(user_input = user_input, document_list = documents))
            if return_time:
                after = time.time()
                time_lis.append(after - before)
        # 调用模型生成回答
        outputs = self.generator.generate(input_text = final_input_prompts, 
                                          max_tokens = max_tokens)

        # 还原输出格式
        if not is_batch:
            outputs = outputs[0]

        return outputs if return_time == False else outputs, time_lis

class RRR:
    def __init__(self, 
                model: "Large_Language_ModelV2" = None, 
                tokenizer: "AutoTokenizer" = None, 
                config: "MyTrainConfig" = None, 
                *args, 
                **kwargs):
        
        if config is None:
            config = MyTrainConfig()

        decision_agent_config = config.to_AERRConfig().decision
        execution_agent_config = config.to_AERRConfig().execution

        self.config = config
        device: torch.device = config.device
        verbose: bool = False
        strategy_params: DecisionStrategyParams = decision_agent_config.strategy_params
        self.batch_size: int = config.batch_size

        print('=' * 40)
        print('Rewriter')
        # 和decision model的配置对齐
        if decision_agent_config.load_without_model:
            self.model = Large_Language_ModelV2(local_dir = decision_agent_config.model_dir, 
                                              device = device,
                                              batch_size = self.batch_size, 
                                              without_model = decision_agent_config.load_without_model, 
                                              with_value_head = True, 
                                              tokenizer = tokenizer)
        

        # 最后的黑盒模型
        self.generator = Generate_Agent()

        
        print('=' * 20)
        print('Loading Index...')
        self.indexer = IndexBuilder(
            device = device,
            # index_load_path = index_load_path,
            # document_load_path = document_load_path
        )

        print('=' * 20)
        print('Finished loading!')
    
    def generate_batch(self, 
                    input_prompts: Union[List[str], str], 
                    need_print: bool = False, 
                    max_tree_length: int = 100, 
                    extract_context_from_template: bool = False, 
                    model: "AutoModelForCausalLMWithValueHead" = None, 
                    generate_func = None, 
                    tokenizer: "AutoTokenizer" = None, 
                    sample_mode: str = "normal", 
                    our_config: "MyTrainConfig" = None, 
                    **kwargs):     
        

        # 首先先搭好整体部分
        query_list = []


        # returns = [output] if is_batch else [output[0]]
        # if sample_mode == "normal":
        #     returns.extend(sampler.get_data())
        # elif sample_mode == "forest":
        #     returns.append(forest)
        # # 最后加了一个questions，因为需要解码，所以放在了最后
        # returns.append(input_prompts)

        pass
        # return *returns, 

class RRRTemplate:
    """用于管理RRR中的上下文，构建输入、拆分输出"""
    def __init__(self):

        pass

    def build_input_context(self):

        pass

    def get_response_context(self):

        pass


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
    

class BGM:
    def __init__(self):
        
        return 


    
class SelfRAG:
    """SelfRAG实现"""

class SearchR1StateManager(AERRStateManager):

    def __init__(self):
        return 
    
class SearchR1Template:
    def __init__(self):
        return 
    
    def buildInputPrompt():
        # 指令：输入模型的指令

        return 
    
    # def get

class SearchR1:
    def __init__(self):

        return 
    
    # SearchR1的管线实现
    # 对齐接口
    def generate_batch(self, 
                    input_prompts: Union[List[str], str], 
                    max_tree_length: int = 100, 
                    extract_context_from_template: bool = False, 
                    model: "AutoModelForCausalLMWithValueHead" = None, 
                    generate_func = None, 
                    tokenizer: "AutoTokenizer" = None, 
                    our_config: "MyTrainConfig" = None, 
                    **kwargs):     
        
        # 那么，SearchR1有如下步骤：
        # 格式微调
        # 然后拿到输入
        # 格式化指令
        # 得到模型输出 x n
        # 然后拿到answer
        # 最后得到结果
        # 再统一输出
        
        # 首先是要格式微调，那么这里需求指令微调，于是需要先写一个指令才行
        
        # 先初始化状态管理器
        state = SearchR1StateManager(input_prompts)

        # 如果没有终止，则反复输入模型
        interaction_nums = 0

        while not state.all_done and interaction_nums < max_tree_length:

            # 然后输入模型，获取需要的数据，包括模型输入、模型输出和耗时
            model_output_prompts, time_costs, model_input_prompts = self.model.generate_batch(state.get_input_prompts())

            # 格式整理
            model_output_prompts = [template.extractOutputPrompts]

            # 更新状态
            state.update(model_input_prompts, model_output_prompts, time_costs)

            interaction_nums += 1
            
        # 终止状态
        state.finish()

        # 获取最终结果
        output = state.get_outputs()

        returns = [output] if is_batch else [output[0]]

        # 获取交互过程
        returns.extend(state.get_training_data(need_init_prompt = True))
        if our_config is not None: 
            state.to_csv(our_config.eval_example_dir, save_interval = our_config.save_training_interaction_interval)

        returns.append(input_prompts) # input prompts 根本没有动过
        return *returns, 
    
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
