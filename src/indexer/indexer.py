
import torch
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import torch
import time
import torch
import os
import concurrent.futures

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Callable, Dict, Union, Tuple, TypeVar, Generic
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging

from .embedder import BGEEmbedder


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexerWorker")

T = TypeVar('T')

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

class Indexer(Generic[T]):
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


class CallbackIndexerWorker(Indexer[T]):
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
