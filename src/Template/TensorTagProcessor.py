

# #######################
# 该文件用于避免调用tokenizer，实现内容的提取
# ###############################


import torch
from typing import List, Tuple, Any, Dict, Optional
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dataclasses import dataclass
import logging


class ContentType(Enum):
    TEXT = "text"
    TAG = "tag"

@dataclass
class ContentSegment:
    """内容段数据结构"""
    type: ContentType
    tag_name: Optional[str] = None
    content: List[int] = None
    children: List['ContentSegment'] = None
    
    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.children is None:
            self.children = []

class BaseProcessor:
    """
    改进的基础标签处理器，支持结构化输出和完整的嵌套提取
    """
    
    def __init__(self, tag_config: dict):
        """
        初始化标签处理器
        
        Args:
            tag_config: 标签配置字典，格式为:
                {
                    "tag_name": (
                        [start_variant1, start_variant2, ...], 
                        [end_variant1, end_variant2, ...]
                    )
                }
                注意：每个variant应该是整数列表，不是字符串编码结果
        """
        self.tag_config = tag_config
        self._build_tag_mappings()
        self._validate_config()
        
    def _build_tag_mappings(self):
        """构建标签映射表用于快速查找"""
        self.start_tag_map = {}  # 开始标签变体映射到标签名 (元组 -> 标签名)
        self.end_tag_map = {}    # 结束标签变体映射到标签名 (元组 -> 标签名)
        self.all_tag_ids = set() # 所有标签ID的集合，用于快速查找
        
        # 为每个标签变体创建映射
        for tag_name, (start_variants, end_variants) in self.tag_config.items():
            # 处理开始标签变体
            for variant in start_variants:
                variant_tuple = tuple(variant)
                self.start_tag_map[variant_tuple] = tag_name
                self.all_tag_ids.update(variant)
            
            # 处理结束标签变体
            for variant in end_variants:
                variant_tuple = tuple(variant)
                self.end_tag_map[variant_tuple] = tag_name
                self.all_tag_ids.update(variant)
        
        # 构建快速查找的数据结构
        self._build_quick_lookup()
    
    def _build_quick_lookup(self):
        """构建快速查找所需的数据结构（按长度分组）"""
        self.start_tags_by_length = {}
        self.end_tags_by_length = {}
        
        for tag_tuple in self.start_tag_map:
            length = len(tag_tuple)
            if length not in self.start_tags_by_length:
                self.start_tags_by_length[length] = []
            self.start_tags_by_length[length].append(tag_tuple)
        
        for tag_tuple in self.end_tag_map:
            length = len(tag_tuple)
            if length not in self.end_tags_by_length:
                self.end_tags_by_length[length] = []
            self.end_tags_by_length[length].append(tag_tuple)
    
    def _validate_config(self):
        """验证配置的合法性"""
        for tag_name, (start_variants, end_variants) in self.tag_config.items():
            if not start_variants or not end_variants:
                raise ValueError(f"标签 '{tag_name}' 必须包含至少一个开始变体和结束变体")
            
            for i, variant in enumerate(start_variants):
                if not variant:
                    raise ValueError(f"标签 '{tag_name}' 的第 {i+1} 个开始变体不能为空")
            
            for i, variant in enumerate(end_variants):
                if not variant:
                    raise ValueError(f"标签 '{tag_name}' 的第 {i+1} 个结束变体不能为空")
    
    def get_processing_order(self):
        """获取标签处理顺序（可被子类重写）"""
        return list(self.tag_config.keys())
    
    def is_tag_id(self, token_id: int) -> bool:
        """判断一个token ID是否属于任何标签"""
        return token_id in self.all_tag_ids
    
    def match_start_tag(self, token_ids: list, position: int) -> tuple:
        """
        在指定位置匹配开始标签
        
        Args:
            token_ids: token ID列表
            position: 开始匹配的位置
            
        Returns:
            (tag_name, variant, variant_length) 如果匹配成功，否则 (None, None, 0)
        """
        # 检查所有可能长度的标签
        for length, tags in self.start_tags_by_length.items():
            if position + length > len(token_ids):
                continue
                
            current_slice = tuple(token_ids[position:position + length])
            if current_slice in tags:
                return self.start_tag_map[current_slice], list(current_slice), length
        
        return None, None, 0
    
    def match_end_tag(self, token_ids: list, position: int) -> tuple:
        """
        在指定位置匹配结束标签
        
        Args:
            token_ids: token ID列表
            position: 开始匹配的位置
            
        Returns:
            (tag_name, variant, variant_length) 如果匹配成功，否则 (None, None, 0)
        """
        # 检查所有可能长度的标签
        for length, tags in self.end_tags_by_length.items():
            if position + length > len(token_ids):
                continue
                
            current_slice = tuple(token_ids[position:position + length])
            if current_slice in tags:
                return self.end_tag_map[current_slice], list(current_slice), length
        
        return None, None, 0

    def process(self, token_ids: list) -> List[Dict]:
        """
        处理token ID序列，返回结构化的内容段列表
        
        Args:
            token_ids: 输入的token ID列表
            
        Returns:
            结构化的内容段列表，每个元素为字典，包含:
            - type: 内容类型 ('text' 或 'tag')
            - tag_name: 标签名（如果是标签内容）
            - content: 内容token IDs
            - start_pos: 在原始序列中的开始位置
            - end_pos: 在原始序列中的结束位置
        """
        result_segments = []
        stack = []  # 栈: (tag_name, start_position, content_segments)
        i = 0
        
        while i < len(token_ids):
            # 检查开始标签
            start_tag_name, start_variant, start_len = self.match_start_tag(token_ids, i)
            if start_tag_name:
                # 如果栈不为空，当前文本内容属于外层标签
                if stack:
                    current_tag_name, current_start, current_segments = stack[-1]
                    # 添加文本段到当前标签的内容中
                    if current_start < i:
                        text_content = token_ids[current_start:i]
                        if text_content:  # 避免空文本段
                            current_segments.append({
                                'type': 'text',
                                'content': text_content,
                                'start_pos': current_start,
                                'end_pos': i
                            })
                
                # 开始新标签，记录开始位置
                stack.append((start_tag_name, i + start_len, []))
                i += start_len
                continue
                
            # 检查结束标签
            end_tag_name, end_variant, end_len = self.match_end_tag(token_ids, i)
            if end_tag_name and stack:
                current_tag_name, current_start, current_segments = stack[-1]
                
                if current_tag_name == end_tag_name:
                    # 添加结束标签前的文本内容
                    if current_start < i:
                        text_content = token_ids[current_start:i]
                        if text_content:
                            current_segments.append({
                                'type': 'text', 
                                'content': text_content,
                                'start_pos': current_start,
                                'end_pos': i
                            })
                    
                    # 创建标签段
                    tag_segment = {
                        'type': 'tag',
                        'tag_name': current_tag_name,
                        'content': current_segments,  # 嵌套的内容段
                        'start_pos': current_start - len(start_variant) if start_variant else current_start,
                        'end_pos': i + end_len
                    }
                    
                    # 出栈
                    stack.pop()
                    
                    # 如果栈为空，添加到最终结果；否则添加到父标签的内容中
                    if not stack:
                        result_segments.append(tag_segment)
                    else:
                        parent_tag_name, parent_start, parent_segments = stack[-1]
                        parent_segments.append(tag_segment)
                        # 更新父标签的当前处理位置
                        stack[-1] = (parent_tag_name, i + end_len, parent_segments)
                    
                    i += end_len
                    continue
                else:
                    raise ValueError(f"标签不匹配: 结束标签 '{end_tag_name}' (期望: '{current_tag_name}')")
            
            # 如果没有匹配到标签且栈为空，直接前进
            if not stack:
                # 寻找下一个标签或文本结束
                next_tag_pos = i
                while next_tag_pos < len(token_ids):
                    start_tag_name, _, start_len = self.match_start_tag(token_ids, next_tag_pos)
                    end_tag_name, _, end_len = self.match_end_tag(token_ids, next_tag_pos)
                    if start_tag_name or end_tag_name:
                        break
                    next_tag_pos += 1
                
                # 添加纯文本段
                if next_tag_pos > i:
                    text_content = token_ids[i:next_tag_pos]
                    if text_content:
                        result_segments.append({
                            'type': 'text',
                            'content': text_content,
                            'start_pos': i,
                            'end_pos': next_tag_pos
                        })
                    i = next_tag_pos
                else:
                    i += 1
            else:
                i += 1
        
        # 处理栈中剩余的内容（不匹配的标签）
        while stack:
            tag_name, start_pos, segments = stack.pop()
            # 将剩余内容作为文本处理
            if start_pos < len(token_ids):
                text_content = token_ids[start_pos:]
                if text_content:
                    if not stack:  # 最外层
                        segments.append({
                            'type': 'text',
                            'content': text_content,
                            'start_pos': start_pos,
                            'end_pos': len(token_ids)
                        })
                        result_segments.append({
                            'type': 'tag',
                            'tag_name': tag_name,
                            'content': segments,
                            'start_pos': start_pos,
                            'end_pos': len(token_ids)
                        })
                    else:  # 内层
                        parent_tag_name, parent_start, parent_segments = stack[-1]
                        parent_segments.append({
                            'type': 'text',
                            'content': text_content,
                            'start_pos': start_pos,
                            'end_pos': len(token_ids)
                        })
        
        return result_segments

    def _process_single(self, token_ids: list) -> list:
        """
        向后兼容的process方法，只返回去标签后的内容
        
        Args:
            token_ids: 输入的token ID列表
            
        Returns:
            处理后的token ID列表（已移除标签）
        """
        structured_result = self.process_structured(token_ids)
        
        def extract_text_content(segments):
            """从结构化的段中提取所有文本内容"""
            text_tokens = []
            for segment in segments:
                if segment['type'] == 'text':
                    text_tokens.extend(segment['content'])
                elif segment['type'] == 'tag':
                    text_tokens.extend(extract_text_content(segment['content']))
            return text_tokens
        
        return extract_text_content(structured_result)

    def flatten_structured_result(self, structured_result: List[Dict]) -> List[Dict]:
        """
        将结构化的结果展平为一维列表，便于处理
        
        Args:
            structured_result: 结构化的处理结果
            
        Returns:
            展平的结果列表
        """
        flattened = []
        
        def flatten_segment(segment, depth=0):
            """递归展平段"""
            flat_segment = {
                'type': segment['type'],
                'tag_name': segment.get('tag_name'),
                'content': segment['content'] if segment['type'] == 'text' else [],
                'start_pos': segment['start_pos'],
                'end_pos': segment['end_pos'],
                'depth': depth
            }
            flattened.append(flat_segment)
            
            if segment['type'] == 'tag':
                for child in segment['content']:
                    flatten_segment(child, depth + 1)
        
        for segment in structured_result:
            flatten_segment(segment)
        
        return flattened

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskResult:
    status: TaskStatus
    result: Optional[List[int]] = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    worker_id: Optional[int] = None

class ProcessorPool:
    """
    改进的多线程处理器池
    - 更好的异常处理
    - 完整的任务状态跟踪
    - 资源安全管理
    - 性能监控

    使用说明：
    tag_config = {
        "Think": (
            [[1, 2, 3], [4, 5, 6]],    # 开始标签变体
            [[7, 8, 9], [10, 11, 12]]   # 结束标签变体
        ),
        "Action": (
            [[13, 14], [15, 16, 17]],
            [[18, 19], [20, 21, 22]]
        )
    }
    
    """
    
    def __init__(self, tag_config: dict, num_threads: int = 1, 
                 queue_timeout: float = 0.1, shutdown_timeout: float = 5.0):
        if num_threads < 1:
            raise ValueError("线程数量必须大于等于1")
        
        self.num_threads = num_threads
        self.tag_config = tag_config
        self.queue_timeout = queue_timeout
        self.shutdown_timeout = shutdown_timeout
        
        # 处理器实例
        self.processors = [BaseProcessor(tag_config) for _ in range(num_threads)]
        
        # 任务管理
        self.task_queue = queue.Queue()
        self.task_results: Dict[int, TaskResult] = {}
        self.task_order: List[int] = []
        self.next_task_id = 0
        
        # 线程和状态管理
        self.workers = []
        self.running = False
        self._shutdown_initiated = False
        
        # 统计信息
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0
        }
        self.stats_lock = threading.Lock()
        
        # 日志
        self.logger = self._setup_logging()
        self._start_workers()
    
    def _setup_logging(self):
        """设置结构化日志"""
        logger = logging.getLogger(f"ProcessorPool.{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [thread:%(thread)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _start_workers(self):
        """启动工作线程"""
        self.running = True
        self._shutdown_initiated = False
        
        for i in range(self.num_threads):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                name=f"ProcessorWorker-{i}",
                daemon=False  # 非守护线程，确保正常退出
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"启动 {self.num_threads} 个工作线程")
    
    def _worker(self, worker_id: int):
        """工作线程执行函数"""
        processor = self.processors[worker_id]
        self.logger.debug(f"工作线程 {worker_id} 启动")
        
        while self.running:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=self.queue_timeout)
                if task is None:  # 退出信号
                    self.task_queue.task_done()
                    break
                
                task_id, token_ids = task
                
                # 更新任务状态
                self.task_results[task_id].status = TaskStatus.PROCESSING
                
                # 处理任务
                start_time = time.time()
                try:
                    result = processor.process(token_ids)
                    processing_time = time.time() - start_time
                    
                    # 更新成功结果
                    self.task_results[task_id] = TaskResult(
                        status=TaskStatus.COMPLETED,
                        result=result,
                        processing_time=processing_time,
                        worker_id=worker_id
                    )
                    
                    # 更新统计
                    with self.stats_lock:
                        self.stats['tasks_completed'] += 1
                        self.stats['total_processing_time'] += processing_time
                    
                    self.logger.debug(f"工作线程 {worker_id} 完成任务 {task_id}, 用时 {processing_time:.4f}s")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    
                    # 更新失败结果
                    self.task_results[task_id] = TaskResult(
                        status=TaskStatus.FAILED,
                        error=e,
                        processing_time=processing_time,
                        worker_id=worker_id
                    )
                    
                    # 更新统计
                    with self.stats_lock:
                        self.stats['tasks_failed'] += 1
                    
                    self.logger.error(f"工作线程 {worker_id} 处理任务 {task_id} 失败: {e}")
                
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # 正常超时，继续循环
                continue
            except Exception as e:
                self.logger.error(f"工作线程 {worker_id} 发生意外错误: {e}")
                # 短暂休眠后继续
                time.sleep(0.01)
        
        self.logger.info(f"工作线程 {worker_id} 退出")
    
    def submit(self, token_ids: List[int]) -> int:
        """提交处理任务"""
        if self._shutdown_initiated:
            raise RuntimeError("ProcessorPool 正在关闭，无法提交新任务")
        if not self.running:
            raise RuntimeError("ProcessorPool 未运行")
        
        task_id = self.next_task_id
        self.next_task_id += 1
        
        # 初始化任务状态
        self.task_results[task_id] = TaskResult(status=TaskStatus.PENDING)
        self.task_order.append(task_id)
        
        # 提交任务
        self.task_queue.put((task_id, token_ids))
        
        # 更新统计
        with self.stats_lock:
            self.stats['tasks_submitted'] += 1
        
        return task_id
    
    def submit_batch(self, batch: List[List[int]]) -> List[int]:
        """批量提交任务"""
        return [self.submit(token_ids) for token_ids in batch]
    
    def wait(self, timeout: Optional[float] = None) -> List[List[int]]:
        """等待所有任务完成并返回结果"""
        if not self.task_order:
            return []
        
        # ✅ 修正：使用超时循环等待
        start_time = time.time()
        while self.task_queue.unfinished_tasks > 0:
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning(f"等待任务完成超时 ({timeout}s)")
                break
            time.sleep(0.01)
        
        # 按顺序收集结果
        results = []
        errors = []
        
        for task_id in self.task_order:
            task_result = self.task_results[task_id]
            
            if task_result.status == TaskStatus.COMPLETED:
                results.append(task_result.result)
            elif task_result.status == TaskStatus.FAILED:
                errors.append((task_id, task_result.error))
        
        # 处理错误
        if errors:
            error_msg = f"{len(errors)} 个任务失败。第一个失败的任务ID: {errors[0][0]}, 错误: {errors[0][1]}"
            if len(errors) > 1:
                error_msg += f" (还有 {len(errors)-1} 个错误)"
            raise RuntimeError(error_msg) from errors[0][1]
        
        # 重置状态（仅在成功时）
        self._reset_state()
        
        return results
    
    def _reset_state(self):
        self.task_results.clear()
        self.task_order.clear()
        self.next_task_id = 0  # ✅ 关键修复
        
    def get_task_status(self, task_id: int) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.task_results:
            return self.task_results[task_id].status
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()
            
            # 计算平均处理时间
            completed_tasks = stats['tasks_completed'] + stats['tasks_failed']
            if completed_tasks > 0:
                stats['avg_processing_time'] = stats['total_processing_time'] / completed_tasks
            else:
                stats['avg_processing_time'] = 0.0
            
            # 添加队列信息
            stats['queue_size'] = self.task_queue.qsize()
            stats['pending_tasks'] = len([t for t in self.task_results.values() 
                                        if t.status == TaskStatus.PENDING])
            stats['active_workers'] = sum(1 for w in self.workers if w.is_alive())
            
            return stats
    
    def shutdown(self, wait: bool = True):
        """安全关闭处理器池"""
        if self._shutdown_initiated:
            return
        
        self._shutdown_initiated = True
        self.running = False
        
        self.logger.info("开始关闭处理器池...")
        
        # 发送退出信号
        for _ in range(self.num_threads):
            try:
                self.task_queue.put_nowait(None)
            except queue.Full:
                self.logger.warning("任务队列已满，无法发送退出信号")
        
        # 等待工作线程退出
        if wait:
            shutdown_start = time.time()
            for i, worker in enumerate(self.workers):
                remaining_time = self.shutdown_timeout - (time.time() - shutdown_start)
                if remaining_time <= 0:
                    self.logger.warning(f"工作线程 {i} 关闭超时")
                    continue
                    
                worker.join(timeout=remaining_time)
                if worker.is_alive():  # ✅ 修正：join后检查
                    self.logger.warning(f"工作线程 {i} 未正常退出")
                else:
                    self.logger.debug(f"工作线程 {i} 已退出")

        self.workers.clear()
        self.logger.info("处理器池已关闭")
    
    def __enter__(self):
        """上下文管理器支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.shutdown(wait=True)

    def process(self, batch: List[List[int]], num_threads: int = None) -> List[List[int]]:
        """
        批量处理任务（自动选择线程数）
        
        Args:
            batch: 要处理的token ID列表列表
            num_threads: 优先使用的线程数（如果为None则使用初始化的线程数）
        
        Returns:
            处理后的token ID列表列表（按输入顺序）
        """
        if num_threads is not None and num_threads != self.num_threads:
            # 临时调整线程数（不推荐频繁调整）
            self.shutdown()
            self.__init__(self.tag_config, num_threads)
        
        # 提交所有任务
        for token_ids in batch:
            self.submit(token_ids)
        
        # 等待结果
        results = self.wait()
        self.shutdown()
        return results