
from ...Template.BaseTemplate import BaseProcessor
import unittest
import queue

class TestBaseProcessor(unittest.TestCase):
    """测试 BaseProcessor 类的各种功能"""
    
    def setUp(self):
        """设置测试用的标签配置"""
        self.tag_config = {
            "Think": (
                [[1, 2], [3, 4, 5]],      # 开始标签的两个变体
                [[6, 7], [8, 9, 10]]      # 结束标签的两个变体
            ),
            "Action": (
                [[11, 12, 13], [14, 15]], # 开始标签的两个变体  
                [[16, 17, 18], [19, 20]]  # 结束标签的两个变体
            )
        }
        self.processor = BaseProcessor(self.tag_config)
    
    def test_initialization(self):
        """测试初始化功能"""
        self.assertEqual(self.processor.tag_config, self.tag_config)
        self.assertIn("Think", self.processor.start_tag_map)
        self.assertIn("Action", self.processor.start_tag_map)
        
    def test_tag_mappings(self):
        """测试标签映射构建"""
        # 测试开始标签映射
        self.assertEqual(self.processor.start_tag_map[tuple([1, 2])], "Think")
        self.assertEqual(self.processor.start_tag_map[tuple([3, 4, 5])], "Think")
        self.assertEqual(self.processor.start_tag_map[tuple([11, 12, 13])], "Action")
        
        # 测试结束标签映射
        self.assertEqual(self.processor.end_tag_map[tuple([6, 7])], "Think")
        self.assertEqual(self.processor.end_tag_map[tuple([16, 17, 18])], "Action")
        
    def test_is_tag_id(self):
        """测试标签ID识别"""
        # 测试存在的标签ID
        self.assertTrue(self.processor.is_tag_id(1))
        self.assertTrue(self.processor.is_tag_id(5))
        self.assertTrue(self.processor.is_tag_id(11))
        self.assertTrue(self.processor.is_tag_id(20))
        
        # 测试不存在的标签ID
        self.assertFalse(self.processor.is_tag_id(0))
        self.assertFalse(self.processor.is_tag_id(100))
        self.assertFalse(self.processor.is_tag_id(21))
        
    def test_match_start_tag(self):
        """测试开始标签匹配"""
        token_ids = [1, 2, 100, 3, 4, 5, 200]
        
        # 匹配第一个位置
        tag_name, length = self.processor.match_start_tag(token_ids, 0)
        self.assertEqual(tag_name, "Think")
        self.assertEqual(length, 2)
        
        # 匹配第三个位置
        tag_name, length = self.processor.match_start_tag(token_ids, 3)
        self.assertEqual(tag_name, "Think")
        self.assertEqual(length, 3)
        
        # 测试不匹配的情况
        tag_name, length = self.processor.match_start_tag(token_ids, 1)
        self.assertIsNone(tag_name)
        self.assertEqual(length, 0)
        
    def test_match_end_tag(self):
        """测试结束标签匹配"""
        token_ids = [6, 7, 100, 8, 9, 10, 200]
        
        # 匹配第一个位置
        tag_name, length = self.processor.match_end_tag(token_ids, 0)
        self.assertEqual(tag_name, "Think")
        self.assertEqual(length, 2)
        
        # 匹配第三个位置
        tag_name, length = self.processor.match_end_tag(token_ids, 3)
        self.assertEqual(tag_name, "Think")
        self.assertEqual(length, 3)
        
    def test_simple_tag_removal(self):
        """测试简单标签移除（无嵌套）"""
        # 测试 Think 标签
        token_ids = [1, 2, 100, 101, 102, 6, 7, 200]
        result = self.processor.process(token_ids)
        expected = [100, 101, 102, 200]
        self.assertEqual(result, expected)
        
        # 测试 Action 标签
        token_ids = [11, 12, 13, 300, 301, 16, 17, 18, 400]
        result = self.processor.process(token_ids)
        expected = [300, 301, 400]
        self.assertEqual(result, expected)
        
    def test_nested_tags(self):
        """测试嵌套标签处理"""
        # Think 内部嵌套 Action
        token_ids = [
            1, 2,              # Think 开始
            100, 
            11, 12, 13,        # Action 开始  
            200, 201,
            16, 17, 18,        # Action 结束
            102,
            6, 7,              # Think 结束
            300
        ]
        result = self.processor.process(token_ids)
        expected = [100, 200, 201, 102, 300]
        self.assertEqual(result, expected)
        
    def test_multiple_variants(self):
        """测试多个标签变体"""
        # 使用 Think 的第二个变体
        token_ids = [3, 4, 5, 400, 401, 8, 9, 10, 500]
        result = self.processor.process(token_ids)
        expected = [400, 401, 500]
        self.assertEqual(result, expected)
        
        # 使用 Action 的第二个变体
        token_ids = [14, 15, 600, 601, 19, 20, 700]
        result = self.processor.process(token_ids)
        expected = [600, 601, 700]
        self.assertEqual(result, expected)
        
    def test_no_tags(self):
        """测试没有标签的情况"""
        token_ids = [100, 101, 102, 103, 104]
        result = self.processor.process(token_ids)
        self.assertEqual(result, token_ids)
        
    def test_empty_input(self):
        """测试空输入"""
        token_ids = []
        result = self.processor.process(token_ids)
        self.assertEqual(result, [])
        
    def test_only_tags(self):
        """测试只有标签的情况"""
        # 只有 Think 标签
        token_ids = [1, 2, 6, 7]
        result = self.processor.process(token_ids)
        self.assertEqual(result, [])
        
        # 嵌套标签
        token_ids = [1, 2, 11, 12, 13, 16, 17, 18, 6, 7]
        result = self.processor.process(token_ids)
        self.assertEqual(result, [])
        
    def test_unmatched_end_tag(self):
        """测试不匹配的结束标签（应该抛出异常）"""
        token_ids = [6, 7, 100]  # 只有结束标签，没有开始标签
        
        with self.assertRaisesRegex(ValueError, "标签不匹配"):
            self.processor.process(token_ids)
            
    def test_unmatched_start_tag(self):
        """测试不匹配的开始标签（应该抛出异常）"""
        token_ids = [1, 2, 100]  # 只有开始标签，没有结束标签
        
        with self.assertRaisesRegex(ValueError, "标签不匹配"):
            self.processor.process(token_ids)
            
    def test_complex_nesting(self):
        """测试复杂嵌套场景"""
        token_ids = [
            1, 2,              # Think 开始
            100,
            11, 12, 13,        # Action 开始
            200,
            3, 4, 5,           # Think 开始（嵌套）
            300,
            8, 9, 10,          # Think 结束（内层）
            201,
            16, 17, 18,        # Action 结束
            101,
            6, 7,              # Think 结束（外层）
            400
        ]
        result = self.processor.process(token_ids)
        expected = [100, 200, 300, 201, 101, 400]
        self.assertEqual(result, expected)
        
    def test_processing_order(self):
        """测试处理顺序获取"""
        order = self.processor.get_processing_order()
        # 应该返回所有标签名
        self.assertEqual(set(order), {"Think", "Action"})
        # 应该返回列表
        self.assertIsInstance(order, list)
        
    def test_invalid_config(self):
        """测试无效配置"""
        # 空变体
        with self.assertRaises(ValueError):
            BaseProcessor({"Think": ([], [[6, 7]])})
        
        # 空标签序列
        with self.assertRaises(ValueError):
            BaseProcessor({"Think": ([[1, 2], []], [[6, 7]])})


from lib.Template.BaseTemplate import ProcessorPool, TaskResult, TaskStatus
import logging
import time

import unittest
import threading
import time
import logging
from unittest.mock import patch, MagicMock

import unittest
import threading
import time
import logging
from unittest.mock import patch, MagicMock, call
import queue

class TestBaseProcessor(BaseProcessor):
    """用于测试的简单处理器，模拟BaseProcessor行为"""
    
    def __init__(self, tag_config):
        super().__init__(tag_config)
        self.process_count = 0
    
    def process(self, token_ids):
        self.process_count += 1
        # 模拟处理：移除标签ID，保留内容ID
        # 假设标签ID是负数，内容ID是正数
        return [x for x in token_ids if x > 0]


class FailingProcessor(TestBaseProcessor):
    """模拟失败的处理器"""
    
    def __init__(self, tag_config, fail_on_count=None, delay=0):
        super().__init__(tag_config)
        self.fail_on_count = fail_on_count
        self.delay = delay
    
    def process(self, token_ids):
        self.process_count += 1
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.fail_on_count and self.process_count == self.fail_on_count:
            raise ValueError(f"Simulated failure on task {self.process_count}")
        
        return super().process(token_ids)


class BlockingProcessor(TestBaseProcessor):
    """模拟阻塞的处理器"""
    
    def __init__(self, tag_config, block_time=1):
        super().__init__(tag_config)
        self.block_time = block_time
    
    def process(self, token_ids):
        time.sleep(self.block_time)
        return super().process(token_ids)


class TestProcessorPool(unittest.TestCase):
    
    def setUp(self):
        # 使用真实的标签配置进行测试
        self.tag_config = {
            "Think": (
                [[-1, -2], [-3, -4, -5]],    # 开始标签
                [[-6, -7], [-8, -9, -10]]    # 结束标签
            ),
            "Action": (
                [[-11, -12, -13], [-14, -15]],
                [[-16, -17, -18], [-19, -20]]
            )
        }
        self.pool = None
        # 设置测试日志
        logging.basicConfig(level=logging.WARNING, force=True)
        self.logger = logging.getLogger("TestProcessorPool")
    
    def tearDown(self):
        if self.pool:
            try:
                self.pool.shutdown(wait=True)
            except Exception as e:
                self.logger.warning(f"Shutdown failed: {e}")
    
    def _create_pool(self, num_threads=2, processor_class=None):
        """创建处理器池的辅助方法"""
        if processor_class:
            # 使用自定义处理器
            with patch.object(ProcessorPool, '_create_processors') as mock_create:
                mock_create.return_value = [processor_class(self.tag_config) for _ in range(num_threads)]
                pool = ProcessorPool(self.tag_config, num_threads=num_threads)
        else:
            pool = ProcessorPool(self.tag_config, num_threads=num_threads)
        return pool
    
    def test_single_task_success(self):
        """测试单个任务成功处理"""
        self.pool = self._create_pool(num_threads=1)
        
        # 提交包含标签的内容
        token_ids = [-1, -2, 100, 200, 300, -6, -7, 400]  # Think标签包裹的内容
        task_id = self.pool.submit(token_ids)
        
        self.assertEqual(task_id, 0)
        self.assertEqual(self.pool.get_task_status(task_id), TaskStatus.PENDING)
        
        results = self.pool.wait()
        
        # 验证结果：标签被移除，内容保留
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], [100, 200, 300, 400])
        
        # 验证统计信息
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_submitted'], 1)
        self.assertEqual(stats['tasks_completed'], 1)
        self.assertGreater(stats['avg_processing_time'], 0)
        self.assertEqual(stats['tasks_failed'], 0)
    
    def test_batch_submission_order(self):
        """测试批量任务提交和顺序保持"""
        self.pool = self._create_pool(num_threads=3)
        
        batch = [
            [-1, -2, 1, 2, -6, -7],           # 任务0
            [-11, -12, -13, 3, 4, -16, -17, -18],  # 任务1
            [5, 6, 7],                        # 任务2（无标签）
            [-3, -4, -5, 8, -8, -9, -10]      # 任务3（不同标签变体）
        ]
        
        task_ids = self.pool.submit_batch(batch)
        self.assertEqual(task_ids, [0, 1, 2, 3])
        
        # 验证任务状态
        for task_id in task_ids:
            self.assertEqual(self.pool.get_task_status(task_id), TaskStatus.PENDING)
        
        results = self.pool.wait()
        
        # 验证结果顺序与输入一致
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0], [1, 2])      # 任务0结果
        self.assertEqual(results[1], [3, 4])      # 任务1结果
        self.assertEqual(results[2], [5, 6, 7])   # 任务2结果
        self.assertEqual(results[3], [8])         # 任务3结果
        
        # 验证统计信息
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_submitted'], 4)
        self.assertEqual(stats['tasks_completed'], 4)
        self.assertEqual(stats['tasks_failed'], 0)
    
    def test_concurrent_processing(self):
        """测试并发处理性能"""
        self.pool = self._create_pool(
            num_threads=4, 
            processor_class=lambda config: BlockingProcessor(config, block_time=0.1)
        )
        
        # 提交多个任务
        num_tasks = 8
        batch = [[i, i+1, i+2] for i in range(num_tasks)]
        
        start_time = time.time()
        task_ids = self.pool.submit_batch(batch)
        results = self.pool.wait()
        end_time = time.time()
        
        # 验证所有任务完成
        self.assertEqual(len(results), num_tasks)
        
        # 验证并发性能：4个线程处理8个任务，每个任务0.1秒
        # 串行需要0.8秒，并发应该显著更快
        processing_time = end_time - start_time
        self.assertLess(processing_time, 0.5)  # 应该远小于串行时间
        self.assertGreater(processing_time, 0.2)  # 但应该大于单个任务时间
    
    def test_task_failure_handling(self):
        """测试任务失败处理"""
        self.pool = self._create_pool(
            num_threads=2,
            processor_class=lambda config: FailingProcessor(config, fail_on_count=2)
        )
        
        # 提交3个任务，第二个会失败
        task_ids = self.pool.submit_batch([[1], [2], [3]])
        
        # 应该抛出异常，且包含失败信息
        with self.assertRaises(RuntimeError) as context:
            self.pool.wait()
        
        self.assertIn("Simulated failure", str(context.exception))
        self.assertIn("1 个任务失败", str(context.exception))
        
        # 验证统计信息
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_submitted'], 3)
        self.assertEqual(stats['tasks_completed'], 2)  # 任务0和任务2成功
        self.assertEqual(stats['tasks_failed'], 1)     # 任务1失败
    
    def test_multiple_failures(self):
        """测试多个任务失败的情况"""
        self.pool = self._create_pool(
            num_threads=3,
            processor_class=lambda config: FailingProcessor(config, fail_on_count=1)
        )
        
        # 所有任务都会失败
        task_ids = self.pool.submit_batch([[1], [2], [3]])
        
        with self.assertRaises(RuntimeError) as context:
            self.pool.wait()
        
        error_msg = str(context.exception)
        self.assertIn("3 个任务失败", error_msg)
        self.assertIn("第一个失败的任务ID: 0", error_msg)
        
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_failed'], 3)
        self.assertEqual(stats['tasks_completed'], 0)
    
    def test_wait_timeout_behavior(self):
        """测试等待超时行为"""
        self.pool = self._create_pool(
            num_threads=1,
            processor_class=lambda config: BlockingProcessor(config, block_time=1)
        )
        
        # 提交长任务
        self.pool.submit([1, 2, 3])
        
        # 设置短超时
        start_time = time.time()
        with self.assertLogs(level='WARNING') as log_context:
            results = self.pool.wait(timeout=0.2)
        
        elapsed_time = time.time() - start_time
        
        # 验证超时行为
        self.assertAlmostEqual(elapsed_time, 0.2, delta=0.1)
        self.assertEqual(len(results), 0)  # 超时时应返回空列表
        
        # 验证日志
        self.assertTrue(any("等待任务完成超时" in message for message in log_context.output))
        
        # 等待任务实际完成
        final_results = self.pool.wait()
        self.assertEqual(len(final_results), 1)
    
    def test_shutdown_safety(self):
        """测试安全关闭流程"""
        self.pool = self._create_pool(num_threads=2)
        
        # 提交一些任务
        task_ids = self.pool.submit_batch([[1], [2], [3]])
        
        # 正常关闭
        self.pool.shutdown(wait=True)
        
        # 验证线程状态
        self.assertFalse(any(worker.is_alive() for worker in self.pool.workers))
        self.assertFalse(self.pool.running)
        
        # 验证无法提交新任务
        with self.assertRaises(RuntimeError):
            self.pool.submit([4])
    
    def test_shutdown_without_wait(self):
        """测试不等待的关闭"""
        self.pool = self._create_pool(
            num_threads=1,
            processor_class=lambda config: BlockingProcessor(config, block_time=0.5)
        )
        
        # 提交长任务
        self.pool.submit([1, 2, 3])
        
        # 不等待关闭
        shutdown_start = time.time()
        self.pool.shutdown(wait=False)
        shutdown_time = time.time() - shutdown_start
        
        # 关闭应该立即返回
        self.assertLess(shutdown_time, 0.1)
        self.assertFalse(self.pool.running)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with ProcessorPool(self.tag_config, num_threads=2) as pool:
            self.assertTrue(pool.running)
            task_id = pool.submit([1, 2, 3])
            results = pool.wait()
            self.assertEqual(results, [[1, 2, 3]])
        
        # 退出上下文后应自动关闭
        self.assertFalse(pool.running)
    
    def test_task_status_tracking(self):
        """测试任务状态跟踪"""
        self.pool = self._create_pool(
            num_threads=1,
            processor_class=lambda config: BlockingProcessor(config, block_time=0.1)
        )
        
        # 提交任务
        task_id = self.pool.submit([1, 2, 3])
        
        # 初始状态应为PENDING
        self.assertEqual(self.pool.get_task_status(task_id), TaskStatus.PENDING)
        
        # 等待一小段时间让任务开始处理
        time.sleep(0.05)
        
        # 状态可能变为PROCESSING或COMPLETED
        status = self.pool.get_task_status(task_id)
        self.assertIn(status, [TaskStatus.PROCESSING, TaskStatus.COMPLETED])
        
        # 等待完成
        self.pool.wait()
        self.assertEqual(self.pool.get_task_status(task_id), TaskStatus.COMPLETED)
    
    def test_empty_wait(self):
        """测试空任务队列的wait()处理"""
        self.pool = self._create_pool(num_threads=2)
        
        results = self.pool.wait()
        self.assertEqual(results, [])
        
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_submitted'], 0)
        self.assertEqual(stats['tasks_completed'], 0)
    
    def test_task_id_continuity(self):
        """测试任务ID连续性"""
        self.pool = self._create_pool(num_threads=1)
        
        # 第一轮任务
        task_ids_1 = self.pool.submit_batch([[1], [2]])
        self.pool.wait()
        
        # 第二轮任务 - ID应该继续递增
        task_ids_2 = self.pool.submit_batch([[3], [4]])
        self.assertEqual(task_ids_2, [2, 3])
        
        results = self.pool.wait()
        self.assertEqual(len(results), 2)
    
    def test_queue_full_handling(self):
        """测试队列满的情况"""
        # 创建很小的队列
        self.pool = ProcessorPool(self.tag_config, num_threads=1, max_queue_size=1)
        self.pool.task_queue = MagicMock()
        self.pool.task_queue.put.side_effect = queue.Full()
        
        with self.assertRaises(queue.Full):
            self.pool.submit([1, 2, 3])
    
    def test_worker_exception_handling(self):
        """测试工作线程异常处理"""
        self.pool = self._create_pool(num_threads=1)
        
        # 模拟工作线程中的意外异常
        with patch.object(self.pool.processors[0], 'process') as mock_process:
            mock_process.side_effect = Exception("Unexpected worker error")
            
            self.pool.submit([1, 2, 3])
            
            # 应该能够正常处理异常并继续
            with self.assertRaises(RuntimeError):
                self.pool.wait()
    
    def test_statistics_accuracy(self):
        """测试统计信息准确性"""
        self.pool = self._create_pool(
            num_threads=2,
            processor_class=lambda config: BlockingProcessor(config, block_time=0.1)
        )
        
        # 提交多个任务
        num_tasks = 4
        task_ids = self.pool.submit_batch([[i] for i in range(num_tasks)])
        
        self.pool.wait()
        
        stats = self.pool.get_stats()
        self.assertEqual(stats['tasks_submitted'], num_tasks)
        self.assertEqual(stats['tasks_completed'], num_tasks)
        self.assertEqual(stats['tasks_failed'], 0)
        self.assertGreater(stats['total_processing_time'], 0)
        self.assertGreater(stats['avg_processing_time'], 0)
        self.assertEqual(stats['queue_size'], 0)
        self.assertEqual(stats['pending_tasks'], 0)
        self.assertEqual(stats['active_workers'], 2)


class TestProcessorPoolEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        self.tag_config = {"Think": ([[-1]], [[-2]])}
        self.pool = None
    
    def tearDown(self):
        if self.pool:
            try:
                self.pool.shutdown(wait=True)
            except Exception as e:
                self.logger.warning(f"Shutdown failed: {e}")
    
    def test_single_thread_performance(self):
        """测试单线程性能"""
        self.pool = ProcessorPool(self.tag_config, num_threads=1)
        
        start_time = time.time()
        task_ids = self.pool.submit_batch([[i] for i in range(10)])
        results = self.pool.wait()
        single_thread_time = time.time() - start_time
        
        self.assertEqual(len(results), 10)
        self.assertGreater(single_thread_time, 0)
    
    def test_many_threads(self):
        """测试大量线程"""
        self.pool = ProcessorPool(self.tag_config, num_threads=8)
        
        task_ids = self.pool.submit_batch([[1], [2], [3]])
        results = self.pool.wait()
        
        self.assertEqual(len(results), 3)
        self.assertEqual(self.pool.get_stats()['active_workers'], 8)
    
    def test_large_batch(self):
        """测试大批量任务"""
        self.pool = ProcessorPool(self.tag_config, num_threads=4)
        
        large_batch = [[i, i+1, i+2] for i in range(100)]
        task_ids = self.pool.submit_batch(large_batch)
        results = self.pool.wait()
        
        self.assertEqual(len(results), 100)
        self.assertEqual(self.pool.get_stats()['tasks_completed'], 100)
    
    def test_empty_tasks(self):
        """测试空任务"""
        self.pool = ProcessorPool(self.tag_config, num_threads=2)
        
        # 提交空任务
        self.pool.submit([])
        results = self.pool.wait()
        
        self.assertEqual(results, [[]])


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)

