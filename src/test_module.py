from Large_language_model import Large_Language_Model, Large_Language_Model_API
from typing import Callable
import sys
import datetime
from io import TextIOWrapper

class LogDiary:
    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def add_log(self, log_entry):
        """添加一条带时间戳的日志记录"""
        self.logs.append(log_entry)

    def activate(self):
        """激活日志记录，替换标准输出流"""
        sys.stdout = StreamToLogger(self.original_stdout, self)
        sys.stderr = StreamToLogger(self.original_stderr, self)

    def deactivate(self):
        """恢复原始输出流"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def save(self, file_path=None):
        """保存日志到文件，若未指定路径则使用默认路径"""
        if not file_path:
            file_path = "console_log.txt"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.logs))
            print(f"✅ 日志已保存到 {file_path}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")

    def get_logs(self):
        """获取所有日志内容"""
        return self.logs


class StreamToLogger(TextIOWrapper):
    def __init__(self, original_stream, log_diary):
        super().__init__(original_stream.buffer)
        self.original_stream = original_stream
        self.log_diary = log_diary

    def write(self, text):
        # 写入原始流（保持控制台输出）
        self.original_stream.write(text)
        self.original_stream.flush()

        # 添加带时间戳的日志
        if text.strip():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {text}"
            self.log_diary.add_log(log_entry)

    def flush(self):
        self.original_stream.flush()

        

def input2output(input_string, pipeline: Callable, desc = None, **kwargs):

    print("=" * 20)
    print(desc if desc is not None else "This is a test")
    print("=" * 20)
    print("Input: ")
    print(input_string)

    result = pipeline(input_string, **kwargs)

    print("=" * 20)
    print("Output: ")
    print(result)

    return 



def test_llm_api():

    system_prompt = ['This is a test.']
    llm = Large_Language_Model_API()
    llm.init_llm(system_prompt = system_prompt)

    # 第一次交互
    test_prompt = "How's everything going today?"
    input2output(test_prompt, llm.chat, desc = 'First Interaction')

    # 第二次交互
    test_prompt = "Not good, not bad. Life always like that..."
    input2output(test_prompt, llm.chat, desc = '2nd Interaction')

    # 获取历史记录
    history = llm.get_history()
    print("=" * 20)
    print("History: ")
    print(history)

    # 获取历史记录
    llm.clear_history()
    history = llm.get_history()
    print("=" * 20)
    print("After Clear History: ")
    print(history)
    return 


def test_llm_inputs2output():
    """
    未修改API前
    如果不超出索引范围，也即初始化一个较大的batchsize后，可以允许多个线程的交互
    一对多是可行的
    
    """

    system_prompt = ['This is a test.', 'This is a test.']
    llm = Large_Language_Model_API()
    llm.init_llm(system_prompt = system_prompt)

    # 第一次交互
    test_prompt = "How's everything going today?"
    input2output(test_prompt, llm.chat, desc = 'First Interaction')

    # 第二次交互
    test_prompt = "Not good, not bad. Life always like that..."
    input2output(test_prompt, llm.chat, desc = '2nd Interaction')

    # 获取历史记录
    history = llm.get_history()
    print("=" * 20)
    print("History: ")
    print(history)

    # 获取历史记录
    llm.clear_history()
    history = llm.get_history()
    print("=" * 20)
    print("After Clear History: ")
    print(history)

    return 


def test_parquet():
    import pandas as pd

    df = pd.read_parquet(r'ambigqa\full\train.parquet')
    print(df)


    return 







if __name__ == "__main__":
    test_parquet()