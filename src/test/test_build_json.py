# Author@ 猫毛
import json
import os
from flashrag.evaluator.metrics import F1_Score, ExactMatch, Precision_Score, Recall_Score
from flashrag.config import Config
import warnings
import random
import numpy as np
from typing import Dict, Any


class Validation_Dataset:
    def __init__(self, dataset_name="custom_qa"):
        self.dataset_name = dataset_name
        self.data = []  # 存储所有问答样本的原始字典
        self.golden_answers = []  # 真实答案列表（字符串或索引）
        self.pred = []  # 模型预测结果列表
        self.choices = []  # 多选题
        self.required_keys = {"question", "pred", "golden_answers"} # 必须包含的键值

    def update(self, value_dict: dict):
        """
        更新数据集，添加一个问答样本。

        Args:
            value_dict (dict): 包含以下键值的字典：
                - question (str): 问题字符串。
                - pred (str): 预测答案。
                - golden_answers (str): 真实答案。
        """
        
        if not self.required_keys.issubset(value_dict.keys()):
            raise ValueError(f"value_dict 必须包含 {self.required_keys} 所有键。")
        
        if not "choices" in value_dict.keys():
            value_dict["choices"] = []

        # 添加原始数据
        self.data.append(value_dict)

        # 提取字段
        self.pred.append(value_dict["pred"])
        self.golden_answers.append(value_dict["golden_answers"])
        self.choices.append(value_dict.get("choices", []))  # 默认为空列表

    @classmethod
    def from_jsonl_file(cls, filename, dataset_name=None, check_valid=False):
        """
        从 JSONL 文件中读取数据并构建 QA_Pred_Dataset 实例。

        参数:
            filename (str): JSONL 文件路径。
            dataset_name (str, optional): 数据集名称（可选，默认从文件名推断）。
            check_valid (bool): 是否检查每行是否为有效 JSON（默认 False）。

        返回:
            QA_Pred_Dataset: 构建完成的实例。
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件 {filename} 不存在。")

        instance = cls(dataset_name=dataset_name or os.path.splitext(os.path.basename(filename))[0])
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    instance.update(entry)
                except json.JSONDecodeError as e:
                    if check_valid:
                        print(f"警告：跳过非法 JSON {i} 行：{line[:50]}... 原因：{e}")
                    else:
                        instance.update({"question": line})  # 不校验时保留原始字符串

        return instance

    def check_required_keys(self):
            """
            检查每个样本是否包含所有必需键。

            参数:
                required_keys (List[str]): 必需键列表，默认为 ['question', 'pred', 'golden_answers']。

            抛出:
                ValueError: 如果有任何样本缺少必需键。
            """
            missing_entries = []

            for idx, entry in enumerate(self.data):
                missing_keys = [key for key in self.required_keys if key not in entry]
                if missing_keys:
                    missing_entries.append({
                        "sample_index": idx,
                        "missing_keys": missing_keys
                    })

            if missing_entries:
                error_msg = "以下样本缺少必需键：\n"
                for entry in missing_entries:
                    error_msg += f"  样本 {entry['sample_index']+1} 缺失键: {', '.join(entry['missing_keys'])}\n"
                raise ValueError(error_msg)

            return True  # 所有样本均满足要求


    def to_jsonl(self, filename, encoding='utf-8'):
        """
        将数据集保存为 JSONL 文件（每行一个 JSON 对象）。

        参数:
            filename (str): 保存文件的路径。
            encoding (str): 文件编码方式，默认 'utf-8'。
        """
        try:
            with open(filename, 'w', encoding=encoding) as f:
                for entry in self.data:
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"数据已成功保存到 {filename}")
        except Exception as e:
            raise RuntimeError(f"保存 JSONL 文件失败：{e}")

    def get_pred(self):
        """返回预测答案列表"""
        return [entry["pred"] for entry in self.data]

    def get_golden_answers(self):
        """返回真实答案列表"""
        return [entry["golden_answers"] for entry in self.data]
    
    def get_choices(self):
        """返回真实答案列表"""
        return [entry["choices"] for entry in self.data]
    
    def _format_dataset(self, max_chars=80, show_choices=True):
        """
        格式化数据集为字符串。
        """
        result = []
        for idx, entry in enumerate(self.data):
            question = (entry["question"][:max_chars] + " ...") if len(entry["question"]) > max_chars else entry["question"]
            prediction = (entry["pred"][:max_chars] + " ...") if len(entry["pred"]) > max_chars else entry["pred"]
            golden = (entry["golden_answers"][:max_chars] + " ...") if len(entry["golden_answers"]) > max_chars else entry["golden_answers"]

            line = f"[Sample {idx+1}]\n"
            line += f"  Question: {question}\n"
            line += f"  Prediction: {prediction}\n"
            line += f"  Golden Answer: {golden}\n"

            if show_choices and self.choices[idx]:
                line += f"  Choices: {', '.join(self.choices[idx])}\n"

            line += "-" * 40 + "\n"
            result.append(line)

        return "".join(result)
    

    def __getitem__(self, index):
        """
        支持索引和切片访问。

        参数:
            index (int or slice): 索引或切片对象。

        返回:
            Union[dict, Validation_Dataset]:
                - 若 index 是 int，返回指定索引的样本字典；
                - 若 index 是 slice，返回切片后的 Validation_Dataset 实例。
        """
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.data))
            sliced_data = self.data[start:stop:step]
            new_dataset = Validation_Dataset(dataset_name=self.dataset_name)
            new_dataset.data = sliced_data
            # 同步更新相关字段（如 pred, golden_answers）
            new_dataset.pred = self.pred[start:stop:step]
            new_dataset.golden_answers = self.golden_answers[start:stop:step]
            new_dataset.choices = self.choices[start:stop:step]
            return new_dataset
        else:
            raise TypeError("索引必须是整数或切片对象。")
        


    
    def __str__(self):
            """
            重写 __str__ 方法，实现 print(dataset) 直接调用。
            默认 max_chars=80，show_choices=True。
            """

            return self._format_dataset(max_chars=80, show_choices=True)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


    
        
class Evaluator:
    def __init__(self):
        config = Config()
        self.f1 = F1_Score(config)
        self.em = ExactMatch(config)

    def compute_metrics(self, validation_dataset: Validation_Dataset):
        """
        计算完整数据集的 F1 和 EM 指标。

        参数:
            validation_dataset (Validation_Dataset): 验证数据集对象。

        返回:
            Tuple[Dict, Dict, List[float], List[float]]:
                - f1_result: 平均 F1 分数；
                - em_result: 平均 EM 分数；
                - f1_per_sample: 每样本 F1 分数；
                - em_per_sample: 每样本 EM 分数。
        """
        pred_list = validation_dataset.get_pred()
        golden_answers_list = validation_dataset.get_golden_answers()
        
        choices_lis = validation_dataset.get_choices()

        class TempData:
            def __init__(self, pred, golden_answers):
                self.pred = pred
                self.golden_answers = golden_answers
                self.choices = choices_lis


        data = TempData(pred_list, golden_answers_list)

        f1_result, f1_per_sample = self.f1.calculate_metric(data)
        em_result, em_per_sample = self.em.calculate_metric(data)

        return f1_result, em_result, f1_per_sample, em_per_sample

    def sample_and_compute_metrics(self, validation_dataset: Validation_Dataset, batch_size):
        """
        从 Validation_Dataset 中随机抽取 batch_size 个样本，并计算 F1 和 EM 指标。

        参数:
            validation_dataset (Validation_Dataset): 验证数据集对象。
            batch_size (int): 抽样样本数量。

        返回:
            Tuple[Dict, Dict, List[float], List[float]]:
                - f1_result: 平均 F1 分数；
                - em_result: 平均 EM 分数；
                - f1_per_sample: 每样本 F1 分数；
                - em_per_sample: 每样本 EM 分数。
        """
        batch_size = min(len(validation_dataset), batch_size)

        # 随机抽样
        indices = random.sample(range(len(validation_dataset)), batch_size)

        # 从数据集中提取抽样数据
        pred_list = [validation_dataset.get_pred()[i] for i in indices]
        golden_answers_list = [validation_dataset.get_golden_answers()[i] for i in indices]
        choices_lis = [validation_dataset.get_choices()[i] for i in indices]


        # 构建临时数据对象并计算指标
        class TempData:
            def __init__(self, pred, golden_answers):
                self.pred = pred
                self.golden_answers = golden_answers

                self.choices = choices_lis


        data = TempData(pred_list, golden_answers_list)

        f1_result, f1_per_sample = self.f1.calculate_metric(data)
        em_result, em_per_sample = self.em.calculate_metric(data)

        return f1_result, em_result, f1_per_sample, em_per_sample
    
def calculate_metric(file_path: str, batch_size: int = 4, **kwargs) -> Dict[str, Any]:
    """
    加载验证数据集并计算评估指标
    
    1. 如果指定文件路径存在，则从 JSONL 文件加载数据集
    2. 否则创建一个空的验证数据集
    3. 使用 Evaluator 对象进行采样并计算指标
    
    Args:
        file_path (str): 验证数据集的 JSONL 文件路径
        
    Returns:
        Dict[str, Any]: 包含所有计算指标的字典，例如 {'f1': 0.85, 'accuracy': 0.92}
        
    Raises:
        FileNotFoundError: 如果文件路径无效且无法创建空数据集时抛出
        ValueError: 如果输入参数不符合预期格式
    """
    try:
        # 验证文件路径有效性
        if not isinstance(file_path, str):
            raise ValueError("file_path 必须是字符串类型")
            
        # 加载或创建数据集
        if os.path.exists(file_path):
            dataset = Validation_Dataset.from_jsonl_file(file_path)
        else:
            dataset = Validation_Dataset()
            
        # 创建评估器并计算指标
        evaluator = Evaluator()
        return evaluator.sample_and_compute_metrics(
            batch_size = batch_size, 
            validation_dataset=dataset
        )
        
    except Exception as e:
        raise RuntimeError(f"计算指标时发生错误: {str(e)}") from e

def update_jsonl_file(file_path: str, value_dict: dict) -> None:
    """
    将单个字典条目追加到 JSONL 文件的末尾，**强制要求包含特定字段**。

    参数:
        file_path (str): JSONL 文件的路径。
        value_dict (dict): 要追加的字典数据，**必须包含以下键**：
            - question (str): 问题字符串。
            - pred (str): 模型预测答案。
            - golden_answers (str): 真实答案。

    返回:
        None

    异常:
        - FileNotFoundError: 如果文件路径对应的目录不存在
        - PermissionError: 如果没有写入权限
        - ValueError: 如果输入字典缺少必需字段
        - RuntimeError: 如果写入过程中发生其他错误
    """

    # 校验字段完整性
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    
    required_keys = {"question", "pred", "golden_answers"}
    if not required_keys.issubset(value_dict.keys()):
        missing = required_keys - value_dict.keys()
        raise ValueError(f"value_dict 必须包含以下键: {required_keys}。当前缺失: {missing}")

    try:
        
        # 以追加模式打开文件，如果文件不存在则自动创建
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将字典转换为 JSON 字符串，并追加换行符
            json_line = json.dumps(value_dict, ensure_ascii=False) + '\n'
            f.write(json_line)
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"文件操作失败: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"输入数据不是有效的 JSON 格式: {e}") from e


def test_calculate_metrics():
    file_path = 'test.jsonl'
    print(calculate_metric(file_path = file_path))

    return 


def test_update_jsonl():

    file_path = 'test.jsonl'
    # 创建测试数据文件
    test_data = [
        {
            "question": "谁写了《哈姆雷特》？",
            "pred": "威廉·莎士比亚",
            "golden_answers": "威廉·莎士比亚"
        },
        {
            "question": "水的化学式是什么？",
            "pred": "H2O",
            "golden_answers": "H2O"
        }
    ]
    for value_dict in test_data:
        update_jsonl_file(file_path, value_dict = value_dict)
    
    validation = Validation_Dataset.from_jsonl_file(file_path)
    print(validation)
    

    return 



def test_dataset_class():
    print('=' * 40)
    print('Test Dataset.from_jsonl_file')
    print('=' * 20)

    if os.path.exists('test.jsonl'):
        dataset = Validation_Dataset.from_jsonl_file('test.jsonl')
    else:
        dataset = Validation_Dataset()
    print('Test Dataset.print')
    print('=' * 20)
    print(dataset)

    print('=' * 20)
    print('Test Dataset.update')
    print('=' * 20)
    dataset.update(
        value_dict = {
            'question': 'This is a question!', 
            'pred': 'This is a pred!',
            'golden_answers': 'This is a golden_answers',
        }
    )
    print(dataset)

    print('=' * 20)
    print('Test Dataset.to_jsonl')
    print('=' * 20)
    dataset.to_jsonl('test.jsonl')

    print('=' * 20)
    print('Test Evaluator.compute_metrics_on_batch')
    print('=' * 20)
    evaluator = Evaluator()
    print(evaluator.sample_and_compute_metrics(batch_size = 4, validation_dataset = dataset))

    return 

def test_metries():
    config = Config()
    calculator1 = F1_Score(config)
    dataset = Validation_Dataset.from_jsonl_file('test.jsonl')
    print(calculator1.calculate_metric(dataset))

    return 

def example():

    # 示例1：使用update_jsonl_file添加新样本到JSONL文件
    # 创建一个符合要求的样本字典
    sample_data = {
        "question": "巴黎是哪个国家的首都？",
        "pred": "法国",
        "golden_answers": "法国",
        "choices": ["德国", "意大利", "法国", "西班牙"]  # 可选字段
    }

    # 将样本写入JSONL文件
    update_jsonl_file("test.jsonl", sample_data)
    # 输出：成功将样本追加到文件末尾

    # 示例2：使用calculate_metric计算评估指标
    # 计算完整数据集的指标（默认采样4个样本）
    metrics = calculate_metric("test.jsonl")
    print(metrics)
    # 输出类似:
    # {'f1': 1.0, 'em': 1.0, 'f1_per_sample': [1.0], 'em_per_sample': [1.0]}

    # 示例3：自定义采样数量计算指标
    custom_metrics = calculate_metric("test.jsonl", batch_size=10)
    print(custom_metrics)
    # 输出类似:
    # {'f1': 1.0, 'em': 1.0, 'f1_per_sample': [1.0, 1.0, ...], 'em_per_sample': [...]}

    # # 示例4：错误处理演示
    # try:
    #     # 尝试写入缺失必要字段的样本
    #     invalid_data = {"question": "无效样本", "pred": "无效预测"}
    #     update_jsonl_file("qa_dataset.jsonl", invalid_data)
    # except ValueError as e:
    #     print(f"错误: {e}")
    # # 输出: 错误: value_dict 必须包含以下键: {'question', 'pred', 'golden_answers'}。当前缺失: {'golden_answers'}

    # try:
    #     # 尝试读取不存在的文件
    #     calculate_metric("nonexistent_file.jsonl")
    # except FileNotFoundError as e:
    #     print(f"错误: {e}")
    # # 输出: 错误: nonexistent_file.jsonl


    return 


if __name__ == '__main__':
    example()