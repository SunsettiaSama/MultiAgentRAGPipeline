# Author@ 猫毛
import json
import os
from flashrag.evaluator.metrics import F1_Score, ExactMatch, Precision_Score, Recall_Score, BLEU, Rouge_L, Rouge_1, Rouge_2, Rouge_Score
from flashrag.config import Config
from flashrag.evaluator import Evaluator as flashrag_evaluator
import warnings
import random
import numpy as np
from typing import Dict, Any
import pandas as pd
from collections.abc import Callable
from tqdm import tqdm
import csv
from pathlib import Path


class Validation_Dataset:
    def __init__(self, dataset_name="custom_qa"):
        self.dataset_name = dataset_name
        self.data = []  # 存储所有问答样本的原始字典
        self.golden_answers = []  # 真实答案列表（字符串或索引）
        self.pred = []  # 模型预测结果列表
        self.choices = []  # 多选题
        self.required_keys = {"questions", "pred", "golden_answers"} # 必须包含的键值

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


    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, 
                       dataset_name=None, 
                       check_valid=False, 
                       ):
        """
        从 pandas DataFrame 中读取数据并构建 Validation_Dataset 实例。

        参数:
            dataframe (pd.DataFrame): 包含问答数据的 DataFrame。
            dataset_name (str, optional): 数据集名称（可选，默认为 "dataframe_dataset"）。
            check_valid (bool): 是否检查 DataFrame 列的完整性（默认 False）。

        返回:
            Validation_Dataset: 构建完成的实例。
        """
        # 设置默认数据集名称
        if dataset_name is None:
            dataset_name = "dataframe_dataset"

        # 创建实例
        instance = cls(dataset_name=dataset_name)

        # 检查必要的列是否存在
        required_columns = instance.required_keys
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if check_valid and missing_cols:
            raise ValueError(f"DataFrame 缺少必要列: {missing_cols}")

        # 遍历每一行
        for _, row in dataframe.iterrows():
            value_dict = {}
            # 添加必须的列
            for col in required_columns:
                value_dict[col] = row[col]
            # 添加可选的 choices 列
            if 'choices' in dataframe.columns:
                value_dict['choices'] = row['choices']
            # 调用 update 方法
            instance.update(value_dict)

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
        # F1_Score, ExactMatch, Precision_Score, Recall_Score, BLEU, Rouge_L, Rouge_Score
        self.f1 = F1_Score(config)
        self.em = ExactMatch(config)
        self.acc = Precision_Score(config)
        self.recall = Recall_Score(config)
        self.bleu = BLEU(config)
        self.R_1 = Rouge_1(config)
        self.R_2 = Rouge_2(config)
        self.R_L = Rouge_L(config)
        self.R_S = Rouge_Score(config)

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
        
        results = {}
        # 定义指标名称、对象和处理方式
        metrics_config = {
            'f1 score': (self.f1, lambda x: x[0]['f1']),  # 直接通过键访问
            'em': (self.em, lambda x: x[0]['em']),
            'precision': (self.acc, lambda x: x[0]['precision']),
            'recall': (self.recall, lambda x: x[0]['recall']),
            'bleu': (self.bleu, lambda x: x[0]['bleu']),
            'Rouge_L': (self.R_L, lambda x: x[0]["rouge-l"]),
            'Rouge_1': (self.R_1, lambda x: x[0]["rouge-1"]),
            'Rouge_2': (self.R_2, lambda x: x[0]["rouge-2"]),
        }

        # 执行统一计算和赋值
        for key, (obj, processor) in metrics_config.items():
            results[key] = processor(obj.calculate_metric(data))

        return results
    
    def compute_em_metrics(self, validation_dataset: Validation_Dataset):
        """
        该接口用于后续的训练部分，传回详细的EM计算结果
        
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
        results = {}
        results = self.em.calculate_metric(data)

        return results
    

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

        results = {}

        results['f1 score'] = self.f1.calculate_metric(data)
        results['em'] = self.em.calculate_metric(data)
        results['precision'] = self.acc.calculate_metric(data)
        results['recall'] = self.recall.calculate_metric(data)
        results['bleu'] = self.bleu.calculate_metric(data)
        results['Rouge_L'] = self.R_L.calculate_metric(data)
        results['Rouge_Score'] = self.R_S.calculate_metric(data)

        return results

    def evaluate_pipeline(self, 
                        ambig_qa_path: int = '/root/data/ambigqa/full/validation.parquet', 
                        pipeline: Callable = None, 
                        if_print: bool = False, 
                        results_save_dir: str = 'results.csv',
                        test: bool = False, 
                        ):
        '''
        该代码用于评估某个Pipeline，数据集可指定，但后续需要修改列名
        
        
        '''
        print('=' * 20)
        print('Start Evaluation...')

        print('=' * 20)
        print('Loading Dataset')

        df = pd.read_parquet(
            path = ambig_qa_path
        )
        if test:
            df = df.iloc[:2]
        
        cache_dataset = pd.DataFrame(
            {
            'id': df['id'],
            'question': df['question'], 
            'golden_answers': df['nq_answer']
            }
        )
        predict_answer = []
        print('=' * 40)
        print('Start Eval:')
        with tqdm(total = len(cache_dataset)) as pbar:
            for index, row in enumerate(cache_dataset.itertuples()):
                id_ = row.id
                question = row.question
                golden_answer = row.golden_answers

                pred = pipeline(question)
                predict_answer.append(pred)
                
                if if_print:
                    print('=' * 20)
                    print(f'[Sample {index}]')
                    print('question: ', question)
                    print('golden_answers: ', golden_answer)
                    print('prediction: ', pred)
                pbar.update()

        cache_dataset['pred'] = predict_answer

        # 接下来写验证的部分
        eval_dataset = Validation_Dataset.from_dataframe(cache_dataset)
        results = self.compute_metrics(eval_dataset)

        self.save_to_csv(results, self.get_pipeline_name(pipeline), results_save_dir)
        return results

    def evaluate_pipeline_batch(self, 
                                ambig_qa_path: str = '/root/data/ambigqa/full/validation.parquet', 
                                pipeline: Callable = None, 
                                if_print: bool = False, 
                                results_save_root: str = './QA_Evaluation/',  # 指定保存目录
                                test: bool = True,
                                batch_size: int = 32,
                                save_full_results_dir: str = 'QA_Prediction.csv',
                                metrics_save_name: str = 'LLM_only_metrics.csv',
                                save_interval: int = 0  # 新增参数：定期保存的间隔（单位：batch）
                                ):  # 新增参数：是否保存完整预测结果
            
        '''
        批量评估 Pipeline，适用于大规模数据集。
        保存所有预测结果和黄金答案到 CSV，增强健壮性。
        Args:
            save_interval: int, 定期保存的间隔（单位：batch）, 0表示不启用定时保存机制
        '''
        print('=' * 20)
        print('Start Evaluation...')

        print('=' * 20)
        print('Loading Dataset')

        try:
            df = pd.read_parquet(path=ambig_qa_path)
        except Exception as e:
            print(f"[ERROR] Failed to load dataset from {ambig_qa_path}: {e}")
            return None

        if test:
            df = df.iloc[:10]  # 测试模式下仅取前2条数据

        cache_dataset = pd.DataFrame({
            'id': df['id'],
            'question': df['question'], 
            'golden_answers': df['nq_answer']
        })

        predict_answer = []
        print('=' * 40)
        print('Start Batch Eval:')

        # 避免父目录不存在
        Path(results_save_root).mkdir(parents=True, exist_ok=True)

        total_samples = len(cache_dataset)
        batch_count = 0  # 新增：记录当前是第几个 batch

        with tqdm(total=(total_samples + batch_size - 1) // batch_size) as pbar:
            for i in range(0, total_samples, batch_size):
                batch = cache_dataset.iloc[i:i+batch_size]
                questions = batch['question'].tolist()

                # 批量调用 pipeline（假设 pipeline 支持批量输入）
                try:
                    batch_preds = pipeline(questions)
                except Exception as e:
                    print(f"[ERROR] Pipeline failed for batch starting at index {i}: {e}")
                    batch_preds = ['ERROR'] * len(questions)  # 用占位符标记失败

                # 验证 pipeline 输出是否为列表
                if not isinstance(batch_preds, list) or len(batch_preds) != len(questions):
                    print(f"[WARNING] Pipeline output is invalid for batch {i}. Expected list of length {len(questions)}, got {batch_preds}")
                    batch_preds = ['ERROR'] * len(questions)

                predict_answer.extend(batch_preds)

                # 打印调试信息（可选）
                if if_print and i == 0:
                    print('=' * 20)
                    print(f'[Sample {i}]')
                    print('questions: ', questions[:2])
                    print('predictions: ', batch_preds[:2])

                # 增加 batch_count
                batch_count += 1

                # 定期保存机制（该机制完成测试，可以使用）
                if save_interval > 0 and batch_count % save_interval == 0 and save_full_results_dir:
                    current_len = len(predict_answer)
                    partial_df = cache_dataset.iloc[:current_len].copy()
                    partial_df['pred'] = predict_answer
                    full_results_path = os.path.join(results_save_root, save_full_results_dir)
                    try:
                        partial_df.to_csv(full_results_path, index=False)
                        pbar.set_description((f"[INFO] Partial results saved to {full_results_path} at batch {batch_count}"))
                    except Exception as e:
                        print((f"[ERROR] Failed to save partial results: {e}"))

                pbar.update()

        # 将预测结果写入 DataFrame
        cache_dataset['pred'] = predict_answer

        print(cache_dataset)
        # 构建验证数据集并计算指标
        try:
            eval_dataset = Validation_Dataset.from_dataframe(cache_dataset)
            results = self.compute_metrics(eval_dataset)
        except Exception as e:
            print(f"[ERROR] Metric computation failed: {e}")
            results = {'error': str(e)}

        # 保存完整数据集（含预测结果和黄金答案）
        if save_full_results_dir:
            full_results_path = os.path.join(results_save_root, save_full_results_dir)
            try:
                cache_dataset.to_csv(full_results_path, index=False)
                print(f"[INFO] Full results saved to {full_results_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save full results: {e}")

        # 保存结果指标
        metrics_path = os.path.join(results_save_root, metrics_save_name)
        try:
            pd.DataFrame([results]).to_csv(metrics_path, index=False)
            print(f"[INFO] Evaluation metrics saved to {metrics_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save metrics: {e}")

        return results

    def get_pipeline_name(self, pipeline):
        """
        根据 pipeline 对象获取其名称。

        支持以下几种场景：
        - 函数对象（function）
        - 类实例（class instance）
        - 用户自定义 name 属性

        参数:
            pipeline (callable): 一个可调用对象，例如函数或类实例。

        返回:
            str: pipeline 的名称。
        """
        if hasattr(pipeline, '__name__'):
            return pipeline.__name__
        elif hasattr(pipeline, '__class__') and hasattr(pipeline.__class__, '__name__'):
            return pipeline.__class__.__name__
        elif hasattr(pipeline, 'name'):
            return pipeline.name
        else:
            raise ValueError("无法自动获取 pipeline 名称，请手动提供名称。")


    def save_to_csv(self, results, pipeline_name, filename='results.csv'):
        """
        将评估结果以 CSV 格式保存，第一列为 pipeline 名称，其余列为指标及其结果。
        
        参数:
            results (dict): 评估结果字典，键为指标名称，值为对应的数值。
            pipeline_name (str): 当前 pipeline 的名称。
            filename (str): CSV 文件保存路径，默认为 'results.csv'。
        """
        # 构造列名：pipeline_name + 所有指标名称
        fieldnames = ['pipeline_name'] + list(results.keys())

        # 检查文件是否存在
        file_exists = os.path.isfile(filename)

        # 打开文件并写入数据
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 如果是新文件，写入表头
            if not file_exists:
                writer.writeheader()

            # 构造当前行数据
            row = {'pipeline_name': pipeline_name}
            row.update(results)

            # 写入当前行
            writer.writerow(row)

def evaluate_LLM_only(test = False):

    from .large_language_model import Large_Language_Model_API

    evaluate_batch_size = 8
    evaluator = Evaluator()

    model = Large_Language_Model_API()
    model.init_llm(system_prompt = [
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

"""
    for i in range(evaluate_batch_size)])


    pipeline = model.chat_without_history
        
    evaluator.evaluate_pipeline_batch(
        pipeline = pipeline, 
        test = test, 
        results_save_root = './QA_Evaluation/' + 'LLM_Only/',
        batch_size = evaluate_batch_size, 
        save_interval = 1
    )

def evaluate_naive_rag(test = False):

    from .large_language_model import Large_Language_Model_API
    from lib.RAG_modules import Naive_RAG

    evaluate_batch_size = 8
    evaluator = Evaluator()

    model = Large_Language_Model_API(
        model = 'gpt-4o-mini'
    )
    model.init_llm(system_prompt = [
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
Document:
[Title]...[Context]...
Your Output: 
1945.

**Example 2**:
User Input:
Who invented the modern three-point seatbelt?
Document:
[Title]...[Context]...
Your Output: 
Nils Bohlin. 

**Example 3**:
User Input:
What is the most widely spoken language in the world?
Document:
[Title]...[Context]...
Your Output: 
Mandarin Chinese. 

Please answer the user's input follow the format upon. 
"""
         for i in range(evaluate_batch_size)
         ])
    
    naive_rag = Naive_RAG(
        model = model, 
    )
    pipeline = naive_rag.generate
        
    evaluator.evaluate_pipeline_batch(
        pipeline = pipeline, 
        test = test, 
        results_save_root = './QA_Evaluation/' + 'naive_rag/',
        batch_size = evaluate_batch_size
    )

def evaluate_query_rewriter(test = False):

    from lib.RAG_modules import Retriever_Augmented_Generation
    RAG = Retriever_Augmented_Generation(
        LLM_model_local_dir = 'Query_Rewriter_llama3_8B'
    )
    evaluator = Evaluator()
    pipeline = RAG.query_rewrite

    evaluator.evaluate_pipeline(
        pipeline = pipeline,
        test = test, 
    )


def evaluate_Our_pipeline():

    from lib.RAG_modules import AERR
    from lib.config import MyTrainConfig

    config = MyTrainConfig().to_AERRConfig()
    config.decision.lora_dir = True
    config.decision.lora_dir = [
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-27-21-38-45",
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-27-23-55-29",
    ]
    # 加载方式
    pipeline = AERR(config)
    pipeline.release()
    pipeline.reload(config.decision)
    output_func = pipeline.generate_batch
    evaluator = Evaluator()
    
    evaluator.evaluate_pipeline(
        ambig_qa_path = '/root/autodl-tmp/data/ambigqa/full/validation.parquet', 
        pipeline = output_func,
        results_save_dir = './QA_Evaluation/AERR', 
    )



