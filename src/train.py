from typing import List, Union, Dict, Any
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, Subset, random_split
from typing import List, Union, Callable, Tuple
import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset
from flashrag.evaluator.metrics import ExactMatch
import random
import time
import yaml
import subprocess
import json
import shlex
import tempfile
import datetime
import warnings
import sys
import tqdm
import copy
import gc


from torch.utils.tensorboard import SummaryWriter

from .RAG_modules import AERR
from .config import *
from .reward import weighted_reward_calculate, validate_json_file
from .dataset import golden_dataset, ConversationTree, from_tree_json_to_sampling_json

class Trainer:
    """
    维护训练配置，包括本地读取、修改、保存 YAML 文件
    运行命令行，进行微调操作
    """
    def __init__(self, 
                 llamafactory_config_path: str = None, 
                 config: LlamaFactoryConfig = None):

        if (llamafactory_config_path is None) and (config is None):
            self.lf_config: LlamaFactoryConfig = LlamaFactoryConfig()
        elif config is not None:
            self.lf_config = config
        elif llamafactory_config_path is not None:
            self.load_config_from_yaml(llamafactory_config_path)

        # 存储已存在的模型目录
        self.exists_model_dir = []
        self.train_config = None
        self.lf_config = None
        self.dec_config = None
        self.warm_start_Lora_dir = None

    def _load_config_from_yaml(self, path: str):
        # 示例方法，实际应根据具体实现加载配置
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def modify_train_config(self, **kwargs):
        """
        修改训练配置，支持嵌套键路径（如 optimizer.learning_rate）。
        如果键路径不存在，则抛出警告，不进行修改。
        """
        for key, value in kwargs.items():
            path = key.split('.')
            parent, last_key = self._get_parent_and_last_key(path)
            if parent is not None:
                if self._key_exists_in_parent(parent, last_key):
                    self._set_value(parent, last_key, value)
                else:
                    warnings.warn(f"配置键 {key} 不存在，无法修改")
            else:
                warnings.warn(f"配置键 {key} 不存在，无法修改")

    def _get_parent_and_last_key(self, path: list):
        """
        根据键路径获取父级对象和最后一个键名。
        如果路径无效，返回 (None, None)
        """
        current = self.lf_config
        for i, part in enumerate(path[:-1]):
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return None, None
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None, None
        return current, path[-1]

    def _key_exists_in_parent(self, parent, key: str) -> bool:
        """检查父级对象中是否存在指定的键"""
        if isinstance(parent, dict):
            return key in parent
        else:
            return hasattr(parent, key)

    def _set_value(self, parent, key: str, value):
        """设置父级对象中指定键的值"""
        if isinstance(parent, dict):
            parent[key] = value
        else:
            setattr(parent, key, value)
    
    def load_config_from_yaml(self, yaml_path: str):
        """从 YAML 文件加载配置并更新 self.lf_config"""
        if not os.path.exists(yaml_path):
            self.lf_config = LlamaFactoryConfig()
            warnings.warn(f"YAML file {yaml_path} not found. Using default config.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        self.lf_config = LlamaFactoryConfig.from_dict(config_dict)

    def log(self, log_dir: str = 'TrainingLogs'):
        """
        将当前配置保存到指定目录下，文件名格式为：config_YYYYMMDD_HHMMSS.yaml

        :param log_dir: 日志保存目录路径
        """
        # 1. 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"config_{timestamp}.yaml"
        log_path = os.path.join(log_dir, log_filename)

        # 2. 确保目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 3. 将配置转换为字典并写入 YAML 文件
        try:
            config_dict = self.lf_config.to_dict()
            with open(log_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f)
            print(f"配置已保存到: {log_path}")
        except Exception as e:
            raise RuntimeError(f"保存配置到 {log_path} 失败: {e}")
        
    def modify_yaml(self, file_path, key, value):
        """
        修改 YAML 文件中的指定键值对。
        
        :param file_path: YAML 文件路径
        :param key: 要修改的键，格式如 "a.b.c"
        :param value: 新的值
        :raises FileNotFoundError: 如果文件不存在
        :raises ValueError: 如果键格式非法
        :raises KeyError: 如果键路径中任意一级不存在
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")

        # 读取 YAML 文件
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # 分割键
        keys = key.split('.')
        if not keys or any(not k.strip() for k in keys):
            raise ValueError("Invalid key format. Key must be non-empty and not contain empty parts.")

        # 遍历键路径，逐层查找
        current = data
        for k in keys[:-1]:
            if not isinstance(current, dict):
                raise KeyError(f"Key path invalid at '{k}': parent is not a dictionary")
            if k not in current:
                raise KeyError(f"Key '{k}' not found in YAML")
            current = current[k]

        # 检查最后一个键是否存在
        last_key = keys[-1]
        if not isinstance(current, dict):
            raise KeyError(f"Parent of key '{last_key}' is not a dictionary")
        if last_key not in current:
            raise KeyError(f"Key '{last_key}' not found in YAML")

        # 修改值
        current[last_key] = value

        # 写回文件
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def run_command(self, command, yaml_path=None):
        """
        执行命令行命令，支持从 YAML 文件读取参数并添加为命令行参数。

        :param command: 要执行的命令字符串。
        :param yaml_path: 可选的 YAML 配置文件路径。
        ##测试完毕##
        """
        # 解析命令字符串为列表
        cmd = shlex.split(command)

        # 如果提供了 YAML 文件路径，则读取并转换为命令行参数
        if yaml_path:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAML 文件 {yaml_path} 不存在。")

            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"YAML 文件格式错误: {e}")

            # 转换 YAML 参数为命令行参数
            params = []
            for key, value in config.items():
                if value is None:
                    continue
                params.extend(["--" + key, str(value)])
            cmd += params

        # 执行命令
        subprocess.run(cmd, check=True)
    
    def run_train_command(self, 
                  yaml_path: Optional[str] = None, 
                  log_dir : str = "/root/autodl-tmp/TrainingLogs", **kwargs):
        """
        执行 LlamaFactory 训练命令，支持动态参数和环境变量设置。

        :param yaml_path: 可选的 YAML 配置文件路径。
        :param **kwargs: 可选参数，用于覆盖 YAML 中的配置或设置环境变量。
        """

        need_save_path = None
        tmp_yaml_path = None
        FinalConfigPath = None

        # 如果未输入yaml_path，则创建临时yaml文件
        if yaml_path is None:
            is_temp_yaml = True 

        # 如果输入了但是发现路径不存在，同样创建临时文件，但指定临时文件路径
        elif not os.path.exists(yaml_path):
            is_temp_yaml = True
            need_save_path = yaml_path
        else:
            FinalConfigPath = yaml_path
            is_temp_yaml = False

        try:
            # 如果未指定 YAML 路径，则使用内部配置生成临时文件
            if is_temp_yaml:
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp_file:
                    tmp_yaml_path = tmp_file.name
                    config_dict = self.lf_config.to_dict()
                    yaml.dump(config_dict, tmp_file)

                FinalConfigPath = tmp_yaml_path

            if need_save_path is not None:

                FinalConfigPath = need_save_path
                # 将临时文件复制到用户指定路径
                with open(tmp_yaml_path, "r", encoding="utf-8") as src:
                    content = src.read()
                with open(yaml_path, "w", encoding="utf-8") as dst:
                    dst.write(content)

            # 将当前config保存到日志中
            self.log(log_dir = log_dir)

            # 实时更新当前config
            if not is_temp_yaml:
                self.load_config_from_yaml(yaml_path)

            # 1. 分离环境变量和命令行参数
            env_vars = []
            cli_args = []

            for key, value in kwargs.items():
                # 判断是否为环境变量（例如 CUDA_VISIBLE_DEVICES）
                if key.upper() in os.environ:
                    env_key = key.upper()
                    env_vars.append(f"{env_key}={value}")
                else:
                    # 转换为命令行参数（例如 learning_rate -> --learning-rate）
                    cli_key = key.replace("_", "-")
                    cli_args.append(f"--{cli_key}")
                    cli_args.append(str(value))

            # 修改FinalConfigPath直至符合命令行需求
            FinalConfigPath = '/' + FinalConfigPath.replace('\\', '/')
  
            # 2. 构建完整命令
            base_cmd = f'llamafactory-cli train {FinalConfigPath}'
            full_cmd = " ".join(env_vars) + " " + base_cmd + " " + " ".join(cli_args) if env_vars else base_cmd + " " + " ".join(cli_args)
            
            # 去头去尾的空格
            full_cmd = full_cmd.strip()
            # 3. 使用 subprocess 执行命令
            subprocess.run(full_cmd, shell=True, check=True)
            
            if tmp_yaml_path is not None:
                # 删除临时文件
                os.remove(tmp_yaml_path)
                tmp_yaml_path = None
                is_temp_yaml = False  # 标记为非临时文件

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"命令执行失败: {e}")

        finally:
            # 仅在内部生成临时 YAML 时删除文件
            if is_temp_yaml and os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)

    def clear_model_cache(self, max_num: int = 2):
        """
        清理缓存，当本地模型数量超过阈值时，删除最旧的模型
        注意，本地实际需求空间大小为nums + 1个，设定的基础上加一个
        """
        # 缓存删除机制
        if len(self.exists_model_dir) > max_num:
            need_del_model_dir = self.exists_model_dir[0]
            self.run_command(f"rm -rf {need_del_model_dir}")
            self.exists_model_dir = self.exists_model_dir[1: ]
            gc.collect()

        return 
    
    def run_train(self, 
                  pipeline: AERR, 
                  config: MyTrainConfig):
        """
        运行训练，使用DPOLora方法来训练模型
        """

        pipeline.release()
        set_working_directory(config.llama_dir)
        # 训练时指定文件夹保存，已经保存的路径会放到exists_model_dir中
        current_lora_dir = self.train_config.model_output_dir + '/Lora/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
        current_fintuned_model_dir = self.train_config.model_output_dir + '/AERR/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
        if not os.path.exists(current_lora_dir):
            os.makedirs(current_lora_dir)
        if not os.path.exists(current_fintuned_model_dir):
            os.makedirs(current_fintuned_model_dir)

        # 把输出的lora_dir置为当前dir
        self.lf_config.output_dir = current_lora_dir
        self.run_train_command()
        self.lf_config.model_name_or_path = current_fintuned_model_dir
        # pipeline重新加载并更换工作目录cmd到原来采样的目录中
        set_working_directory(config.my_cmd)
        
        # 同样是错位加载
        self.dec_config.lora_dir = current_lora_dir
        pipeline.reload(self.dec_config)
        self.dec_config.model_dir = current_fintuned_model_dir

        # 保存并清理模型缓存
        pipeline.save_model(current_fintuned_model_dir)
        self.exists_model_dir.append(current_fintuned_model_dir)
        self.clear_model_cache()

        return 
    
    def train(self, pipeline: AERR = None, lora_dir = None, **kwargs):
        """
        可以直接对pipeline进行训练，调整为通用的训练框架
        确保有release和reload函数，以释放或者加载内存
        以及确保有sampling函数，该训练框架应当与树状结构配套使用
        """
        config = MyTrainConfig(**kwargs)
        if not lora_dir == None:
            config.lora_dir = lora_dir
        self.train_config = config
        if self.warm_start_Lora_dir is not None:
            config.lora_dir = self.warm_start_Lora_dir
            
        self.lf_config = copy.deepcopy(config.to_LlamaFactoryConfig())
        self.dec_config = config.to_AERRConfig().decision

        # 启用tensorboard
        if config.UseTensorboard:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir = config.tensorboard_log + timestamp)

        print('=' * 20)
        print('Loading AERR')
        if pipeline == None:
            pipelieConfig = config.to_AERRConfig()
            pipelieConfig.test = False
            pipeline = AERR(pipelieConfig)
        if not config.lora_dir is None:
            pipeline.release()
            pipeline.reload(self.dec_config)

            # 重新加载lora时，应记得保存到本地
            save_dir = self.lf_config.output_dir + '/AERR/BaseModel' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
            pipeline.save_model(save_dir)
            self.lf_config.model_name_or_path = save_dir
            self.exists_model_dir.append(save_dir)

        # 已存在交互过后的数据集时，则进行训练，再进行交互
        if config.model_run_train_first:
            self.run_train(pipeline = pipeline, config = config)

        print('=' * 20)
        print('Loading Dataset')
        # 加载数据集
        golden_QA = golden_dataset.from_path(
            path=config.ambig_qa_dir,
            question_column=config.dataset_question_column,
            golden_answer_column=config.dataset_golden_answer_column,
        )
        
        # 创建交互树
        tree = ConversationTree()
        print('=' * 20)
        print('Start Interaction & training!')
        with tqdm.tqdm(total = config.epochs) as pbar:

            for epoch in range(config.epochs):
                # 数据裁剪抽样
                golden_QA.restore_data()
                golden_QA.shuffle()
                golden_QA.shorten_data(n_samples=config.dataset_shorten_nums)
                conversation_examples = []

                weighted_reward_list = []
                format_reward_list = []
                time_list = []
                f1_reward_list = []

                # 交互部分
                for _ in range(len(golden_QA)):
                    pbar.set_description(f'Current Sample/Total Sample: {_}/{len(golden_QA)}')
                    # 一个个进行采样即可
                    questions, golden_answers = golden_QA.get_data(batch_size = 1)
                    question, golden_answer = questions[0], golden_answers[0]
                    
                    # 进行DPO采样
                    tree, output = pipeline.sampling(question, 
                                                    sampling_nums = config.sampling_num, 
                                                    sampling_decay = config.sampling_num_decay, 
                                                    test = config.test, 
                                                    tree = tree, 
                                                    max_tree_length = config.max_tree_length, 
                                                    max_tokens = config.max_tokens,
                                                    temperature = config.temperature, 
                                                    top_p = config.top_p)

                    # 引入格式评分
                    # tree.evaluate_format_score()
                    format_rewards = tree.get_all_nodes_format_scores()
                    format_reward_list.extend(format_rewards)
                    time_costs = tree.get_cumulative_times_last_layer()
                    
                    # 权重奖励计算！
                    reward, (f1_rewards, time) = weighted_reward_calculate(question = question, 
                                                        golden_answer = golden_answer, 
                                                        predictions = output, # 有问题
                                                        time = time_costs, 
                                                        alpha = config.alpha, 
                                                        return_rewards = True)
                    
                    weighted_reward_list.append(float(np.mean(reward)))
                    f1_reward_list.append(float(np.mean(f1_rewards)))
                    time_list.append(float(np.mean(time)))
                    tree.add_reward(reward)
                    
                    # 采样最后一层交互结果
                    conversation_examples.extend([{"input": tree.layers[-1][i]["input"], 
                                                    "response": tree.layers[-1][i]["response"]}
                                                        for i in range(tree.total_bottom_nodes())])

                    # 依照分数构造pair并保存到本地，以供后续训练
                    tree.pair2json(config.train_dataset_save_dir, 
                                    mode = 'w' if (_ == 0 and epoch != 0) else 'a', 
                                    system = pipeline.decision_agent._init_prompt(), 
                                    top_percent = config.build_pair_percent, 
                                    gap = config.build_pair_gap, 
                                    garbage_threshold = config.build_pair_garbage_threshold, 
                                    activate_critic = config.activate_critic,
                                    data_format = config.data_format, 
                                    ) # 仅训练决策模型
                    
                    # SFT2DPO Data
                    if config.add_to_sampling_path and config.data_format == "sft":
                        if _ == 0:
                            dir_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        # 依照分数构造pair并保存到本地，以供后续训练
                        tree.pair2json(config.sft2dpo_sampling_json_dir + dir_time + "Samplingdata.json", 
                                        mode = 'a', 
                                        system = pipeline.decision_agent._init_prompt(), 
                                        top_percent = config.build_pair_percent, 
                                        gap = config.build_pair_gap, 
                                        garbage_threshold = config.build_pair_garbage_threshold, 
                                        activate_critic = config.activate_critic,
                                        data_format = "dpo", 
                                        ) # 仅训练决策模型
                        
                    # 清空树
                    tree.clear()
                # 采样
                if config.add_to_sampling_path:
                    from_tree_json_to_sampling_json(input_file_path = config.train_dataset_save_dir, 
                                                    output_file_path = config.sampling_json_path)

                # 验证结果
                # 将采样实例保存到本地
                save_to_jsonl(conversation_examples, config.eval_example_path)
                # 硬编码一个add_json

                # 可视化
                if config.UseTensorboard:
                    writer.add_scalar('Weighted Score', 
                                    scalar_value = np.mean(weighted_reward_list), 
                                    global_step = epoch)
                    writer.add_scalar('Format Score', 
                                    scalar_value = np.mean(format_reward_list), 
                                    global_step = epoch)
                    writer.add_scalar('F1 Score', 
                                    scalar_value = np.mean(f1_reward_list), 
                                    global_step = epoch)
                    writer.add_scalar('Time Reward Score', 
                                    scalar_value = np.mean(time_list), 
                                    global_step = epoch)
                    
                pbar.set_description(f'Current Weighted Score: {np.mean(weighted_reward_list)}')
                pbar.update(1)
                self.run_train(pipeline = pipeline, 
                               config = config)


        # 最后记得关掉tensorboard
        if config.UseTensorboard:
            writer.close()


        return 
    
    def warm_start(self):
        set_working_directory('/root/LLaMA-Factory/')
        self.lf_config = LlamaFactoryConfig.from_yaml("/root/autodl-tmp/config/SFTTrainingConfig.yaml")
        output_dir = "/root/autodl-tmp/AfterTraining/FormatLoraResult" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "V2/"
        self.lf_config.output_dir = output_dir
        self.run_train_command()
        self.warm_start_Lora_dir = output_dir

        set_working_directory('/root/autodl-tmp/')

        return output_dir
    
def save_to_jsonl(data: List[Dict], file_path: str) -> None:
    """
    将字典列表保存为 JSONL 文件（每行一个 JSON 对象）。

    参数:
        data (List[Dict]): 要保存的字典列表。
        file_path (str): 输出文件路径（以 `.jsonl` 结尾）。
    """
    # 获取文件所在目录路径
    directory = os.path.dirname(file_path)

    # 递归创建目录（若路径不存在）
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 写入 JSONL 文件
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


def set_working_directory(project_root: str) -> bool:
    """
    修改当前工作目录为指定路径。

    参数:
        project_root (str): 项目根目录路径（如 '/root/LLaMA-Factory'）。

    返回:
        bool: 如果路径有效且成功修改，返回 True；否则返回 False。
    """
    abs_path = os.path.abspath(project_root)
    
    if not os.path.isdir(abs_path):
        return False
    try:
        os.chdir(abs_path)
        return True
    except Exception as e:
        return False
    
def testTrainerTrain():
    import datetime
    import copy 
    exists_model_dir = []

    config = MyTrainConfig(test = False)
    
    lf_config = config.to_LlamaFactoryConfig()
    trainer = Trainer(config = copy.deepcopy(lf_config))

    sampling_path = os.path.curdir
    training_path = '/root/LLaMA-Factory/'
    # 当前工作目录
    set_working_directory(training_path)
    trainer.run_train()

def eval_format():
    """检查格式"""

    config = MyTrainConfig(test = False)
    # 测试格式分数是否正常
    config.dataset_shorten_nums = 100
    config.sampling_num = 1
    config.max_tree_length = 1
    config.eval_example_path = '/root/autodl-tmp/QA_Evaluation/format_example.jsonl'
    lf_config = LlamaFactoryConfig.from_yaml(config.yaml_path)
    
    # 启用tensorboard
    if config.UseTensorboard:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir = config.tensorboard_log + timestamp)

    # 创建 Trainer 对象
    print('=' * 20)
    print('Loading Trainer')
    trainer = Trainer(config = copy.deepcopy(lf_config))

    print('=' * 20)
    print('Loading AERR')
    pipeline = AERR(config.to_AERRConfig())
    if not config.lora_dir is None:
        pipeline.release()
        dec_config = config.to_AERRConfig().decision
        pipeline.reload_lora(dec_config)
        
    print('=' * 20)
    print('Loading Dataset')
    # 加载数据集
    golden_QA = golden_dataset.from_path(
        path=config.ambig_qa_dir,
        question_column=config.dataset_question_column,
        golden_answer_column=config.dataset_golden_answer_column,
    )
    
    # 创建交互树
    tree = ConversationTree()
    sampling_path = os.path.curdir
    training_path = '/root/LLaMA-Factory/'

    print('=' * 20)
    print('Start Interaction & training!')
    with tqdm.tqdm(golden_QA) as pbar:
        exists_model_dir = []
        scores_for_saving = [0]

            
        if len(exists_model_dir) >= 2:
            need_del_model_dir = exists_model_dir[0]
            trainer.run_command(f"rm -rf {need_del_model_dir}")
            exists_model_dir = exists_model_dir[1: ]
            gc.collect()

        # 随机抽取100个样本，进行热启动
        golden_QA.restore_data()
        golden_QA.shuffle()
        golden_QA.shorten_data(n_samples=config.dataset_shorten_nums)

        format_scores = []
        conversation_examples = []
        mean_format_score_list = []
        
        # 交互部分
        for _ in range(len(golden_QA)):
            pbar.set_description(f'Current Sample/Total Sample: {_}/{len(golden_QA)}')
            # 一个个进行采样即可
            questions, golden_answers = golden_QA.get_data(batch_size = 1)
            question, golden_answer = questions[0], golden_answers[0]
            
            # 进行DPO采样
            tree, output = pipeline.sampling(question, 
                                            sampling_nums = config.sampling_num, 
                                            test = config.test, 
                                            temperature = config.temperature, 
                                            tree = tree, 
                                            max_tree_length = config.max_tree_length, 
                                            max_tokens = config.max_tokens)

            # 评估采样格式得分
            tree.evaluate_format_score()
        
            # 记录格式分数
            format_scores.extend([tree.layers[-1][i]["reward"] 
                                for i in range(tree.total_bottom_nodes())])
            conversation_examples.extend([{"input": tree.layers[-1][i]["input"], 
                                            "response": tree.layers[-1][i]["response"]}
                                                for i in range(tree.total_bottom_nodes())])
            
        # 验证结果
        # 将验证结果保存到本地
        format_scores_mean = np.mean(format_scores)
        print("=" * 20)
        print("Current Format Score: ", str(format_scores_mean))

        # time_scores_mean = np.mean(time_scores)
        save_to_jsonl(conversation_examples, 
                        config.eval_example_path)
        
        saved_string = [
            "Model Name or Path: ", config.model_path, 
            "\nLora Name or Path", config.lora_dir, 
            "\nDataset Nums: ", str(config.dataset_shorten_nums), 
            "\nMean Format Score:", str(format_scores_mean)
        ]

        with open("/root/autodl-tmp/QA_Evaluation/format_reward_output.txt", 'w') as file:
            file.write(''.join(saved_string))
