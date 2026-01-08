from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, Type, List, Any, TypeVar, Generic, Union
import torch
from abc import ABC
import os
import numpy as np

if not "datetime" in globals():
    import datetime

T = TypeVar('T')

@dataclass
class BaseConfig(ABC, Generic[T]):
    """
    抽象基类，为所有配置类提供统一的 todict 接口。
    """

    def todict(self, exclude_types: List[Type] = []) -> dict:
        """
        递归将数据类转换为字典，但保留 exclude_types 中的类型为实例。

        :param exclude_types: 不转换为字典的类型列表
        :return: 转换后的字典
        """
        if not isinstance(exclude_types, List):
            exclude_types = [exclude_types]
        result = {}

        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                result[field_name] = None
            elif is_dataclass(field_value) and type(field_value) not in exclude_types:
                # 如果是数据类且不在 exclude_types 中，则递归转换
                result[field_name] = field_value.todict(exclude_types)
            
            elif type(field_value) in exclude_types:
                # 如果是 exclude_types 中的类型，则保留实例
                result[field_name] = field_value
            else:
                # 其他类型直接放入字典
                result[field_name] = field_value

        return result
    
@dataclass
class GenerativeConfig(BaseConfig):
    model_dir: str = None
    batchsize: int = 1

@dataclass
class StrategyParams(BaseConfig):
    max_tokens: int = 1000
    temperature: float = 1.0
    
@dataclass
class DecisionStrategyParams(BaseConfig):
    do_sample: bool = True
    num_return_sequences: int = 1 # 每个输入生成一个结果
    max_tokens: int = 1000
    temperature: float = 1.0

@dataclass
class APIStrategyParams(BaseConfig):
    max_tokens: int = 2000

@dataclass
class DecisionConfig(BaseConfig):
    model_dir: str = "./Qwen3-8B/"
    lora_dir: str = None # "/root/autodl-tmp/AfterTraining/Lora/2025-08-22-13-02-34/"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy_params: DecisionStrategyParams = field(default_factory=DecisionStrategyParams)
    batch_size: int = 32
    test: bool = False
    load_api: bool = False
    api_model: str = None
    load_without_model: bool = False

@dataclass
class ExecutionConfig(BaseConfig):
    # 使用Cpu作为indexer的加载位置
    indexer_device: torch.device = torch.device('cpu')
    model_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir: Optional[str] = None
    strategy_params: APIStrategyParams = field(default_factory=APIStrategyParams)
    index_load_path: str = "./wikipedia_BGE_L2.contriever"
    document_load_path: str = "./psgs_w100.tsv"
    batchsize: int = 1
    verbose: bool = False
    test: bool = False

@dataclass
class AERRConfig(BaseConfig):
    test: bool = False
    
    init_generate_model: bool = True
    generative: GenerativeConfig = field(default_factory=GenerativeConfig)

    init_decision_model: bool = True
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    
    init_execution_model: bool = True
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    def __post_init__(self):
        """将 test 参数同步到子模块配置"""
        self.decision.test = self.test
        self.execution.test = self.test

        
# ==========================================Pipeline Config================================================
@dataclass
class PipelineConfig:
    """
    通用训练流程配置类，用于参数化 TrainPipeline 的行为。
    """
    model_dir: str = "./default_model/"  # 模型路径
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    batch_size: int = 32  # 批量大小
    max_epochs: int = 10  # 最大训练轮次

    # 可扩展字段（子类可添加自己的配置）


# =================================================LlamaFactory Config==================================================
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict
import yaml
import torch
@dataclass(frozen = False)
class LlamaFactoryConfig:
    """
    统一配置类，包含所有训练和推理相关参数。
    """

    # Model configuration
    model_name_or_path: str = "/root/autodl-tmp/qwen2.5_1.5B/"
    trust_remote_code: bool = True
    
    # Method configuration
    stage: str = "dpo"
    do_train: bool = True
    finetuning_type: str = "lora"
    lora_rank: int = 8
    lora_target: str = "all"
    pref_beta: float = 0.1
    pref_loss: str = "sigmoid"  # choices: [sigmoid (dpo), orpo, simpo]
    
    # Dataset configuration
    dataset: str = "dpo_en_demo"
    template: str = "qwen"
    cutoff_len: int = 2048
    max_samples: int = 1000
    overwrite_cache: bool = True
    preprocessing_num_workers: int = 16
    dataloader_num_workers: int = 4
    
    # Output configuration
    output_dir: str = "/root/autodl-tmp/AfterTraining/DPO_QWEN2.5_1.5B_No"
    logging_steps: int = 10
    save_steps: int = 500
    plot_loss: bool = True
    overwrite_output_dir: bool = True
    save_only_model: bool = False
    report_to: str = "none"  # choices: [none, wandb, tensorboard, swanlab, mlflow]
    
    # Training configuration
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5.0e-6
    num_train_epochs: float = 3.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    ddp_timeout: int = 180000000
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation configuration
    eval_dataset: Optional[str] = None
    val_size: Optional[float] = 0.1
    per_device_eval_batch_size: Optional[int] = 1
    eval_strategy: Optional[str] = "steps"
    eval_steps: Optional[int] = 500

    def to_dict(self) -> Dict:
        """
        将配置对象转换为字典。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # 获取类的所有字段名
        valid_fields = {f.name for f in fields(cls)}
        # 过滤 config_dict 中多余的字段
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        # 补充缺失的字段默认值
        for field in fields(cls):
            if field.name not in filtered_config:
                filtered_config[field.name] = field.default
        return cls(**filtered_config)

    def to_yaml(self, file_path: str):
        """
        将配置对象写入 YAML 文件。
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        从 YAML 文件加载配置。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def __getitem__(self, key: str) -> Any:
        """
        支持字典式访问：config["key"]

        参数:
            key (str): 字段名称

        返回:
            Any: 字段值

        抛出:
            KeyError: 如果字段不存在
        """
        for field in fields(self.__class__):
            if field.name == key:
                return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        支持字典式修改：config["key"] = value

        参数:
            key (str): 字段名称
            value (Any): 要设置的值

        抛出:
            KeyError: 如果字段不存在
        """
        for field in fields(self.__class__):
            if field.name == key:
                setattr(self, key, value)
                return
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

@dataclass
class MyTrainConfig:
    load_api: bool = False
    api_model: str = None # "gpt-4o"太贵哩

    # 训练参数
    batch_size: int = 16
    max_tokens: int = 1024
    epochs: int = 20
    temperature: float = 1.0 # 1.4
    top_p: float = 0.9 # 0.9
    time_reward_refresh_step: int = 15 # 时间奖励刷新交互轮数

    # 树参数
    sample_mode = "normal" # 要么 normal 要么 forest 森林采样已经实现
    baseline_reward: float = 0.6 # 采样的baseline部分，要求至少高于这个baseline的部分才会得到学习
    normal_sampling_num: int = 2 # 普通采样下的sampling的次数

    build_pair_percent: float = 0.1
    build_pair_gap: float = 0 # =0时关闭该功能
    build_pair_garbage_threshold: float = 100 # 取大于2的值时关闭该功能
    activate_critic: bool = True # 是否启用api的格式筛选
    activate_filter: bool = False # 是否启用json文件格式筛选？但该方法是失效的
    max_tree_length: int = 4 # 固定为4轮吧，不能再改了，更长的无法接受
    sampling_num: int = 4 # 乘数、成长
    sampling_num_decay: int = 4 # 4 # 除数、衰减

    # 可视化
    UseTensorboard: bool = True
    tensorboard_log: str = "/root/tf-logs/" + "1030/"# + datetime.datetime.now().strftime("%m%d") # + "1026/" # 
    curr_step: int = 0

    # 模型加载配置
    model_path: str = "/root/autodl-tmp/Qwen3-8B"
    lora_dir: str =  "/root/autodl-tmp/AfterTraining/AERR/HotpotQA/1208/checkpoint-5680" # None 
    
    # 模型保存配置
    model_output_dir: str = "/root/autodl-tmp/AfterTraining/GoldenFormatLoraAdaptor8B"
    model_cache_num: int = 2
    model_run_train_first: bool = False
    
    # 工作环境参数
    llama_dir: str = '/root/LLaMA-Factory/'
    my_cmd: str = '/root/autodl-tmp/'
    
    # 奖励参数
    alpha: float = 0.1  # 精确度和时间的折减系数
    mean_time_baseline: float = 500.0 # 平均耗时，作为参考
    
    # 数据集参数
    ambig_qa_dir: str = "/root/autodl-tmp/data/ambigqa/full/validation.parquet"
    dataset_question_column: str = 'question'
    dataset_golden_answer_column: str = 'nq_answer'
    dataset_shorten_nums: int = 50
    add_to_sampling_path: bool = True # 是否保存到用于采样的json文件中
    sampling_json_path: str = "/root/autodl-tmp/data/Samplingdata.json" # 后续采样文件地址
    sft2dpo_sampling_json_dir: str = "/root/autodl-tmp/data/SFT2DPO/"

    # 验证
    eval_interval: int = 1000 # 验证间隔，其实这里是倍数
    save_training_interaction_interval: int = 20 # 训练数据采样间隔
    eval_example_dir: str = "/root/autodl-tmp/QA_Evaluation/AmbigQA/AERR/1214/" # 示例结果

    def __init__(self, train_mode: str = "sft", test = False):
        if train_mode == "dpo":
            # 数据集和配置文件
            self.data_format = "dpo"
            self.train_dataset_save_dir: str = "/root/autodl-tmp/data/CacheTrainingData/mydataset.json"
            self.yaml_path: str = "/root/autodl-tmp/config/DPOTrainingConfig.yaml"

        elif train_mode == "sft":
            self.data_format = "sft"
            self.train_dataset_save_dir: str = "/root/autodl-tmp/data/CacheTrainingData/SFTData.json"
            self.yaml_path: str = "/root/autodl-tmp/config/SFTTrainingConfig.yaml"

        self.test = test
        if test == True:
            self.dataset_shorten_nums = 3

    def load_yaml(self, yaml_path: str = None, merge: bool = True) -> None:
        """
        从 YAML 文件加载配置并合并到当前实例
        
        参数:
            yaml_path: 自定义YAML文件路径，若为None则使用实例自身的yaml_path
            merge: 是否合并配置（True: 保留实例原有值，仅覆盖YAML中存在的字段；False: 完全替换为YAML配置）
        """
        # 确定最终的YAML路径
        final_yaml_path = yaml_path or self.yaml_path
        if not final_yaml_path or not os.path.exists(final_yaml_path):
            raise FileNotFoundError(f"YAML配置文件不存在: {final_yaml_path}")

        # 读取YAML文件
        with open(final_yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}

        # 验证YAML配置格式
        if not isinstance(yaml_config, dict):
            raise ValueError(f"YAML配置文件格式错误，预期为字典，实际为: {type(yaml_config)}")

        # 获取当前实例的所有字段（dataclass）
        instance_fields = {f.name: f for f in fields(self)}

        # 遍历YAML配置并合并到实例
        for key, value in yaml_config.items():
            # 跳过不存在的字段
            if key not in instance_fields:
                print(f"警告: YAML中的字段 {key} 不存在于 MyTrainConfig，已忽略")
                continue

            # 合并模式：仅当值不为None时覆盖（或强制替换）
            if not merge or value is not None:
                # 类型检查和转换（保证类型安全）
                field_type = instance_fields[key].type
                try:
                    # 处理基础类型转换
                    converted_value = self._convert_type(value, field_type)
                    setattr(self, key, converted_value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"字段 {key} 类型转换失败: 预期 {field_type}，实际 {type(value)}，值: {value}\n错误: {e}"
                    )

        # 特殊处理test模式的dataset_shorten_nums（确保优先级）
        if self.test and 'dataset_shorten_nums' in yaml_config and merge:
            print("警告: test模式下强制将dataset_shorten_nums设为3，覆盖YAML配置")
            self.dataset_shorten_nums = 3

    def _convert_type(self, value: Any, target_type: Any) -> Any:
        """
        类型转换辅助方法，确保YAML中的值符合字段类型要求
        
        参数:
            value: YAML中读取的值
            target_type: 目标类型（dataclass字段类型）
        
        返回:
            转换后的值
        """
        # 处理Optional类型（如str | None）
        if hasattr(target_type, '__origin__') and target_type.__origin__ is Union:
            args = target_type.__args__
            # 检查是否是 Optional[X] (X | None)
            if len(args) == 2 and args[1] is type(None):
                target_type = args[0]
                if value is None:
                    return None

        # 基础类型转换
        if target_type is int and not isinstance(value, int):
            return int(value)
        elif target_type is float and not isinstance(value, float):
            return float(value)
        elif target_type is bool and not isinstance(value, bool):
            # 处理YAML中的布尔值（如"true"/"false"字符串）
            if isinstance(value, str):
                lower_val = value.lower()
                if lower_val in ('true', 'false'):
                    return lower_val == 'true'
            raise ValueError(f"无法转换为布尔值: {value}")
        elif target_type is str and not isinstance(value, str):
            return str(value)
        else:
            # 无需转换或复杂类型（如list/dict）直接返回
            return value
        
    def to_LlamaFactoryConfig(self) -> LlamaFactoryConfig:
        """
        将当前 MyTrainConfig 转换为 LlamaFactoryConfig 实例，对齐 model_path 和 model_output_dir。

        返回:
            LlamaFactoryConfig: 对齐后的配置对象。
        """
        # 构建 LlamaFactoryConfig 实例
        lf_config = LlamaFactoryConfig().from_yaml(self.yaml_path)
        lf_config.model_name_or_path = self.model_path
        lf_config.output_dir = self.model_output_dir
        return lf_config
    
    def to_AERRConfig(self) -> AERRConfig:
        """
        """
        current_dir = os.getcwd()

        # 处理相对路径为绝对路径
        model_path = self.model_path
        model_output_dir = self.model_output_dir

        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, model_path)
        if not os.path.isabs(model_output_dir):
            model_output_dir = os.path.join(current_dir, model_output_dir)

        # 验证路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"`model_path` `{model_path}` 不存在，请检查配置。")
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir, exist_ok=True)  # 自动创建输出目录（可选）

        # 处理相对路径为绝对路径
        model_output_dir = self.model_output_dir

        # 构建 LlamaFactoryConfig 实例
        config = AERRConfig()
        config.decision.model_dir = model_path
        config.decision.batch_size = self.batch_size
        config.decision.strategy_params.max_tokens = self.max_tokens
        config.decision.strategy_params.temperature = self.temperature
        # config.decision.strategy_params.top_p = self.top_p
        config.decision.lora_dir = self.lora_dir
        config.decision.load_api = self.load_api
        config.decision.api_model = self.api_model
        config.test = self.test

        return config

