from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, Type, List, Any, TypeVar, Generic
import torch
from abc import ABC
import os
import numpy as np


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
    max_tokens: int = 200
    temperature: float = 1.0
    
@dataclass
class DecisionStrategyParams(StrategyParams):
    do_sample: bool = True, 
    num_return_sequences: int = 1,  # 每个输入生成一个结果

@dataclass
class DecisionConfig(BaseConfig):
    model_dir: str = "./Qwen3-8B/"
    lora_dir: str = None # "/root/autodl-tmp/AfterTraining/Lora/2025-08-22-13-02-34/"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy_params: DecisionStrategyParams = field(default_factory=DecisionStrategyParams)
    batch_size: int = 1
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
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
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
    """
    DPO 训练配置类，封装所有训练参数。
    """
    load_api: bool = False
    api_model: str = None # "gpt-4o"太贵哩

    # 训练参数
    batch_size: int = 32
    max_tokens: int = 300
    epochs: int = 20
    temperature: float = 1.4
    top_p: float = 0.9

    # 树参数
    build_pair_percent: float = 0.1
    build_pair_gap: float = 0 # =0时关闭该功能
    build_pair_garbage_threshold: float = 100 # 取大于2的值时关闭该功能
    activate_critic: bool = True # 是否启用api的格式筛选
    activate_filter: bool = False # 是否启用json文件格式筛选？但该方法是失效的

    # 模型加载配置
    model_path: str = "/root/autodl-tmp/Qwen3-1.7B"
    lora_dir: str =  "/root/autodl-tmp/AfterTraining/GoldenFormatLora/"
    
    # 模型保存配置
    model_output_dir: str = "/root/autodl-tmp/AfterTraining/"
    model_cache_num: int = 2
    model_run_train_first: bool = False
    
    # 工作环境参数
    llama_dir: str = '/root/LLaMA-Factory/'
    my_cmd: str = '/root/autodl-tmp/'
    
    # 奖励参数
    alpha: float = 0.0  # 精确度和时间的折减系数
    
    # 数据集参数
    ambig_qa_dir: str = '/root/autodl-tmp/data/ambigqa/full/train.parquet'
    dataset_question_column: str = 'question'
    dataset_golden_answer_column: str = 'nq_answer'
    dataset_shorten_nums: int = 50
    add_to_sampling_path: bool = True # 是否保存到用于采样的json文件中
    sampling_json_path: str = "/root/autodl-tmp/data/Samplingdata.json" # 后续采样文件地址
    sft2dpo_sampling_json_dir: str = "/root/autodl-tmp/data/SFT2DPO/"

    # 验证
    eval_interval: int = 3 # 验证间隔
    eval_example_path: str = "/root/autodl-tmp/eval/example.jsonl" # 示例结果

    # 可视化
    UseTensorboard: bool = True
    tensorboard_log: str = '/root/tf-logs/'

    def __init__(self, train_mode: str = "dpo", test = False):
        if train_mode == "dpo":
            # 数据集和配置文件
            self.data_format = "dpo"
            self.train_dataset_save_dir: str = "/root/autodl-tmp/data/CacheTrainingData/mydataset.json"
            self.yaml_path: str = "/root/autodl-tmp/config/DPOTrainingConfig.yaml"

            self.max_tree_length: int = 3
            self.sampling_num: int = 16 # 乘数、成长
            self.sampling_num_decay: int = 7 # 除数、衰减

        elif train_mode == "sft":
            self.data_format = "sft"
            self.train_dataset_save_dir: str = "/root/autodl-tmp/data/CacheTrainingData/SFTData.json"
            self.yaml_path: str = "/root/autodl-tmp/config/SFTTrainingConfig.yaml"

            self.max_tree_length: int = 4
            self.sampling_num: int = 4 # 乘数、成长
            self.sampling_num_decay: int = 2 # 除数、衰减

        self.test = test
        if test == True:
            self.dataset_shorten_nums = 3

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
        config.decision.lora_dir = self.lora_dir
        config.decision.load_api = self.load_api
        config.decision.api_model = self.api_model
        config.test = self.test

        return config

