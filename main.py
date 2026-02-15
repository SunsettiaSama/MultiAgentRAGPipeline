from src.RAG_modules import *
from src.dataset_evaluation import *
import os
from src.train import *
from src.dataset import build_garbage_dataset, generate_garbage
import traceback
import datetime
from omegaconf import OmegaConf
from src.llamafactory.train.tuner import run_exp
from src.config import MyTrainConfig

def save_format_golden_model():
    config = MyTrainConfig()
    llm = Large_Language_Model(local_dir = config.model_path)
    llm.reload_lora(config.to_AERRConfig().decision)

    llm.save_model("/root/autodl-tmp/AfterTraining/GoldenFormatModel")


def train_for_sleep(sleep = False, warm_start = False, **kwargs):

    trainer = Trainer()
    if warm_start:
        trainer.warm_start()

    # 这里暂时存放lora_dir，允许多个lora_dir加载
    # 这是一整条训练过程
    need_load_lora_dir = [
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-17-39/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-32-37/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-08-31-23-48-09/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-02-58-11/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-05-34-37/", 
        "/root/autodl-tmp/AfterTraining/Lora/2025-09-01-07-30-42/",
    ]
    if sleep:
        try:
            trainer.train(lora_dir = need_load_lora_dir, **kwargs)
            os.system("/usr/bin/shutdown")
        except Exception as e:
            log_file = "log.txt"
            with open(log_file, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Error occurred:\n")
                f.write("\n")
                f.write("Error occurred:\n")
                traceback.print_exc(file=f)
                f.write("\n")  
                os.system("/usr/bin/shutdown")
    else:
        trainer.train(lora_dir = need_load_lora_dir, **kwargs)

def my_ppo_train(llama_factory_config_path = None, AERR_config_path = None):
    
    # llama的config
    if llama_factory_config_path is not None:
        llama_config = OmegaConf.load(llama_factory_config_path)
    else:
        llama_config = OmegaConf.load("/root/autodl-tmp/config/PPOTrainingConfig.yaml")
    
    # 我们框架的config
    AERR_config = MyTrainConfig()
    if AERR_config_path is not None:
        AERR_config.load_yaml(AERR_config_path, merge = True)

    config_dict = OmegaConf.to_container(llama_config, resolve=True)
    from src.llamafactory.hparams import get_train_args
    from src.llamafactory.train.ppo import run_ppo
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config_dict)

    run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, our_config = AERR_config)

    return 

def sft(yaml_path = "/root/autodl-tmp/config/SFTTrainingConfig.yaml"):
    config = OmegaConf.load(yaml_path)

    config_dict = OmegaConf.to_container(config, resolve=True)
    from src.llamafactory.hparams import get_train_args
    from src.llamafactory.train.sft import run_sft
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config_dict)
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
    
if __name__ == '__main__':
    my_ppo_train(llama_factory_config_path = "/root/autodl-tmp/config/AERR/alpha0_1/llama_config.yaml", 
                 AERR_config_path = "/root/autodl-tmp/config/AERR/alpha0_1/our_config.yaml")
    
    # warm_start()
    
    # from src.dataset import csv2format_dataset
    # csv2format_dataset(input_path = "/root/autodl-tmp/QA_Evaluation/AERR/interaction_history-ckpt1000.csv", 
    #                    output_dir = "/root/autodl-tmp/data/format_data/", 
    #                    ignore_index = True, 
    #                    )
    # test_LLM()

    # evaluate_ourPipeline(sampling_nums = None)

    # evaluate_LLM_only()
    # gc.collect()

    # evaluate_naive_rag()
    # gc.collect()

    # evaluate_ourPipeline_by_API()
    # gc.collect()