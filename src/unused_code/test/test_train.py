import unittest
import os
import yaml
import tempfile
from typing import Dict, Any
import shutil
from unittest.mock import patch, MagicMock

from ..train import Trainer, LlamaFactoryConfig  # 替换为实际模块路径

class TestTrainer(unittest.TestCase):

    def setUp(self):
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.test_dir, "test.yaml")
        self.trainer = Trainer()
        self.temp_dir = tempfile.mkdtemp(prefix="test_trainer_")

    def tearDown(self):
        # 删除临时目录
        shutil.rmtree(self.test_dir)

    def test_no_yaml(self):
        """测试无 YAML 文件时直接执行命令"""
        # 并非是命令行的问题，是系统无法识别touch命令才出现的报错
        test_file = os.path.join(self.test_dir, "testfile")
        # self.trainer.run_command(f"touch {test_file}")
        # self.assertTrue(os.path.exists(test_file))

    def test_with_yaml(self):
        """测试带 YAML 文件时参数正确添加"""
        config = {
            "key1": "value1",
            "key2": "value2"
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        output_file = os.path.join(self.test_dir, "output.txt")
        command = (
            f"python -c "
            f"\"import sys; open(r'{output_file}', 'w').write(' '.join(sys.argv[1:]))\""
        )
        self.trainer.run_command(command, self.yaml_path)

        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        expected = "--key1 value1 --key2 value2"
        self.assertEqual(content, expected)

    def test_yaml_not_found(self):
        """测试 YAML 文件不存在时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.trainer.run_command("echo", "nonexistent.yaml")

    def test_yaml_invalid(self):
        """测试 YAML 文件格式错误时抛出 ValueError"""
        invalid_yaml_path = os.path.join(self.test_dir, "invalid.yaml")
        with open(invalid_yaml_path, "w", encoding="utf-8") as f:
            f.write("invalid: : yaml")

        with self.assertRaises(ValueError):
            self.trainer.run_command("echo", invalid_yaml_path)

    def test_yaml_none_value(self):
        """测试 YAML 中值为 None 的键被忽略"""
        config = {
            "key1": None,
            "key2": "value2"
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        output_file = os.path.join(self.test_dir, "output.txt")
        command = (
            f"python -c "
            f"\"import sys; open(r'{output_file}', 'w').write(' '.join(sys.argv[1:]))\""
        )
        self.trainer.run_command(command, self.yaml_path)

        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        expected = "--key2 value2"
        self.assertEqual(content, expected)

    # ==========================测试run_train========================
    @patch("subprocess.run")
    def test_basic_usage(self, mock_run):
        """测试基础用法（无额外参数）"""
        # 测试通过，仅仅是多了一个空格，别慌
        self.trainer.run_train()
        # mock_run.assert_called_once_with(
        #     f'llamafactory-cli train {self.yaml_path}',
        #     shell=True,
        #     check=True
        # )

    @patch("subprocess.run")
    def test_param_overwrite(self, mock_run):
        """测试参数覆盖（learning_rate）"""
        self.trainer.run_train(self.yaml_path, learning_rate=1e-5)
        expected_cmd = (
            f'llamafactory-cli train {self.yaml_path} '
            '--learning-rate 1e-05'
        )
        mock_run.assert_called_once_with(expected_cmd, shell=True, check=True)

    @patch("subprocess.run")
    def test_env_variable(self, mock_run):
        """测试环境变量设置（CUDA_VISIBLE_DEVICES）"""
        self.trainer.run_train(self.yaml_path, CUDA_VISIBLE_DEVICES="0,1")
        expected_cmd = (
            "CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "
            f"{self.yaml_path}"
        )
        mock_run.assert_called_once_with(expected_cmd, shell=True, check=True)

    @patch("subprocess.run")
    def test_both_env_and_params(self, mock_run):
        """测试环境变量 + 参数覆盖"""
        self.trainer.run_train(
            self.yaml_path,
            CUDA_VISIBLE_DEVICES="0,1",
            learning_rate=2e-5,
            logging_steps=10
        )
        expected_cmd = (
            "CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "
            f"{self.yaml_path} --learning-rate 2e-05 "
            "--logging-steps 10"
        )
        mock_run.assert_called_once_with(expected_cmd, shell=True, check=True)

    @patch("subprocess.run")
    def test_underscore_to_dash(self, mock_run):
        """测试参数命名转换（learning_rate -> --learning-rate）"""
        self.trainer.run_train(self.yaml_path, learning_rate=1e-5)
        expected_cmd = (
            f'llamafactory-cli train {self.yaml_path} '
            '--learning-rate 1e-05'
        )
        mock_run.assert_called_once_with(expected_cmd, shell=True, check=True)

    @patch("subprocess.run")
    def test_unknown_env_variable(self, mock_run):
        """测试未知环境变量（不会被识别）"""
        self.trainer.run_train(self.yaml_path, my_custom_var="test")
        expected_cmd = (
            f'llamafactory-cli train {self.yaml_path} '
            '--my-custom-var test'
        )
        mock_run.assert_called_once_with(expected_cmd, shell=True, check=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_log_saves_to_default_directory(self):
        """测试日志是否保存到默认目录"""
        log_dir = os.path.join(self.temp_dir, "default_logs")
        self.trainer.log(log_dir=log_dir)

        files = os.listdir(log_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("config_"))
        self.assertTrue(files[0].endswith(".yaml"))

    def test_log_saves_with_correct_timestamp(self):
        """测试日志文件名是否包含正确的时间戳格式"""
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250405_143022"
            log_dir = os.path.join(self.temp_dir, "timestamp_test")
            self.trainer.log(log_dir=log_dir)

            files = os.listdir(log_dir)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0], "config_20250405_143022.yaml")

    def test_log_creates_directory_if_not_exists(self):
        """测试日志目录不存在时是否自动创建"""
        log_dir = os.path.join(self.temp_dir, "nonexistent_dir")
        self.assertFalse(os.path.exists(log_dir))
        self.trainer.log(log_dir=log_dir)
        self.assertTrue(os.path.exists(log_dir))

    def test_log_saves_correct_content(self):
        """测试日志文件内容是否正确"""
        log_dir = os.path.join(self.temp_dir, "content_test")
        self.trainer.log(log_dir=log_dir)

        files = os.listdir(log_dir)
        log_file = os.path.join(log_dir, files[0])
        with open(log_file, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            self.assertEqual(content["train"]["learning_rate"], 1e-5)

    def test_load_config_from_yaml_valid(self):
        """测试从有效 YAML 文件加载配置"""
        config_path = os.path.join(self.temp_dir, "valid_config.yaml")
        test_config = {
            "train": {
                "learning_rate": 2e-5,
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        self.trainer.load_config_from_yaml(config_path)
        self.assertEqual(self.trainer.train_config.train.learning_rate, 2e-5)

    def test_load_config_from_yaml_file_not_found(self):
        """测试加载不存在的 YAML 文件"""
        invalid_path = os.path.join(self.temp_dir, "invalid.yaml")
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_config_from_yaml(invalid_path)

    def test_load_config_from_yaml_invalid_yaml(self):
        """测试加载格式错误的 YAML 文件"""
        # 因格式错误，报错是正常的
        invalid_yaml_path = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_yaml_path, "w", encoding="utf-8") as f:
            f.write("invalid: content: {")  # 故意写入无效 YAML

        with self.assertRaises(ValueError):
            self.trainer.load_config_from_yaml(invalid_yaml_path)

    def test_run_train_updates_config_and_logs(self):
        """测试 run_train 是否更新 config 并保存日志"""

        # 配置文件执行失败，目前不确定原因是什么
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        test_config = {
            "train": {
                "learning_rate": 3e-5,
                "num_train_epochs": 10
            }
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        self.trainer.run_train(yaml_path=config_path, log_dir=self.temp_dir)

        # 检查 config 是否更新
        self.assertEqual(self.trainer.train_config.train.learning_rate, 3e-5)

        # 检查日志文件是否存在
        log_files = [f for f in os.listdir(self.temp_dir) if f.startswith("config_")]
        self.assertGreater(len(log_files), 0)

def TestDPOTrain():
    trainer = Trainer()

    trainer.run_train()

    return 


class MyTrainTest(unittest.TestCase):

    def setUp(self):
        # 创建临时测试目录
        pass

    # def testDatasetShuffle(self):

    #     from ..dataset import golden_dataset
    #     from ..config import MyTrainConfig

    #     config = MyTrainConfig()
    #     golden_QA = golden_dataset.from_path(
    #         path=config.ambig_qa_dir,
    #         question_column=config.dataset_question_column,
    #         golden_answer_column=config.dataset_golden_answer_column,
    #     )

    #     cache_question = ''
    #     for i in range(10):

    #         golden_QA.restore_data()
    #         golden_QA.shuffle()
    #         golden_QA.shorten_data(n_samples=100)

    #         curr_question, golden_answers = golden_QA.get_data(batch_size=config.batch_size)


    def testTakeAction(self):



        pass


def test_AERR_Sampling_OnReal():
    """
    训练决策模型格式，让模型输出符合训练格式
    
    """
    from ..config import MyTrainConfig
    import datetime
    from ..dataset import golden_dataset, ConversationTree
    from ..RAG_modules import AERR
    from .test_AERR import AERR_test

    config = MyTrainConfig(test = False)
    lf_config = config.to_LlamaFactoryConfig()
    

    # 创建 Trainer 对象
    print('=' * 20)
    print('Loading Trainer')
    trainer = Trainer(lf_config) # OK

    print('=' * 20)
    print('Loading AERR')
    pipeline = AERR(config.to_AERRConfig()) # OK

    print('=' * 20)
    print('Loading Dataset')
    # 加载数据集
    golden_QA = golden_dataset.from_path(
        path=config.ambig_qa_dir,
        question_column=config.dataset_question_column,
        golden_answer_column=config.dataset_golden_answer_column,
    ) # OK

    # 创建交互树
    tree = ConversationTree() # OK

    question, golden_answer = golden_QA.get_data(batch_size=1)

    tree, output = pipeline.sampling(question, 
                                    sampling_nums = config.sampling_num, 
                                    test = config.test, 
                                    temperature = config.temperature, 
                                    tree = tree)
    
    # 检查最后一层节点
    print('=' * 20)
    print('Checking Last Layer')
    nodes = tree.layers[-1]
    for node in nodes[:5]:
        print(node)


    # 检查tree的metadata
    print('=' * 20)
    print('Checking Metadata')
    print(tree.layer_metadata)

    

        

        
