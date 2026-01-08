#!/usr/bin/python
# -*- coding: UTF-8 -*-

import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import random
import torch
import os
import numpy as np
import openai
from tqdm import tqdm
import json
import argparse
import ast
import re
from tqdm import tqdm
from collections import Counter
import string
import sys
import time
from .utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from .metrics import match, accuracy
import yaml
from argparse import Namespace

seed = 633

DEFAULT_CONFIG = {
    "model_name": None,
    "input_file": None,
    "output_file": None,
    "task": None,
    "device": "cuda",
    "max_new_tokens": 15,
    "tokenizer_path": None,
    "download_dir": ".cache",
    "ndocs": 10,
    "world_size": 1,
    "dtype": "half",
    "threshold": None,
    "use_seqscore": False,
    "use_groundness": False,
    "use_utility": False,
    "beam_width": 2,
    "max_depth": 2,
    "w_rel": 1.0,
    "w_sup": 1.0,
    "w_use": 1.0,
    "mode": "default",
    "metric": None
}



torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def call_model_rerank_w_scores_batch(prompt, evidences, model, max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5, lora_request = None, 
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
        preds = model.generate([prompt], sampling_params, lora_request = lora_request)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True

    elif mode == "no_retrieval":
        do_retrieve = False

    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred

    if do_retrieve is True:
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences]
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params, lora_request = lora_request)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params, lora_request = lora_request)

        pred = preds[0].outputs[0].text

    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve

def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences

def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            # 先处理instruction的空值（None/空字符串都视为无指令）
            if instruction and instruction.strip():  # 非空且去除空白后也非空
                # 拼接指令+输入模板，换行符用\n统一，避免跨行拼接的反斜杠
                prompt = f"{instruction.strip()}\n\n## Input:\n\n{item['input']}"
            else:
                # 无有效指令时，直接使用问题作为prompt
                prompt = item["input"]
            
            # 推荐：创建item副本，避免修改原字典引发副作用（可选但更安全）
            new_item = item.copy()
            new_item["instruction"] = prompt
            new_data.append(new_item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(
        input_data, task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    if args.dtype is not None:
        model = LLM(model=gpt, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
    else:
        model = LLM(model=gpt, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold, max_depth=args.max_depth, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    for i, row in tqdm(enumerate(input_data)):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        pred, results, do_retrieve = generate(
            prompt, evidences, max_new_tokens=args.max_new_tokens,)
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if i % 10 == 0:
            print("average: {}".format(np.mean(metric_results)))
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(final_results)))


def load_config(config_path):
    """加载YAML配置文件，并合并默认配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件解析错误: {e}")
    
    # 合并配置（用户配置覆盖默认配置）
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in user_config.items() if v is not None})
    
    # 必要参数检查
    required_keys = ["model_name", "input_file", "output_file", "task", "metric"]
    missing_keys = [k for k in required_keys if config[k] is None]
    if missing_keys:
        raise ValueError(f"缺少必要配置项: {', '.join(missing_keys)}")
    
    # 模式合法性检查
    valid_modes = ['adaptive_retrieval', 'no_retrieval', 'always_retrieve']
    if config["mode"] not in valid_modes:
        raise ValueError(f"mode必须是以下值之一: {valid_modes}")
    
    return config

# ===================== 可调用的主函数 =====================
def our_main(yaml_path: str = "/root/autodl-tmp/config/selfrag/evaluation.yaml"):
    """
    Self-RAG 主函数（支持外部代码调用）
    
    Args:
        yaml_path: YAML配置文件路径，默认值为 "config.yaml"
    """
    # 直接使用传入的yaml路径加载配置（不再解析命令行）
    config = load_config(yaml_path)
    
    # 提取配置参数（与原代码变量名保持一致）
    model_name = config["model_name"]
    lora_path = config["lora_adapter_path"]
    input_file = config["input_file"]
    output_file = config["output_file"]
    task = config["task"]
    device = config["device"]
    max_new_tokens = config["max_new_tokens"]
    tokenizer_path = config["tokenizer_path"] or model_name  # 兼容tokenizer_path未配置的情况
    download_dir = config["download_dir"]
    ndocs = config["ndocs"]
    world_size = config["world_size"]
    dtype = config["dtype"]
    threshold = config["threshold"]
    use_seqscore = config["use_seqscore"]
    use_groundness = config["use_groundness"]
    use_utility = config["use_utility"]
    beam_width = config["beam_width"]
    max_depth = config["max_depth"]
    w_rel = config["w_rel"]
    w_sup = config["w_sup"]
    w_use = config["w_use"]
    mode = config["mode"]
    metric = config["metric"]

    # 加载输入数据
    if input_file.endswith(".json"):
        input_data = json.load(open(input_file, 'r', encoding='utf-8'))
    else:
        input_data = load_jsonlines(input_file)

    input_data = preprocess_input_data(input_data, task=task)
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    model = LLM(
        model=model_name,
        download_dir=download_dir,
        dtype=dtype,
        tensor_parallel_size=world_size
    )
    if lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(lora_name = 'test', 
                                   lora_path = lora_path, 
                                   lora_int_id = 1)
    else:
        lora_request = None

    # 获取特殊token
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=use_groundness, use_utility=use_utility
    )

    # 生成函数（保持原有逻辑）
    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(
            prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
            rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
            threshold=threshold, max_depth=max_depth, use_seqscore=use_seqscore, lora_request = lora_request, 
            w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, closed=task in ["fever", "arc_c"]
        )

    # 推理主循环
    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0

    for i, row in tqdm(enumerate(input_data), desc="处理数据"):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=ndocs)
        pred, results, do_retrieve = generate(prompt, evidences, max_new_tokens=max_new_tokens)
        
        # 清理预测结果
        if isinstance(pred, str) and pred.startswith(("#", ":")):
            pred = pred[1:]
        
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieve:
            count += 1

        # 处理答案格式
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if isinstance(row["answer"], str) else row["answer"]
        
        # 计算指标
        # 这里可以插入我们的计算指标
        if metric == "accuracy":
            metric_result = accuracy(pred, row["output"])
        elif metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError(f"不支持的指标: {metric}")
        
        metric_results.append(metric_result)

        # 每10条记录保存临时结果
        if i % 10 == 0:
            avg_metric = np.mean(metric_results)
            print(f"当前平均指标: {avg_metric:.4f}")
            final_results = {
                "preds": preds, "prompts": prompts, "metric_results": metric_results,
                "all_results": all_results, "golds": golds, "metric": metric,
                "metric_mean": avg_metric, "scores": scores
            }
            with open(f"{output_file}_tmp", "w", encoding='utf-8') as outfile:
                json.dump(final_results, outfile, ensure_ascii=False, indent=2)

    # 保存最终结果
    final_avg = np.mean(metric_results)
    final_results = {
        "preds": preds, "prompts": prompts, "metric_results": metric_results,
        "all_results": all_results, "golds": golds, "metric": metric,
        "metric_mean": final_avg, "scores": scores,
        "retrieval_frequency": count / len(input_data) if input_data else 0
    }
    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(final_results, outfile, ensure_ascii=False, indent=2)

    # 打印最终结果
    print(f"最终指标平均值: {final_avg:.4f}")
    print(f"检索频率: {count / len(input_data):.4f}")
    
    # 返回最终结果（方便外部调用时获取结果）
    return final_results

# ===================== 命令行入口 =====================
def cli_main():
    """命令行运行入口（兼容原有命令行方式）"""
    parser = argparse.ArgumentParser(description="Self-RAG 运行脚本（YAML配置版）")
    parser.add_argument('--config', type=str, default="config.yaml",
                        help="配置文件路径（默认: config.yaml）")
    cli_args = parser.parse_args()
    
    # 调用our_main函数
    our_main(cli_args.config)

# 主入口（兼容命令行和代码调用）
if __name__ == "__main__":
    cli_main()
