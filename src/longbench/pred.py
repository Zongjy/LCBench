# pred_v1.py
import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 从src/longbench/pred.py到项目根目录
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.common import truncate_input_tokens


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Evaluate a language model on LongBench tasks using OpenAI-compatible API"
    )
    parser.add_argument("--base_dir", type=str, help="Base directory for LongBench")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_url", type=str, required=True, help="API URL")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument(
        "--desc",
        type=str,
        default=None,
        help="Optional description for model name directory",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of datasets or 'all'",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    return parser.parse_args()


class LongBenchConfig:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or f"/home/{os.getenv('USER')}/LCBench"
        self.data_dir = Path(self.base_dir) / "data" / "longbench"
        self.save_dir = Path(self.base_dir) / "result" / "longbench"

        # 加载配置文件
        self.config_dir = Path(self.base_dir) / "config"
        self.model2path = json.load(
            open(Path(self.config_dir) / "model2path.json", "r")
        )
        self.model2maxlen = json.load(
            open(Path(self.config_dir) / "model2maxlen.json", "r")
        )
        self.dataset2prompt = json.load(
            open(Path(self.config_dir) / "longbench" / "dataset2prompt.json", "r")
        )
        self.dataset2maxlen = json.load(
            open(Path(self.config_dir) / "longbench" / "dataset2maxlen.json", "r")
        )


class LongBenchPredictor:
    def __init__(self, config, model_name, api_url, api_key, temperature=0.0):
        self.config = config
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model2path[model_name], trust_remote_code=True
        )
        self.client = OpenAI(base_url=api_url, api_key=api_key, timeout=3600)

    def build_prompt(self, json_obj: Dict, prompt_format: str) -> str:
        """
        构建完整的 LongBench 任务 prompt

        Args:
            json_obj: 输入示例字典
            prompt_format: prompt 模板字符串

        Returns:
            str: 完整的 prompt 字符串
        """
        try:
            return prompt_format.format(**json_obj)
        except KeyError as e:
            print(
                f"Skipped sample due to missing key in json_obj for prompt format: {e}"
            )
            return ""

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        """发送聊天请求并返回响应。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            max_tokens: 最大生成 token 数

        Returns:
            str: 模型的响应文本
        """
        # 从消息中提取 prompt（假设只有一个用户消息）
        prompt = messages[0]["content"] if messages and "content" in messages[0] else ""

        max_len = self.config.model2maxlen.get(self.model_name, 2048) - max_tokens
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if len(input_ids) > max_len:
            input_ids = truncate_input_tokens(input_ids, max_len, manner="middle")
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            # 更新消息中的内容
            messages = [{"role": "user", "content": prompt}]

        model_path = self.config.model2path.get(self.model_name)
        if not model_path:
            print(f"Model {self.model_name} not found in model2path.")
            return ""

        tries = 0
        while tries < 5:
            tries += 1
            try:
                completion = self.client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f'PID {os.getpid()} Error Occurs: "{str(e)}"        Retry ...')
                time.sleep(1)
        else:
            print(f"PID {os.getpid()} Max tries. Failed.")
            return ""

    def get_pred(
        self, data, prompt_format, max_new_tokens, out_path, lock, rank, world_size
    ):
        data_subset = data[rank::world_size]
        print(
            f"Rank {rank}/{world_size} (PID: {os.getpid()}) processing {len(data_subset)} samples on {self.api_url} and writing to {out_path}..."
        )

        tqdm_position = rank % 8
        for json_obj in tqdm(
            data_subset,
            desc=f"Rank {rank} Progress",
            unit="sample",
            position=tqdm_position,
        ):
            prompt = self.build_prompt(json_obj, prompt_format)
            if not prompt:
                print(
                    f"Rank {rank} (PID: {os.getpid()}) Skipped sample due to prompt build failure."
                )
                continue

            output = self.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
            if output == "":
                print(
                    f"Rank {rank} (PID: {os.getpid()}) chat returned empty string for a sample."
                )

            with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump(
                        {
                            "pred": output,
                            "answers": json_obj.get("answers", "N/A"),
                            "all_classes": json_obj.get("all_classes", "N/A"),
                            "length": json_obj.get("length", "N/A"),
                            "rank": rank,
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")

    def load_dataset_safe(self, dataset_name, is_eval=False):
        """安全地加载数据集，包含错误处理"""
        try:
            dataset_path = str(self.config.data_dir / "LongBench.py")
            print(f"Loading dataset {dataset_name} from {dataset_path}")

            data = load_dataset(
                dataset_path,
                f"{dataset_name}_e" if is_eval else dataset_name,
                split="test",
                trust_remote_code=True,
            )
            return list(data)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {str(e)}")
            return None

    def process_dataset(
        self,
        dataset,
        prompt_format,
        max_new_tokens,
        out_path,
        lock,
        rank,
        world_size,
        is_eval=False,
    ):
        """处理单个数据集"""
        print(f"Rank {rank} (PID: {os.getpid()}): Processing dataset {dataset}...")

        # 加载数据集
        data_all = self.load_dataset_safe(dataset, is_eval)
        if not data_all:
            print(f"Rank {rank}: No data loaded for {dataset}. Skipping.")
            return

        # 处理数据
        self.get_pred(
            data_all,
            prompt_format,
            max_new_tokens,
            out_path=out_path,
            lock=lock,
            rank=rank,
            world_size=world_size,
        )
        print(f"Rank {rank}: Finished dataset {dataset}.")


def seed_everything(seed):
    """Seeds the random number generators for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    # torch seeding commented out as per original script's comment
    # import torch
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid RANK ({rank}) or WORLD_SIZE ({world_size})")

    print(f"Process Rank {rank}/{world_size} (PID: {os.getpid()}): Starting")

    # 初始化配置和预测器
    config = LongBenchConfig(args.base_dir)
    predictor = LongBenchPredictor(
        config=config,
        model_name=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    # 设置输出目录
    os.makedirs(config.save_dir, exist_ok=True)
    model_name = args.model + ("_" + args.desc if args.desc else "")
    pred_dir = os.path.join(config.save_dir, "pred_e" if args.e else "pred")
    os.makedirs(pred_dir, exist_ok=True)

    # 确定要处理的数据集
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = (
            [d for d in config.dataset2prompt.keys() if not d.endswith("_e")]
            if args.datasets == "all"
            else args.datasets.split(",")
        )

    # 处理每个数据集
    lock = multiprocessing.Lock()
    for dataset in tqdm(
        datasets, desc="Processing Datasets", unit="dataset", disable=rank != 0
    ):
        out_dir = os.path.join(pred_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset}.jsonl")

        prompt_format = config.dataset2prompt.get(dataset)
        max_new_tokens = config.dataset2maxlen.get(dataset)

        predictor.process_dataset(
            dataset=dataset,
            prompt_format=prompt_format,
            max_new_tokens=max_new_tokens,
            out_path=out_path,
            lock=lock,
            rank=rank,
            world_size=world_size,
            is_eval=args.e,
        )

    print(f"Rank {rank}: All datasets processed.")
    print(
        f"Please run `python src/longbench/eval.py --model {args.model} --desc {args.desc} --base_dir {args.base_dir}` to evaluate the results."
    )


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    main(args)
