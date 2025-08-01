# pred_v1.py
import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


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
from src.utils.longbench import (
    ALL_TASKS,
    load_data,
)


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
        "--task",
        type=str,
        nargs='+',
        required=True,
        help="""Which task(s) to use. Can be:
  - "all" to evaluate on all tasks
  - One or more task names separated by spaces
  - Comma-separated task names in quotes
Available tasks:
    narrativeqa,
    qasper,
    multifieldqa_en,
    multifieldqa_zh,
    hotpotqa,
    2wikimqa,
    musique,
    dureader,
    gov_report,
    qmsum,
    multi_news,
    vcsum,
    passage_count
    passage_retrieval_en
    passage_retrieval_zh
    trec
    triviaqa
    samsum
    lsht
    lcc
    repobench-p
Examples:
    --task all
    --task 2wikimqa qasper
    --task "2wikimqa,qasper"
    """,
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

    def build_prompt(
        self, 
        json_obj: Dict, 
        task_name: str,
        tokenizer=None,
        max_tokens: Optional[int] = None,
        prompt_format: Optional[str] = None,
    ) -> str:
        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = prompt_format.format(**json_obj)
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = prompt_format.format(**json_obj)
        
        if tokenizer and max_tokens:
            tokens = tokenizer.encode(prompt)
            tokens = truncate_input_tokens(tokens, max_tokens, manner="middle")
            prompt = tokenizer.decode(tokens)
        return prompt

    def post_process(self, pred: str, task_name: str) -> str:
        if task_name == "samsum":
            return pred.split("\n")[0].strip()
        else:
            return pred

    def chat(self, prompt: str, **kwargs) -> str:
        max_gen_tokens = kwargs.get("max_gen_tokens")
        temperature = kwargs.get("temperature", self.temperature)
        
        tries = 0
        while tries < 5:
            tries += 1
            try:
                completion = self.client.completions.create(
                    model=self.config.model2path.get(self.model_name),
                    prompt=prompt,
                    max_tokens=max_gen_tokens,
                    temperature=temperature
                )
                return completion.choices[0].text
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f'PID {os.getpid()} Error Occurs: "{str(e)}"        Retry ...')
                time.sleep(1)
        else:
            print(f"PID {os.getpid()} Max tries. Failed.")
            return ""

    def process_task(
        self,
        task_name: str,
        dataset: List[Dict],
        out_path: str,
        lock: multiprocessing.Lock,
        rank: int,
        world_size: int,
    ):
        data_subset = dataset[rank::world_size]
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
            prompt = self.build_prompt(
                json_obj=json_obj,
                task_name=task_name,
                tokenizer=self.tokenizer, 
                max_tokens=self.config.model2maxlen.get(self.model_name) 
                            - self.config.dataset2maxlen.get(task_name) - 128,
                prompt_format=self.config.dataset2prompt.get(task_name)
            )
            if not prompt:
                print(
                    f"Rank {rank} (PID: {os.getpid()}) Skipped sample due to prompt build failure."
                )
                continue

            output = self.chat(
                prompt=prompt,
                temperature=self.temperature,
                max_gen_tokens=self.config.dataset2maxlen.get(task_name),
            )
            if output == "":
                print(
                    f"Rank {rank} (PID: {os.getpid()}) chat returned empty string for a sample."
                )

            pred = self.post_process(output, task_name)
            result = {
                "pred": pred,
                "answers": json_obj.get("answers", "N/A"),
                "all_classes": json_obj.get("all_classes", "N/A"),
                "length": json_obj.get("length", "N/A"),
                "rank": rank,
            }

            with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")


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

    # # 设置输出目录
    # model_name = args.model + ("_" + args.desc if args.desc else "")
    # pred_dir = os.path.join(config.save_dir, model_name)
    # os.makedirs(pred_dir, exist_ok=True)

    # 确定要处理的数据集
    if args.e:
        tasks = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        if len(args.task) == 1 and args.task[0] == "all":
            tasks = ALL_TASKS
        else:
            tasks = []
            for task_item in args.task:
                if ',' in task_item:
                    tasks.extend([t.strip() for t in task_item.split(',') if t.strip()])
                else:
                    tasks.append(task_item.strip())
            invalid_tasks = [t for t in tasks if t not in ALL_TASKS]
            if invalid_tasks:
                print(f"错误：无效的任务名称: {invalid_tasks}")
                print(f"可用的任务: {ALL_TASKS}")
                return
                
            tasks = list(dict.fromkeys(tasks))

    datasets = load_data()
    
    lock = multiprocessing.Lock()
    for task_name in tqdm(tasks, desc="Processing Tasks", unit="task", disable=rank != 0):
        out_dir = (
            Path(args.base_dir)
            / "result"
            / "longbench"
            / (f"{args.model}_{args.desc}" if args.desc else args.model)
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{task_name}.jsonl"
        dataset = datasets[task_name]

        predictor.process_task(
            task_name=task_name,
            dataset=list(dataset),
            out_path=str(out_path),
            lock=lock,
            rank=rank,
            world_size=world_size,
        )

    print(f"Rank {rank}: All tasks processed.")
    print(
        f"Please run `python src/longbench/eval.py --model {args.model} --desc {args.desc} --base_dir {args.base_dir}` to evaluate the results."
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
