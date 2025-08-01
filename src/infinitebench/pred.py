import json
import multiprocessing
import os
import sys
import time
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 从src/infinitebench/pred.py到项目根目录
sys.path.insert(0, str(project_root))

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.common import truncate_input_tokens
from src.utils.infinitebench import (
    ALL_TASKS,
    get_answer,
    load_data,
)


def parse_args() -> Namespace:
    p = ArgumentParser(
        description="Evaluate a language model on a conversational task using multiple APIs",
        formatter_class=RawTextHelpFormatter,
    )
    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Specify the base directory to use for evaluation",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the model to use for evaluation",
    )
    p.add_argument(
        "--task",
        type=str,
        nargs='+',
        required=True,
        help="""Which task(s) to use. Can be:
  - "all" to evaluate on all tasks
  - One or more task names separated by spaces
  - Comma-separated task names in quotes
Available tasks:
    passkey
    number_string
    kv_retrieval
    longdialogue_qa_eng
    longbook_sum_eng
    longbook_choice_eng
    longbook_qa_eng
    longbook_qa_chn
    math_find
    math_calc
    code_run
    code_debug
Examples:
    --task all
    --task passkey number_string
    --task "passkey,number_string,kv_retrieval"
    """,
    )
    p.add_argument("--api_url", type=str, required=True, help="API URL")
    p.add_argument("--api_key", type=str, required=True, help="API Key")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Specify the temperature to use for evaluation",
    )
    p.add_argument(
        "--desc",
        type=str,
        default=None,
        help="Optional description for model name directory",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return p.parse_args()


class InfiniteBenchConfig:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or f"/home/{os.getenv('USER')}/InfiniteBench"
        self.save_dir = Path(self.base_dir) / "result" / "infinitebench"

        self.config_dir = Path(self.base_dir) / "config"
        self.model2path = json.load(
            open(Path(self.config_dir) / "model2path.json", "r")
        )
        self.model2maxlen = json.load(
            open(Path(self.config_dir) / "model2maxlen.json", "r")
        )
        self.dataset2prompt = json.load(
            open(Path(self.config_dir) / "infinitebench" / "dataset2prompt.json", "r")
        )
        self.dataset2maxlen = json.load(
            open(Path(self.config_dir) / "infinitebench" / "dataset2maxlen.json", "r")
        )


class InfiniteBenchPredictor:
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
        json_obj: dict,
        task_name: str,
        tokenizer=None,
        max_tokens: Optional[int] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        import re

        if task_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", json_obj["input"])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            prompt = prompt_template.format(
                func=func,
                func_call=func_call,
                context=json_obj["context"],
            )
        elif task_name in ["code_debug", "code_debug_qa"]:
            code = json_obj["context"]
            if task_name == "code_debug":
                prompt = prompt_template.format(
                    context=code,
                    OPTION_A=json_obj["options"][0],
                    OPTION_B=json_obj["options"][1],
                    OPTION_C=json_obj["options"][2],
                    OPTION_D=json_obj["options"][3],
                )
            else:
                prompt = prompt_template.format(context=code)
        elif task_name == "longdialogue_qa_eng":
            script = json_obj["context"]
            prompt = prompt_template.format(context=script)
        elif task_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            book = json_obj["context"]
            if task_name == "longbook_choice_eng":
                prompt = prompt_template.format(
                    question=json_obj["input"],
                    context=book,
                    OPTION_A=json_obj["options"][0],
                    OPTION_B=json_obj["options"][1],
                    OPTION_C=json_obj["options"][2],
                    OPTION_D=json_obj["options"][3],
                )
            elif task_name == "longbook_qa_eng":
                prompt = prompt_template.format(
                    question=json_obj["input"],
                    context=book,
                )
            elif task_name == "longbook_sum_eng":
                prompt = prompt_template.format(context=book)
            elif task_name == "longbook_qa_chn":
                prompt = prompt_template.format(
                    question=json_obj["input"],
                    context=book,
                )
        elif task_name == "math_calc":
            prompt = prompt_template.format(context=json_obj["context"])
        elif task_name == "math_find":
            find_result = re.findall(r"The .+ of", json_obj["input"])
            assert find_result, f"Cannot find the target number in {json_obj['input']}"
            target_number = find_result[0].lower()[:-3]
            prefix = f"What is {target_number} in the following list?"
            prompt = prompt_template.format(
                prefix=prefix,
                context=json_obj["context"],
                input=json_obj["input"],
            )
        elif task_name == "kv_retrieval":
            format_dict = {
                "context": json_obj["content"] if "content" in json_obj else json_obj["context"],
                "input": json_obj["input"],
                "key": json_obj["input"][6:44]
            }
            prompt = prompt_template.format(**format_dict)
        else:
            if "content" in json_obj:
                content = json_obj["content"]
                del json_obj["content"]
                json_obj["context"] = content

            format_dict = {
                "context": json_obj["context"],
                "input": json_obj["input"],
            }
            prompt = prompt_template.format(**format_dict)

        if tokenizer and max_tokens:
            tokens = tokenizer.encode(prompt)
            tokens = truncate_input_tokens(tokens, max_tokens, manner="middle")
            prompt = tokenizer.decode(tokens)
        return prompt, len(json_obj["context"])

    def chat(self, prompt: str, **kwargs) -> str:
        max_gen_tokens = kwargs.get("max_gen_tokens")
        temperature = kwargs.get("temperature", self.temperature)

        tries = 0
        while tries < 5:
            tries += 1
            try:
                completion = self.client.completions.create(
                    model=self.config.model2path[self.model_name],
                    prompt=prompt,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
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
            f"Rank {rank}/{world_size} (PID: {os.getpid()}) processing {len(data_subset)} samples for task {task_name}..."
        )

        tqdm_position = rank % 8
        for i, json_obj in enumerate(
            tqdm(
                data_subset,
                desc=f"Rank {rank} {task_name}",
                unit="sample",
                position=tqdm_position,
            )
        ):
            prompt, context_length = self.build_prompt(
                json_obj=json_obj,
                task_name=task_name,
                tokenizer=self.tokenizer,
                max_tokens=self.config.model2maxlen.get(self.model_name)
                            - self.config.dataset2maxlen.get(task_name) - 128,
                prompt_template=self.config.dataset2prompt.get(task_name),
            )

            try:
                output = self.chat(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_gen_tokens=self.config.dataset2maxlen.get(task_name),
                )

                ans = get_answer(json_obj, task_name)

                result = {
                    "id": rank * len(data_subset) + i,
                    "pred": output,
                    "length": context_length,
                    "answers": ans if isinstance(ans, list) else [ans],
                    "rank": rank,
                }

                with lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")

            except Exception as e:
                print(f"Rank {rank} ERROR: {e}")


def main():
    args = parse_args()

    # 获取并发参数
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid RANK ({rank}) or WORLD_SIZE ({world_size})")

    print(f"Process Rank {rank}/{world_size} (PID: {os.getpid()}): Starting")

    config = InfiniteBenchConfig(args.base_dir)
    client = InfiniteBenchPredictor(
        config=config,
        model_name=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    datasets = load_data()
    
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
    

    lock = multiprocessing.Lock()
    for task in tqdm(tasks, desc="Processing Tasks", unit="task", disable=rank != 0):
        out_dir = (
            Path(args.base_dir)
            / "result"
            / "infinitebench"
            / (f"{args.model}_{args.desc}" if args.desc else args.model)
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{task}.jsonl"
        dataset = datasets[task]

        # 并发处理任务
        client.process_task(
            task_name=task,
            dataset=list(dataset),
            out_path=str(out_path),
            lock=lock,
            rank=rank,
            world_size=world_size,
        )

    print(f"Rank {rank}: All tasks processed.")
    print(
        f"Please run `python src/infinitebench/eval.py --model {args.model} --desc {args.desc} --base_dir {args.base_dir}` to evaluate the results."
    )


if __name__ == "__main__":
    main()
