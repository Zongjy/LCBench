# eval.py
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # 从src/longbench/eval.py到项目根目录
sys.path.insert(0, str(project_root))

import numpy as np

from src.utils.common import (
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
)
from src.utils.longbench import (
    classification_score,
    code_sim_score,
    count_score,
    retrieval_score,
    retrieval_zh_score,
)

USER = os.getenv("USER")

# 数据集到评估指标的映射
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def preprocess_prediction(prediction: str, dataset: str) -> str:
    """预处理预测结果"""
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        if prediction is None:
            return ""
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def calculate_single_score(
    prediction: str,
    ground_truths: List[str],
    dataset: str,
    all_classes: Optional[List[str]] = None,
) -> float:
    """计算单个样本的得分"""
    prediction = preprocess_prediction(prediction, dataset)
    return max(
        dataset2metric[dataset](prediction, gt, all_classes=all_classes)
        for gt in ground_truths
    )


def scorer_e(
    dataset: str,
    predictions: List[str],
    answers: List[List[str]],
    lengths: List[int],
    all_classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """按长度分组的评分函数"""
    scores = {"0-4k": [], "4-8k": [], "8k+": []}

    for pred, gts, length in zip(predictions, answers, lengths):
        score = calculate_single_score(pred, gts, dataset, all_classes)
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)

    return {k: round(100 * np.mean(v), 2) for k, v in scores.items()}


def scorer(
    dataset: str,
    predictions: List[str],
    answers: List[List[str]],
    all_classes: Optional[List[str]] = None,
) -> float:
    """计算整体评分"""
    total_score = sum(
        calculate_single_score(pred, gts, dataset, all_classes)
        for pred, gts in zip(predictions, answers)
    )
    return round(100 * total_score / len(predictions), 2)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    return parser.parse_args(args)


def get_dataset_name(filename: str) -> str:
    """从文件名中提取数据集名称"""
    base_name = filename[: -len(".jsonl")]
    parts = base_name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else base_name


def load_prediction_file(file_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """加载预测文件数据"""
    data = {"predictions": [], "answers": [], "lengths": [], "all_classes": None}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data["predictions"].append(item.get("pred", ""))
                data["answers"].append(item.get("answers", []))
                if args.e and "length" in item:
                    data["lengths"].append(item["length"])
                if data["all_classes"] is None and "all_classes" in item:
                    data["all_classes"] = item["all_classes"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line in {file_path}: {e}")
                exit(1)
    return data


def main():
    args = parse_args()
    scores = {}
    model_name = args.model + ("_" + args.desc if args.desc else "")
    save_dir = os.path.join(args.base_dir, "result", "longbench")

    base_pred_path = os.path.join(save_dir, model_name)

    if not os.path.exists(base_pred_path):
        print(f"Error: Prediction directory not found at {base_pred_path}")
        exit()

    # 收集所有数据集的数据
    dataset_data = {}
    for file_path in glob.glob(os.path.join(base_pred_path, "*.jsonl")):
        dataset_name = get_dataset_name(os.path.basename(file_path))
        if dataset_name not in dataset_data:
            dataset_data[dataset_name] = load_prediction_file(file_path, args)

    # 计算每个数据集的得分
    print("Calculating scores for datasets:")
    for dataset_name, data in dataset_data.items():
        if not data["predictions"]:
            print(f"  No valid data found for dataset {dataset_name}. Skipping.")
            continue

        print(
            f"  Calculating score for {dataset_name} with {len(data['predictions'])} samples."
        )
        score = (
            scorer_e(
                dataset_name,
                data["predictions"],
                data["answers"],
                data["lengths"],
                data["all_classes"],
            )
            if args.e
            else scorer(
                dataset_name, data["predictions"], data["answers"], data["all_classes"]
            )
        )
        scores[dataset_name] = score

    # 保存结果
    out_path = os.path.join(save_dir, model_name, "result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
