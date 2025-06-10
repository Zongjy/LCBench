"""
InfiniteBench specific utility functions.

This module contains functions specific to InfiniteBench evaluation,
including task definitions, data loading, and specialized scoring functions.
"""

import re
from typing import Dict, List, Union

from datasets import Features, Sequence, Value, load_dataset

from .common import f1_score, normalize_answer

# All available tasks in InfiniteBench
ALL_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
]


def load_data():
    """
    Load InfiniteBench dataset from HuggingFace.
    
    Returns:
        Loaded dataset with proper features schema
    """
    # Define the features schema
    ft = Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )

    # Load the dataset with the specified features
    dataset = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
    return dataset


def get_answer(eg: Dict, data_name: str) -> Union[str, List[str]]:
    """
    Extract the correct answer format for different InfiniteBench tasks.
    
    Args:
        eg: Example dictionary containing answer and options
        data_name: Name of the dataset/task
        
    Returns:
        Formatted answer(s)
    """
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg["options"].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg["options"].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ["A", "B", "C", "D"]:
                ret = eg["answer"]
            else:
                raise ValueError("Invalid answer format")
        else:
            raise ValueError("Invalid answer type")
        return ret

    return eg["answer"]


def first_int_match(prediction: str) -> int:
    """
    Check if the first integer in prediction matches ground truth.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth string
        
    Returns:
        1 if first integer matches, 0 otherwise
    """
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def in_match(prediction: str, ground_truth: str) -> int:
    """
    Check if ground truth is contained in prediction.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth string
        
    Returns:
        1 if ground truth is in prediction, 0 otherwise
    """
    if ground_truth in prediction:
        return 1
    return 0


def qa_f1_score(line: Dict) -> float:
    """
    Calculate F1 score for QA tasks with potentially multiple ground truths.
    This is InfiniteBench's specific implementation that differs from the common one.
    
    Args:
        line: Dictionary containing 'pred' and 'std_out' keys
        
    Returns:
        Maximum F1 score across all ground truths
    """
    prediction = line["pred"]

    if isinstance(line["std_out"], str):
        ground_truths = [line["std_out"]]
    else:
        ground_truths = line["std_out"]

    score = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        score = max(score, f1_score(prediction_tokens, ground_truth_tokens))

    return score