"""
LongBench specific utility functions.

This module contains scoring functions and utilities specific to LongBench evaluation.
"""

import difflib
import re
from typing import Any, Dict, List

from fuzzywuzzy import fuzz


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Count the number of correct numerical answers in prediction.

    Args:
        prediction: Model prediction
        ground_truth: Correct answer
        **kwargs: Additional arguments

    Returns:
        Score as ratio of correct numbers to total numbers found
    """
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Score for retrieval tasks (English).

    Args:
        prediction: Model prediction
        ground_truth: Ground truth with format "Paragraph X"
        **kwargs: Additional arguments

    Returns:
        Score as ratio of correct paragraph numbers
    """
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Score for retrieval tasks (Chinese).

    Args:
        prediction: Model prediction
        ground_truth: Ground truth with format "段落X"
        **kwargs: Additional arguments

    Returns:
        Score as ratio of correct paragraph numbers
    """
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Calculate code similarity score using fuzzy string matching.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth code
        **kwargs: Additional arguments

    Returns:
        Similarity score between 0 and 1
    """
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Score for classification tasks.

    Args:
        prediction: Model prediction
        ground_truth: Correct class label
        **kwargs: Must contain 'all_classes' key with list of all possible classes

    Returns:
        Classification score
    """
    em_match_list = []
    all_classes = kwargs["all_classes"]

    # Find all classes mentioned in prediction
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)

    # Remove shorter matches that are substrings of longer ones
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)

    if em_match_list:
        if ground_truth in em_match_list:
            score = 1.0 / len(em_match_list)
        else:
            score = 0.0
    else:
        # InfLLM fallback: use similarity matching
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)

    return score
