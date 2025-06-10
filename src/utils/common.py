"""
Common utility functions used by both LongBench and InfiniteBench.

This module contains shared utility functions for text normalization,
scoring metrics, and file operations.
"""

import json
import re
import string
from collections import Counter
from typing import Any, Dict, Generator, List, Union

import jieba
from rouge import Rouge


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Used for English text normalization.

    Args:
        s: Input string to normalize

    Returns:
        Normalized string
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """
    Lower text and remove punctuation, extra whitespace.
    Used for Chinese text normalization.

    Args:
        s: Input string to normalize

    Returns:
        Normalized string
    """

    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
            "„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Calculate ROUGE-L F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        **kwargs: Additional arguments (for compatibility)

    Returns:
        ROUGE-L F1 score
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except (ValueError, TypeError, RecursionError):
        # 如果出现递归错误或其他错误，返回0分
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Calculate ROUGE-L F1 score for Chinese text.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        **kwargs: Additional arguments (for compatibility)

    Returns:
        ROUGE-L F1 score for Chinese text
    """
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(
    prediction: Union[str, List[str]], ground_truth: Union[str, List[str]], **kwargs
) -> float:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted tokens or string
        ground_truth: Ground truth tokens or string
        **kwargs: Additional arguments (for compatibility)

    Returns:
        F1 score
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Calculate F1 score for QA tasks (English).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        **kwargs: Additional arguments (for compatibility)

    Returns:
        F1 score
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    Calculate F1 score for QA tasks (Chinese).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        **kwargs: Additional arguments (for compatibility)

    Returns:
        F1 score
    """
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def iter_jsonl(fname: str, cnt: int = None) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over lines in a JSONL file.

    Args:
        fname: Path to JSONL file
        cnt: Maximum number of lines to read (None for all)

    Yields:
        Parsed JSON objects
    """
    i = 0
    with open(fname, "r", encoding="utf8") as fin:
        for line in fin:
            if cnt is not None and i == cnt:
                break
            yield json.loads(line)
            i += 1


def dump_jsonl(data: List[Dict[str, Any]], fname: str) -> None:
    """
    Write data to a JSONL file.

    Args:
        data: List of dictionaries to write
        fname: Output file path
    """
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_json(data: Any, fname: str) -> None:
    """
    Write data to a JSON file.

    Args:
        data: Data to write
        fname: Output file path
    """
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def truncate_input_tokens(
    input_ids: List[int], max_length: int, manner: str = "middle"
) -> List[int]:
    """
    Truncate input token IDs to maximum length.

    Args:
        input_ids: List of token IDs to truncate
        max_length: Maximum allowed length
        manner: Truncation method ("middle" supported)

    Returns:
        Truncated token IDs list
    """
    if len(input_ids) <= max_length:
        return input_ids
    if manner == "middle":
        return input_ids[0 : max_length // 2] + input_ids[-max_length // 2 :]
    else:
        # 默认从末尾截断
        return input_ids[:max_length]
