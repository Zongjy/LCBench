import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from src.utils import (
    ALL_TASKS,
    dump_json,
    # InfiniteBench specific
    first_int_match,
    # Common utilities
    iter_jsonl,
    qa_f1_zh_score,
    rouge_score,
)
from src.utils.infinitebench import qa_f1_score  # InfiniteBench specific version


def parse_args() -> Namespace:
    p = ArgumentParser()
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
        "--desc",
        type=str,
        default="",
        help="Specify the description of the model",
    )
    return p.parse_args()


def split_retrieval_answer(pred: str):
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return words


def get_score_one_kv_retrieval(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return label in words


def get_score_one_passkey(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_run(pred, label) -> bool:
    """
    Returns the score of one example in Code.Run.
    """
    if isinstance(label, list):
        label = label[0]
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred = int(words[-1])
        return label == pred
    except Exception:
        return False


def get_score_one_code_debug(pred, label) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    pred = pred.strip()
    label_c = label[1]
    fn_name = label[0]
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred == label_c:
            return True
    ans_prefixes = [
        "answer is:",
        "is:",
        "answer:",
        "correct option is:",
    ]
    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(pred, label) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(first_num) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred, label) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return 1
    return 0


def get_score_one_longbook_choice_eng(pred, label) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one_longbook_qa_eng(pred, label) -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(pred: str, label: str) -> float:
    return rouge_score(pred, label)


def get_score_one_longbook_qa_chn(pred, label) -> float:
    return qa_f1_zh_score(pred, label)


def get_score_one_math_calc(pred, label) -> float:
    assert isinstance(label, list), f"Expected list, got {type(label)}"
    if isinstance(label[0], list):
        label = label[0]
    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            pred_nums.append(int(item))

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label)


def get_score_one(pred: str, label: str, task_name: str) -> float:
    """
    Computes the score for one prediction.
    Returns one float (zero and one for boolean values).
    """
    NAME_TO_SCORE_GETTER = {
        # Retrieve
        "kv_retrieval": get_score_one_kv_retrieval,
        "kv_retrieval_prefix": get_score_one_kv_retrieval,
        "kv_retrieval_both": get_score_one_kv_retrieval,
        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        # Code
        "code_run": get_score_one_code_run,
        "code_debug": get_score_one_code_debug,
        # Longbook
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        # Math
        "math_find": get_score_one_math_find,
        "math_calc": get_score_one_math_calc,
    }
    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label)
    return float(score)


def get_labels(preds: list) -> list[str]:
    possible_label_keys = ["ground_truth", "label"]
    for label_key in possible_label_keys:
        if label_key in preds[0]:
            return [x.get(label_key, "XXXXXXXXXX") for x in preds]
    raise ValueError(f"Cannot find label in {preds[0]}")


def get_preds(preds: list) -> list[str]:
    pred_strings = []
    possible_pred_keys = ["prediction", "pred"]
    for pred in preds:
        this_pred = "NO PREDICTION"
        for pred_key in possible_pred_keys:
            if pred_key in pred:
                this_pred = pred[pred_key]
                break
        else:
            raise ValueError(f"Cannot find prediction in {pred}")
        pred_strings.append(this_pred)
    return pred_strings


def get_score(labels: list, preds: list, data_name: str, model_name: str) -> float:
    """
    Computes the average score for a task.
    """
    assert len(labels) == len(preds)
    scores = []
    for label, pred in tqdm(zip(labels, preds)):
        score = get_score_one(pred, label, data_name, model_name)
        scores.append(score)
    return sum(scores) / len(scores)


def compute_scores(preds_path, data_name: str, model_name: str):
    print("Loading prediction results from", preds_path)
    preds = list(iter_jsonl(preds_path))
    labels = get_labels(preds)
    preds = get_preds(preds, data_name)

    acc = get_score(labels, preds, data_name, model_name)
    print(f"Average score for {data_name}: {acc}")
    return acc


if __name__ == "__main__":
    args = parse_args()

    result_dir = Path(
        args.base_dir,
        "result",
        "infinitebench",
        f"{args.model}_{args.desc}" if args.desc else args.model,
    )

    acc_list = []
    for task in ALL_TASKS:
        preds_path = result_dir / f"{task}.jsonl"
        assert preds_path.exists(), f"Predictions not found in: {preds_path}"
        acc_list.append({task: compute_scores(preds_path, task, args.model)})

    dump_json(acc_list, result_dir / "result.json")
