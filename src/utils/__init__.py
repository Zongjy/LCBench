"""
LCBench utilities package.

This package contains common utility functions and benchmark-specific utilities
for LongBench and InfiniteBench evaluations.
"""

from .common import (
    # Text normalization
    normalize_answer,
    normalize_zh_answer,
    # Scoring functions
    rouge_score,
    rouge_zh_score,
    f1_score,
    qa_f1_score,
    qa_f1_zh_score,
    # File operations
    iter_jsonl,
    dump_jsonl,
    dump_json,
    # Input processing
    truncate_input_tokens,
)

from .longbench import (
    # LongBench specific scoring functions
    count_score,
    retrieval_score,
    retrieval_zh_score,
    code_sim_score,
    classification_score,
)

from .infinitebench import (
    # InfiniteBench specific functions
    ALL_TASKS,
    load_data,
    get_answer,
    first_int_match,
    in_match,
)

__all__ = [
    # Common utilities
    "normalize_answer",
    "normalize_zh_answer",
    "rouge_score",
    "rouge_zh_score",
    "f1_score",
    "qa_f1_score",
    "qa_f1_zh_score",
    "iter_jsonl",
    "dump_jsonl",
    "dump_json",
    "truncate_input_tokens",
    # LongBench specific
    "count_score",
    "retrieval_score",
    "retrieval_zh_score",
    "code_sim_score",
    "classification_score",
    # InfiniteBench specific
    "ALL_TASKS",
    "load_data",
    "get_answer",
    "first_int_match",
    "in_match",
] 