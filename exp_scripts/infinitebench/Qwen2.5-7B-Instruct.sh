#!/bin/bash

# 基础配置
N=4
MODEL_NAME="Qwen2.5-7B-Instruct"
BASE_DIR=$(pwd)

# API配置
API_URL="http://localhost:11451/v1"
API_KEY="None"

# 实验配置
TASK="kv_retrieval,longdialogue_qa_eng,longbook_sum_eng,longbook_choice_eng,longbook_qa_eng,longbook_qa_chn,math_find,math_calc,code_run,code_debug"
DESC="full"
TEMPERATURE=0.0

export TOKENIZERS_PARALLELISM=true

torchrun --nproc_per_node=$N $BASE_DIR/src/infinitebench/pred.py \
        --base_dir $BASE_DIR \
        --model $MODEL_NAME \
        --task $TASK \
        --desc $DESC \
        --api_url $API_URL \
        --api_key $API_KEY \
        --temperature $TEMPERATURE \
