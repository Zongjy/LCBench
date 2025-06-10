#!/bin/bash

# 基础配置
N=4
MODEL_NAME="Qwen2.5-7B-Instruct"
BASE_DIR=$(pwd)

# API配置
API_URL="http://localhost:11452/v1"
API_KEY="None"

# 实验配置
TASK="all"
DESC="full"
TEMPERATURE=0.0

export TOKENIZERS_PARALLELISM=true

PYTHONPATH=$BASE_DIR torchrun --nproc_per_node=$N $BASE_DIR/src/infinitebench/pred.py \
        --base_dir $BASE_DIR \
        --model $MODEL_NAME \
        --task $TASK \
        --desc $DESC \
        --api_url $API_URL \
        --api_key $API_KEY \
        --temperature $TEMPERATURE \
