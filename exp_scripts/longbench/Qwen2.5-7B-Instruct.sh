#!/bin/bash

# 基础配置
N=1
MODEL_NAME="Qwen2.5-7B-Instruct"
BASE_DIR=$(pwd)
RUN_FILE="$BASE_DIR/src/longbench/pred.py"

# API配置
API_URL="http://localhost:11451/v1"
API_KEY="None"

# 实验配置
DATASETS="qasper"
DESC="all-25"
TEMPERATURE=0.0

export TOKENIZERS_PARALLELISM=true

# 运行实验
torchrun --nproc_per_node=$N $RUN_FILE \
    --base_dir $BASE_DIR \
    --model $MODEL_NAME \
    --datasets $DATASETS \
    --desc $DESC \
    --api_url $API_URL \
    --api_key $API_KEY \
    --temperature $TEMPERATURE \