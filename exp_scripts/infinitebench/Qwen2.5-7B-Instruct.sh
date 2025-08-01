#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
# 基础配置
N=8
MODEL_NAME="Qwen2.5-7B-Instruct"
BASE_DIR=$(pwd)

# API配置
API_URL="http://localhost:11452/v1"
API_KEY="None"

# 实验配置
TASK="all"

DESC="psdf-r10-b128-quest"
TEMPERATURE=0.0

export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0


torchrun \
        --nproc_per_node=$N \
        --master_port 29502 \
        $BASE_DIR/src/infinitebench/pred.py \
        --base_dir $BASE_DIR \
        --model $MODEL_NAME \
        --task $TASK \
        --desc $DESC \
        --api_url $API_URL \
        --api_key $API_KEY \
        --temperature $TEMPERATURE \
