#!/bin/bash

# 基础配置
N=4
MODEL_NAME="Qwen2.5-7B-Instruct"
BASE_DIR=$(pwd)

# API配置
API_URL="http://localhost:11451/v1"
API_KEY="None"

# 实验配置
# 使用Chat_template
# DATASETS="narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news,vcsum,passage_count,passage_retrieval_en,passage_retrieval_zh"
DATASETS="narrativeqa"

# 不使用Chat_template
# DATASETS="trec,triviaqa,samsum,lsht,lcc,repobench-p"

DESC="test"
TEMPERATURE=0

export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0


# python -m debugpy --listen 5678 --wait-for-client $BASE_DIR/src/longbench/pred.py \
torchrun \
    --nproc_per_node=$N \
    --master_port 29509 \
    $BASE_DIR/src/longbench/pred.py \
    --base_dir $BASE_DIR \
    --model $MODEL_NAME \
    --datasets $DATASETS \
    --desc $DESC \
    --api_url $API_URL \
    --api_key $API_KEY \
    --temperature $TEMPERATURE \