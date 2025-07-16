import os
import multiprocessing
import subprocess
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 基础配置
N = 16  # 进程数量
MODEL_NAME = "Qwen2.5-7B-Instruct"
BASE_DIR = "/home/ziyang/LCBench"

# API配置
API_URL = "http://localhost:11452/v1"
API_KEY = "None"

# 实验配置
# 使用Chat_template
# DATASETS="narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news,vcsum,passage_count,passage_retrieval_en,passage_retrieval_zh"
DATASETS = "gov_report"

# 不使用Chat_template
# DATASETS="trec,triviaqa,samsum,lsht,lcc,repobench-p"

DESC = "test"
TEMPERATURE = 0

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_pred_main(rank, world_size, base_dir, model_name, api_url, api_key, datasets, desc, temperature, e=False):
    """在子进程中运行pred_main函数"""
    env = os.environ.copy()
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    
    cmd = [
        sys.executable, "-m", "src.longbench.pred",
        "--base_dir", base_dir,
        "--model", model_name,
        "--api_url", api_url,
        "--api_key", api_key,
        "--datasets", datasets,
        "--temperature", str(temperature),
    ]
    
    if desc:
        cmd.extend(["--desc", desc])
    
    if e:
        cmd.append("--e")
    
    print(f"Starting process {rank}/{world_size}")
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Process {rank}/{world_size} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Process {rank}/{world_size} failed with return code {e.returncode}")
        raise

def main():
    """主函数：启动多进程执行pred_main"""
    world_size = N
    print(f"Starting {world_size} processes for parallel execution...")
    
    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(
            target=run_pred_main,
            args=(
                rank,
                world_size,
                BASE_DIR,
                MODEL_NAME,
                API_URL,
                API_KEY,
                DATASETS,
                DESC,
                TEMPERATURE,
            ),
            kwargs={"e": False}  # 设置为True如果使用LongBench-E
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("All processes completed!")

if __name__ == "__main__":
    main()
    
