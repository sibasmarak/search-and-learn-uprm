#! /bin/bash

export SEED=0 
# export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
export CONFIG=recipes/Qwen2.5-32B-Instruct/best_of_n.yaml
export HF_HOME=/iopsstor/scratch/cscs/spanigra/huggingface

CUDA_VISIBLE_DEVICES=0,1 python scripts/test_time_compute.py $CONFIG \
    --n=16 \
    --num_samples=500 \
    --seed=$SEED \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=5 \
    --gpu_memory_utilization=0.9 \
    --output_dir=/capstor/scratch/cscs/spanigra/search_and_learn/data/Qwen2.5-32B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters

# best_of_n
sbatch recipes/launch_array.slurm recipes/Qwen2.5-14B-Instruct/best_of_n.yaml \
    --n=256 \
    --seed=2 \
    --num_samples=500 \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=5 \
    --gpu_memory_utilization=0.5 \
    # --gpu_memory_utilization=0.15

# dvts
sbatch recipes/launch_array.slurm recipes/Qwen2.5-7B-Instruct/dvts.yaml \
    --n=256 \
    --seed=2 \
    --num_samples=500 \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=2 \
    --search_batch_size=5 \
    --gpu_memory_utilization=0.5

# merging dataset revisions on Hugging Face Hub
python scripts/merge_chunks.py \
    --dataset_name=sibasmarakp/Llama-3.2-1B-Instruct-best_of_n-completions \
    --filter_strings seed-0

# merging local chunks of a dataset and pushing to Hugging Face Hub
python scripts/merge_local_chunks.py \
  --data_completions_dir /capstor/scratch/cscs/spanigra/search_and_learn/data/Qwen2.5-1.5B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/seed-0-dvts-prm-batch-size-2 \
  --seed 0 \
  --repo_id sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions \
  --split train \
  --expected_num_samples 500

# evaluating
export DATASET_ID=sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions
export DATASET_CONFIG=HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--seed-0--agg_strategy-last 
export VOTING_N="1 2 4 8 16 32 64 128 256"

cd Qwen2.5-Math
python evaluation/evaluate_hf.py \
    --dataset_id $DATASET_ID \
    --dataset_config $DATASET_CONFIG \
    --voting_n $VOTING_N