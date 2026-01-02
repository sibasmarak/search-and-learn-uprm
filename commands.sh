#! /bin/bash
export SEED=0 
# export CONFIG=recipes/Llama-3.1-405B-Instruct/best_of_n.yaml
export CONFIG=recipes/Qwen2.5-7B-Instruct/best_of_n.yaml
export HF_HOME=/capstor/scratch/cscs/spanigra/huggingface

CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py $CONFIG \
    --n=1 \
    --seed=$SEED \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=20 \
    --gpu_memory_utilization=0.5 \
    --temperature=0.0 \
    --dataset_name=HuggingFaceH4/aime_2024 \
    --dataset_split=train \
    --output_dir=/capstor/scratch/cscs/spanigra/search_and_learn/data/AIME-2024-CoT/Qwen2.5-7B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/range-0-30
    # --dataset_config=OE_TO_maths_en_COMP \

# best_of_n / dvts
sbatch recipes/launch_array.slurm recipes/Qwen2.5-14B-Instruct/best_of_n.yaml \
    --n=256 \
    --seed=2 \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=5 \
    --gpu_memory_utilization=0.5 \
    --dataset_name=HuggingFaceH4/aime_2024 \
    --dataset_split=train # \
    # --dataset_config=OE_TO_maths_en_COMP

# merging local chunks of a dataset and pushing to Hugging Face Hub
for seed in 0 1 2; do
    python scripts/merge_local_chunks.py \
        --data_completions_dir /capstor/scratch/cscs/spanigra/search_and_learn/data/AIME-2024/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/seed-$seed-dvts \
        --seed $seed \
        --filename dvts_completions.jsonl \
        --repo_id sibasmarakp/Qwen2.5-14B-Instruct-uPRM-T80-adapters-dvts-completions \
        --split train \
        --dataset_name HuggingFaceH4_aime_2024 \
        --temperature 0.8 \
        --expected_num_samples 30
done

# Multi-node, for zero-shot (temperature = 0.0)
sbatch recipes/launch_multi_node.slurm recipes/Llama-3.1-405B-Instruct/best_of_n.yaml \
    --n=1 \
    --seed=0 \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=20 \
    --gpu_memory_utilization=0.8 \
    --temperature=0.0 \
    --dataset_name=HuggingFaceH4/aime_2024 \
    --dataset_split=train \
    --output_dir=/capstor/scratch/cscs/spanigra/search_and_learn/data/AIME-2024-CoT/Llama-3.1-405B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/range-0-30

# evaluating
export DATASET_ID=sibasmarakp/Qwen2.5-7B-Instruct-uPRM-T80-adapters-best_of_n-completions
export VOTING_N="1 2 4 8 16 32 64 128 256"

cd Qwen2.5-Math
for seed in 0; do
    export DATASET_CONFIG=HuggingFaceH4_aime_2024_CoT--T-0.0--top_p-1.0--n-1--seed-$seed--agg_strategy-last 
    python evaluation/evaluate_hf.py \
        --dataset_id $DATASET_ID \
        --dataset_config $DATASET_CONFIG \
        --voting_n $VOTING_N \
        --benchmark aime24
done