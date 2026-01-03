#!/bin/bash
#SBATCH --account=a132
#SBATCH --partition=normal
#SBATCH --job-name=test_time_compute    # Job name
#SBATCH --array=1-20                  # %10 Array range (this creates 20 tasks with IDs from 1 to 20, with max 8 tasks concurrently)
#SBATCH --gres=gpu:1                    # Number of GPUs (per node)
#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=11:00:00                  # Time limit hrs:min:sec

source ~/.bashrc
set -x -e
conda activate sal
export HF_HOME=/capstor/scratch/cscs/spanigra/huggingface

# Define the array of input files (assuming 10 input files)
STEP=35
ENDPOINT=$((20 * STEP - STEP))
STARTS=($(seq 0 $STEP $ENDPOINT))

DATASET_START=${STARTS[$SLURM_ARRAY_TASK_ID-1]}
DATASET_END=$(((${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP) < 674 ? (${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP) : 674)) # for OlympiadBench
# DATASET_END=$((${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP)) # for AIME-2024, MATH-500

echo "DATASET_START: $DATASET_START"
echo "DATASET_END: $DATASET_END"

CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --push_to_hub=false \
    --output_dir=/capstor/scratch/cscs/spanigra/search_and_learn/data/OlympiadBench_min/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters/seed-2-best_of_n/range-$DATASET_START-$DATASET_END