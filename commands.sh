#! /bin/bash

export SEED=0 
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
# export CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml
# python scripts/test_time_compute.py $CONFIG \
#     --n=256 \
#     --num_samples=500 \
#     --seed=$SEED


python scripts/test_time_compute.py $CONFIG \
    --n=4 \
    --num_samples=500 \
    --seed=$SEED \
    --prm_path=agadetskii/Qwen2.5-14B-Instruct-uPRM-T80-adapters \
    --prm_batch_size=4 \
    --search_batch_size=25 \
    --output_dir=/mlbio_scratch/panigrah/search_and_learn/data/Llama-3.2-1B-Instruct/Qwen2.5-14B-Instruct-uPRM-T80-adapters