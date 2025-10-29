#!/bin/bash

MODEL_ROOT="/app/saved_models/vrft/fgvc_aircraft"  # root path for saved models
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# BASE_MODEL="google/gemma-3-4b-it"
CHECKPOINT="checkpoint-400"  # checkpoint name for saved models


# ==== configurations ====
# zero_shot="True"
num_shot=0
eval_type="${num_shot}_shot_two_steps"  # "sft" or everything else
use_cat_list="True"

# ==== dataset and output paths ====
DATA_ROOT="/data2/raja/"
# dataset="oxford_flowers"  # oxford_flowers, oxford-iiit-pet, CUB_200_2011

## === generation settings ===
temperature=1.0
max_new_tokens=1024

splits=("base_val" "new_val")  # splits to evaluate on
num_return_sequences=(2)  # number of sequences to return
EXP_NAMES=("Qwen2_5-VL-7B-Instruct_GRPO_aircraft_base_qwen_mcq")
# datasets=("CUB_200_2011" "oxford-iiit-pet" "stanford_cars" "fgvc_aircraft")  # datasets to evaluate on
datasets=("fgvc_aircraft")

for dataset in "${datasets[@]}"; do
    for EXP_NAME in "${EXP_NAMES[@]}"; do
        for split in "${splits[@]}"; do
            for NSEQ in "${num_return_sequences[@]}"; do
                python two_step_inference.py \
                    --model_root "$MODEL_ROOT" \
                    --base_model "$BASE_MODEL" \
                    --exp_name "$EXP_NAME" \
                    --checkpoint "$CHECKPOINT" \
                    --num_shot "$num_shot" \
                    --eval_type "$eval_type" \
                    --use_cat_list "$use_cat_list" \
                    --data_root "$DATA_ROOT" \
                    --dataset "$dataset" \
                    --split "$split" \
                    --num_return_sequences "$NSEQ" \
                    --temperature "$temperature" \
                    --max_new_tokens "$max_new_tokens" 
            done
        done
    done
done