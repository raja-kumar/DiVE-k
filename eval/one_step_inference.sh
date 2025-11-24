#!/bin/bash

# ==== model details ====
MODEL_ROOT="/app/saved_models/vrft/"  # root path for saved models
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# BASE_MODEL="google/gemma-3-12b-it"
CHECKPOINT="checkpoint-209"  # checkpoint name for saved models


# ==== configurations ====
num_shot=0
eval_type="${num_shot}_shot_one_step"  # "sft" or everything else


# ==== dataset and output paths ====
DATA_ROOT="/data2/raja/"
datasets=("CUB_200_2011") #"oxford_flowers" "CUB_200_2011" "oxford-iiit-pet" "stanford_cars" "fgvc_aircraft"


## === generation settings ===
max_new_tokens=1024

splits=("base_val" "new_val")  # splits to evaluate on
EXP_NAMES=("baseline") # experiment names
use_cat_lists=("True")  # whether to use category list in the prompt

for dataset in "${datasets[@]}"; do
    MODEL_ROOT="${MODEL_ROOT}/${dataset}/"
    for EXP_NAME in "${EXP_NAMES[@]}"; do
        for split in "${splits[@]}"; do
            for use_cat_list in "${use_cat_lists[@]}"; do
                python classification_inference.py \
                    --model_root "$MODEL_ROOT" \
                    --base_model "$BASE_MODEL" \
                    --exp_name "$EXP_NAME" \
                    --checkpoint "$CHECKPOINT" \
                    --num_shot "$num_shot" \
                    --eval_type "$eval_type" \
                    --data_root "$DATA_ROOT" \
                    --dataset "$dataset" \
                    --split "$split" \
                    --max_new_tokens "$max_new_tokens" \
                    --use_cat_list "$use_cat_list"
            done
        done
    done
done