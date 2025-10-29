#!/bin/bash

MODEL_ROOT="/app/saved_models/vrft/ckpts/"  # root path for saved models
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT="checkpoint-291"  # checkpoint name for saved models


# ==== configurations ====
num_shot=0
eval_type="baseline"  # name of the evaluation. used as output folder name
use_cat_list="True"

# ==== dataset and output paths ====
DATA_ROOT="/data2/raja/"

## === generation settings ===
temperature=1.0
max_new_tokens=1024  # maximum number of new tokens to generate`

# datasets=("oxford_flowers" "CUB_200_2011" "oxford-iiit-pet" "stanford_cars" "fgvc_aircraft")  # datasets to evaluate on
datasets=("oxford_flowers")
splits=("base_train_mini")  # splits to evaluate on
num_return_sequences=(20)  # number of sequences to return
EXP_NAMES=("baseline")

for dataset in "${datasets[@]}"; do
    for EXP_NAME in "${EXP_NAMES[@]}"; do
        for split in "${splits[@]}"; do
            for NSEQ in "${num_return_sequences[@]}"; do
                python topk_gen.py \
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