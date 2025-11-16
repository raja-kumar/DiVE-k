cd /app/DiVE-k/src/divek/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_test.txt"

export DATA_PATH=/data2/datasets/CUB_200_2011/zero_shot/subsample_base_train_dataset
export HARD_DATA_PATH=/data2/raja/oxford_flowers/zero_shot_mcq/hard_subsample_base_train_dataset
export CKPT_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export SAVE_PATH=/app/saved_models/vrft/ckpts/Qwen2-VL-2B-test
export RUN_NAME=Qwen2-VL-2B_test
export CHECKPOINT_PATH=/app/saved_models/vrft/ckpts/Qwen2-VL-2B-test/checkpoint-1/

# --master_addr="127.0.0.1" \
# --master_port="12345" \

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name  ${RUN_NAME}\
    --save_steps 200 \
    --save_only_model true \
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
    --reward_funcs "format" "mcq" \
    --data_name "CUB_200_2011" \
    --class_to_idx_path "/data2/datasets/CUB_200_2011/class_2_idx.json" \
    --category_names_path "/data2/datasets/CUB_200_2011/zero_shot/base_categories.txt" \
    # --use_hard_examples true \
    # --hard_dataset_name ${HARD_DATA_PATH} \
