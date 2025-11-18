cd /app/DiVE-k/src/divek/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_Qwen2_5-VL-7B-flower_base_e_2_e_hard_negative.txt"

export DATA_PATH=/data2/raja/oxford_flowers/zero_shot/subsample_base_train_hard_negative_dataset
export CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export SAVE_PATH=/app/saved_models/vrft/oxford_flowers/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_e_2_e_hard_negative
export RUN_NAME=Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_e_2_e_hard_negative

torchrun --nproc_per_node="3" \
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
    --num_train_epochs 3.14 \
    --run_name  ${RUN_NAME}\
    --save_steps 100 \
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
    --reward_funcs "format" "mcq" \
    --max_completion_length 1024 \
    --data_name "oxford_flowers" \
    --class_to_idx_path "/data2/raja/oxford_flowers/class_2_idx.json" \
    --category_names_path "/data2/raja/oxford_flowers/zero_shot/base_categories.txt" \
    --save_only_model true \