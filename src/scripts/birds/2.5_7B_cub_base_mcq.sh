cd /app/Visual-RFT/src/virft/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_qwen2_5_7B_GRPO_cub_base_mcq.txt"

export DATA_PATH=/data2/raja/CUB_200_2011/zero_shot/subsample_base_train_mcq_dataset
# export CKPT_PATH="Qwen/Qwen2-VL-2B-Instruct"
export CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export SAVE_PATH=/app/saved_models/vrft/CUB_200_2011/Qwen2_5-VL-7B-Instruct_GRPO_cub_base_mcq
export RUN_NAME=Qwen2_5-VL-7B_GRPO_cub_base_mcq

# --master_addr="127.0.0.1" \
# --master_port="12345" \

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
    --num_train_epochs 1 \
    --run_name  ${RUN_NAME}\
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
    --reward_funcs "format" "mcq" \
    --max_completion_length 1024 \