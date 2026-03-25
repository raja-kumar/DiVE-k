cd /app/DiVE-k/src/divek/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_Qwen2_5-VL-7B-combined_base_qwen_mcq_test.txt"

export DATA_PATH=/data2/raja/combined_datasets/combined_mcq_2400_dataset
export CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export SAVE_PATH=/app/saved_models/vrft/combined/Qwen2_5-VL-7B-Instruct_GRPO_combined_base_qwen_mcq_test
export RUN_NAME=Qwen2_5-VL-7B-Instruct_GRPO_combined_base_qwen_mcq_test
# export CHECKPOINT_PATH=/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_qwen_mcq/checkpoint-200/

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
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
    --reward_funcs "format" "mcq" \
    --max_completion_length 1024 \