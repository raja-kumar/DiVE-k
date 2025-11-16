cd /app/DiVE-k/src/divek/

export DEBUG_MODE="true"
export LOG_PATH="./logs/debug_log_gemma-3-12B-cub_base_gemma_mcq.txt"

export DATA_PATH=/data2/raja/CUB_200_2011/gemma_mcq/subsample_base_train_pass_20_mcq_dataset
# export DATA_PATH=/data2/raja/CUB_200_2011/gemma_mcq/subsample_base_train_pass_20_mcq_dataset
# export CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export CKPT_PATH="google/gemma-3-12b-it"
export SAVE_PATH=/app/raja/saved_models/vrft/CUB_200_2011/gemma-3-12B_GRPO_cub_base_gemma_mcq
export RUN_NAME=gemma-3-12B_GRPO_cub_base_gemma_mcq
# export CHECKPOINT_PATH=/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_qwen_mcq/checkpoint-200/
# --master_addr="127.0.0.1" \
# --master_port="12345" \

torchrun --nproc_per_node="1" \
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
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1.35 \
    --run_name  ${RUN_NAME}\
    --save_steps 200 \
    --num_generations 4 \
    --deepspeed local_scripts/zero3_offload.json \
    --reward_funcs "format" "mcq" \
    --max_completion_length 1024 \
    --bf16 \