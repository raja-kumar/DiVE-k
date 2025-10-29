# Set your environment variables or replace them directly in the command
export SAVE_PATH="./output_model"
export CKPT_PATH="gpt2" # Using a smaller model like gpt2 for this example
export DATA_PATH="dummy_dataset" # The script uses a dummy dataset internally
export RUN_NAME="gpt2-finetune-test"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    main.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name  ${RUN_NAME}\
    --save_steps 100 \
    --deepspeed zero3_offload.json \
    --resume_from_checkpoint "./output_model/checkpoint-62/" \