#!/bin/bash

# Array of JSON files to evaluate

OUTPUT_FILE_ROOT="./output/oxford_flowers/0_shot_two_steps/"
# OUTPUT_FILE_ROOT="./output/test/"
OUTPUT_FILES=(
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_e_2_e_hard_negative_final_weights_base_val_pass20.json"
    "${OUTPUT_FILE_ROOT}/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_e_2_e_hard_negative_final_weights_new_test_pass20.json"
)

# OUTPUT_FILES=(
#     "${OUTPUT_FILE_ROOT}/test.json"
# )
# STEP3_OUTPUT_FILE="/app/Visual-RFT/classification/output/CUB_200_2011/step3/Qwen2_5-VL-7B-Instruct_GRPO_combined_base_qwen_mcq_checkpoint-400_base_val.json"

# Assuming OUTPUT_FILES is an array of file paths
# Example: OUTPUT_FILES=("file1.json" "file2.json" "file3.json")

for OUTPUT_FILE in "${OUTPUT_FILES[@]}"; do
    echo "Starting evaluation for: $OUTPUT_FILE"
    # Run the Python script in the background by adding '&' at the end
    python llm_eval.py --output_file "$OUTPUT_FILE" --one_answer "True" --one_step "False" &
    # python llm_eval_v2.py --output-file "$OUTPUT_FILE" --eval-topk &
done

# Wait for all background jobs started in the loop to finish
wait

echo "All evaluations are complete."