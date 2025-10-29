#!/bin/bash

# Array of JSON files to evaluate

OUTPUT_FILE_ROOT="/app/Visual-RFT/classification/output/fgvc_aircraft/closed_source_models"
OUTPUT_FILES=(
    "${OUTPUT_FILE_ROOT}/gemini-2.5-prosubsample_base_val_no_reasoning.json"
    "${OUTPUT_FILE_ROOT}/gemini-2.5-prosubsample_new_val_no_reasoning.json"
    "${OUTPUT_FILE_ROOT}/gpt-5subsample_base_val_no_reasoning.json"
    "${OUTPUT_FILE_ROOT}/gpt-5subsample_new_val_no_reasoning.json"
    "${OUTPUT_FILE_ROOT}/grok-4subsample_base_val_no_reasoning.json"
    "${OUTPUT_FILE_ROOT}/grok-4subsample_new_val_no_reasoning.json"
)

# STEP3_OUTPUT_FILE="/app/Visual-RFT/classification/output/CUB_200_2011/step3/Qwen2_5-VL-7B-Instruct_GRPO_combined_base_qwen_mcq_checkpoint-400_base_val.json"

# Assuming OUTPUT_FILES is an array of file paths
# Example: OUTPUT_FILES=("file1.json" "file2.json" "file3.json")

for OUTPUT_FILE in "${OUTPUT_FILES[@]}"; do
    echo "Starting evaluation for: $OUTPUT_FILE"
    # Run the Python script in the background by adding '&' at the end
    python llm_eval.py --output_file "$OUTPUT_FILE" --one_answer "True" --one_step "True" &
    # python llm_eval_v2.py --output-file "$OUTPUT_FILE" --eval-topk &
done

# Wait for all background jobs started in the loop to finish
wait

echo "All evaluations are complete."