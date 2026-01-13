#!/bin/bash
export HIP_VISIBLE_DEVICES=7

# --- General Configuration ---
MODEL_NAME_OR_PATH="./pretrain/Phi-3-mini-4k-instruct"
VIT_PATH="./pretrain/clip-vit-large-patch14-336"
HLORA_PATH="./pretrain/HealthGPT-M3/com_hlora_weights.bin" # Path to your HLora weights file

# --- Dataset Configuration ---
# Path to your dataset XLSX file
DATA_PATH="./dataset/M3CoTBench.xlsx"
# Path to your image directory
IMAGE_DIR="./dataset/images" 

# --- Output Configuration ---
# Define output directory
OUTPUT_DIR="./final_output"
mkdir -p "$OUTPUT_DIR" # Create output directory

# --- Execute Evaluation ---

# Mode 1: Direct Answer
echo "Running evaluation in 'direct' mode..."
python3 batch_infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP32" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_path "${OUTPUT_DIR}/HealthGPT_direct.json" \
    --eval_mode "direct" \
    --max_new_tokens 1024 # Adjust as needed

echo "Direct evaluation finished. Results saved to ${OUTPUT_DIR}/predictions_direct.json"

# Mode 2: Chain-of-Thought
echo "Running evaluation in 'cot' mode..."
python3 batch_infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP32" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_path "${OUTPUT_DIR}/HealthGPT_cot.json" \
    --eval_mode "cot" \
    --max_new_tokens 2048 # CoT mode may require more tokens to generate the reasoning process

echo "CoT evaluation finished. Results saved to ${OUTPUT_DIR}/predictions_cot.json"