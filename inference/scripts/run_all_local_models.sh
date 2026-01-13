#!/bin/bash

# Exit immediately if any command fails
set -e

# SCRIPT_DIR will be the absolute path where the script file is located (e.g., /.../project/scripts_new)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# PROJECT_ROOT is the parent directory of the script's location
PROJECT_ROOT=$( dirname "${SCRIPT_DIR}" )

echo "##########################################################################"
echo "###        Starting batch parallel execution of all models             ###"
echo "###        (8 GPUs, 4 parallel groups)                                 ###"
echo "##########################################################################"

# --- Logs and Global Configuration ---
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
echo "Logs will be saved in directory '${LOG_DIR}'."

# 1. Define the mode to run for all tasks ('all', 'cot', 'direct')
MODE_TO_RUN="all"

# 2. Define GPUs and Ports for 4 parallel slots
# (Note: Ports must be unique as they run simultaneously)
# Your single model script will use base_port and base_port+1
GPU_GROUP_1="0,1"
PORT_GROUP_1=10580 # Will use 10580, 10581

GPU_GROUP_2="2,3"
PORT_GROUP_2=10582 # Will use 10582, 10583

GPU_GROUP_3="4,5"
PORT_GROUP_3=10584 # Will use 10584, 10585

GPU_GROUP_4="6,7"
PORT_GROUP_4=10586 # Will use 10586, 10587

echo "Config: 4 Parallel Groups. Mode: ${MODE_TO_RUN}"
echo " - Group 1: GPU ${GPU_GROUP_1}, Base Port ${PORT_GROUP_1}"
echo " - Group 2: GPU ${GPU_GROUP_2}, Base Port ${PORT_GROUP_2}"
echo " - Group 3: GPU ${GPU_GROUP_3}, Base Port ${PORT_GROUP_3}"
echo " - Group 4: GPU ${GPU_GROUP_4}, Base Port ${PORT_GROUP_4}"
echo ""

# --- Helper Function: To start a model in the background ---
# $1: model_name
# $2: gpu_ids
# $3: base_port
run_model_in_background() {
    local model_name="$1"
    local gpu_ids="$2"
    local base_port="$3"
    
    # Dynamically handle slashes for log filenames (if present in model name)
    local log_file_name=$(echo "${model_name}" | tr '/' '_')
    local LOG_PATH="${LOG_DIR}/${log_file_name}.log"
    
    echo "   -> Starting: ${model_name} (GPU ${gpu_ids}, Port ${base_port}) ... Log: ${LOG_PATH}"
    
    # Call child script and run in background (Note the & at the end)
    "${SCRIPT_DIR}/run_local_gpu_model.sh" \
      "${model_name}" \
      "${gpu_ids}" \
      "${MODE_TO_RUN}" \
      "${base_port}" \
      > "${LOG_PATH}" 2>&1 &
}

# ================================================================================
# --- Batch 1 (4 models) ---
echo "========================================================================"
echo "--- Starting Batch 1 ---"
run_model_in_background "LLaVA-Med-7B"     ${GPU_GROUP_1} ${PORT_GROUP_1}
run_model_in_background "HuatuoGPT-V-7B"   ${GPU_GROUP_2} ${PORT_GROUP_2}
run_model_in_background "HealthGPT-3.8B"   ${GPU_GROUP_3} ${PORT_GROUP_3}
run_model_in_background "Lingshu-7B"       ${GPU_GROUP_4} ${PORT_GROUP_4}

echo "Waiting for Batch 1 (4 tasks) to complete..."
wait # Wait for all 4 background tasks in Batch 1 to complete
echo "--- Batch 1 Completed ---"


# ================================================================================
# --- Batch 2 (4 models) ---
echo "========================================================================"
echo "--- Starting Batch 2 ---"
run_model_in_background "Lingshu-32B"      ${GPU_GROUP_1} ${PORT_GROUP_1}
run_model_in_background "MedGemma-4B"      ${GPU_GROUP_2} ${PORT_GROUP_2}
run_model_in_background "MedGemma-27B"     ${GPU_GROUP_3} ${PORT_GROUP_3}
run_model_in_background "LLaVA-OV-7B"      ${GPU_GROUP_4} ${PORT_GROUP_4}

echo "Waiting for Batch 2 (4 tasks) to complete..."
wait # Wait for Batch 2 completion
echo "--- Batch 2 Completed ---"


# ================================================================================
# --- Batch 3 (4 models) ---
echo "========================================================================"
echo "--- Starting Batch 3 ---"
run_model_in_background "LLaVA-CoT-11B"          ${GPU_GROUP_1} ${PORT_GROUP_1}
run_model_in_background "Qwen3-VL-30B-Thinking"  ${GPU_GROUP_2} ${PORT_GROUP_2}
run_model_in_background "Qwen3-VL-30B-Instruct"  ${GPU_GROUP_3} ${PORT_GROUP_3}
run_model_in_background "Qwen3-VL-8B-Thinking"   ${GPU_GROUP_4} ${PORT_GROUP_4}

echo "Waiting for Batch 3 (4 tasks) to complete..."
wait # Wait for Batch 3 completion
echo "--- Batch 3 Completed ---"


# ================================================================================
# --- Batch 4 (Last 3 models) ---
echo "========================================================================"
echo "--- Starting Batch 4 (Final Batch) ---"
run_model_in_background "Qwen3-VL-8B-Instruct"   ${GPU_GROUP_1} ${PORT_GROUP_1}
run_model_in_background "InternVL3_5-8B"         ${GPU_GROUP_2} ${PORT_GROUP_2}
run_model_in_background "InternVL3_5-30B-A3B"    ${GPU_GROUP_3} ${PORT_GROUP_3}
# Group 4 (GPU 6,7) is idle in this batch

echo "Waiting for Batch 4 (3 tasks) to complete..."
wait # Wait for Batch 4 completion
echo "--- Batch 4 Completed ---"


# ================================================================================
echo ""
echo "##########################################################################"
echo "###                  All models executed successfully!                 ###"
echo "##########################################################################"