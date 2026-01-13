#!/bin/bash

# Exit immediately if any command fails
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname "${SCRIPT_DIR}" )

# --- Script Arguments ---
# $1: Model Name (e.g., "LLaVA-CoT")
# $2: GPU Device IDs to use (e.g., "0,1")
# $3: Run mode (optional, 'cot', 'direct', or 'all'. Default: 'all')
# $4: Base port number (optional. Default: 10580)
MODEL_NAME=$1
GPU_DEVICES=$2
MODE_ARG=${3:-"all"}      # Defaults to "all" if $3 is not provided
BASE_PORT_ARG=${4:-10580} # Defaults to 10580 if $4 is not provided

if [ -z "$MODEL_NAME" ] || [ -z "$GPU_DEVICES" ]; then
  echo "Error: Must provide model name (arg 1) and GPU device IDs (arg 2)."
  echo ""
  echo "Usage: $0 <model_name> <gpu_ids> [mode] [base_port]"
  echo "  [mode]: 'cot', 'direct', or 'all' (default: 'all')"
  echo "  [base_port]: Base port number (default: 10580)"
  exit 1
fi

# --- Set modes to run based on mode argument ---
MODES_TO_RUN=() # Initialize an empty array
if [ "$MODE_ARG" == "all" ]; then
  MODES_TO_RUN=("cot" "direct")
elif [ "$MODE_ARG" == "cot" ]; then
  MODES_TO_RUN=("cot")
elif [ "$MODE_ARG" == "direct" ]; then
  MODES_TO_RUN=("direct")
else
  echo "Error: Invalid mode '$MODE_ARG'. Only 'cot', 'direct', or 'all' are supported."
  exit 1
fi

# --- Environment Configuration ---
export HIP_VISIBLE_DEVICES=${GPU_DEVICES}
# export CUDA_VISIBLE_DEVICES=${GPU_DEVICES} # If you are using NVIDIA cards, this might be more common
NUM_PROCESSES=$(echo "${GPU_DEVICES}" | tr ',' '\n' | wc -l)
OUTPUT_DIR="${PROJECT_ROOT}/final_output"

# --- Port Allocation Logic (Modified) ---
# Use passed $4 (or default) as the port for cot mode
# direct mode port is cot mode port + 1
COT_PORT=$((BASE_PORT_ARG))
DIRECT_PORT=$((COT_PORT + 1))

echo "========================================================================"
echo "Preparing to execute local GPU task: ${MODEL_NAME}"
echo "  - Using GPU: ${HIP_VISIBLE_DEVICES}"
echo "  - Number of processes to start: ${NUM_PROCESSES}"
echo "  - Run mode (original arg): ${MODE_ARG}"
echo "  - Modes to execute: ${MODES_TO_RUN[*]}"
echo "  - Base port (original arg): ${BASE_PORT_ARG}"
echo "  - CoT mode port: ${COT_PORT}"
echo "  - Direct mode port: ${DIRECT_PORT}"
echo "========================================================================"

# --- Loop through selected modes (Modified) ---
for MODE in "${MODES_TO_RUN[@]}"; do
  
  # Dynamically select port based on current loop mode
  if [ "$MODE" == "cot" ]; then
    PORT=${COT_PORT}
  elif [ "$MODE" == "direct" ]; then
    PORT=${DIRECT_PORT}
  else
    # This case should theoretically not happen as we checked it above
    echo "Internal Error: Unknown mode $MODE"
    exit 1
  fi

  echo "--- Starting mode: ${MODE} (Using port: ${PORT}) ---"
  
  python -m accelerate.commands.launch  \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port=${PORT} \
    "${PROJECT_ROOT}/main.py" \
    --model "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --mode "${MODE}" \
    --resume_from_checkpoint

  echo "--- Mode: ${MODE} execution completed ---"
done

echo "========================================================================"
echo "Local GPU task ${MODEL_NAME} completed for all modes!"
echo "========================================================================"