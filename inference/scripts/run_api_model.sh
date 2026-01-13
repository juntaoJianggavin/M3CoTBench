#!/bin/bash

# Exit immediately if any command fails
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname "${SCRIPT_DIR}" )

# --- Script Arguments ---
MODEL_NAME=$1
MAIN_PORT=${2}  # Port number as the second argument
NUM_PROCESSES_PER_API_JOB=${3:-4}  # Number of processes as the third argument, defaults to 4

if [ -z "$MODEL_NAME" ] || [ -z "$MAIN_PORT" ]; then
  echo "Error: Please provide the model name and main process port number."
  echo "Usage: $0 <model_name> <port_number> [num_processes (default 4)]"
  exit 1
fi

OUTPUT_DIR="${PROJECT_ROOT}/final_output"

echo "========================================================================"
echo "Preparing to execute API task: ${MODEL_NAME}"
echo "Using ${NUM_PROCESSES_PER_API_JOB} internal parallel processes, coordinating on port ${MAIN_PORT}."
echo "========================================================================"

MODES=("cot" "direct")
for MODE in "${MODES[@]}"; do
  echo "--- Starting mode: ${MODE} ---"
  
  python -m accelerate.commands.launch \
    --num_processes ${NUM_PROCESSES_PER_API_JOB} \
    --main_process_port ${MAIN_PORT} \
    "${PROJECT_ROOT}/main.py" \
      --model "${MODEL_NAME}" \
      --output_dir "${OUTPUT_DIR}" \
      --mode "${MODE}" \
      --resume_from_checkpoint

  echo "--- Mode: ${MODE} completed ---"
done

echo "========================================================================"
echo "API task ${MODEL_NAME} completed for all modes!"
echo "========================================================================"