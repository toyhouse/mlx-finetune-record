#!/bin/bash
set -eo pipefail

# Default values
MODEL_NAME=""
# BASE_MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="data/GASING"
ADAPTER_PATH=""
FUSED_PATH=""
MODE="all"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
TRAINING_ITERS=200  # Number of training iterations

# Print usage
usage() {
    echo "Usage: $0 -n MODEL_NAME [-b BASE_MODEL] [-d DATA_PATH] [-m MODE]"
    echo "  -n MODEL_NAME     Name of the model (required)"
    echo "  -b BASE_MODEL     Base model to use (default: ${BASE_MODEL})"
    echo "  -d DATA_PATH      Path to training data (default: ${DATA_PATH})"
    echo "  -m MODE          Mode to run: train, fuse, create, all (default: all)"
    exit 1
}

# Parse arguments
while getopts "n:b:d:m:h" opt; do
    case $opt in
        n) MODEL_NAME="$OPTARG" ;;
        b) BASE_MODEL="$OPTARG" ;;
        d) DATA_PATH="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is required"
    usage
fi

# Set dependent paths
ADAPTER_PATH="adapters/${MODEL_NAME}_lora"
FUSED_PATH="fused_model/${MODEL_NAME}_fused"

# Create necessary directories
mkdir -p "${LOG_DIR}" "data" "adapters" "fused_model"

# Validate data path
if [ ! -e "${DATA_PATH}" ]; then
    echo "Warning: Training data not found at ${DATA_PATH}"
    echo "Please ensure your training data exists before running training mode"
fi

activate_venv() {
    source "${VENV_DIR}/bin/activate"
}

train_model() {
    echo "Starting LoRA training..."
    if [ ! -e "${DATA_PATH}" ]; then
        echo "Error: Training data not found at ${DATA_PATH}"
        exit 1
    fi
    
    python -m mlx_lm.lora \
        --model "${BASE_MODEL}" \
        --train \
        --data "${DATA_PATH}" \
        --batch-size 4 \
        --iters ${TRAINING_ITERS} \
        --adapter-path "${ADAPTER_PATH}" \
        --save-every 50 \
        --learning-rate 1e-4 \
        --steps-per-report 10 \
        | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
}

fuse_model() {
    echo "Fusing adapter to base model..."
    mkdir -p "${FUSED_PATH}" "${FUSED_PATH}_hf"
    
    python -m mlx_lm.fuse \
        --model "${BASE_MODEL}" \
        --save-path "${FUSED_PATH}" \
        --adapter-path "${ADAPTER_PATH}" \
        --hf-path "${FUSED_PATH}_hf"
    
    # Copy all necessary files from fused model to HF path
    echo "Copying model files to HF format..."
    cp "${FUSED_PATH}"/* "${FUSED_PATH}_hf/"
    
    # Copy tokenizer files from base model to HF path
    python -c "
from huggingface_hub import snapshot_download
from shutil import copy2
import os

model_path = snapshot_download('${BASE_MODEL}')
for file in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt']:
    src = os.path.join(model_path, file)
    if os.path.exists(src):
        copy2(src, os.path.join('${FUSED_PATH}_hf', file))
"
    echo "Model fusion complete"
}

create_ollama_model() {
    echo "Creating Ollama model package..."
    
    # Ensure modelfiles directory exists
    mkdir -p modelfiles
    
    # Create Modelfile from template
    cp modelfiles/ShortAnswer_Template.txt modelfiles/${MODEL_NAME}_Modelfile

    # Replace placeholder with actual path
    sed -i '' "s|{FUSED_MODEL_PATH}|${PWD}/fused_model/${MODEL_NAME}_fused_hf|g" modelfiles/${MODEL_NAME}_Modelfile
    
    # Create the Ollama model
    ollama create "${MODEL_NAME}" \
        -f "modelfiles/${MODEL_NAME}_Modelfile" \
        && echo "Model created: ${MODEL_NAME}"
}

main() {
    activate_venv
    
    case "${MODE}" in
        "train")
            train_model
            ;;
        "fuse")
            fuse_model
            ;;
        "create")
            create_ollama_model
            ;;
        "all")
            train_model && fuse_model && create_ollama_model
            ;;
        *)
            echo "Invalid mode: ${MODE}"
            exit 1
            ;;
    esac
}

main "$@"