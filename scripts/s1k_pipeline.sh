#!/bin/bash
set -eo pipefail

# Default values
MODEL_NAME=""
BASE_MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATA_DIR="${PWD}/data/s1k_prob"
DATA_PATH="${DATA_DIR}"  # Point to directory, not file
ADAPTER_PATH=""
FUSED_PATH=""
MODE="all"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
TRAINING_ITERS=100

# Add explicit path verification
prepare_data() {
    echo "Preparing dataset..."
    python Python/prepare_s1_prob.py
    
    # Verify both train and validation files exist
    if [ ! -s "${DATA_DIR}/train.jsonl" ]; then
        echo "Error: Training file missing or empty: ${DATA_DIR}/train.jsonl"
        exit 1
    fi
    if [ ! -s "${DATA_DIR}/valid.jsonl" ]; then  # Changed to valid.jsonl
        echo "Error: Validation file missing or empty: ${DATA_DIR}/valid.jsonl"
        exit 1
    fi
    echo "Training files content:"
    echo "train.jsonl:"
    cat "${DATA_DIR}/train.jsonl"
    echo "valid.jsonl:"  # Changed output message
    cat "${DATA_DIR}/valid.jsonl"
}

# Train the model
train_model() {
    echo "Training model: $MODEL_NAME"
    mkdir -p "models/${MODEL_NAME}"
    ADAPTER_PATH="models/${MODEL_NAME}/adapter"
    
    echo "Training with data directory: ${DATA_PATH}"
    python -m mlx_lm.lora \
        --model "${BASE_MODEL}" \
        --train \
        --data "${DATA_PATH}" \
        --batch-size 1 \
        --iters 2 \
        --save-every 1 \
        --adapter-path "${ADAPTER_PATH}" \
        --val-batches 1 \
        --max-seq-length 32
}

# Fuse the adapter with the base model
fuse_model() {
    echo "Fusing model: $MODEL_NAME"
    ADAPTER_PATH="models/${MODEL_NAME}/adapter"
    FUSED_PATH="models/${MODEL_NAME}/fused"
    mkdir -p "${FUSED_PATH}"
    
    # Check if adapter exists
    if [ ! -d "${ADAPTER_PATH}" ]; then
        echo "Error: Adapter directory not found: ${ADAPTER_PATH}"
        exit 1
    fi
    
    # Check if adapter config exists
    if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
        echo "Error: adapter_config.json not found in ${ADAPTER_PATH}"
        ls -la "${ADAPTER_PATH}"
        exit 1
    fi
    
    python -m mlx_lm.fuse \
        --model "${BASE_MODEL}" \
        --save-path "${FUSED_PATH}" \
        --adapter-path "${ADAPTER_PATH}" \
        --hf-path "${FUSED_PATH}_hf"
    
    # Copy all necessary files from fused model to HF path
    echo "Copying model files to HF format..."
    cp "${FUSED_PATH}"/* "${FUSED_PATH}_hf/"
    
}


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
    exit 1
fi

# Activate virtual environment
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Virtual environment not found!"
        exit 1
    fi
}


create_ollama_model() {
    # Convert model name to lowercase and replace invalid characters with underscores
    OLLAMA_MODEL_NAME=$(echo "${MODEL_NAME}_MOST_RECENT")
    echo "Creating Ollama model package..."
    mkdir -p "${MODEL_NAME}"
    cp "${FUSED_PATH}/config.json" "${MODEL_NAME}/config.json"
    cp "${FUSED_PATH}/model.safetensors" "${MODEL_NAME}/model.safetensors"
    
    echo "FROM ${PWD}/models/${MODEL_NAME}_model/fused"

    # Ensure modelfiles directory exists
    mkdir -p modelfiles
    
    # Create Modelfile with absolute paths
    cat > "modelfiles/${MODEL_NAME}_Modelfile" << EOF
FROM ${PWD}/models/${MODEL_NAME}_model/fused

PARAMETER temperature 0.7
PARAMETER top_p 0.7
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"

SYSTEM You are a friendly and patient math tutor for students aged 10-15. You explain math concepts using clear, step-by-step instructions and real-world examples. You encourage students to think through problems and verify their understanding. You maintain a positive and encouraging tone throughout the conversation.
EOF
    
    # Create the Ollama model using sanitized name
    ollama create "${OLLAMA_MODEL_NAME}" \
        -f "modelfiles/${MODEL_NAME}_Modelfile" \
        && echo "Model created: ${OLLAMA_MODEL_NAME}"
}


# Main execution flow
main() {
    activate_venv
    prepare_data  # Add data preparation before training
    
    case "$MODE" in
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
            train_model
            fuse_model
            create_ollama_model
            ;;
        *)
            echo "Invalid mode: $MODE"
            exit 1
            ;;
    esac
}

main "$@"