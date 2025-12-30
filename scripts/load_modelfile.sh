#!/bin/bash

# Path to the Modelfile
MODELFILE="/Users/bkoo/Documents/Development/AIProjects/mlx-finetune-record/modelfiles/lckoo_example_Modelfile"

# Extract the GGUF path from the Modelfile
GGUF_PATH=$(grep '^FROM' "$MODELFILE" | awk '{print $2}')

# Extract the model name from the GGUF path
MODEL_NAME=$(basename "$GGUF_PATH" .gguf)

# Create a temporary Modelfile in Ollama format
TEMP_MODELFILE="/tmp/${MODEL_NAME}_ollama.Modelfile"
{
    echo "FROM $GGUF_PATH"
    grep '^PARAMETER' "$MODELFILE"
    grep '^SYSTEM' "$MODELFILE"
} > "$TEMP_MODELFILE"

# First, pull a base model that supports GGUF conversion
BASE_MODEL="llama2"  # You can change this to another supported base model
echo "Pulling base model $BASE_MODEL..."
ollama pull $BASE_MODEL

# Create the model in Ollama using the base model and GGUF file
echo "Creating model $MODEL_NAME in Ollama..."
ollama create "$MODEL_NAME" -f "$TEMP_MODELFILE"

if [ $? -eq 0 ]; then
    # Run the model
    echo "Running model $MODEL_NAME..."
    ollama run "$MODEL_NAME"
else
    echo "Failed to create model. Please ensure:"
    echo "1. The GGUF file exists at $GGUF_PATH"
    echo "2. You have a compatible base model installed"
    echo "3. The GGUF file is in the correct format"
fi