#!/bin/bash
# Export Ollama model to GGUF
MODEL_NAME=small_indo_teacher
OUTPUT_PATH=/Users/Henrykoo/Documents/mlx-finetune-record/huggingface/small-indo-teacher-gguf/small_indo_teacher.gguf

# Find the model's storage location
MODEL_PATH=$(find ~/.ollama/models -name "*small_indo_teacher*" -type d | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Model not found in Ollama storage"
    exit 1
fi

# Copy the model file to the output path
cp "$MODEL_PATH/blobs/"* "$OUTPUT_PATH"
