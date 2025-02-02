#!/bin/bash

# Exit on error
set -e

# Check if transformers is installed
if ! pip show transformers > /dev/null; then
    echo "Installing transformers package..."
    pip install transformers
fi

# Create output directories
mkdir -p ./gguf_model
mkdir -p ./hf_converted

echo "Converting MLX model to Hugging Face format..."
python -c "
from transformers import AutoConfig
import torch
import json
import os

# Load the model config and weights
model_path = './fused_model/qwen2.5_coder_fused'
output_path = './hf_converted/qwen_math'

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Copy the config and tokenizer files
for file in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
    if os.path.exists(os.path.join(model_path, file)):
        os.system(f'cp {os.path.join(model_path, file)} {os.path.join(output_path, file)}')

# Load and convert weights if they exist
if os.path.exists(os.path.join(model_path, 'model.safetensors')):
    os.system(f'cp {os.path.join(model_path, 'model.safetensors')} {os.path.join(output_path, 'model.safetensors')}')
elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
    os.system(f'cp {os.path.join(model_path, 'pytorch_model.bin')} {os.path.join(output_path, 'pytorch_model.bin')}')
"

echo "Converting to GGUF format..."
echo "Installing required packages..."
pip install llama-cpp-python

# Clone llama.cpp for the conversion script
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

# Install the conversion script requirements
cd llama.cpp
pip install -e .
cd ..

# Convert using the conversion script
python llama.cpp/convert_hf_to_gguf.py ./hf_converted/qwen_math \
    --outfile ./gguf_model/qwen_math.gguf \
    --outtype q8_0

echo "Creating Ollama model..."
ollama create qwen_math_0.5B -f Modelfile