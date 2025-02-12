#!/bin/bash

# Check if llama.cpp is installed
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make
    cd ..
else
    echo "llama.cpp already exists"
fi

# Install required Python packages
echo "Installing required Python packages..."
pip install -r llama.cpp/requirements.txt

# Create GGUF directory if it doesn't exist
mkdir -p ./fused_model/GGUF

# Convert MLX model to GGUF format
echo "Converting MLX model to GGUF format..."
python3 llama.cpp/convert_hf_to_gguf.py \
    --outfile ./fused_model/GGUF/qwen_coder.gguf \
    --outtype q8_0 \
    --model-dir ./fused_model/qwen2.5_coder_fused

echo "Done! GGUF model saved to ./fused_model/GGUF/qwen_coder.gguf"
