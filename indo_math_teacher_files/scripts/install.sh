#!/bin/bash

echo "Installing Indo Math Teacher model with Gasing method..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first: https://ollama.com"
    exit 1
fi

# Create directory for model files
mkdir -p model_files

# Download model files
echo "Downloading model files..."
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/added_tokens.json -o model_files/added_tokens.json
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/config.json -o model_files/config.json
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/merges.txt -o model_files/merges.txt
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/model.safetensors -o model_files/model.safetensors
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/special_tokens_map.json -o model_files/special_tokens_map.json
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/tokenizer.json -o model_files/tokenizer.json
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/tokenizer_config.json -o model_files/tokenizer_config.json
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/vocab.json -o model_files/vocab.json

# Download Modelfile
echo "Downloading Modelfile..."
curl -L https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/configs/Modelfile_Indo_Math_Teacher -o Modelfile_Indo_Math_Teacher

# Create the Ollama model
ollama create indo_math_teacher -f Modelfile_Indo_Math_Teacher

echo "Indo Math Teacher installed successfully!"
echo "Run with: ollama run indo_math_teacher"
echo ""
echo "Example: Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"
