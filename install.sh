#!/bin/bash

echo "Installing Indo Math Teacher model with Gasing method..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first: https://ollama.com"
    exit 1
fi

# Create temporary Modelfile
cat > Modelfile.temp << EOL
FROM Qwen/Qwen2.5-Math-1.5B
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{ .System }}\n\n{{ .Prompt }}"
STOP "[/INST]"
STOP ">>> "
STOP "\n\nHuman:"
STOP "\nHuman:"
EOL

# Create the model in Ollama
ollama create indo_math_teacher -f Modelfile.temp

# Clean up
rm Modelfile.temp

echo "Indo Math Teacher installed successfully!"
echo "Run with: ollama run indo_math_teacher"
echo ""
echo "Example: Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"
