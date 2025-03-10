#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")/.."

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait a moment for Ollama to start
sleep 5

# Run the test script
echo "Running model math ability test..."
python scripts/test_model_math_ability.py

# Stop Ollama service
kill $OLLAMA_PID

# Deactivate virtual environment
deactivate

echo "Setup and testing completed!"
