#!/bin/bash

# Activate the virtual environment if necessary
# source /path/to/your/venv/bin/activate

# Run Qwen/QwQ-32B model
python -c "import mlx; from mlx_lm import load; model, tokenizer = load('Qwen/QwQ-32B'); input_data = 'Your input text here'; output = model.run(input_data); print(output)"
