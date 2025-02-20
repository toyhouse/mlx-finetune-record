#!/bin/bash

# Validate input
if [ -z "$1" ]; then
    echo "Error: Missing base model path argument"
    echo "Usage: $0 <path_to_base_model>"
    exit 1
fi

BASE_MODEL="$1"
FILTERED_WEIGHTS="${BASE_MODEL}/weights.npz"

echo "Converting PyTorch weights to MLX format for ${BASE_MODEL}"
python3 -c '
import sys
import mlx.core as mx
import torch
import numpy as np
from transformers import AutoModelForCausalLM

model_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
state_dict = {}

print("Converting weights...")
for k, v in model.state_dict().items():
    if k.startswith("score."):
        continue
    # Convert PyTorch tensor to NumPy
    np_array = v.cpu().float().numpy()
    # Convert to MLX array directly
    try:
        mlx_array = mx.array(np_array)
        state_dict[k] = mlx_array
        print(f"Converted {k}: shape={np_array.shape}, dtype={np_array.dtype}")
    except Exception as e:
        print(f"Error converting {k}: {e}")
        continue

print(f"Total converted weights: {len(state_dict)}")
print(f"Saving weights to: {output_path}")

# Save as MLX arrays
try:
    mx.save(output_path, state_dict)
except Exception as e:
    print(f"Error saving weights: {e}")
    # Try alternative saving method
    try:
        print("Attempting alternative save method...")
        arrays = {k: v.numpy() for k, v in state_dict.items()}
        np.savez(output_path, **arrays)
        print("Save successful using NumPy format")
    except Exception as e2:
        print(f"Alternative save failed: {e2}")
' "$BASE_MODEL" "$FILTERED_WEIGHTS"

echo "MLX weights saved to: ${FILTERED_WEIGHTS}"