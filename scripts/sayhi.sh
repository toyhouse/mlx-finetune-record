#!/bin/bash

mlx_lm.generate \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --prompt "Hi" \
    --adapter adapters/