#!/bin/bash

mlx_lm.generate \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --prompt "Hello, can you say hi in Chinese?" \
    --adapter adapters/