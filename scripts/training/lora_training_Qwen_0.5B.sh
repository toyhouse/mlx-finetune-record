#!/bin/bash

# Run LoRA fine-tuning for Qwen2.5-Coder model
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/GASING_converted" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
