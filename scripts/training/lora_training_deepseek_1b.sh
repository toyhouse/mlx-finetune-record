#!/bin/bash

# Run LoRA fine-tuning for Qwen2.5-Coder model
mlx_lm.lora \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --train \
    --data "./jsonl/GASING" \
    --learning-rate 1e-4 \
    --iters 100 \
    --fine-tune-type full