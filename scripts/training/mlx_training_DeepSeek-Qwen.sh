#!/bin/bash

# Run LoRA fine-tuning for DeepSeek-R1-Distill-Qwen-1.5B" model
mlx_lm.lora \
    --model "deepseek-ai/deepseek-r1-distill-qwen-1.5b" \
    --train \
    --data "./jsonl/GASING" \
    --fine-tune-type lora \
    --learning-rate 1e-4 \
    --iters 100 \
    --batch-size 4 \
    --max-seq-length 2048 \
    --steps-per-eval 50 \
    --steps-per-report 10 \
    --adapter-path adapters/