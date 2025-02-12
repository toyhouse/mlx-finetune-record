#!/bin/bash

mlx_lm.fuse \
    --model "deepseek-ai/deepseek-r1-distill-qwen-1.5b" \
    --save-path ./fused_model/deepseek-r1_fused/ \
    --adapter-path adapters/