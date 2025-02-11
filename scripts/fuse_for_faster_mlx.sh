#!/bin/bash

mlx_lm.fuse \
    --model "deepseek-ai/deepseek-r1-distill-qwen-1.5b" \
    --save-path ./fused_model/deepseek_gasing_faster_fused/ \
    --adapter-path adapters/ \
    --hf-path ./fused_model/deepseek_gasing_faster_hf