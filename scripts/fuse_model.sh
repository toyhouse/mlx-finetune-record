#!/bin/bash

mlx_lm.fuse \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --save-path ./fused_model/qwen2.5_coder_fused/ \
    --adapter-path adapters/