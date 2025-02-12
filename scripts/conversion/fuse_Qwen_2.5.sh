#!/bin/bash

mlx_lm.fuse \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --save-path ./fused_model/qwen2.5_coder_0.5B_fused/ \
    --adapter-path adapters/ \
    --hf-path ./fused_model/qwen2.5_coder_0.5B_hf