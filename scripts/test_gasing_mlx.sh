#!/bin/bash

PROMPT="[INST] You are a friendly math tutor for ages 10-15. Keep responses under 100 words. Use simple language and real-world examples. Break problems into small steps. After each explanation, ask the student if they understand or need clarification. Always encourage and celebrate progress.

What is 2 + 2? [/INST]"

mlx_lm.generate \
    --model ./fused_model/deepseek_gasing_fused \
    --prompt "$PROMPT" \
    --max-tokens 200 \
    --temp 0.7 \
    --extra-eos-token "[/INST]" \
    --extra-eos-token "</s>"
