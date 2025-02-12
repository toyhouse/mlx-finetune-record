#!/bin/bash

# Default system prompt
SYSTEM_PROMPT="You are a friendly math tutor for ages 10-15. Keep responses under 100 words. Use simple language and real-world examples. Break problems into small steps. After each explanation, ask the student if they understand or need clarification. Always encourage and celebrate progress."

# Function to format the prompt
format_prompt() {
    local user_input="$1"
    echo "```
${SYSTEM_PROMPT}

${user_input}
```
assistant"
}

# Check if input is provided as argument
if [ $# -eq 0 ]; then
    # Interactive mode
    echo "GASing Math Tutor (MLX)"
    echo "Type 'exit' to quit"
    echo "---------------------"
    
    while true; do
        echo -n ">>> "
        read -r user_input
        
        # Check for exit command
        if [ "$user_input" = "exit" ]; then
            echo "Goodbye!"
            exit 0
        fi
        
        # Format the prompt and run the model
        formatted_prompt=$(format_prompt "$user_input")
        mlx_lm.generate \
            --model ./fused_model/deepseek_gasing_fused \
            --prompt "$formatted_prompt" \
            --max-tokens 200 \
            --temp 0.7 \
            --extra-eos-token "```
        
        echo ""
    done
else
    # Single query mode
    formatted_prompt=$(format_prompt "$1")
    mlx_lm.generate \
        --model ./fused_model/deepseek_gasing_fused \
        --prompt "$formatted_prompt" \
        --max-tokens 200 \
        --temp 0.7 \
        --extra-eos-token "```
fi
