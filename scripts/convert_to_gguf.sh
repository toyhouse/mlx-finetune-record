#!/bin/bash

# Input parameters
INPUT_MODEL_PATH="$1"
OUTPUT_PATH="$2"
REPO_ID="$3"

# Model details
MODEL_NAME="small_indo_teacher"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Create Modelfile
cat > "$OUTPUT_PATH/Modelfile" << EOL
FROM $INPUT_MODEL_PATH
PARAMETER temperature 0.6
PARAMETER top_p 0.8
SYSTEM You are a friendly and patient English language teacher who helps students improve their language skills through interactive and engaging conversations. Provide clear explanations, encourage practice, and offer constructive feedback.
EOL

# Create README
cat > "$OUTPUT_PATH/README.md" << EOL
---
language: 
  - en
tags:
  - ollama
  - gguf
  - language-model
  - small-model
  - english-teacher
license: apache-2.0
model-index:
  - name: Small Indo Teacher GGUF
    results:
      - task: 
          name: Language Teaching
          type: text-generation
        metrics:
          - type: Engagement
            value: High
            mode: qualitative

# Small Indo Teacher - GGUF Model

## Model Details
- **Base Model**: Qwen/Qwen1.5-1.8B-Chat
- **Format**: GGUF (Ollama-compatible)
- **Training**: Fine-tuned on video transcripts
- **Purpose**: English language teaching

## Usage with Ollama
\`\`\`bash
ollama run small_indo_teacher
\`\`\`

### Python Example
\`\`\`python
import ollama

response = ollama.chat(model='small_indo_teacher', messages=[
    {
        'role': 'user',
        'content': 'Can you help me improve my English grammar?'
    }
])
print(response['message']['content'])
\`\`\`

## Model Capabilities
- Provide clear language explanations
- Encourage language practice
- Offer constructive feedback
- Engage students interactively

## Limitations
- Best used for language learning and practice
- Performance may vary based on specific language contexts

## License
Apache 2.0
EOL

# Create Ollama model
ollama create "$MODEL_NAME" -f "$OUTPUT_PATH/Modelfile"

# Export model metadata
ollama show "$MODEL_NAME" > "$OUTPUT_PATH/model_info.json"

# Find and export GGUF file
GGUF_PATH="$OUTPUT_PATH/$MODEL_NAME.gguf"

# Attempt to export the model
find ~/.ollama/models -name "*$MODEL_NAME*" -type d | while read -r MODEL_DIR; do
    if [ -d "$MODEL_DIR/blobs" ]; then
        # Combine all blob files into a single GGUF file
        cat "$MODEL_DIR/blobs/"* > "$GGUF_PATH"
        break
    fi
done

# Check if GGUF file was created
if [ ! -f "$GGUF_PATH" ]; then
    echo "Failed to create GGUF file"
    exit 1
fi

# Upload to Hugging Face
huggingface-cli upload "$REPO_ID" "$OUTPUT_PATH" --repo-type model

echo "Successfully converted and uploaded GGUF model to $REPO_ID"
