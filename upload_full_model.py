#!/usr/bin/env python3
"""
Script to upload the complete Indo Math Teacher model to Hugging Face Hub.
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder

# Get Hugging Face username
api = HfApi()
user_info = api.whoami()
username = user_info["name"]

# Configuration
model_name = "indo-math-teacher-complete"
repo_id = f"{username}/{model_name}"
model_path = "./fused_model/Qwen2.5-Math-1.5B_indo_tutor"
ollama_config_path = "./configs/deployment_configs/indo_math_ollama_config.yml"

# Create repository
try:
    create_repo(repo_id, private=False, exist_ok=True)
    print(f"Repository {repo_id} created or already exists.")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Prepare temporary directory for upload
temp_dir = "./temp_upload_full"
os.makedirs(temp_dir, exist_ok=True)

# Copy model files
if os.path.exists(model_path):
    # Copy all model files
    for file in os.listdir(model_path):
        src = os.path.join(model_path, file)
        dst = os.path.join(temp_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {file}")
    print(f"Copied model files from {model_path}")
else:
    print(f"Model path {model_path} not found!")
    exit(1)

# Create installation script
install_script = """#!/bin/bash

echo "Installing Indo Math Teacher model with Gasing method..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first: https://ollama.com"
    exit 1
fi

# Download model files
MODEL_DIR="$HOME/.ollama/models/indo_math_teacher"
mkdir -p "$MODEL_DIR"

echo "Downloading model files..."
FILES=(
    "added_tokens.json"
    "config.json"
    "merges.txt"
    "model.safetensors"
    "model.safetensors.index.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
)

HF_REPO="USER_REPO_PLACEHOLDER"

for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    curl -L "https://huggingface.co/$HF_REPO/resolve/main/$file" -o "$MODEL_DIR/$file"
done

# Create Modelfile
cat > Modelfile.temp << EOL
FROM qwen/Qwen2.5-1.5B
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{ .System }}\\n\\n{{ .Prompt }}"
STOP "[/INST]"
STOP ">>> "
STOP "\\n\\nHuman:"
STOP "\\nHuman:"
EOL

# Create the model in Ollama
ollama create indo_math_teacher -f Modelfile.temp

echo "Indo Math Teacher installed successfully!"
echo "Run with: ollama run indo_math_teacher"
echo ""
echo "Example: Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"
"""

# Replace placeholder with actual repo
install_script = install_script.replace("USER_REPO_PLACEHOLDER", repo_id)

with open(os.path.join(temp_dir, "install.sh"), "w", encoding="utf-8") as f:
    f.write(install_script)
print("Created installation script")

# Create Modelfile
with open(ollama_config_path, "r") as f:
    import yaml
    ollama_config = yaml.safe_load(f)

modelfile_content = f"""FROM Qwen/Qwen2.5-Math-1.5B
SYSTEM "{ollama_config['system_prompt']}"

PARAMETER temperature {ollama_config['temperature']}
PARAMETER top_p {ollama_config['top_p']}
PARAMETER num_predict {ollama_config.get('num_predict', 200)}

# Set stop sequences
TEMPLATE "{{ .System }}\\n\\n{{ .Prompt }}"
STOP "{ollama_config['stop']}"
"""

for stop in ollama_config.get('additional_stops', []):
    modelfile_content += f'STOP "{stop}"\n'

with open(os.path.join(temp_dir, "Modelfile"), "w", encoding="utf-8") as f:
    f.write(modelfile_content)
print("Created Modelfile")

# Create README.md
readme_content = """# Indo Math Teacher with Gasing Method (Complete Model)

This repository contains the complete fine-tuned model files for Indo Math Teacher, which teaches mathematics using the Gasing method in Bahasa Indonesia.

## Installation

### Quick Installation

For easy installation, run:

```bash
curl -s https://raw.githubusercontent.com/USERNAME/indo-math-teacher-complete/main/install.sh | bash
```

### Manual Installation

If you prefer to install manually:

1. Install [Ollama](https://ollama.com)
2. Download all model files from this repository
3. Create a `Modelfile` with the following content:

```
FROM Qwen/Qwen2.5-Math-1.5B
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{ .System }}\\n\\n{{ .Prompt }}"
STOP "[/INST]"
STOP ">>> "
STOP "\\n\\nHuman:"
STOP "\\nHuman:"
```

4. Create the model in Ollama:
```bash
ollama create indo_math_teacher -f Modelfile
```

5. Run the model:
```bash
ollama run indo_math_teacher
```

## Example Usage

**User**: "Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"

**Model**: "Simpan 8 di kepala dan 5 di tangan. Pasangan 8 untuk mencapai 10 adalah 2. Ambil 2 dari 5 di tangan. Sekarang di kepala kita punya 10, dan di tangan tersisa 3. Jadi, 10 ditambah 3 sama dengan 13."

## About the Gasing Method

Gasing (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.
"""

readme_content = readme_content.replace("USERNAME", username)

with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Created README.md")

# Upload to Hugging Face
try:
    print(f"Starting upload to {repo_id}...")
    print(f"This might take a while as the model files are large (around 3GB)...")
    
    upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".*"],
        commit_message="Upload complete Indo Math Teacher model with Gasing method"
    )
    print(f"Successfully uploaded model to {repo_id}")
    print(f"View your model at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Error uploading model: {e}")

# Clean up
shutil.rmtree(temp_dir)
print("Cleaned up temporary files")
