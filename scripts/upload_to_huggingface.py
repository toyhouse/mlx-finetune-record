#!/usr/bin/env python3
"""
Script to upload the Indo Math Teacher model to Hugging Face Hub.
Uploads both the LoRA adapter and the necessary files for Ollama integration.
"""

import os
import shutil
import json
from huggingface_hub import HfApi, create_repo, upload_folder

# Get Hugging Face username
api = HfApi()
user_info = api.whoami()
username = user_info["name"]

# Configuration
model_name = "indo-math-teacher-gasing"
repo_id = f"{username}/{model_name}"
adapter_path = "./adapters/Qwen/Qwen2.5-Math-1.5B_indo_tutor_20250327_153426"  # Updated to the most recent adapter
base_model = "Qwen/Qwen2.5-Math-1.5B"
ollama_config_path = "./configs/deployment_configs/indo_math_ollama_config.yml"

# Create repository
try:
    create_repo(repo_id, private=False, exist_ok=True)
    print(f"Repository {repo_id} created or already exists.")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Prepare temporary directory for upload
temp_dir = "./temp_upload"
os.makedirs(temp_dir, exist_ok=True)

# Copy adapter files
if os.path.exists(adapter_path):
    # Copy adapter files
    shutil.copytree(adapter_path, os.path.join(temp_dir, "adapter"), dirs_exist_ok=True)
    print(f"Copied adapter files from {adapter_path}")
else:
    print(f"Adapter path {adapter_path} not found!")
    exit(1)

# Create model card
model_card = f"""---
language:
  - id
  - en
tags:
  - gasing-method
  - math-tutor
  - indonesian
  - education
  - ollama
license: apache-2.0
base_model: {base_model}
---

# Indo Math Teacher - Gasing Method

This model is a fine-tuned version of {base_model} specialized in teaching mathematics using the Gasing method in Bahasa Indonesia.

## Model Description

The Gasing method (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.

This model can:
- Solve basic arithmetic problems directly and simply
- Explain problem-solving steps in a clear manner
- Use the Gasing method's approach to mathematical education

## Usage with Ollama

To use this model with Ollama:

1. Install [Ollama](https://ollama.com) on your device
2. Create a Modelfile:

```
FROM {base_model}
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

3. Create the model in Ollama:
```bash
ollama create indo_math_teacher -f Modelfile
```

4. Run the model:
```bash
ollama run indo_math_teacher
```

## Example Usage

**User**: "Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"

**Model**: "Simpan 8 di kepala dan 5 di tangan. Pasangan 8 untuk mencapai 10 adalah 2. Ambil 2 dari 5 di tangan. Sekarang di kepala kita punya 10, dan di tangan tersisa 3. Jadi, 10 ditambah 3 sama dengan 13."

## Model Limitations

- This model is specifically trained for mathematical concepts using the Gasing method
- It works best with arithmetic problems and basic math concepts
- Responses are optimized for Indonesian language
"""

with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(model_card)
print("Created model card")

# Create Modelfile for Ollama
with open(ollama_config_path, "r") as f:
    import yaml
    ollama_config = yaml.safe_load(f)

modelfile_content = f"""FROM {base_model}
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

# Create config.json with model metadata
config = {
    "base_model": base_model,
    "model_type": "lora",
    "system_prompt": ollama_config['system_prompt'],
    "temperature": ollama_config['temperature'],
    "top_p": ollama_config['top_p'],
    "num_predict": ollama_config.get('num_predict', 200)
}

with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print("Created config.json")

# Upload to Hugging Face
try:
    upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".*"],
        commit_message="Upload Indo Math Teacher with Gasing method"
    )
    print(f"Successfully uploaded model to {repo_id}")
    print(f"View your model at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Error uploading model: {e}")

# Clean up
shutil.rmtree(temp_dir)
print("Cleaned up temporary files")
