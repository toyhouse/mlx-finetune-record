#!/usr/bin/env python3
"""
Script to upload the Indo Math Teacher GGUF model to Hugging Face.
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder, login

# Set Hugging Face repository details - username will be determined at runtime
MODEL_PATH = "../gguf_model/indo_math_teacher_bf16.gguf"

# README content for the GGUF model repository
README_CONTENT = """# Indo Math Teacher (GGUF)

A fine-tuned math teaching model using the Gasing method for teaching mathematics in Bahasa Indonesia.

## Model Description

This is the GGUF version of the Indo Math Teacher model, fine-tuned from Qwen2.5-Math-1.5B to provide step-by-step mathematical explanations using the Gasing method in Bahasa Indonesia.

### Gasing Method

The Gasing method (Gampang, ASyik, dan menyenaNGkan - Easy, Fun, and Enjoyable) is an approach to teaching mathematics that focuses on:
- Building strong foundational understanding through visual and practical examples
- Using step-by-step explanations
- Making math enjoyable and accessible

## Usage

You can use this model with llama.cpp, text-generation-webui, or any other tool that supports GGUF models.

### Using with llama.cpp

1. Clone llama.cpp repository:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

2. Download the model:
```bash
# Replace USERNAME with the actual repository username
wget https://huggingface.co/USERNAME/indo-math-teacher-gguf/resolve/main/indo_math_teacher_bf16.gguf
```

3. Run the model:
```bash
./main -m indo_math_teacher_bf16.gguf -n 1024 -p "Kamu adalah guru matematika yang menggunakan metode Gasing. Tolong jelaskan cara menghitung 7 x 8:"
```

### Using with Ollama

Create a Modelfile:
```
FROM gguf:indo_math_teacher_bf16.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are a math teacher using the Gasing method."
```

Create the model and chat with it:
```bash
ollama create indo-math-teacher -f Modelfile
ollama run indo-math-teacher
```

## Examples

### Basic Multiplication
**Input**: "Bagaimana cara mengalikan 7 x 8 dengan metode Gasing?"

**Output**: 
```
Untuk mengalikan 7 x 8 dengan metode Gasing, kita bisa menggunakan pendekatan berikut:

1. Pertama, kita ingat bahwa 5 x 8 = 40
2. 7 berarti 5 + 2
3. Jadi 7 x 8 = (5 x 8) + (2 x 8) = 40 + 16 = 56

Atau cara lain:
1. 7 x 8 = 7 x (10 - 2) = 70 - 14 = 56

Dengan metode Gasing, kita memecah perhitungan menjadi langkah-langkah yang lebih mudah sehingga siswa dapat memahami konsep perkalian dengan lebih baik.
```

## Model Details

- **Base Model**: Qwen2.5-Math-1.5B
- **Training Data**: Collection of mathematical explanations using the Gasing method in Bahasa Indonesia
- **Format**: BF16 GGUF (Full Precision)
- **Context Length**: 4096 tokens
- **License**: The same license as the base model (Qwen2.5)

## Available Versions

This model is available in two quantization formats:
1. **indo_math_teacher_bf16.gguf** (this version): Full BF16 precision, best quality for math reasoning
2. **indo_math_teacher_q8_0.gguf**: 8-bit quantized, smaller file size with slight quality reduction
"""

def main():
    """Upload the GGUF model to Hugging Face."""
    # First check if token exists or prompt for it
    try:
        api = HfApi()
        whoami_info = api.whoami()
        username = whoami_info.get("name", None)
        print(f"Using existing Hugging Face token for user: {username}")
        
        if not username:
            username = input("Couldn't detect username. Please enter your Hugging Face username: ")
    except Exception:
        # Token doesn't exist or is invalid
        token = input("Enter your Hugging Face token: ")
        login(token=token)
        # Get username after login
        api = HfApi()
        whoami_info = api.whoami()
        username = whoami_info.get("name", None)
        if not username:
            username = input("Please enter your Hugging Face username: ")
    
    repo_id = f"{username}/indo-math-teacher-gguf"
    print(f"Creating repository: {repo_id}")
    
    # Update README with correct username
    updated_readme = README_CONTENT.replace("USERNAME", username)
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare README.md
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme)
        
        # Copy model file
        model_filename = os.path.basename(MODEL_PATH)
        dst_path = os.path.join(temp_dir, model_filename)
        print(f"Copying model from {MODEL_PATH} to {dst_path}")
        shutil.copy2(MODEL_PATH, dst_path)
        
        # Upload to Hugging Face
        print(f"Uploading files to {repo_id}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[".*"],
            commit_message="Upload Indo Math Teacher BF16 GGUF model (full precision)"
        )
        
        print(f"Upload complete! Model available at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
