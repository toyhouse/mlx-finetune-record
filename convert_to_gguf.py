#!/usr/bin/env python3
"""
Script to convert the Indo Math Teacher model to GGUF format and upload to Hugging Face.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configure paths
model_path = "./fused_model/Qwen2.5-Math-1.5B_indo_tutor"
output_dir = "./gguf_model"
temp_dir = "./temp_gguf_upload"

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Check if llama.cpp is installed
llama_cpp_path = input("Enter the path to your llama.cpp directory (or press Enter to use ./llama.cpp): ") or "./llama.cpp"
llama_cpp_path = os.path.abspath(llama_cpp_path)

if not os.path.exists(llama_cpp_path):
    print(f"llama.cpp not found at {llama_cpp_path}")
    clone_llama = input("Would you like to clone llama.cpp? (y/n): ").lower()
    if clone_llama == 'y':
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_path])
        subprocess.run(["make", "-j"], cwd=llama_cpp_path)
    else:
        print("Please install llama.cpp manually and run this script again.")
        sys.exit(1)

# Convert to GGUF
print(f"Converting model from {model_path} to GGUF format...")

# For Qwen models, we need to use a specific conversion script
convert_script = os.path.join(llama_cpp_path, "convert-hf-to-gguf.py")

if not os.path.exists(convert_script):
    print(f"Conversion script not found at {convert_script}")
    print("Please ensure llama.cpp is properly installed.")
    sys.exit(1)

# Define output file path
output_file = os.path.join(output_dir, "indo_math_teacher_q4_k_m.gguf")

# Run conversion command
cmd = [
    "python3", convert_script,
    "--outfile", output_file,
    "--outtype", "q4_k_m",  # A good balance of size and quality
    model_path
]

print(f"Running conversion command: {' '.join(cmd)}")
result = subprocess.run(cmd)

if result.returncode != 0:
    print("Conversion failed. Please check error messages above.")
    sys.exit(1)

print(f"Successfully converted model to GGUF format: {output_file}")

# Upload to Hugging Face
print("Preparing to upload GGUF model to Hugging Face...")

# Get Hugging Face username
api = HfApi()
user_info = api.whoami()
username = user_info["name"]

# Configure repository
model_name = "indo-math-teacher-gguf"
repo_id = f"{username}/{model_name}"

# Create repository
create_repo(repo_id, private=False, exist_ok=True)
print(f"Repository {repo_id} created or already exists.")

# Copy GGUF model to temp directory
shutil.copy2(output_file, os.path.join(temp_dir, Path(output_file).name))
print(f"Copied GGUF model to temporary directory")

# Create README.md
readme_content = """# Indo Math Teacher with Gasing Method (GGUF Format)

This model is a fine-tuned version of Qwen2.5-Math-1.5B specialized in teaching mathematics using the Gasing method in Bahasa Indonesia, converted to GGUF format for broader compatibility.

## Model Description

The Gasing method (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.

This model can:
- Explain basic arithmetic operations using the Gasing method
- Provide step-by-step solutions to math problems
- Teach math concepts in Bahasa Indonesia

## Using with llama.cpp

```bash
# Download the model
wget https://huggingface.co/{repo_id}/resolve/main/indo_math_teacher_q4_k_m.gguf

# Run with llama.cpp
./main -m indo_math_teacher_q4_k_m.gguf -p "You are a math teacher using the Gasing method\\n\\nHuman: Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!\\n\\nAssistant:" -n 200 -t 0.1
```

## Using with Ollama

Create a Modelfile:

```
FROM ./indo_math_teacher_q4_k_m.gguf
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{{{.System}}}}\\n\\n{{{{.Prompt}}}}"
STOP "[/INST]"
STOP ">>> "
STOP "\\n\\nHuman:"
STOP "\\nHuman:"
```

Then create and run the model:

```bash
ollama create indo_math_teacher -f Modelfile
ollama run indo_math_teacher
```

## Example Usage

**User**: "Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"

**Model**: "Simpan 8 di kepala dan 5 di tangan. Pasangan 8 untuk mencapai 10 adalah 2. Ambil 2 dari 5 di tangan. Sekarang di kepala kita punya 10, dan di tangan tersisa 3. Jadi, 10 ditambah 3 sama dengan 13."

## About the Gasing Method

Gasing (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.

## License

This model is released under the Apache 2.0 license.
"""

readme_content = readme_content.replace("{repo_id}", repo_id)

with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Created README.md")

# Create installation script
install_script = """#!/bin/bash

echo "Installing Indo Math Teacher model (GGUF format)..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first: https://ollama.com"
    exit 1
fi

# Download GGUF model
echo "Downloading GGUF model (this might take a while)..."
curl -L "https://huggingface.co/{repo_id}/resolve/main/indo_math_teacher_q4_k_m.gguf" -o "./indo_math_teacher_q4_k_m.gguf"

# Create Modelfile
cat > Modelfile << EOL
FROM ./indo_math_teacher_q4_k_m.gguf
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{{{.System}}}}\\n\\n{{{{.Prompt}}}}"
STOP "[/INST]"
STOP ">>> "
STOP "\\n\\nHuman:"
STOP "\\nHuman:"
EOL

# Create the model in Ollama
ollama create indo_math_teacher -f Modelfile

echo "Indo Math Teacher installed successfully!"
echo "Run with: ollama run indo_math_teacher"
echo ""
echo "Example: Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"
"""

install_script = install_script.replace("{repo_id}", repo_id)

with open(os.path.join(temp_dir, "install.sh"), "w", encoding="utf-8") as f:
    f.write(install_script)
print("Created installation script")

# Upload to Hugging Face
print(f"Starting upload to {repo_id}...")
print(f"This might take a while as the GGUF model file is large...")

try:
    upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".*"],
        commit_message="Upload Indo Math Teacher model in GGUF format"
    )
    print(f"Successfully uploaded model to {repo_id}")
    print(f"View your model at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Error uploading model: {e}")

# Clean up
shutil.rmtree(temp_dir)
print("Cleaned up temporary files")
print("\nDone! Your model has been converted to GGUF format and uploaded to Hugging Face.")
