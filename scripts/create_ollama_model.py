import argparse
import yaml
import subprocess
import os
import shutil
import tempfile
import sys

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description='Create an Ollama model from a fused model and deployment config.')
    parser.add_argument('--fused_model_path', type=str, required=True, help='Path to the fused model.')
    parser.add_argument('--deployment_config', type=str, required=True, help='Path to the deployment configuration file.')
    
    args = parser.parse_args()
    
    # Check if fused model path exists
    fused_model_path = os.path.abspath(args.fused_model_path)
    if not os.path.exists(fused_model_path):
        print(f"Error: Fused model path '{fused_model_path}' does not exist.")
        sys.exit(1)
    
    # Load deployment configuration
    deployment_config = load_yaml_config(args.deployment_config)
    
    # Verify that the platform is 'ollama'
    if deployment_config.get('platform', '').lower() != 'ollama':
        raise ValueError(f"Expected platform 'ollama', but got '{deployment_config.get('platform')}'")
    
    # Extract parameters from deployment config
    model_name = deployment_config.get('model_name')
    temperature = deployment_config.get('temperature', 0.7)
    top_p = deployment_config.get('top_p', 0.7)
    start_token = deployment_config.get('start', '')
    stop_token = deployment_config.get('stop', '')
    system_prompt = deployment_config.get('system_prompt', '')
    modelfile_template_path = deployment_config.get('modelfile_template')
    
    # Validate model name (simple check)
    if not model_name or not model_name.islower() or ' ' in model_name:
        print(f"Warning: Model name '{model_name}' may not be valid for Ollama. Model names should be lowercase with no spaces.")
    
    print(f"Creating Ollama model with the following parameters:")
    print(f"Model Name: {model_name}")
    print(f"Fused Model Path: {fused_model_path}")
    print(f"System Prompt: {system_prompt}")
    
    # Read the modelfile template
    with open(modelfile_template_path, 'r') as file:
        modelfile_content = file.read()
    
    # Replace placeholders in the template
    replacements = {
        '{fuse_model_path}': fused_model_path,
        '{temperature}': str(temperature),
        '{top_p}': str(top_p),
        '{start}': start_token,
        '{stop}': stop_token,
        '{system_prompt}': system_prompt,
        '{SYSTEM_PROMPT}': system_prompt  # Adding this for backward compatibility
    }
    
    for placeholder, value in replacements.items():
        modelfile_content = modelfile_content.replace(placeholder, value)
    
    # Create a temporary directory for the Modelfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the Modelfile
        modelfile_path = os.path.join(tmpdir, 'Modelfile')
        with open(modelfile_path, 'w') as file:
            file.write(modelfile_content)
        
        # Display the Modelfile content for debugging
        print(f"\nGenerated Modelfile content:")
        print("-" * 40)
        with open(modelfile_path, 'r') as file:
            print(file.read())
        print("-" * 40)
        
        # Create the Ollama model
        command = ['ollama', 'create', model_name, '-f', modelfile_path]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if command was successful
        if result.returncode != 0:
            print(f"Error: Failed to create Ollama model. Exit code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            sys.exit(1)
        else:
            print(f"Stdout: {result.stdout}")
            print(f"Ollama model '{model_name}' created successfully!")

if __name__ == '__main__':
    main()
