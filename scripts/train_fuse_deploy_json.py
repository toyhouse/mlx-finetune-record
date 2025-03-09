import argparse
import subprocess
import json
import os
import tempfile
import sys
from datetime import datetime

def load_json_config(file_path):
    """Load JSON configuration from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def train_model(config):
    """Train the model using mlx_lm.lora."""
    print("\n" + "="*50)
    print("STEP 1: TRAINING MODEL")
    print("="*50)
    
    # Extract training configuration
    model_name = config['model']['name']
    training_data_name = config['data']['name']
    training_data_path = config['data']['path']
    training_config = config['training']
    
    # Generate timestamp for saving the adapter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_save_path = training_config.get('adapter_path', './adapters/{model_name}_{data_name}')
    adapter_path = adapter_save_path.format(model_name=model_name, data_name=training_data_name) + f"_{timestamp}"
    
    # Print information
    print(f'Model     : {model_name}')
    print(f'Data      : {training_data_path}')
    print(f'Training  : {training_config}')
    print(f'Adapter Save Path: {adapter_path}')
    
    # Execute the mlx_lm.lora command
    command = [
        'mlx_lm.lora',
        '--model', model_name,
        '--train',
        '--data', training_data_path,
        '--learning-rate', str(training_config.get('learning_rate')),
        '--iters', str(training_config.get('iterations')),
        '--fine-tune-type', training_config.get('fine-tune-type'),
        '--adapter-path', adapter_path
    ]
    
    print(f"Running training command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during training: {result.stderr}")
        sys.exit(1)
    else:
        print(f"Training completed successfully!")
        print(f"Adapter saved to: {adapter_path}")
    
    return adapter_path

def fuse_model(config, adapter_path):
    """Fuse the model with adapter weights."""
    print("\n" + "="*50)
    print("STEP 2: FUSING MODEL")
    print("="*50)
    
    # Extract model and fusion configuration
    model_name = config['model']['name']
    
    # Determine output path for fused model
    output_path = config.get('fuse', {}).get('output_path')
    if not output_path:
        # Generate default output path
        data_name = config['data']['name']
        output_path = f"./fused_model/{model_name.split('/')[-1]}_{data_name}"
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Fusing model '{model_name}' with adapter from '{adapter_path}'")
    print(f"Output will be saved to: {output_path}")
    
    # Construct the mlx_lm.fuse command
    command = [
        'mlx_lm.fuse',
        '--model', model_name,
        '--save-path', output_path,
        '--adapter-path', adapter_path
    ]
    
    print(f"Running fusion command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during fusion: {result.stderr}")
        sys.exit(1)
    else:
        print(f"Fusion completed successfully!")
        print(f"Fused model saved to: {output_path}")
    
    return output_path

def create_ollama_model(config, fused_model_path):
    """Create an Ollama model from a fused model."""
    print("\n" + "="*50)
    print("STEP 3: CREATING OLLAMA MODEL")
    print("="*50)
    
    # Check if fused model path exists
    fused_model_path = os.path.abspath(fused_model_path)
    if not os.path.exists(fused_model_path):
        print(f"Error: Fused model path '{fused_model_path}' does not exist.")
        sys.exit(1)
    
    # Extract deployment configuration
    deployment_config = config.get('deployment', {})
    
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

def main():
    parser = argparse.ArgumentParser(description='Train, fuse, and deploy a model to Ollama using a JSON configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('--skip_train', action='store_true', help='Skip the training step and use an existing adapter.')
    parser.add_argument('--skip_fuse', action='store_true', help='Skip the fusion step and use an existing fused model.')
    parser.add_argument('--adapter_path', type=str, help='Path to existing adapter weights (required if skip_train is True).')
    parser.add_argument('--fused_model_path', type=str, help='Path to existing fused model (required if skip_fuse is True).')
    
    args = parser.parse_args()
    
    # Load the configuration
    config = load_json_config(args.config)
    
    # Step 1: Train the model (unless skipped)
    adapter_path = args.adapter_path
    if not args.skip_train:
        adapter_path = train_model(config)
    else:
        if not adapter_path:
            print("Error: --adapter_path must be provided when --skip_train is used.")
            sys.exit(1)
        print(f"Skipping training, using existing adapter: {adapter_path}")
    
    # Step 2: Fuse the model (unless skipped)
    fused_model_path = args.fused_model_path
    if not args.skip_fuse:
        fused_model_path = fuse_model(config, adapter_path)
    else:
        if not fused_model_path:
            print("Error: --fused_model_path must be provided when --skip_fuse is used.")
            sys.exit(1)
        print(f"Skipping fusion, using existing fused model: {fused_model_path}")
    
    # Step 3: Create the Ollama model
    create_ollama_model(config, fused_model_path)
    
    print("\n" + "="*50)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == '__main__':
    main()
