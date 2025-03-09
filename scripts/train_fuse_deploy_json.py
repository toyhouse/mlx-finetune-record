import argparse
import subprocess
import json
import os
import tempfile
import sys
from datetime import datetime
import multiprocessing

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
        return None
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
        return None
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
        return False
    
    # Extract deployment configuration
    deployment_config = config.get('deployment', {})
    
    # Verify that the platform is 'ollama'
    if deployment_config.get('platform', '').lower() != 'ollama':
        print(f"Skipping Ollama deployment for non-Ollama platform")
        return False
    
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
            return False
        else:
            print(f"Stdout: {result.stdout}")
            print(f"Ollama model '{model_name}' created successfully!")
            return True

def process_single_config(config, skip_train=False, skip_fuse=False, adapter_path=None, fused_model_path=None):
    """Process a single configuration file."""
    # Step 1: Train the model (unless skipped)
    if not skip_train:
        adapter_path = train_model(config)
        if adapter_path is None:
            print(f"Training failed for configuration: {config.get('model', {}).get('name', 'Unknown')}")
            return False
    else:
        if not adapter_path:
            print("Error: --adapter_path must be provided when --skip_train is used.")
            return False
    
    # Step 2: Fuse the model (unless skipped)
    if not skip_fuse:
        fused_model_path = fuse_model(config, adapter_path)
        if fused_model_path is None:
            print(f"Fusion failed for configuration: {config.get('model', {}).get('name', 'Unknown')}")
            return False
    else:
        if not fused_model_path:
            print("Error: --fused_model_path must be provided when --skip_fuse is used.")
            return False
    
    # Step 3: Create the Ollama model
    create_ollama_model(config, fused_model_path)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train, fuse, and deploy multiple models using JSON configurations.')
    parser.add_argument('--configs', type=str, nargs='+', required=True, help='Paths to JSON configuration files.')
    parser.add_argument('--skip_train', action='store_true', help='Skip the training step and use existing adapters.')
    parser.add_argument('--skip_fuse', action='store_true', help='Skip the fusion step and use existing fused models.')
    parser.add_argument('--parallel', action='store_true', help='Run training in parallel.')
    
    args = parser.parse_args()
    
    # Load configurations
    configs = [load_json_config(config_path) for config_path in args.configs]
    
    # Determine whether to run in parallel or sequentially
    if args.parallel:
        # Use multiprocessing to train models in parallel
        with multiprocessing.Pool() as pool:
            # Prepare arguments for each configuration
            pool_args = [(config, args.skip_train, args.skip_fuse, None, None) for config in configs]
            
            # Run training in parallel
            results = pool.starmap(process_single_config, pool_args)
        
        # Check if all configurations were processed successfully
        if not all(results):
            print("Some configurations failed to process.")
            sys.exit(1)
    else:
        # Process configurations sequentially
        for config in configs:
            success = process_single_config(config, args.skip_train, args.skip_fuse)
            if not success:
                print(f"Failed to process configuration: {config.get('model', {}).get('name', 'Unknown')}")
                sys.exit(1)
    
    print("\n" + "="*50)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == '__main__':
    main()
