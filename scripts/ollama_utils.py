"""
Utilities for creating and managing Ollama modelfiles.
"""
import argparse
import os
import subprocess
import tempfile
import yaml
import sys

def load_yaml_config(file_path):
    """Load YAML configuration from a file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_modelfile(model_path, system_prompt, template_path=None, parameters=None):
    """
    Generate Ollama Modelfile content.
    
    Args:
        model_path (str): Path to the model
        system_prompt (str): System prompt for the model
        template_path (str, optional): Path to a Modelfile template
        parameters (dict, optional): Additional parameters for the Modelfile
        
    Returns:
        str: The content of the Modelfile
    """
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        replacements = {
            '{fuse_model_path}': model_path,
            '{system_prompt}': system_prompt,
            '{SYSTEM_PROMPT}': system_prompt,  # For backward compatibility
        }
        
        # Add other parameters if provided
        if parameters:
            for key, value in parameters.items():
                replacements[f'{{{key}}}'] = str(value)
        
        # Replace all placeholders
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
            
        return content
    else:
        # Create a basic Modelfile if no template is provided
        content = f"FROM {model_path}\n\n"
        
        # Add parameters if provided
        if parameters:
            for key, value in parameters.items():
                content += f"PARAMETER {key} {value}\n"
            content += "\n"
        
        # Add system prompt
        content += f"SYSTEM {system_prompt}\n"
        
        return content

def create_ollama_model(model_name, modelfile_content, verbose=False):
    """
    Create an Ollama model using the provided Modelfile content.
    
    Args:
        model_name (str): Name for the Ollama model
        modelfile_content (str): Content of the Modelfile
        verbose (bool, optional): Whether to display verbose output
        
    Returns:
        bool: True if the model was created successfully, False otherwise
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        modelfile_path = os.path.join(tmpdir, 'Modelfile')
        
        # Write Modelfile content to the temporary file
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        if verbose:
            print(f"Generated Modelfile content:")
            print("-" * 40)
            print(modelfile_content)
            print("-" * 40)
        
        # Create the Ollama model
        command = ['ollama', 'create', model_name, '-f', modelfile_path]
        
        if verbose:
            print(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error creating Ollama model: {result.stderr}")
            return False
        else:
            if verbose:
                print(f"Ollama model '{model_name}' created successfully.")
            return True

def main():
    parser = argparse.ArgumentParser(description="Utilities for creating Ollama models.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model to use.')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the Ollama model.')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant.', help='System prompt for the model.')
    parser.add_argument('--template', type=str, help='Path to a Modelfile template.')
    parser.add_argument('--config', type=str, help='Path to a deployment configuration file.')
    parser.add_argument('--verbose', action='store_true', help='Display verbose output.')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config and os.path.exists(args.config):
        config = load_yaml_config(args.config)
        model_name = config.get('model_name', args.model_name)
        system_prompt = config.get('system_prompt', args.system_prompt)
        template_path = config.get('modelfile_template', args.template)
        
        # Extract parameters
        parameters = {}
        for key in ['temperature', 'top_p', 'start', 'stop']:
            if key in config:
                parameters[key] = config[key]
    else:
        model_name = args.model_name
        system_prompt = args.system_prompt
        template_path = args.template
        parameters = None
    
    # Generate Modelfile content
    modelfile_content = generate_modelfile(
        args.model_path, 
        system_prompt, 
        template_path,
        parameters
    )
    
    # Create the Ollama model
    success = create_ollama_model(model_name, modelfile_content, args.verbose)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
