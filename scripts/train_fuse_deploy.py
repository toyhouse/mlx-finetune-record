import os
import json
from datetime import datetime
import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import subprocess
import sys
import traceback
import yaml
import tempfile
import argparse
import subprocess
import sys
from datetime import datetime

def load_yaml_config(file_path):
    """Load YAML configuration from a file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def test_model_math_ability(model_name: str, output_path: str):
    """
    Test the math ability of a trained Ollama model with pure equations
    
    :param model_name: Name of the Ollama model to test
    :param output_path: Path to save the markdown report
    """
    print("\n" + "="*50)
    print("STEP 4: TESTING MODEL MATH ABILITY")
    print("="*50)
    
    # Load API key from environment
    load_dotenv()
    
    # Initialize Gemini with the provided API key
    try:
        gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    except Exception as gemini_init_error:
        print(f"Warning: Could not initialize Gemini API: {gemini_init_error}")
        gemini = None

    # Pure equation questions
    math_questions = [
        "7 + 9",
        "42 + 58",
        "23 + 45 + 32",
        "7 + 5"  # One-digit addition problem
    ]
    
    # Prepare markdown content
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"# Math Ability Test: {model_name}\n\n"
    markdown_content += f"**Test Timestamp:** {current_time}\n\n"
    markdown_content += "## Addition Problem Responses\n\n"
    
    # Verify Ollama model exists and is running
    try:
        import subprocess
        
        # Check if Ollama service is running
        print("Checking Ollama service...")
        ollama_ps = subprocess.run(['pgrep', 'ollama'], capture_output=True, text=True)
        if ollama_ps.returncode != 0:
            print("Ollama service is not running. Attempting to start...")
            subprocess.Popen(['ollama', 'serve'], start_new_session=True)
            import time
            time.sleep(5)  # Wait for service to start
        
        # List available models
        print("Checking available Ollama models...")
        models_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        print(models_result.stdout)
        
        # Verify specific model exists
        if model_name not in models_result.stdout:
            print(f"Error: Model '{model_name}' not found in Ollama models.")
            markdown_content += f"**ERROR:** Model '{model_name}' not found in Ollama models.\n"
            
            # Write markdown to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return
    except Exception as startup_error:
        print(f"Error checking Ollama service: {startup_error}")
        markdown_content += f"**ERROR:** Failed to check Ollama service: {startup_error}\n"
        
        # Write markdown to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return
    
    # Test each math question
    for idx, question in enumerate(math_questions, 1):
        try:
            print(f"Sending question {idx}: {question}")
            
            # Generate response from the model
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system', 
                        'content': 'Solve the math equation directly and precisely.'
                    },
                    {
                        'role': 'user', 
                        'content': question
                    }
                ]
            )
            
            # Extract model's response
            model_response = response['message']['content']
            print(f"Model response: {model_response}")
            
            # Evaluate response with Gemini if available
            if gemini:
                try:
                    evaluation_prompt = ChatPromptTemplate.from_template("""
                    Evaluate the mathematical solution to the problem: {question}
                    
                    Model's Solution: {model_response}
                    
                    Provide a detailed evaluation focusing on:
                    1. Correctness of the solution
                    2. Clarity of explanation
                    3. Mathematical reasoning
                    
                    Give a narrative assessment of the solution's strengths and areas for improvement.
                    """)
                    
                    evaluation_chain = evaluation_prompt | gemini
                    evaluation = evaluation_chain.invoke({
                        'question': question,
                        'model_response': model_response
                    })
                    
                    # Add to markdown
                    markdown_content += f"### Problem {idx}: {question}\n\n"
                    markdown_content += f"**Model's Solution:**\n```\n{model_response}\n```\n\n"
                    markdown_content += f"**Evaluation:**\n{evaluation.content}\n\n"
                except Exception as eval_error:
                    print(f"Warning: Gemini evaluation failed for question {question}: {eval_error}")
                    markdown_content += f"### Problem {idx}: {question}\n\n"
                    markdown_content += f"**Model's Solution:**\n```\n{model_response}\n```\n\n"
                    markdown_content += f"**Evaluation:** *Gemini evaluation unavailable*\n\n"
            else:
                # If Gemini is not available, just add the model's response
                markdown_content += f"### Problem {idx}: {question}\n\n"
                markdown_content += f"**Model's Solution:**\n```\n{model_response}\n```\n\n"
                markdown_content += f"**Evaluation:** *Gemini evaluation unavailable*\n\n"
        
        except Exception as e:
            error_msg = f"Error testing question {question}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            markdown_content += f"### Problem {idx}: Error\n\n"
            markdown_content += f"**Error:** {error_msg}\n\n"
    
    # Write markdown to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Math ability test report saved to {output_path}")

def train_model(model_config_path, data_config_path, training_config_path, adapter_path=None):
    """Train the model using mlx_lm.lora."""
    print("\n" + "="*50)
    print("STEP 1: TRAINING MODEL")
    print("="*50)
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tqdm'], check=True)
        from tqdm import tqdm
    
    # Load configurations from YAML files
    model_config = load_yaml_config(model_config_path)
    model_name = model_config.get('name', 'Unknown Model')
    data_config = load_yaml_config(data_config_path)
    training_data_name = data_config.get('name', 'unknown_data')
    training_config = load_yaml_config(training_config_path)
    
    # Generate timestamp for saving the adapter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_save_path = training_config.get('adapter_path', './adapters/{model_name}_{data_name}')
    adapter_path = adapter_save_path.format(model_name=model_name, data_name=training_data_name) + f"_{timestamp}"
    
    # Print information
    print(f'Model     : {model_name}')
    print(f'Data      : ./data/{training_data_name}')
    print(f'Training  : {training_config}')
    print(f'Adapter Save Path: {adapter_path}')
    
    # Execute the mlx_lm.lora command
    command = [
        'mlx_lm.lora',
        '--model', model_name,
        '--train',
        '--data', f'./data/{training_data_name}',
        '--learning-rate', str(training_config.get('learning_rate')),
        '--iters', str(training_config.get('iterations')),
        '--fine-tune-type', training_config.get('fine-tune-type'),
        '--adapter-path', adapter_path
    ]
    
    print(f"Running training command: {' '.join(command)}")
    
    # Use subprocess.Popen to capture output and create progress bar
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True)
    
    # Total iterations for progress bar
    total_iterations = training_config.get('iterations', 500)
    
    # Create progress bar
    with tqdm(total=total_iterations, desc="Training Progress", unit="iter", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        current_iteration = 0
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                # Check if the line indicates progress
                if 'iter' in output.lower():
                    try:
                        # Extract current iteration from the output
                        current_iteration += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error updating progress bar: {e}")
                
                # Print the output
                print(output.strip())
        
        # Check for any errors
        stderr_output = process.stderr.read()
        if stderr_output:
            print("Error output:", stderr_output)
    
    # Wait for the process to complete and get return code
    return_code = process.poll()
    
    if return_code != 0:
        print(f"Error during training: Return code {return_code}")
        sys.exit(1)
    else:
        print(f"Training completed successfully!")
        print(f"Adapter saved to: {adapter_path}")
    
    return adapter_path

def fuse_model(model_config_path, adapter_path, output_path):
    """Fuse the model with adapter weights."""
    print("\n" + "="*50)
    print("STEP 2: FUSING MODEL")
    print("="*50)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model configuration
    model_config = load_yaml_config(model_config_path)
    model_name = model_config.get('name', 'Unknown Model')
    
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

def create_ollama_model(fused_model_path, deployment_config_path):
    """Create an Ollama model from a fused model."""
    print("\n" + "="*50)
    print("STEP 3: CREATING OLLAMA MODEL")
    print("="*50)
    
    # Check if fused model path exists
    fused_model_path = os.path.abspath(fused_model_path)
    if not os.path.exists(fused_model_path):
        print(f"Error: Fused model path '{fused_model_path}' does not exist.")
        sys.exit(1)
    
    # Load deployment configuration
    deployment_config = load_yaml_config(deployment_config_path)
    
    # Verify that the platform is 'ollama'
    if deployment_config.get('platform', '').lower() != 'ollama':
        raise ValueError(f"Expected platform 'ollama', but got '{deployment_config.get('platform')}'")
    
    # Extract parameters from deployment config
    model_name = deployment_config.get('model_name')
    temperature = deployment_config.get('temperature', 0.7)
    top_p = deployment_config.get('top_p', 0.7)
    start_token = deployment_config.get('start', '')
    stop_token = deployment_config.get('stop', '')
    additional_stops = deployment_config.get('additional_stops', [])
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
        '{system_prompt}': system_prompt
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
    
    return model_name

def main():
    parser = argparse.ArgumentParser(description='Train, fuse, and deploy a model to Ollama.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--training_config', type=str, required=True, help='Path to the training configuration file.')
    parser.add_argument('--deployment_config', type=str, required=True, help='Path to the deployment configuration file.')
    parser.add_argument('--skip_train', action='store_true', help='Skip the training step and use an existing adapter.')
    parser.add_argument('--skip_fuse', action='store_true', help='Skip the fusion step and use an existing fused model.')
    parser.add_argument('--adapter_path', type=str, help='Path to existing adapter weights (required if skip_train is True).')
    parser.add_argument('--fused_model_path', type=str, help='Path to existing fused model (required if skip_fuse is True).')
    parser.add_argument('--output_path', type=str, help='Path to save the fused model. If not provided, a default path will be used.')
    parser.add_argument('--test_output', type=str, help='Optional path to save test results.')
    
    args = parser.parse_args()
    
    # Step 1: Train the model (unless skipped)
    adapter_path = args.adapter_path
    if not args.skip_train:
        adapter_path = train_model(args.model_config, args.data_config, args.training_config)
    else:
        if not adapter_path:
            print("Error: --adapter_path must be provided when --skip_train is used.")
            sys.exit(1)
        print(f"Skipping training, using existing adapter: {adapter_path}")
    
    # Step 2: Fuse the model (unless skipped)
    fused_model_path = args.fused_model_path
    if not args.skip_fuse:
        # Determine output path for fused model
        if args.output_path:
            output_path = args.output_path
        else:
            # Extract model and data names for default output path
            model_config = load_yaml_config(args.model_config)
            model_name = model_config.get('name', 'Unknown_Model').split('/')[-1]
            data_config = load_yaml_config(args.data_config)
            data_name = data_config.get('name', 'unknown_data')
            output_path = f"./fused_model/{model_name}_{data_name}"
        
        fused_model_path = fuse_model(args.model_config, adapter_path, output_path)
    else:
        if not fused_model_path:
            print("Error: --fused_model_path must be provided when --skip_fuse is used.")
            sys.exit(1)
        print(f"Skipping fusion, using existing fused model: {fused_model_path}")
    
    # Step 3: Create the Ollama model
    model_name = create_ollama_model(fused_model_path, args.deployment_config)
    
    # Step 4: Test the model (optional)
    if args.test_output:
        test_model_math_ability(
            model_name, 
            args.test_output
        )
    
    print("\n" + "="*50)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == '__main__':
    main()
