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

def run_train_fuse_deploy(
    model_config_path='./configs/model_config.yaml', 
    data_config_path='./configs/data_config.yaml', 
    training_config_path='./configs/training_config.yaml', 
    deployment_config_path='./configs/deployment_config.yaml'
):
    """
    Run the training, fusion, and deployment process using train_fuse_deploy.py
    
    :param model_config_path: Path to model configuration YAML
    :param data_config_path: Path to data configuration YAML
    :param training_config_path: Path to training configuration YAML
    :param deployment_config_path: Path to deployment configuration YAML
    :return: Deployed model name
    """
    print("\n" + "="*50)
    print("PREPARING MODEL FOR TESTING")
    print("="*50)
    
    # Construct the command to run train_fuse_deploy.py
    command = [
        sys.executable,  # Use the current Python interpreter
        './scripts/train_fuse_deploy.py',
        '--model_config', os.path.abspath(model_config_path),
        '--data_config', os.path.abspath(data_config_path),
        '--training_config', os.path.abspath(training_config_path),
        '--deployment_config', os.path.abspath(deployment_config_path)
    ]
    
    # Detailed logging of paths and files
    print("Checking configuration files:")
    for path in [model_config_path, data_config_path, training_config_path, deployment_config_path]:
        abs_path = os.path.abspath(path)
        print(f"- {path}: {'EXISTS' if os.path.exists(abs_path) else 'MISSING'}")
        if os.path.exists(abs_path):
            print(f"  Full path: {abs_path}")
    
    # Check data directory
    data_dir = './data/math_addition'
    print(f"\nChecking data directory: {os.path.abspath(data_dir)}")
    if os.path.exists(data_dir):
        print("Contents:")
        for item in os.listdir(data_dir):
            print(f"- {item}")
    else:
        print("Data directory does not exist!")
    
    # Run the training and deployment process
    print("\nRunning training and deployment process...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print full output for debugging
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    
    # Check if the process was successful
    if result.returncode != 0:
        print("Error in training/deployment process:")
        print(result.stderr)
        sys.exit(1)
    
    # Load deployment config to get the model name
    import yaml
    with open(deployment_config_path, 'r') as file:
        deployment_config = yaml.safe_load(file)
    
    model_name = deployment_config.get('model_name', 'small_english_teacher:latest')
    
    print(f"\nModel '{model_name}' is now ready for testing!")
    return model_name

def test_model_math_ability(model_name: str, output_path: str):
    """
    Test the math ability of a trained Ollama model with pure equations
    
    :param model_name: Name of the Ollama model to test
    :param output_path: Path to save the markdown report
    """
    # Load API key from environment
    load_dotenv()
    
    # Initialize Gemini with the provided API key
    gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

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
    
    # Test each math question
    for idx, question in enumerate(math_questions, 1):
        try:
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
            
            # Evaluate response with Gemini
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
        
        except Exception as e:
            markdown_content += f"### Problem {idx}: Error\n\n"
            markdown_content += f"**Error:** {str(e)}\n\n"
            traceback.print_exc()
    
    # Write markdown to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Math ability test report saved to {output_path}")

def main():
    # First, run the training and deployment process
    model_name = run_train_fuse_deploy()
    
    # Then test the model
    test_model_math_ability(
        model_name, 
        '/Users/Henrykoo/Documents/mlx-finetune-record/results/model_math_ability.md'
    )

if __name__ == '__main__':
    main()
