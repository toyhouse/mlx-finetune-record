import os
import json
from datetime import datetime
import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

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
        "7 + 5"  
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
    
    # Write markdown to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Math ability test report saved to {output_path}")

def main():
    test_model_math_ability(
        'small_english_teacher:latest', 
        '/Users/Henrykoo/Documents/mlx-finetune-record/results/model_math_ability.md'
    )

if __name__ == '__main__':
    main()
