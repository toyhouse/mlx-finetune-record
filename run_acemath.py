import os
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

class AceMathModel:
    def __init__(self, model_path: str):
        """Initialize the AceMath model.
        
        Args:
            model_path: Path to the model on Hugging Face or local directory
        """
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load(model_path)
        print("Model loaded successfully!")
        
        # Model configuration
        self.max_length = 512

    def generate(self, prompt: str, max_tokens=100, temperature=0.7):
        """Generate a response to a math problem. 
        
        Args:
            prompt: The math problem to solve
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated solution text
        """
        try:
            # Format prompt for instruction model
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            print("\nThinking...", end="", flush=True)
            
            # Create a sampler with the specified temperature
            sampler = make_sampler(temp=temperature)
            
            # Generate response using mlx_lm
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler
            )
            
            return response
            
        except Exception as e:
            return f"Error generating solution: {str(e)}"

def main():
    # Change this to your local model path if needed
    model_path = "nvidia/AceMath-7B-Instruct"
    
    try:
        model = AceMathModel(model_path)
        
        print("\nMath Assistant (type 'exit' to quit)")
        while True:
            try:
                problem = input("\nQuestion: ")
                if problem.lower() in ['exit', 'quit']:
                    break
                    
                solution = model.generate(problem)
                print(f"\rSolution: {solution}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        print("Please ensure you have the model files available locally or check your internet connection.")

if __name__ == "__main__":
    main()