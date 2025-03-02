from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Set the maximum number of tokens
MAX_TOKENS = 500

# Set the model path to the fused model
MODEL_PATH = "./fused_model/Phi-4-mini-instruct_calculator"

# Define the system prompt
SYSTEM_PROMPT = "You are a helpful calculator assistant. Always show your work."
USER_PROMPT = "Calculate 245 + 372"

def main():    
    # Load the model and tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    
    # Construct the message list with a system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    print(f"Prompt: {prompt}")
    
    # Create a sampler with the specified temperature
    sampler = make_sampler(temp=0.2)
    
    # Generate the response
    print("Generating response...")
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler
    )
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
