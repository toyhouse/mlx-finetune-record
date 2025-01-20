from mlx_lm import load, generate

# Set the maximum number of tokens
MAX_TOKENS = 500

# Set the model name
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Define the system prompt
SYSTEM_PROMPT = "You are a knowledgeable assistant. Please respond concisely."
USER_PROMPT = "Who is Ada Lovelace?"

def main():    
    # Load the model and tokenizer
    model, tokenizer = load(MODEL_NAME)
    
    # Construct the message list with a system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Generate the response
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=MAX_TOKENS, 
        verbose=True
    )
    
    # Print the response
    print(response)

if __name__ == "__main__":
    main()
