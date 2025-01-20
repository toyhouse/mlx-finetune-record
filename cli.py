import argparse
from mlx_lm import load, generate

# Set the maximum number of tokens
MAX_TOKENS = 500

# Define the system prompt
SYSTEM_PROMPT = (
    "You are a very silly assistant that wanders off topic nearly every sentence, "
    "related to the previous sentence. You do give long rambly answers"
)

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Provide a user prompt for the language model.")
    
    # Command-line arguments
    parser.add_argument("prompt", nargs="*", help="The user prompt to pass to the model.")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        help="The model name to load.")

    # Parse arguments
    args = parser.parse_args()

    # Combine all prompt parts into one string, or use a default if none provided
    if args.prompt:
        user_prompt = " ".join(args.prompt)
    else:
        user_prompt = "Who is Ada Lovelace?"
    
    return user_prompt, args.model

def main():
    # Parse arguments
    user_prompt, model_name = parse_arguments()
    
    # Load the model and tokenizer
    model, tokenizer = load(model_name)
    
    # Construct the messages with a system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate the response
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        verbose=True
    )
    
    # Print the response
    print(response)

if __name__ == "__main__":
    main()
