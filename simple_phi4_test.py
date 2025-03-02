from mlx_lm import load, generate

# Load the model and tokenizer
model_path = "./fused_model/Phi-4-mini-instruct_calculator"
model, tokenizer = load(model_path)

# Simple prompt
prompt = "[INST] Calculate 245 + 372 [/INST]"

# Generate response
response = generate(model, tokenizer, prompt=prompt)

print(response)
