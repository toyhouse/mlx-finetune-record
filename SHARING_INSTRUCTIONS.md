# Indo Math Teacher - Gasing Method

This is a fine-tuned model that teaches mathematics using the Gasing method in Bahasa Indonesia.

## How to Use This Model

### Option 1: Use the Modelfile (Easiest)

1. Install [Ollama](https://ollama.com/)
2. Save this Modelfile to your computer:

```
FROM Qwen/Qwen2.5-Math-1.5B
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{.System}}\n\n{{.Prompt}}"
STOP "[/INST]"
STOP ">>> "
STOP "\n\nHuman:"
STOP "\nHuman:"
```

3. Create the model:
```bash
ollama create indo_math_teacher -f Modelfile
```

4. Run the model:
```bash
ollama run indo_math_teacher
```

### Option 2: Fine-tune Your Own Copy (Advanced)

This model was fine-tuned using MLX and LoRA. The adapter files are available at:
https://huggingface.co/Lckoo1230/indo-math-teacher-gasing

To fine-tune your own copy:
1. Clone the repository: `git clone https://github.com/ml-explore/mlx-examples`
2. Follow the instructions in the `mlx-lm` folder for fine-tuning
3. Use the adapter files from Hugging Face

## Example Prompts

- "Jelaskan cara menghitung 7 ditambah 5 menggunakan metode Gasing!"
- "Bagaimana cara menghitung 8 ditambah 4 dengan metode Gasing?"
- "Jelaskan penjumlahan 9 ditambah 6 menggunakan metode Gasing!"

## About the Gasing Method

Gasing (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.

The method particularly excels at teaching basic arithmetic by using mental models like "keeping numbers in your head" and "taking numbers on your hand" to make addition and other operations intuitive.
