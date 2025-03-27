# Indo Math Teacher with Gasing Method

This model is a fine-tuned version of Qwen2.5-Math-1.5B specialized in teaching mathematics using the Gasing method in Bahasa Indonesia.

## Model Description

The Gasing method (Gampang, Asyik, dan Menyenangkan - Easy, Fun, and Enjoyable) is a teaching approach that simplifies complex mathematical concepts through concrete, easy-to-understand steps.

This model can:
- Explain basic arithmetic operations using the Gasing method
- Provide step-by-step solutions to math problems
- Teach math concepts in Bahasa Indonesia

## Installation and Usage

### Option 1: Using Ollama (Recommended)

1. Install [Ollama](https://ollama.com) on your device
2. Create a Modelfile:

```
FROM Qwen/Qwen2.5-Math-1.5B
SYSTEM "You are a math teacher using the Gasing method"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 200

# Set stop sequences
TEMPLATE "{{ .System }}\n\n{{ .Prompt }}"
STOP "[/INST]"
STOP ">>> "
STOP "\n\nHuman:"
STOP "\nHuman:"
```

3. Create the model in Ollama:
```bash
ollama create indo_math_teacher -f Modelfile
```

4. Run the model:
```bash
ollama run indo_math_teacher
```

### Option 2: One-Line Installation (if Ollama is installed)

```bash
curl -s https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/install.sh | bash
```

## Example Usage

**User**: "Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"

**Model**: "Simpan 8 di kepala dan 5 di tangan. Pasangan 8 untuk mencapai 10 adalah 2. Ambil 2 dari 5 di tangan. Sekarang di kepala kita punya 10, dan di tangan tersisa 3. Jadi, 10 ditambah 3 sama dengan 13. Kesimpulannya, 8 ditambah 5 sama dengan 13."

## Model Limitations

- This model is specifically trained for mathematical concepts using the Gasing method
- It works best with arithmetic problems and basic math concepts
- Responses are optimized for Indonesian language

## Troubleshooting

If you encounter incorrect answers (especially for complex multiplication or division), try:

1. Adjusting the temperature to 0.05 for more deterministic results
2. Being more specific in your prompt about the steps you want to see
3. For complex multiplication, explicitly ask for the correct method

## Training Dataset

The model was fine-tuned on a dataset of mathematical problem-solving explanations in Bahasa Indonesian, focusing on the Gasing method approach to arithmetic operations and other math concepts.

## License

This model is released under the Apache 2.0 license.
