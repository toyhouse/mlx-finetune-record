---
language: id
license: apache-2.0
datasets:
  - custom_indo_math_dataset
tags:
  - mathematics
  - education
  - gasing-method
  - bahasa-indonesia
  - mlx
library_name: mlx
pipeline_tag: text-generation
---

# Indo Math Teacher (Guru Matematika Indonesia)

A mathematics tutor model fine-tuned to teach mathematics using the Gasing method in Bahasa Indonesia.

## Model Description

Indo Math Teacher is a specialized model for teaching mathematics using the Gasing method - an approach that makes math learning easier, more practical, and more fun for students. The model is fine-tuned from Qwen2.5-Math-1.5B to provide high-quality mathematical reasoning in Bahasa Indonesia.

## Usage

The model can be used with Ollama or any GGUF-compatible inference framework.

### Installation with Ollama

```bash
curl -s https://huggingface.co/Lckoo1230/indo-math-teacher-complete/resolve/main/scripts/install.sh | bash
```

### Example

**User**: "Jelaskan cara menghitung 8 ditambah 5 menggunakan metode Gasing!"

**Model**: "Simpan 8 di kepala dan 5 di tangan. Pasangan 8 untuk mencapai 10 adalah 2. Ambil 2 dari 5 di tangan. Sekarang di kepala kita punya 10, dan di tangan tersisa 3. Jadi, 10 ditambah 3 sama dengan 13. Kesimpulannya, 8 ditambah 5 sama dengan 13."

## Training Methodology

The model was fine-tuned using MLX on a custom dataset of mathematical problems and explanations in Bahasa Indonesia, with a focus on the Gasing teaching methodology.

## Limitations

- The model specializes in mathematics education and may not perform well on general knowledge tasks
- Currently optimized for elementary and middle school mathematics
- Best results with clear, well-formatted questions
