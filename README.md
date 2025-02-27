# MLX Fine-tuning Record

A comprehensive toolkit for fine-tuning and deploying Apple MLX-based language models, with a primary focus on the Qwen model series. This project supports both full fine-tuning and LoRA (Low-Rank Adaptation) approaches, providing a complete pipeline from data preparation to model deployment.

## Features

- **Multiple Model Support**: Works with various models including Qwen2.5-Coder (0.5B and 7B variants)
- **Flexible Fine-tuning Approaches**:
  - Full fine-tuning for complete model adaptation
  - LoRA (Low-Rank Adaptation) for efficient parameter-efficient tuning
- **Complete Pipeline**:
  - Data preparation and management
  - Model training with customizable parameters
  - Adapter management for LoRA models
  - Model fusion for deployment
  - Integration with Ollama for local deployment
- **Utilities**:
  - Command-line interface for easy model interaction
  - JSONL dataset manipulation tools
  - Comprehensive testing scripts

## Project Structure

```
mlx-finetune-record/
├── adapters/           # Trained model adapters (LoRA weights)
├── cli.py              # Command-line interface for model interaction
├── data/               # Structured data for various tasks
├── fused_model/        # Fused models (base model + adapters)
├── infer.py            # Inference script for model testing
├── jsonl/              # Training and validation datasets in JSONL format
├── logs/               # Training and inference logs
├── merge_jsonl.py      # Utility for merging and processing JSONL files
├── modelfiles/         # Ollama model configuration files
├── models/             # Storage for downloaded model weights
├── requirements.txt    # Python dependencies
├── samples/            # Example inputs and outputs for verification
├── scripts/            # Utility scripts for various operations
│   ├── conversion/     # Scripts for model format conversion
│   ├── model_creation/ # Scripts for creating and preparing base models
│   ├── testing/        # Model evaluation scripts
│   └── training/       # Fine-tuning scripts for different models and approaches
└── verifiers/          # Tools to verify model performance
```

## Getting Started

### Prerequisites

- MacOS with Apple Silicon (M1/M2/M3) for MLX acceleration
- Python 3.9+
- Ollama (for model deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mlx-finetune-record.git
   cd mlx-finetune-record
   ```

2. **Set up the environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Data Preparation

Place your training data in the `jsonl` directory with the following format:
```json
{
    "text": "System instruction\n\nHuman: prompt\nAssistant: response\n\n"
}
```

Use the `merge_jsonl.py` utility to combine or process multiple data files:
```bash
python merge_jsonl.py --input-dir ./data/raw --output-file ./jsonl/combined.jsonl
```

## Fine-tuning Models

### LoRA Fine-tuning (Recommended for efficiency)

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/your_dataset" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
```

### Full Fine-tuning (For maximum adaptation)

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/your_dataset" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
```

## Model Testing

### Testing with LoRA Adapters

```bash
python -m mlx_lm.generate \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --max-tokens 500 \
    --adapter-path adapters \
    --prompt "Your test prompt here"
```

### Testing Base Model (No Adapters)

```bash
python -m mlx_lm.generate \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --max-tokens 500 \
    --prompt "Your test prompt here"
```

### Using the CLI Interface

```bash
python cli.py "Your prompt here" --model "model_name"
```

## Deployment with Ollama

### 1. Fuse Model with Adapters

```bash
mlx_lm.fuse \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --save-path ./fused_model/your_model_name/ \
    --adapter-path adapters/
```

### 2. Create and Use Ollama Model

```bash
ollama create your_model_name -f modelfiles/your_modelfile
ollama run your_model_name
```

## Advanced Usage

### Pipeline Scripts

The `scripts` directory contains several ready-to-use pipelines:

- `qwen_pipeline.sh`: Complete pipeline for Qwen models
- `s1k_pipeline.sh`: Pipeline optimized for S1K datasets
- `Alternative_pipeline.sh`: Alternative approach with different parameters

Example:
```bash
bash scripts/qwen_pipeline.sh
```

### Model Conversion

Convert between different model formats:

```bash
bash scripts/conversion/convert_to_gguf.sh
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Apple MLX Framework](https://github.com/ml-explore/mlx)
- [Qwen model series](https://huggingface.co/Qwen)
- [Ollama project](https://github.com/ollama/ollama)
- [MLX-LM Library](https://github.com/ml-explore/mlx-examples)