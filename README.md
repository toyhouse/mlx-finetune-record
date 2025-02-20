# MLX Fine-tuning Record

A project for fine-tuning and deploying MLX-based language models, with a focus on the Qwen model series. This project supports both full fine-tuning and LoRA (Low-Rank Adaptation) approaches, with tools for model creation, training, and deployment using Ollama.

## Project Structure

```
mlx-finetune-record/
├── adapters/           # Directory for storing trained model adapters
├── fused_model/        # Directory for fused models (base + adapters)
├── jsonl/             # Training and testing data in JSONL format
├── modelfiles/        # Ollama model configuration files
├── scripts/          # Training and utility scripts
│   ├── model_creation/    # Scripts for creating base models
│   └── training/          # Fine-tuning scripts
├── samples/          # Example inputs and outputs
└── verifiers/        # Verification tools and tests
```

## Features

- Support for Qwen2.5-Coder models (0.5B and 7B variants)
- Multiple fine-tuning approaches:
  - Full fine-tuning
  - LoRA (Low-Rank Adaptation)
- Training data management in JSONL format
- Model deployment using Ollama
- Utility scripts for model creation and testing

## Prerequisites

- MLX framework
- Ollama
- Python 3.x
- Required Python packages (specified in requirements.txt)

## Quick Start

1. **Prepare Your Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**
   - Place your training data in the `jsonl` directory
   - Format should follow the JSONL specification with text field containing:
     - System instruction
     - Human prompt
     - Assistant response

3. **Fine-tune the Model**
   
   For LoRA fine-tuning:
   ```bash
   mlx_lm.lora \
       --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
       --train \
       --data "./jsonl/your_data_directory" \
       --num-layers 4 \
       --learning-rate 1e-5 \
       --iters 100 \
       --fine-tune-type lora
   ```

   For full fine-tuning:
   ```bash
   mlx_lm.lora \
       --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
       --train \python
       --data "./jsonl/your_data_directory" \
       --learning-rate 1e-5 \
       --iters 100 \
       --fine-tune-type full
   ```

4. **Test the Model**
   ```bash
   python -m mlx_lm.generate \
       --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
       --max-tokens 500 \
       --adapter-path adapters \
       --prompt "Your test prompt here"
   ```

## Deployment with Ollama

1. **Fuse Model with Adapters**
   ```bash
   mlx_lm.fuse \
       --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
       --save-path ./fused_model/your_model_name/ \
       --adapter-path adapters/
   ```

2. **Create Ollama Model**
   ```bash
   ollama create your_model_name -f Modelfile
   ```

## Data Format

Training data should be in JSONL format with the following structure:
```json
{
    "text": "System instruction\n\nHuman: prompt\nAssistant: response\n\n"
}
```

## Scripts

- `scripts/model_creation/`: Scripts for creating base models
- `scripts/training/`: Fine-tuning scripts for different model sizes
- `merge_jsonl.py`: Utility for merging multiple JSONL files
- `cli.py`: Command-line interface for model operations
- `infer.py`: Inference script for testing models

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MLX framework by Apple
- Qwen model series
- Ollama project

## Qwen2.5-Coder-7B-Instruct

### Fine Tuning

#### Full Fine Tune
```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
```

### Lora Fine Tune

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
```

### Testing the Adapters

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "what's 254-7?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "who is ada lovelace?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "which number is bigger 3.9 or 3.11?"
```

### No Adapters

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "what's 254-7?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "who is ada lovelace?"
```

## Qwen2.5-Coder-7B-Instruct

### Fine Tuning

#### Full Fine Tune
```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
```

### Lora Fine Tune

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
```


mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --train \
    --data "./json/test" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
    

python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
                --max-tokens 500 \
               --adapter-path adapters \
               --prompt "What is 990 * 75 + 12?"


## Ollama runtime creation by first fusing based model with training data

### First test if the adapters directory has proper content
mlx_lm.generate --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
            --prompt "Hi" --adapter adapters/

### Then fuse the adapters with base model
mlx_lm.fuse --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
            --save-path ./fused_model/qwen2.5_coder_fused/ \
            --adapter-path adapters/

### Finally create the gguf file using the ollama command
ollama create qwen_math_0.5B -f Modelfile            