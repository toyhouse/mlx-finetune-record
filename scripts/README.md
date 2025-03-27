# MLX Fine-tuning Scripts

This directory contains various scripts for MLX model creation, training, and testing.

## Training Commands

### Qwen 1.5B Model Training
```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/Qwen_1.5B.yaml \
    --data_config configs/data_configs/videotranscript_config.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/indo_ollama_config.yml \
    --test_output ./results/qwen_1.5B_test_results.md
```

### Qwen 0.5B Model Training
```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --data_config configs/data_configs/videotranscript_config.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml \
    --test_output ./results/qwen_0.5B_test_results.md
```

### NVIDIA ACE Math Model Training
```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/nvidia_acemath.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/nvidia_instruct_deploy.yaml \
    --test_output ./results/nvidia_acemath_test_results.md
```

### Phi4 Mini Model Training
```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/phi4_mini.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/phi4-mini_lora.yaml \
    --deployment_config configs/deployment_configs/phi4_ollama.yaml \
    --test_output ./results/phi4_mini_test_results.md
```

python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/phi4_mini.yaml \
    --data_config configs/data_configs/videotranscript_config.yaml \
    --training_config configs/training_configs/phi4-mini_lora.yaml \
    --deployment_config configs/deployment_configs/phi4_ollama.yaml \
    --test_output ./results/phi4_mini_test_results.md

### Qwen 2.5 1.5B Model Training
```bash
     python3 scripts/train_fuse_deploy.py \
        --model_config configs/model_configs/Qwen2.5_1.5B.yaml \
        --data_config configs/data_configs/indo_tutor_config.yaml \
        --training_config configs/training_configs/lora_config.yaml \
        --deployment_config configs/deployment_configs/indo_ollama_config.yml \
        --test_output ./results/qwen2.5_1.5B_test_results.md\
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/indo_ollama_config.yml \
    --test_output ./results/qwen2.5_1.5B_test_results.md
```

### Optional Testing
To include model testing after training, add the `--test_output` parameter:
```bash
python scripts/train_fuse_deploy.py \
    ... \
    --test_output ./results/model_test_results.md
```

```bash
mlx_lm.lora --model nvidia/AceMath-7B-Instruct --train --data ./data/calculator --learning-rate 1e-4 --iters 20 --fine-tune-type full --adapter-path ./adapters/nvidia/AceMath-7B-Instruct_calculator_20250301_102505