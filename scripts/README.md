# MLX Fine-tuning Scripts

This directory contains various scripts for MLX model creation, training, and testing.

For a sample train, fuse, deploy session:

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml
```
```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/Qwen_1.5B.yaml \
    --data_config configs/data_configs/videotranscript_config.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/indo_ollama_config.yml
```

Another example for using a different base model

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/nvidia_acemath.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/nvidia_instruct_deploy.yaml
```

Another example for using a different base model

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/phi4_mini.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/phi4-mini_lora.yaml \
    --deployment_config configs/deployment_configs/phi4_ollama.yaml
```



```bash
mlx_lm.lora --model nvidia/AceMath-7B-Instruct --train --data ./data/calculator --learning-rate 1e-4 --iters 20 --fine-tune-type full --adapter-path ./adapters/nvidia/AceMath-7B-Instruct_calculator_20250301_102505
```