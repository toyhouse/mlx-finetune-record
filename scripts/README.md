# MLX Fine-tuning Scripts

This directory contains various scripts for MLX model creation, training, and testing.

## Directory Structure

### model_creation/
Scripts for creating and initializing different models:
- create_deepseek_for_gasing.sh: Creates DeepSeek model for GASING
- create_faster_mlx.sh: Creates faster MLX model
- create_ollama_*.sh: Creates Ollama-based models

### training/
Scripts for model training and fine-tuning:
- faster_training.sh: Fast training implementation
- lora_training.sh: LoRA-based training scripts
- mlx_training_DeepSeek-Qwen.sh: Training script for DeepSeek-Qwen
- train_GASING.sh: GASING-specific training

### conversion/
Scripts for model conversion and fusion:
- convert_training_data.py: Converts training data to required format
- fuse_*.sh: Various model fusion scripts

### testing/
Scripts for testing and running models:
- run_gasing_mlx.sh: Runs GASING MLX model
- test_gasing_mlx.sh: Tests GASING MLX model
- fact_based_mlx.sh: Fact-based testing script
