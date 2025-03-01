# MLX Fine-tuning Pipeline

A comprehensive, configurable pipeline for fine-tuning and deploying Apple MLX-based language models. This project provides a unified workflow to train, adapt, and deploy various language models with minimal configuration changes.

## 🌟 Features

- **Single Configurable Pipeline**: Use one unified process for model training, fine-tuning, and deployment
- **Multi-Model Support**: Works with various MLX-compatible models (Qwen, Llama, etc.)
- **Flexible Training Options**:
  - Full fine-tuning for complete model adaptation
  - LoRA (Low-Rank Adaptation) for efficient parameter-efficient tuning
- **Modular Configuration System**: Change models, datasets, and training parameters without code modifications
- **Ollama Integration**: Streamlined deployment to Ollama for local inference
- **EXO Parallel Processing**: Leverage EXO for efficient parallel processing of training tasks

## 📋 Project Structure

```
mlx-finetune-record/
├── configs/                    # Configuration files for different models and tasks
│   ├── model_configs/          # Model-specific configurations
│   │   ├── Qwen_0.5B.yaml      # Configuration for Qwen 0.5B
│   │   ├── Qwen_7B.yaml        # Configuration for Qwen 7B
│   │   └── ...
│   ├── data_configs/           # Data-specific configurations
│   │   ├── calculator.yaml     # Configuration for calculator dataset
│   │   ├── code.yaml           # Configuration for code dataset
│   │   └── ...
│   ├── training_configs/       # Training configurations
│   │   ├── lora_config.yaml    # LoRA training settings
│   │   ├── full_config.yaml    # Full fine-tuning settings
│   │   ├── exo_parallel.yaml   # EXO parallel processing configuration
│   │   └── ...
│   ├── deployment_configs/     # Deployment configurations
│   │   ├── ollama_config.yaml  # Ollama deployment settings
│   │   └── ...
│   └── benchmark_configs/      # Benchmark configurations
│       ├── accuracy.yaml       # Accuracy benchmark settings
│       └── ...
├── adapters/                   # Trained model adapters (LoRA weights)
├── data/                       # Structured data for various tasks
├── fused_model/                # Fused models (base model + adapters)
├── modelfiles/                 # Ollama model configuration files
├── pretrained_models/          # Storage for downloaded model weights
├── benchmarks/                 # Benchmark data and results
│   ├── datasets/               # Test datasets for benchmarking
│   ├── results/                # Benchmark results in JSON/CSV format
│   ├── visualizations/         # Generated charts and graphs
│   └── reports/                # HTML/PDF benchmark reports
├── scripts/                    # Core scripts for training, evaluation, and deployment
│   ├── train.py                # Main training script (configurable)
│   ├── eval.py                 # Evaluation script
│   ├── fuse.py                 # Script to fuse adapters with the base model
│   ├── ollama_utils.py         # Utilities for creating Ollama modelfiles
│   ├── benchmark.py            # Benchmarking script
│   ├── monitor_exo.py          # Script to monitor EXO performance
│   ├── create_ollama_model.py  # Script to create Ollama models from fused models
│   ├── train_fuse_deploy.py    # All-in-one script for training, fusing, and deployment
│   └── ...
├── utils/                      # Utility functions and modules
│   ├── config_loader.py        # Module to load configurations
│   ├── data_loader.py          # Module to load and preprocess data
│   └── ...
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- MacOS with Apple Silicon (M1/M2/M3/M4) for MLX acceleration
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

## ⚙️ Configuration

This project uses YAML files for all configuration aspects. Here's how to set them up:

### 1. Model Configuration

Create or modify a file in `configs/model_configs/`:

```yaml
# configs/model_configs/Qwen_0.5B.yaml
name: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
architecture: "transformer"
tokenizer: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
max_sequence_length: 2048
```

### 2. Data Configuration

Create or modify a file in `configs/data_configs/`:

```yaml
# configs/data_configs/calculator.yaml
name: "calculator"
train_data: "./data/calculator/train.jsonl"
validation_data: "./data/calculator/validation.jsonl"
format: "jsonl"
text_field: "text"
format_template: "System instruction\n\nHuman: {prompt}\nAssistant: {response}\n\n"
```

### 3. Training Configuration

Create or modify a file in `configs/training_configs/`:

```yaml
# configs/training_configs/lora_config.yaml
name: "lora_default"
type: "lora"
num_layers: 4
learning_rate: 1e-5
batch_size: 8
iterations: 100
adapter_save_path: "./adapters/{model_name}_{data_name}"
```

### 4. Deployment Configuration

Create or modify a file in `configs/deployment_configs/`:

```yaml
# configs/deployment_configs/ollama_config.yaml
name: "qwen_calculator"
platform: "ollama"
model_name: "qwen_calc"
system_prompt: "You are a helpful calculator assistant."
fused_model_path: "./fused_model/{model_name}_{data_name}"
modelfile_template: "./templates/ollama_modelfile.template"
```

## ⚡ Parallel Processing with EXO

The pipeline integrates with EXO for efficient parallel processing of training tasks. Here's how to configure it:

### EXO Configuration Example
```yaml
# configs/training_configs/exo_parallel.yaml
parallel:
  framework: "exo"
  config:
    num_workers: 4
    worker_allocation:
      - devices: ["GPU0", "GPU1"]
        memory: 16GB
      - devices: ["GPU2", "GPU3"] 
        memory: 16GB
    communication:
      protocol: "nccl"
      compression: "fp16"
    batch_processing:
      strategy: "dynamic_batching"
      max_batch_size: 64
    memory_management:
      cache_size: 8GB
      prefetch: true
```

### Key EXO Configuration Parameters
| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `num_workers` | Number of parallel workers | 2-4 per GPU |
| `worker_allocation` | Hardware resource allocation | Match GPU count |
| `communication.protocol` | Inter-worker communication | NCCL for NVIDIA, MPS for Apple |
| `batch_processing.strategy` | Parallel batch handling | dynamic_batching |

### Enabling EXO Parallelism
Add to your training config:
```yaml
training:
  use_exo: true
  exo_config: "configs/training_configs/exo_parallel.yaml"
```

### EXO-Specific Runtime Options
```bash
python scripts/train.py \
    --exo-backend "metal" \
    --exo-batch-split "auto" \
    --exo-memory-optimization "aggressive"
```

### Monitoring EXO Performance
```bash
python scripts/monitor_exo.py \
    --interval 1 \
    --output-format "dashboard"
```

## 🔄 Training Pipeline

### Running the Full Pipeline

The entire process can be run with a single command:

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml
```

This will:
1. Load the specified model
2. Process the data according to the data configuration
3. Train the model using the specified training approach (LoRA or full fine-tuning)
4. Save the adapter weights
5. Fuse the adapter with the base model
6. Create an Ollama modelfile for deployment

### Running Individual Steps

Each step can also be run separately:

#### Training Only

```bash
python scripts/train.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml
```

#### Fusing Model with Adapters

```bash
python scripts/fuse.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --adapter_path ./adapters/Qwen2.5-Coder-0.5B-Instruct_calculator \
    --output_path ./fused_model/Qwen2.5-Coder-0.5B-Instruct_calculator
```

#### Creating Ollama Model

```bash
python scripts/create_ollama_model.py \
    --fused_model_path ./fused_model/Qwen2.5-Coder-0.5B-Instruct_calculator \
    --deployment_config configs/deployment_configs/ollama_config.yaml
```

## 📊 Evaluation

Evaluate the model performance with the evaluation script:

```bash
python scripts/eval.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --adapter_path ./adapters/Qwen2.5-Coder-0.5B-Instruct_calculator \
    --test_data ./data/calculator/test.jsonl
```

## 📈 Benchmarking

The project includes a comprehensive benchmarking system to evaluate and compare model performance across different metrics and configurations.

### Benchmark Configuration

Create a benchmark configuration file in `configs/benchmark_configs/`:

```yaml
# configs/benchmark_configs/accuracy.yaml
name: "accuracy_benchmark"
metrics:
  - "exact_match"
  - "token_accuracy"
  - "semantic_similarity"
test_datasets:
  - path: "./benchmarks/datasets/calculator_test.jsonl"
    name: "calculator_basic"
  - path: "./benchmarks/datasets/calculator_complex.jsonl"
    name: "calculator_complex"
output:
  format: "json"
  save_path: "./benchmarks/results/{model_name}_{benchmark_name}_{timestamp}.json"
visualize: true
visualization_config:
  type: "bar_chart"
  save_path: "./benchmarks/visualizations/{model_name}_{benchmark_name}_{timestamp}.png"
```

### Running Benchmarks

Run benchmarks on your trained model using:

```bash
python scripts/benchmark.py \
    --model_config configs/model_configs/Qwen_0.5B.yaml \
    --adapter_path ./adapters/Qwen2.5-Coder-0.5B-Instruct_calculator \
    --benchmark_config configs/benchmark_configs/accuracy.yaml
```

For comparative benchmarking across multiple models:

```bash
python scripts/benchmark.py \
    --benchmark_config configs/benchmark_configs/accuracy.yaml \
    --models_list configs/benchmark_configs/models_to_compare.yaml
```

### Benchmark Results Storage

Benchmark results are stored in structured formats (JSON/CSV) in the `benchmarks/results/` directory with a naming convention that includes the model name, benchmark type, and timestamp.

Sample results format:

```json
{
  "model": "Qwen2.5-Coder-0.5B-Instruct_calculator",
  "timestamp": "2025-02-27T08:30:45",
  "benchmark": "accuracy_benchmark",
  "metrics": {
    "exact_match": 0.85,
    "token_accuracy": 0.92,
    "semantic_similarity": 0.88
  },
  "datasets": {
    "calculator_basic": {
      "exact_match": 0.92,
      "token_accuracy": 0.95,
      "semantic_similarity": 0.91
    },
    "calculator_complex": {
      "exact_match": 0.78,
      "token_accuracy": 0.89,
      "semantic_similarity": 0.85
    }
  },
  "config": {
    "model_config": "configs/model_configs/Qwen_0.5B.yaml",
    "benchmark_config": "configs/benchmark_configs/accuracy.yaml"
  },
  "runtime": {
    "total_time": 325.6,
    "average_inference_time": 0.325
  }
}
```

### Visualization and Reporting

The benchmarking system automatically generates visualizations for easy comparison:

```bash
python scripts/generate_report.py \
    --results_dir ./benchmarks/results \
    --filter "model=Qwen*,benchmark=accuracy*" \
    --output_format "html" \
    --output_path ./benchmarks/reports/accuracy_comparison.html
```

Sample reports include:
- Performance comparisons across models
- Radar charts for multi-metric evaluation
- Before/after visualizations for fine-tuning improvements
- Time series of performance across training iterations

### Performance Tracking Dashboard

For continuous monitoring of model improvements, use the included dashboard:

```bash
python scripts/run_dashboard.py --port 8050
```

This launches a web interface at http://localhost:8050 that displays:
- Historical performance of models
- Comparison between different fine-tuning approaches
- Performance across different test datasets
- Resource utilization metrics (memory, inference time)

## 🔍 Example Configuration Files

### Model Config - Qwen2.5-Coder-0.5B

```yaml
name: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
architecture: "transformer"
tokenizer: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
max_sequence_length: 2048
```

### Data Config - Calculator

```yaml
name: "calculator"
train_data: "./data/calculator/train.jsonl"
validation_data: "./data/calculator/validation.jsonl"
format: "jsonl"
text_field: "text"
format_template: "System instruction\n\nHuman: {prompt}\nAssistant: {response}\n\n"
```

### Training Config - LoRA

```yaml
name: "lora_default"
type: "lora"
num_layers: 4
learning_rate: 1e-5
batch_size: 8
iterations: 100
adapter_save_path: "./adapters/{model_name}_{data_name}"
lora_rank: 8
lora_alpha: 16
```

### Deployment Config - Ollama

```yaml
name: "qwen_calculator"
platform: "ollama"
model_name: "qwen_calc"
system_prompt: "You are a helpful calculator assistant."
modelfile_template: "./templates/ollama_modelfile.template"
```

## 📦 Data Format

Training data should be in JSONL format with the following structure:

```json
{
    "text": "System instruction\n\nHuman: What is 2+2?\nAssistant: 2+2 equals 4.\n\n"
}
```

## 🧪 Customizing the Pipeline

### Adding New Models

1. Create a new model configuration file in `configs/model_configs/`
2. Ensure the model is compatible with MLX
3. Use the model in the training pipeline

### Adding New Datasets

1. Format your data in JSONL format
2. Create a new data configuration file in `configs/data_configs/`
3. Use the data in the training pipeline

### Customizing Training

1. Modify or create a new training configuration file in `configs/training_configs/`
2. Adjust parameters like learning rate, iterations, etc.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Apple MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-LM Library](https://github.com/ml-explore/mlx-examples)
- [Qwen model series](https://huggingface.co/Qwen)
- [Ollama project](https://github.com/ollama/ollama)

## 🔄 Complete End-to-End Workflow

The `train_fuse_deploy.py` script provides a comprehensive all-in-one solution for the entire model fine-tuning and deployment workflow, allowing you to train, fuse, and deploy models in a single command.

### Features of the Combined Script

1. **Modular Design**: The script is divided into three main functions corresponding to each step: training, fusing, and creating an Ollama model.
2. **Flexible Execution**: You can run all steps sequentially or skip specific steps with the `--skip_train` and `--skip_fuse` flags.
3. **Improved Error Handling**: Each step checks for errors and provides informative output.
4. **Clear Progress Indicators**: The script includes section headers that make it clear which step is currently executing.
5. **Automated Path Management**: The script generates reasonable default paths for adapter weights and fused models based on the configuration files.

### Basic Usage

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/qwen2_config.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml
```

or 


```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/phi4_mm.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/phi4_ollama.yaml
```

### Optional Arguments

- `--skip_train`: Skip the training step and use an existing adapter (requires `--adapter_path`)
- `--skip_fuse`: Skip the fusion step and use an existing fused model (requires `--fused_model_path`)
- `--adapter_path`: Path to existing adapter weights (required if skipping training)
- `--fused_model_path`: Path to existing fused model (required if skipping fusion)
- `--output_path`: Custom path to save the fused model

### Example Workflows

#### Skip Training Step (using existing adapter)

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/qwen2_config.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml \
    --skip_train \
    --adapter_path ./adapters/Qwen2.5-Coder-0.5B-Instruct_calculator
```

#### Skip Training and Fusion (just create Ollama model)

```bash
python scripts/train_fuse_deploy.py \
    --model_config configs/model_configs/qwen2_config.yaml \
    --data_config configs/data_configs/calculator.yaml \
    --training_config configs/training_configs/lora_config.yaml \
    --deployment_config configs/deployment_configs/ollama_config.yaml \
    --skip_train \
    --skip_fuse \
    --adapter_path ./adapters/Qwen2.5-Coder-0.5B-Instruct_calculator_20250227_123456 \
    --fused_model_path ./fused_model/Qwen2.5-Coder-0.5B-Instruct_calculator
```

This script streamlines your workflow by combining all three steps into a single command while maintaining the flexibility to run only the steps you need.
