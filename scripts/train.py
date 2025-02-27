import argparse
import subprocess
import yaml
from datetime import datetime

# Utility function to load YAML configuration
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Define the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified configurations.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--training_config', type=str, required=True, help='Path to the training configuration file.')
    parser.add_argument('--deployment_config', type=str, required=True, help='Path to the deployment configuration file.')

    args = parser.parse_args()

    # Load configurations from YAML files
    model_config = load_yaml_config(args.model_config)
    model_name = model_config.get('name', 'Unknown Model')
    data_config = load_yaml_config(args.data_config)
    training_data_name = data_config.get('name', 'unknown_data')
    training_config = load_yaml_config(args.training_config)
    adapter_save_path = training_config.get('adapter_path', './adapters/{model_name}_{data_name}')

    # Generate timestamp for saving the adapter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_path = adapter_save_path.format(model_name=model_name, data_name=training_data_name) + f"_{timestamp}"

    # Print the model name and training data directory
    print(f'Model     : {model_name}')
    print(f'Data      : ./data/{training_data_name}')
    print(f'Training  : {training_config}')
    print(f'Deployment: {args.deployment_config}')
    print(f'Adapter Save Path: {adapter_path}')

    # Execute the mlx_lm.lora command
    command = [
        'mlx_lm.lora',
        '--model', model_name,
        '--train',
        '--data', f'./data/{training_data_name}',
        '--learning-rate', str(training_config.get('learning_rate')),
        '--iters', str(training_config.get('iterations')),
        '--fine-tune-type', training_config.get('fine-tune-type'),
        '--adapter-path', adapter_path
    ]
    subprocess.run(command)
