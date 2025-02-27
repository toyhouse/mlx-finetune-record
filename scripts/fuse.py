import argparse
import subprocess
import yaml
        
# Define the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse the model with the adapter weights.')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the adapter weights.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the fused model.')

    args = parser.parse_args()

    # Load model configuration from YAML file
    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    # Construct the mlx_lm.fuse command
    command = [
        'mlx_lm.fuse',
        '--model', model_config['name'],
        '--save-path', args.output_path,
        '--adapter-path', args.adapter_path
    ]

    # Execute the fuse command
    subprocess.run(command)
