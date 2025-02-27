"""
Module to load and process configurations from YAML files.
"""
import yaml
import os
from typing import Dict, Any, Optional

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file is not valid YAML
    """
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config if config else {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}")
            raise

def get_config_path(config_type: str, config_name: str) -> Optional[str]:
    """
    Get the path to a configuration file.
    
    Args:
        config_type (str): Type of configuration (model, data, training, deployment, benchmark)
        config_name (str): Name of the configuration file (without extension)
        
    Returns:
        Optional[str]: Path to the configuration file, or None if it doesn't exist
    """
    type_mapping = {
        'model': 'model_configs',
        'data': 'data_configs',
        'training': 'training_configs',
        'deployment': 'deployment_configs',
        'benchmark': 'benchmark_configs'
    }
    
    if config_type not in type_mapping:
        raise ValueError(f"Invalid configuration type: {config_type}")
    
    config_dir = os.path.join('configs', type_mapping[config_type])
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    
    return config_path if os.path.exists(config_path) else None

def load_config(config_type: str, config_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a configuration by type and name.
    
    Args:
        config_type (str): Type of configuration (model, data, training, deployment, benchmark)
        config_name (str): Name of the configuration file (without extension)
        
    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary, or None if file doesn't exist
    """
    config_path = get_config_path(config_type, config_name)
    
    if config_path:
        return load_yaml_config(config_path)
    else:
        print(f"Configuration '{config_name}' of type '{config_type}' not found.")
        return None
