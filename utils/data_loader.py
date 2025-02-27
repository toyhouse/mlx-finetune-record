"""
Module to load and preprocess data for model training and evaluation.
"""
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import yaml

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of JSON objects
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def format_data(data: List[Dict[str, Any]], text_field: str, format_template: Optional[str] = None) -> List[str]:
    """
    Format data according to a template.
    
    Args:
        data (List[Dict[str, Any]]): List of data objects
        text_field (str): Name of the field containing the text
        format_template (Optional[str]): Template for formatting the text
        
    Returns:
        List[str]: List of formatted text
    """
    formatted_data = []
    
    for item in data:
        if text_field in item:
            if format_template and isinstance(item[text_field], dict):
                # If template is provided and the text field is a dictionary,
                # use the template for formatting
                formatted_text = format_template.format(**item[text_field])
            else:
                # Otherwise, use the text field directly
                formatted_text = item[text_field]
            
            formatted_data.append(formatted_text)
    
    return formatted_data

def load_and_preprocess_data(data_config_path: str) -> Tuple[List[str], Optional[List[str]]]:
    """
    Load and preprocess data according to a configuration.
    
    Args:
        data_config_path (str): Path to the data configuration file
        
    Returns:
        Tuple[List[str], Optional[List[str]]]: Tuple of training and validation data
    """
    # Load data configuration
    data_config = load_yaml_config(data_config_path)
    
    # Extract relevant parameters
    train_data_path = data_config.get('train_data')
    validation_data_path = data_config.get('validation_data')
    data_format = data_config.get('format', 'jsonl')
    text_field = data_config.get('text_field', 'text')
    format_template = data_config.get('format_template')
    
    # Load training data
    if train_data_path and os.path.exists(train_data_path):
        if data_format == 'jsonl':
            train_data = load_jsonl_data(train_data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        train_texts = format_data(train_data, text_field, format_template)
    else:
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")
    
    # Load validation data if available
    validation_texts = None
    if validation_data_path and os.path.exists(validation_data_path):
        if data_format == 'jsonl':
            validation_data = load_jsonl_data(validation_data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        validation_texts = format_data(validation_data, text_field, format_template)
    
    return train_texts, validation_texts
