#!/usr/bin/env python
"""
Command-line interface for the translation system.
"""

import os
import sys
import argparse
from pathlib import Path

from translation import config
from translation.translator import Translator
from translation.ollama_client import OllamaClient


def list_available_models():
    """List all available models in the Ollama instance."""
    client = OllamaClient()
    models = client.list_models()
    
    if models:
        print("Available Ollama models:")
        for model in models:
            print(f"  - {model}")
    else:
        print("No models found or could not connect to Ollama.")
        print("Make sure Ollama is running and accessible at http://localhost:11434")


def translate_file(args):
    """Translate a single file."""
    translator = Translator(model=args.model)
    translator.translate_jsonl_file(args.input_file, args.output_file)


def translate_directory(args):
    """Translate all files in a directory."""
    translator = Translator(model=args.model)
    translator.translate_directory(args.input_dir, args.output_dir)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Translate Indonesian content to English using Ollama LLMs"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available Ollama models")
    
    # Translate file command
    file_parser = subparsers.add_parser("translate-file", help="Translate a single JSONL file")
    file_parser.add_argument("input_file", help="Path to the input JSONL file")
    file_parser.add_argument("output_file", help="Path to the output JSONL file")
    file_parser.add_argument(
        "--model", 
        default=config.DEFAULT_MODEL,
        help=f"Ollama model to use (default: {config.DEFAULT_MODEL})"
    )
    
    # Translate directory command
    dir_parser = subparsers.add_parser("translate-dir", help="Translate all JSONL files in a directory")
    dir_parser.add_argument(
        "--input-dir", 
        default=config.INPUT_DIR,
        help=f"Path to the input directory (default: {config.INPUT_DIR})"
    )
    dir_parser.add_argument(
        "--output-dir", 
        default=config.OUTPUT_DIR,
        help=f"Path to the output directory (default: {config.OUTPUT_DIR})"
    )
    dir_parser.add_argument(
        "--model", 
        default=config.DEFAULT_MODEL,
        help=f"Ollama model to use (default: {config.DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        list_available_models()
    elif args.command == "translate-file":
        translate_file(args)
    elif args.command == "translate-dir":
        translate_directory(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
