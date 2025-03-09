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


def list_supported_languages():
    """List all supported languages for translation."""
    print("Supported languages:")
    for code, name in config.SUPPORTED_LANGUAGES.items():
        print(f"  - {name} ({code})")


def translate_file(args):
    """Translate a single file."""
    translator = Translator(model=args.model, source_lang=args.source_lang, target_lang=args.target_lang)
    translator.translate_jsonl_file(args.input_file, args.output_file)


def translate_directory(args):
    """Translate all files in a directory."""
    translator = Translator(model=args.model, source_lang=args.source_lang, target_lang=args.target_lang)
    translator.translate_directory(args.input_dir, args.output_dir)


def detect_language(args):
    """Detect the language of a text file."""
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        client = OllamaClient(model=args.model)
        detected_language = client.detect_language(text)
        language_code = client.get_language_code(detected_language)
        
        print(f"Detected language: {detected_language} ({language_code})")
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 1
    
    return 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Translate content between multiple languages using Ollama LLMs"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available Ollama models")
    
    # List languages command
    list_langs_parser = subparsers.add_parser("list-languages", help="List supported languages for translation")
    
    # Translate file command
    file_parser = subparsers.add_parser("translate-file", help="Translate a single JSONL file")
    file_parser.add_argument("input_file", help="Path to the input JSONL file")
    file_parser.add_argument("output_file", help="Path to the output JSONL file")
    file_parser.add_argument(
        "--model", 
        default=config.DEFAULT_MODEL,
        help=f"Ollama model to use (default: {config.DEFAULT_MODEL})"
    )
    file_parser.add_argument(
        "--source-lang", 
        default=config.DEFAULT_SOURCE_LANGUAGE,
        help=f"Source language code or 'auto' for auto-detection (default: {config.DEFAULT_SOURCE_LANGUAGE})"
    )
    file_parser.add_argument(
        "--target-lang", 
        default=config.DEFAULT_TARGET_LANGUAGE,
        help=f"Target language code (default: {config.DEFAULT_TARGET_LANGUAGE}). Use list-languages to see available codes."
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
        default=None,
        help=f"Path to the output directory (default: {config.OUTPUT_DIR.format(lang_code='<target_lang>')})"
    )
    dir_parser.add_argument(
        "--model", 
        default=config.DEFAULT_MODEL,
        help=f"Ollama model to use (default: {config.DEFAULT_MODEL})"
    )
    dir_parser.add_argument(
        "--source-lang", 
        default=config.DEFAULT_SOURCE_LANGUAGE,
        help=f"Source language code or 'auto' for auto-detection (default: {config.DEFAULT_SOURCE_LANGUAGE})"
    )
    dir_parser.add_argument(
        "--target-lang", 
        default=config.DEFAULT_TARGET_LANGUAGE,
        help=f"Target language code (default: {config.DEFAULT_TARGET_LANGUAGE}). Use list-languages to see available codes."
    )
    
    # Detect language command
    detect_parser = subparsers.add_parser("detect-language", help="Detect the language of a text file")
    detect_parser.add_argument(
        "input_file", 
        help="Path to the input text file"
    )
    detect_parser.add_argument(
        "--model", 
        default=config.DEFAULT_MODEL,
        help=f"Ollama model to use (default: {config.DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        list_available_models()
    elif args.command == "list-languages":
        list_supported_languages()
    elif args.command == "translate-file":
        translate_file(args)
    elif args.command == "translate-dir":
        translate_directory(args)
    elif args.command == "detect-language":
        detect_language(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
