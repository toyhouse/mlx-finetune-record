"""
Translator module for translating JSONL files from Indonesian to English.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from translation import config
from translation.ollama_client import OllamaClient


class Translator:
    """Translator for JSONL files from Indonesian to English."""
    
    def __init__(self, model: str = config.DEFAULT_MODEL):
        """
        Initialize the translator.
        
        Args:
            model: The model to use for translation
        """
        self.model = model
        self.client = OllamaClient(model=model)
        
    def translate_jsonl_file(self, input_file: str, output_file: str) -> None:
        """
        Translate a JSONL file from Indonesian to English.
        
        Args:
            input_file: Path to the input JSONL file
            output_file: Path to the output JSONL file
        """
        print(f"Translating {input_file} to {output_file}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        translated_lines = []
        
        for i, line in enumerate(lines):
            print(f"Processing entry {i+1}/{total_lines}...")
            
            try:
                # Parse JSON
                entry = json.loads(line.strip())
                
                # Translate the text content
                translated_entry = self._translate_entry(entry)
                
                # Add to translated lines
                translated_lines.append(json.dumps(translated_entry, ensure_ascii=False))
                
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {i+1}, skipping...")
                continue
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in translated_lines:
                f.write(line + '\n')
        
        print(f"Translation completed: {output_file}")
    
    def _translate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a single JSONL entry.
        
        Args:
            entry: The JSONL entry to translate
            
        Returns:
            The translated entry
        """
        # Make a copy of the entry to avoid modifying the original
        translated_entry = entry.copy()
        
        # Extract the text content
        text = entry.get('text', '')
        
        if text:
            # Split the text into major sections (system, human, assistant)
            # The format appears to be:
            # System instruction in English
            # Human: Indonesian content
            # Assistant: Indonesian content
            
            # First, identify the main sections by splitting on double newlines
            sections = []
            current_section = []
            current_prefix = None
            
            # Split by double newlines first to get the main sections
            raw_parts = text.split('\n\n')
            
            for part in raw_parts:
                # Check if this is a new section (starts with Human: or Assistant:)
                if part.startswith('Human:') or part.startswith('Assistant:'):
                    # If we have a current section, add it to sections
                    if current_section:
                        sections.append((current_prefix, '\n\n'.join(current_section)))
                        current_section = []
                    
                    # Extract the prefix (Human: or Assistant:)
                    prefix_end = part.find(':') + 1
                    current_prefix = part[:prefix_end]
                    
                    # Add the content (without prefix) to the current section
                    current_section.append(part[prefix_end:].strip())
                elif current_prefix and current_section:  # This is a continuation of the current section
                    current_section.append(part)
                else:  # This is a system message or other non-conversation part
                    if current_section:  # Finish any previous section
                        sections.append((current_prefix, '\n\n'.join(current_section)))
                        current_section = []
                    sections.append((None, part))  # None prefix means keep as is
                    current_prefix = None
            
            # Add the last section if there is one
            if current_section:
                sections.append((current_prefix, '\n\n'.join(current_section)))
            
            # Now translate each section
            translated_sections = []
            
            for prefix, content in sections:
                if prefix:  # This is a Human: or Assistant: section
                    # Translate the entire content as one block
                    translated_content = self.client.translate(content)
                    
                    if translated_content:
                        # Reconstruct with the original prefix
                        translated_sections.append(f"{prefix}{translated_content}")
                    else:
                        # If translation failed, keep the original
                        translated_sections.append(f"{prefix}{content}")
                else:  # This is a system message or other non-conversation part
                    # Keep as is
                    translated_sections.append(content)
            
            # Combine the translated sections
            translated_text = '\n\n'.join(translated_sections)
            translated_entry['text'] = translated_text
        
        return translated_entry
    
    def translate_directory(self, input_dir: str = config.INPUT_DIR, 
                           output_dir: str = config.OUTPUT_DIR) -> None:
        """
        Translate all JSONL files in a directory.
        
        Args:
            input_dir: Path to the input directory
            output_dir: Path to the output directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Get all JSONL files in the input directory
        jsonl_files = list(input_path.glob('*.jsonl'))
        
        if not jsonl_files:
            print(f"No JSONL files found in {input_dir}")
            return
        
        print(f"Found {len(jsonl_files)} JSONL files to translate")
        
        for file in jsonl_files:
            input_file = str(file)
            output_file = str(output_path / file.name)
            
            self.translate_jsonl_file(input_file, output_file)
