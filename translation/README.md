# Indonesian to English Translation System

This system translates Indonesian content to English using Ollama LLMs while preserving all metadata in the original files.

## Features

- Translate JSONL files from Indonesian to English
- Preserve all metadata in the original files
- Support for multiple Ollama models
- Command-line interface for easy use
- Configurable translation parameters

## Prerequisites

- Python 3.6+
- Ollama installed and running locally
- At least one LLM loaded in Ollama

## Installation

No additional installation is required if you're already in the project environment. The translation system uses the standard Python libraries and the `requests` package which should be available in the project environment.

## Configuration

The translation system is configured in `config.py`. You can modify the following parameters:

- `AVAILABLE_MODELS`: List of available Ollama models
- `DEFAULT_MODEL`: Default model to use for translation
- `SOURCE_LANGUAGE`: Source language (default: Indonesian)
- `TARGET_LANGUAGE`: Target language (default: English)
- `INPUT_DIR`: Input directory containing JSONL files
- `OUTPUT_DIR`: Output directory for translated JSONL files
- `TRANSLATION_PROMPT`: Prompt template for translation
- `BATCH_SIZE`: Number of entries to process in one go
- `REQUEST_TIMEOUT`: Request timeout in seconds
- `MAX_RETRIES`: Maximum retries for failed requests
- `RETRY_DELAY`: Delay between retries in seconds

## Usage

### List Available Models

```bash
python -m translation.cli list-models
```

### Translate a Single File

```bash
python -m translation.cli translate-file /path/to/input.jsonl /path/to/output.jsonl --model llama3
```

### Translate All Files in a Directory

```bash
python -m translation.cli translate-dir --input-dir /path/to/input_dir --output-dir /path/to/output_dir --model llama3
```

### Default Usage

To translate all files in the default directories with the default model:

```bash
python -m translation.cli translate-dir
```

## Example

```bash
# Translate all files in data/videotranscript to data/en_transcription using llama3
python -m translation.cli translate-dir --model llama3
```

## Programmatic Usage

You can also use the translation system programmatically in your Python code:

```python
from translation import Translator

# Initialize the translator with a specific model
translator = Translator(model="llama3")

# Translate a single file
translator.translate_jsonl_file(
    "data/videotranscript/train.jsonl", 
    "data/en_transcription/train.jsonl"
)

# Translate all files in a directory
translator.translate_directory(
    "data/videotranscript", 
    "data/en_transcription"
)
```
