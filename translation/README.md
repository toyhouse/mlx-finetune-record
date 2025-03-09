# Multi-Language Translation System

This system translates content between multiple languages using Ollama LLMs while preserving all metadata in the original files.

## Features

- Translate JSONL files between multiple languages (English, Simplified Chinese, Traditional Chinese, German, Spanish, French, etc.)
- Automatic source language detection with specific support for distinguishing between Simplified and Traditional Chinese
- Preserve all metadata in the original files
- Support for multiple Ollama models
- Command-line interface for easy use
- Configurable translation parameters
- Dynamic output directory naming based on target language

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
- `SUPPORTED_LANGUAGES`: Dictionary of supported language codes and names
- `DEFAULT_SOURCE_LANGUAGE`: Default source language code (default: 'auto' for automatic detection)
- `DEFAULT_TARGET_LANGUAGE`: Default target language code (default: 'en' for English)
- `INPUT_DIR`: Input directory containing JSONL files
- `OUTPUT_DIR`: Output directory template for translated JSONL files (uses target language code)
- `TRANSLATION_PROMPT`: Prompt template for translation
- `LANGUAGE_DETECTION_PROMPT`: Prompt template for language detection
- `BATCH_SIZE`: Number of entries to process in one go
- `REQUEST_TIMEOUT`: Request timeout in seconds
- `MAX_RETRIES`: Maximum retries for failed requests
- `RETRY_DELAY`: Delay between retries in seconds

## Usage

### List Available Models

```bash
python -m translation.cli list-models
```

### List Supported Languages

```bash
python -m translation.cli list-languages
```

### Detect Language of a Text File

```bash
python -m translation.cli detect-language /path/to/file.txt
```

### Translate a Single File

```bash
python -m translation.cli translate-file /path/to/input.jsonl /path/to/output.jsonl --model phi4 --source-lang auto --target-lang en
```

### Translate All Files in a Directory

```bash
python -m translation.cli translate-dir --input-dir /path/to/input_dir --target-lang zh --model phi4
```

### Default Usage

To translate all files in the default directories with the default model (to English):

```bash
python -m translation.cli translate-dir
```

## Examples

```bash
# Translate all files in data/videotranscript to data/en_transcription using llama3
python -m translation.cli translate-dir --model llama3

# Translate files to Simplified Chinese
python -m translation.cli translate-dir --target-lang zh-cn

# Translate files to Traditional Chinese
python -m translation.cli translate-dir --target-lang zh-tw

# Translate files from English to German
python -m translation.cli translate-dir --source-lang en --target-lang de

# Detect language of a file
python -m translation.cli detect-language data/videotranscript/train.jsonl
```

## Programmatic Usage

You can also use the translation system programmatically in your Python code:

```python
from translation import Translator

# Initialize the translator with a specific model and languages
translator = Translator(
    model="llama3",
    source_lang="auto",  # Auto-detect source language
    target_lang="en"     # Translate to English
)

# Translate a single file
translator.translate_jsonl_file(
    "data/videotranscript/train.jsonl", 
    "data/en_transcription/train.jsonl"
)

# Translate all files in a directory (uses target_lang to create output directory)
translator.translate_directory(
    "data/videotranscript"
    # Output directory will be automatically created based on target language
)

# Translate to Simplified Chinese
translator = Translator(model="llama3", target_lang="zh-cn")
translator.translate_directory("data/videotranscript")
# Output will be in data/zh_simplified_transcription

# Translate to Traditional Chinese
translator = Translator(model="llama3", target_lang="zh-tw")
translator.translate_directory("data/videotranscript")
# Output will be in data/zh_traditional_transcription
```
