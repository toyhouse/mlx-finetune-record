"""
Configuration for the translation system.
"""

# Available Ollama models
AVAILABLE_MODELS = [
    "llama3",
    "mistral",
    "phi3",
    "gemma",
    "llama2",
    "mixtral",
    "qwen",
    "yi"
]

# Default model to use
DEFAULT_MODEL = "llama3"

# Source and target languages
SOURCE_LANGUAGE = "Indonesian"
TARGET_LANGUAGE = "English"

# Input and output directories
INPUT_DIR = "../data/videotranscript"
OUTPUT_DIR = "../data/en_transcription"

# Translation prompt template
TRANSLATION_PROMPT = """
You are a professional translator from {source_language} to {target_language}.
Translate the following text from {source_language} to {target_language}.

Important instructions:
1. Preserve the original meaning, tone, and style.
2. Maintain all formatting, including numbered lists, bullet points, and paragraphs.
3. Translate ALL of the text, including any multi-line content or lists.
4. Do not add any explanations, notes, or commentary.
5. Do not include phrases like 'Translation:' or 'Here is the translation:' in your response.
6. Return only the translated text.

Text to translate:
{text}
"""

# Batch size for processing (number of entries to process in one go)
BATCH_SIZE = 10

# Request timeout in seconds
REQUEST_TIMEOUT = 60

# Maximum retries for failed requests
MAX_RETRIES = 3

# Delay between retries in seconds
RETRY_DELAY = 2
