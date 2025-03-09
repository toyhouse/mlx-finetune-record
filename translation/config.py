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

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "de": "German",
    "id": "Indonesian",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "th": "Thai"
}

# Default source and target languages
DEFAULT_SOURCE_LANGUAGE = "auto"  # 'auto' means auto-detect
DEFAULT_TARGET_LANGUAGE = "en"    # English by default

# Input and output directories
INPUT_DIR = "../data/videotranscript"
OUTPUT_DIR = "../data/{lang_code}_transcription"  # Format string with language code

# Translation prompt templates
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

# Language detection prompt
LANGUAGE_DETECTION_PROMPT = """
Identify the language of the following text. Respond with only the language name in English (e.g., 'English', 'Simplified Chinese', 'Traditional Chinese', 'German', 'Indonesian', 'Japanese', etc.).
If the text is in Chinese, please specify whether it is Simplified Chinese or Traditional Chinese.
Do not include any additional text, explanations, or punctuation in your response.

Text:
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
