"""
Client for interacting with Ollama API.
"""

import json
import time
import requests
from typing import Dict, Any, Optional, List

from translation import config


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model: str = config.DEFAULT_MODEL, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            model: The model to use for translation
            base_url: The base URL of the Ollama API
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.language_cache = {}
        
    def list_models(self) -> List[str]:
        """
        List all available models in the Ollama instance.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, retries: int = config.MAX_RETRIES) -> Optional[str]:
        """
        Generate a response from the model.
        
        Args:
            prompt: The prompt to send to the model
            retries: Number of retries for failed requests
            
        Returns:
            The generated response or None if failed
        """
        url = f"{self.api_url}/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=data, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json().get("response", "")
            except requests.RequestException as e:
                if attempt < retries:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {config.RETRY_DELAY} seconds...")
                    time.sleep(config.RETRY_DELAY)
                else:
                    print(f"Failed to generate response after {retries} retries: {e}")
                    return None
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: The text to detect language for
            
        Returns:
            The detected language name (e.g., 'English', 'Simplified Chinese', 'Traditional Chinese')
        """
        # Use a sample of the text for detection (first 200 characters)
        sample = text[:200]
        
        # Check if we already have this sample in cache
        if sample in self.language_cache:
            return self.language_cache[sample]
        
        # Prepare the prompt for language detection
        prompt = config.LANGUAGE_DETECTION_PROMPT.format(text=sample)
        
        # Generate the response
        detected_language = self.generate(prompt)
        
        if detected_language:
            # Clean up the response
            detected_language = detected_language.strip()
            
            # Handle Chinese detection specifically
            if "chinese" in detected_language.lower():
                if not ("simplified" in detected_language.lower() or "traditional" in detected_language.lower()):
                    # If it just says "Chinese" without specifying, try to determine which type
                    is_traditional = self._is_traditional_chinese(sample)
                    detected_language = "Traditional Chinese" if is_traditional else "Simplified Chinese"
            
            # Store in cache
            self.language_cache[sample] = detected_language
            
            return detected_language
        else:
            # Default to English if detection fails
            return "English"
    
    def _is_traditional_chinese(self, text: str) -> bool:
        """
        Determine if Chinese text is Traditional or Simplified.
        This is a heuristic approach based on character frequency.
        
        Args:
            text: The Chinese text to analyze
            
        Returns:
            True if likely Traditional Chinese, False if likely Simplified Chinese
        """
        # Common characters that differ between Traditional and Simplified
        traditional_chars = set('國說壹會來這個時後麼東關樣學張華發們產還實與點經當對處聲體裡應當國會來這個時後麼東關樣學長發們產還實與點經當對處聲體裡應')
        simplified_chars = set('国说一会来这个时后么东关样学张华发们产还实与点经当对处声体里应当国会来这个时后么东关样学长发们产还实与点经当对处声体里应')
        
        # Count occurrences of traditional and simplified characters
        trad_count = sum(1 for char in text if char in traditional_chars)
        simp_count = sum(1 for char in text if char in simplified_chars)
        
        # If there are more traditional characters, it's likely Traditional Chinese
        return trad_count > simp_count
    
    def get_language_code(self, language_name: str) -> str:
        """
        Get the language code for a language name.
        
        Args:
            language_name: The language name (e.g., 'English', 'Simplified Chinese')
            
        Returns:
            The language code (e.g., 'en', 'zh-cn', 'zh-tw')
        """
        # Convert to lowercase for case-insensitive matching
        language_name_lower = language_name.lower()
        
        # Special handling for Chinese variants
        if "chinese" in language_name_lower:
            if "simplified" in language_name_lower:
                return "zh-cn"
            elif "traditional" in language_name_lower:
                return "zh-tw"
            else:
                # If just "Chinese" without specifying, default to Simplified
                return "zh-cn"
        
        # Look up the language code for other languages
        for code, name in config.SUPPORTED_LANGUAGES.items():
            if name.lower() == language_name_lower:
                return code
        
        # Default to English if not found
        return "en"
    
    def translate(self, text: str, source_lang: str = config.DEFAULT_SOURCE_LANGUAGE, 
                 target_lang: str = config.DEFAULT_TARGET_LANGUAGE) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: The text to translate
            source_lang: The source language code or 'auto' for auto-detection
            target_lang: The target language code
            
        Returns:
            The translated text or None if translation failed
        """
        # Handle multi-line text by preserving the original structure
        # Clean up the text by removing extra whitespace but preserving newlines
        cleaned_text = "\n".join([line.strip() for line in text.split("\n")])
        
        # Auto-detect the source language if needed
        if source_lang == "auto":
            detected_language = self.detect_language(cleaned_text)
            source_language_name = detected_language
            # Get the language code for the detected language
            source_lang = self.get_language_code(detected_language)
        else:
            # Convert language code to full name
            source_language_name = config.SUPPORTED_LANGUAGES.get(source_lang, "English")
        
        # Convert target language code to full name
        target_language_name = config.SUPPORTED_LANGUAGES.get(target_lang, "English")
        
        # Skip translation if source and target languages are the same
        if source_language_name.lower() == target_language_name.lower():
            return cleaned_text
        
        # Prepare the translation prompt
        prompt = config.TRANSLATION_PROMPT.format(
            source_language=source_language_name,
            target_language=target_language_name,
            text=cleaned_text
        )
        
        # Generate the translation
        translated_text = self.generate(prompt)
        
        # If translation was successful, clean up any potential prefixes
        if translated_text:
            # Some models might prefix with "Translation:" or similar
            prefixes_to_remove = [
                "Translation:", 
                "Translated text:", 
                f"{target_language_name} translation:",
                "Here is the translation:"
            ]
            for prefix in prefixes_to_remove:
                if translated_text.lower().startswith(prefix.lower()):
                    translated_text = translated_text[len(prefix):].strip()
        
        return translated_text
