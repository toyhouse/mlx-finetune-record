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
    
    def translate(self, text: str, source_lang: str = config.SOURCE_LANGUAGE, 
                 target_lang: str = config.TARGET_LANGUAGE) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: The text to translate
            source_lang: The source language
            target_lang: The target language
            
        Returns:
            The translated text or None if translation failed
        """
        # Handle multi-line text by preserving the original structure
        # Clean up the text by removing extra whitespace but preserving newlines
        cleaned_text = "\n".join([line.strip() for line in text.split("\n")])
        
        prompt = config.TRANSLATION_PROMPT.format(
            source_language=source_lang,
            target_language=target_lang,
            text=cleaned_text
        )
        
        translated_text = self.generate(prompt)
        
        # If translation was successful, clean up any potential "Translation:" prefix
        # that the model might add to the response
        if translated_text:
            # Some models might prefix with "Translation:" or similar
            prefixes_to_remove = ["Translation:", "Translated text:", "English translation:"]
            for prefix in prefixes_to_remove:
                if translated_text.startswith(prefix):
                    translated_text = translated_text[len(prefix):].strip()
        
        return translated_text
