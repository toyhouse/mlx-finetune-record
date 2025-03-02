"""
Base utilities and common functionality for the agentic workflow.
"""

import importlib.util
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.llms.ollama import Ollama
from rich.console import Console
from rich.progress import Progress

# Create console for nice output formatting
console = Console()


class Agent(ABC):
    """
    Abstract base class for all agents in the workflow.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the agent with a model name.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model = None
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Set up the agent, initializing any required models or resources.
        
        Returns:
            True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data with this agent.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed output
        """
        pass
    
    def invoke_with_timeout(self, prompt: str, timeout_seconds: int = 30) -> Tuple[str, bool]:
        """
        Invoke the model with a timeout to prevent hanging.
        
        Args:
            prompt: The prompt to send to the model
            timeout_seconds: Maximum time to wait for a response
            
        Returns:
            A tuple of (response, timed_out)
        """
        if not hasattr(self, 'model') or self.model is None:
            console.print(f"[bold red]Model not initialized for {self.__class__.__name__}[/bold red]")
            return "Error: Model not initialized", False
        
        response = None
        timed_out = False
        
        # Use a thread to handle timeout
        def invoke_model():
            nonlocal response
            try:
                response = self.model.invoke(prompt)
            except Exception as e:
                console.print(f"[bold red]Error during model invocation: {str(e)}[/bold red]")
        
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Processing with {self.model_name}...", total=100
            )
            
            # Start the model invocation in a separate thread
            model_thread = threading.Thread(target=invoke_model)
            model_thread.daemon = True
            model_thread.start()
            
            # Wait for the thread to complete or timeout
            start_time = time.time()
            while model_thread.is_alive():
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    timed_out = True
                    break
                
                # Update progress bar
                progress_value = min(int((elapsed / timeout_seconds) * 90), 90)
                progress.update(task, completed=progress_value)
                time.sleep(0.1)
            
            progress.update(task, completed=100)
        
        return response or "Error: Failed to generate a response", timed_out
