"""
Formatter Agent - Validates and formats math questions.
"""

from typing import Tuple

from langchain_community.llms.ollama import Ollama
from rich.console import Console
from rich.progress import Progress

from .base import Agent, console

class FormatterAgent(Agent):
    """
    Agent responsible for validating and formatting math questions.
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        """
        Initialize the formatter agent.
        
        Args:
            model_name: Name of the Ollama model to use (default: deepseek-r1:1.5b)
        """
        super().__init__(model_name)
    
    def setup(self) -> bool:
        """
        Set up the formatter agent by loading the Ollama model.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            self.model = Ollama(model=self.model_name)
            console.print(f"[green]✓ {self.model_name} loaded for question formatting[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading model: {str(e)}[/bold red]")
            console.print("[yellow]No fallback model configured[/yellow]")
            return False
    
    def process(self, question: str) -> Tuple[str, bool, str]:
        """
        Format and validate a math question.
        
        Args:
            question: The original math question
            
        Returns:
            A tuple of (formatted_question, is_math, explanation)
        """
        console.print("[bold]Step 1:[/bold] Formatting and validating your question...")
        
        prompt = f"""
        Your task is to determine if this is a math problem, and reformat it for better clarity if needed.
        
        QUESTION: {question}
        
        Respond strictly in the following format:
        IS_MATH: [Yes/No]
        
        FORMATTED_QUESTION: [The reformatted question, with better structure if needed]
        
        EXPLANATION: [Brief explanation of your determination, and what changes you made to formatting if any]
        """
        
        try:
            response, timed_out = self.invoke_with_timeout(prompt)
            
            if timed_out:
                console.print("[yellow]Validation timed out, proceeding with original question[/yellow]")
                return question, True, "Validation timed out, proceeding as if this is a math question."

            # Parse the response
            is_math = False
            formatted_question = question
            explanation = "Could not parse the response properly."
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("IS_MATH:"):
                    is_math = "yes" in line.lower()
                elif line.startswith("FORMATTED_QUESTION:"):
                    formatted_question = line[len("FORMATTED_QUESTION:"):].strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line[len("EXPLANATION:"):].strip()
            
            if not formatted_question:
                formatted_question = question
            
            if is_math:
                console.print("[green]✓ This is a math question[/green]")
                console.print(f"[blue]Formatted question:[/blue] {formatted_question}")
            else:
                console.print("[yellow]This doesn't appear to be a math question[/yellow]")
                console.print(f"[blue]Explanation:[/blue] {explanation}")
            
            return formatted_question, is_math, explanation
            
        except Exception as e:
            console.print(f"[bold red]Error formatting question: {str(e)}[/bold red]")
            return question, True, f"Error during validation: {str(e)}"
