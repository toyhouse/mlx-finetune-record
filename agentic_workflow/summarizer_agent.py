"""
Summarizer Agent - Summarizes solutions to be concise and clear.
"""

from langchain_community.llms.ollama import Ollama
from rich.console import Console
from rich.progress import Progress

from .base import Agent, console

class SummarizerAgent(Agent):
    """
    Agent responsible for summarizing solutions to be concise and clear.
    """
    
    def __init__(self, model_name: str = "phi4"):
        """
        Initialize the summarizer agent.
        
        Args:
            model_name: Name of the Ollama model to use (default: phi4)
        """
        super().__init__(model_name)
    
    def setup(self) -> bool:
        """
        Set up the summarizer agent by loading the Ollama model.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            self.model = Ollama(model=self.model_name)
            console.print(f"[green]✓ {self.model_name} loaded for summarization[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading {self.model_name}: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to phi4 for summarization[/yellow]")
            try:
                self.model_name = "phi4"
                self.model = Ollama(model="phi4")
                return True
            except Exception as e:
                console.print(f"[bold red]Error loading fallback model: {str(e)}[/bold red]")
                return False
    
    def process(self, question: str, solution: str) -> str:
        """
        Summarize the solution to be less than 512 tokens.
        
        Args:
            question: The original question
            solution: The detailed solution
            
        Returns:
            A concise summary of the solution
        """
        console.print(f"[bold]Step 4:[/bold] Summarizing solution with {self.model_name}...")
        
        prompt = f"""
        Please summarize this math solution to be less than 512 tokens while preserving the key steps and explanation.
        
        QUESTION: {question}
        
        DETAILED SOLUTION: {solution}
        
        Produce a concise but complete summary that includes:
        1. The main approach used
        2. Key steps in the solution
        3. The final answer
        4. Any important insights
        
        Keep the summary clear and educational while being brief.
        """
        
        response, timed_out = self.invoke_with_timeout(prompt)
        
        if timed_out:
            console.print(f"[bold red]Summarization timed out[/bold red]")
            fallback_response = f"""
            Due to a timeout, here's a brief summary:
            
            For the question: {question}
            
            The detailed solution provided a step-by-step approach to solve this problem.
            The answer can be found in the full solution.
            
            Please refer to the complete solution for details.
            """
            return fallback_response.strip()
        
        # Estimate token count (rough approximation: ~4 chars per token)
        estimated_tokens = len(response) / 4
        console.print(f"[green]✓ Summary generated with {self.model_name} (~{int(estimated_tokens)} tokens)[/green]")
        
        return response
