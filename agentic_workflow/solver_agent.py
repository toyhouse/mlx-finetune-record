"""
Solver Agent - Solves math problems with detailed explanation.
"""

from langchain_community.llms.ollama import Ollama
from rich.console import Console
from rich.progress import Progress

from .base import Agent, console

class SolverAgent(Agent):
    """
    Agent responsible for solving math problems with detailed explanations.
    """
    
    def __init__(self, model_name: str = "qwen_deepseek"):
        """
        Initialize the solver agent.
        
        Args:
            model_name: Name of the Ollama model to use (default: qwen_deepseek)
        """
        super().__init__(model_name)
    
    def setup(self) -> bool:
        """
        Set up the solver agent by loading the Ollama model.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            self.model = Ollama(model=self.model_name)
            console.print(f"[green]✓ {self.model_name} loaded for problem solving[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading {self.model_name}: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to phi4 for solving[/yellow]")
            try:
                self.model_name = "phi4"
                self.model = Ollama(model="phi4")
                return True
            except Exception as e:
                console.print(f"[bold red]Error loading fallback model: {str(e)}[/bold red]")
                return False

    def process(self, problem: str) -> str:
        """
        Solve a math problem with detailed explanation.
        
        Args:
            problem: The formatted math problem to solve
            
        Returns:
            The solution with explanation
        """
        console.print(f"[bold]Step 2:[/bold] Solving with {self.model_name} model...")
        
        prompt = f"""
        Please solve this math problem step-by-step with clear explanations:
        
        PROBLEM: {problem}
        
        - Start by understanding what the problem is asking
        - Break down the problem into steps
        - Show your work for each step
        - Explain your reasoning
        - Verify your answer
        
        Provide a detailed explanation that would help someone understand how to solve this problem.
        """
        
        response, timed_out = self.invoke_with_timeout(prompt)
        
        if timed_out:
            console.print(f"[bold red]Operation timed out[/bold red]")
            # Fall back to a more reliable model
            fallback_model_name = "phi4"  # Phi-4 is typically more reliable
            console.print(f"[yellow]Falling back to {fallback_model_name} model...[/yellow]")
            
            try:
                fallback_model = Ollama(model=fallback_model_name)
                with Progress() as progress:
                    task = progress.add_task(f"[cyan]Processing with fallback {fallback_model_name}...", total=None)
                    response = fallback_model.invoke(prompt)
                    progress.update(task, completed=100)
                
                console.print(f"[green]✓ Solution generated with fallback model {fallback_model_name}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error with fallback model: {str(e)}[/bold red]")
                return f"Error: Operation timed out and fallback model failed."
        else:
            console.print(f"[green]✓ Solution generated with {self.model_name}[/green]")
            
        return response
