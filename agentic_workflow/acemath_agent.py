"""
AceMath Agent - Uses MLX-based AceMath model for alternative solutions.
"""

import importlib.util
from typing import Optional

from rich.console import Console
from rich.progress import Progress

from .base import Agent, console

class AceMathAgent(Agent):
    """
    Agent responsible for generating alternative solutions using the MLX AceMath model.
    """
    
    def __init__(self):
        """
        Initialize the AceMath agent.
        """
        super().__init__("AceMath-7B-Instruct")
        self.math_model = None
        self.tokenizer = None
        
    def _check_mlx_imports(self) -> bool:
        """
        Check if MLX dependencies are available.
        
        Returns:
            True if MLX is available, False otherwise
        """
        try:
            mlx_spec = importlib.util.find_spec("mlx")
            mlx_lm_spec = importlib.util.find_spec("mlx_lm")
            return mlx_spec is not None and mlx_lm_spec is not None
        except ImportError:
            return False
    
    def setup(self) -> bool:
        """
        Set up the AceMath agent by loading the MLX-based AceMath model.
        
        Returns:
            True if setup was successful, False otherwise
        """
        console.print("[bold blue]Setting up MLX AceMath model...[/bold blue]")
        
        if not self._check_mlx_imports():
            console.print("[bold red]MLX or MLX_LM not found. Can't use AceMath model.[/bold red]")
            return False
        
        try:
            # Import only when needed to avoid errors if MLX is not installed
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
            import mlx.core as mx
            import mlx.nn as nn
            from transformers import AutoTokenizer
            
            model_path = "nvidia/AceMath-7B-Instruct"
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Loading AceMath model...", total=None)
                self.math_model, self.tokenizer = load(model_path)
                progress.update(task, completed=100)
                
            console.print("[green]✓ AceMath model loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error loading AceMath model: {str(e)}[/bold red]")
            console.print("[yellow]Will not use AceMath for alternative solutions[/yellow]")
            return False
    
    def process(self, problem: str) -> str:
        """
        Solve a math problem using the MLX-based AceMath model.
        
        Args:
            problem: The formatted math problem to solve
            
        Returns:
            The solution to the math problem
        """
        console.print("[bold]Step 3:[/bold] Getting alternative solution with AceMath model...")
        
        if not self.math_model or not self.tokenizer:
            return "AceMath model is not loaded. Cannot provide alternative solution."
        
        try:
            # Import only when needed
            from mlx_lm.utils import generate
            from mlx_lm.sample_utils import make_sampler
            
            # Format prompt for instruction model - using raw string to avoid issues with brackets
            formatted_prompt = f"[INST] {problem} [/INST]"
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing with AceMath...", total=None)
                
                # Create a sampler with the specified temperature
                sampler = make_sampler(temp=0.7)
                
                # Generate response using mlx_lm
                response = generate(
                    self.math_model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=200,
                    sampler=sampler
                )
                
                progress.update(task, completed=100)
            
            console.print("[green]✓ Alternative solution generated with AceMath[/green]")
            return response
            
        except Exception as e:
            console.print(f"[bold red]Error solving with AceMath: {str(e)}[/bold red]")
            return f"Error: {str(e)}"
