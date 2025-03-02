"""
Main workflow module that orchestrates the operation of all agents.
"""

from typing import Dict, Tuple, Optional, Any

from rich.console import Console
from rich.prompt import Prompt

from .acemath_agent import AceMathAgent
from .formatter_agent import FormatterAgent
from .solver_agent import SolverAgent
from .summarizer_agent import SummarizerAgent
from .ui import display_welcome, prompt_for_model_selection, display_results

console = Console()

class MathWorkflow:
    """
    Orchestrates the agentic math workflow using multiple agents in sequence.
    """
    
    def __init__(
        self, 
        formatter_model: str = "phi4", 
        solver_model: str = "qwen_deepseek", 
        summarizer_model: str = "phi4", 
        use_mlx: bool = True
    ):
        """
        Initialize the workflow by creating all required agents.
        
        Args:
            formatter_model: Name of the model for the formatter agent (default: phi4)
            solver_model: Name of the model for the solver agent (default: qwen_deepseek)
            summarizer_model: Name of the model for the summarizer agent (default: phi4)
            use_mlx: Whether to use the MLX-based AceMath model (default: True)
        """
        self.models = {
            "formatter": formatter_model,
            "solver": solver_model,
            "summarizer": summarizer_model,
            "use_mlx": use_mlx
        }
        
        self.formatter = None
        self.solver = None
        self.summarizer = None
        self.acemath = None
        
        # Initialize all agents
        self.setup_agents()
    
    def setup_agents(self) -> None:
        """
        Set up all agents needed for the workflow.
        """
        console.print("[bold blue]Setting up math workflow agents...[/bold blue]")
        
        # Initialize formatter agent
        self.formatter = FormatterAgent(self.models["formatter"])
        self.formatter.setup()
        
        # Initialize solver agent
        self.solver = SolverAgent(self.models["solver"])
        self.solver.setup()
        
        # Initialize summarizer agent
        self.summarizer = SummarizerAgent(self.models["summarizer"])
        self.summarizer.setup()
        
        # Initialize AceMath agent if enabled
        if self.models["use_mlx"]:
            self.acemath = AceMathAgent()
            if not self.acemath.setup():
                console.print("[yellow]Disabling MLX-based alternative solutions[/yellow]")
                self.models["use_mlx"] = False
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a math question through the complete workflow.
        
        Args:
            question: The math question to process
            
        Returns:
            A dictionary containing the results from each step
        """
        results = {
            "original_question": question,
            "is_math": False,
            "formatted_question": "",
            "explanation": "",
            "primary_solution": "",
            "alternative_solution": "",
            "summary": ""
        }
        
        # Step 1: Format and validate the question
        formatted_question, is_math, explanation = self.formatter.process(question)
        results["formatted_question"] = formatted_question
        results["is_math"] = is_math
        results["explanation"] = explanation
        
        if not is_math:
            console.print("[yellow]Skipping the solving step as this doesn't appear to be a math question.[/yellow]")
            return results
        
        # Step 2: Solve the problem with primary solver
        primary_solution = self.solver.process(formatted_question)
        results["primary_solution"] = primary_solution
        
        # Step 3: Get alternative solution (if enabled)
        if self.models["use_mlx"] and self.acemath:
            alternative_solution = self.acemath.process(formatted_question)
            results["alternative_solution"] = alternative_solution
        else:
            # Use a different model as alternative
            # If we have a different model available as a fallback
            alt_model = "gemma" if self.models["solver"] != "gemma" else "phi4"
            alt_solver = SolverAgent(alt_model)
            if alt_solver.setup():
                alternative_solution = alt_solver.process(formatted_question)
                results["alternative_solution"] = alternative_solution
            else:
                results["alternative_solution"] = "Alternative solution not available."
        
        # Step 4: Summarize the solution
        summary = self.summarizer.process(formatted_question, primary_solution)
        results["summary"] = summary
        
        return results
    
    def change_models(self) -> None:
        """
        Allow the user to change the models used for each step.
        """
        console.print("\n[bold blue]Change Models[/bold blue]")
        
        # Get new model selections
        new_models = prompt_for_model_selection(self.models)
        
        # Only set up new agents if models have changed
        if new_models != self.models:
            self.models = new_models
            console.print("\n[bold]Reloading agents with new model selections...[/bold]")
            self.setup_agents()
        else:
            console.print("\n[yellow]No changes to models.[/yellow]")
    
    def run_interactive(self) -> None:
        """
        Run the interactive command-line interface.
        """
        display_welcome(
            self.models["formatter"],
            self.models["solver"],
            self.models["summarizer"],
            self.models["use_mlx"]
        )
        
        while True:
            question = Prompt.ask(
                "\n[bold cyan]Ask a math question[/bold cyan] "
                "(or type 'models' to change models, 'exit' to quit)"
            )
            
            if question.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            if question.lower() == "models":
                self.change_models()
                continue
            
            console.print("\n[bold]Processing your request...[/bold]")
            
            # Process the question through the workflow
            results = self.process_question(question)
            
            # Display the results if it was a math question
            if results["is_math"]:
                display_results(
                    results["primary_solution"],
                    results["alternative_solution"],
                    results["summary"]
                )
            else:
                console.print(
                    f"\n[yellow]This doesn't appear to be a math question: "
                    f"{results['explanation']}[/yellow]"
                )


if __name__ == "__main__":
    # Create and run the workflow
    workflow = MathWorkflow()
    workflow.run_interactive()
