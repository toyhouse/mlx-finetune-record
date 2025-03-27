"""
UI components for the agentic workflow.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

def display_welcome(formatter_model: str, solver_model: str, summarizer_model: str, use_mlx: bool = True) -> None:
    """
    Display a welcome message with information about the workflow.
    
    Args:
        formatter_model: Name of the formatter model
        solver_model: Name of the solver model
        summarizer_model: Name of the summarizer model
        use_mlx: Whether MLX is being used
    """
    mlx_text = "- [bold]AceMath (MLX)[/bold]: Provides an alternative solution" if use_mlx else ""
    
    console.print(Panel.fit(
        "[bold yellow]Agentic Math Workflow[/bold yellow]\n\n"
        "This workflow uses multiple agents in sequence to process and solve math problems:\n"
        f"- [bold]{formatter_model}[/bold]: Validates and formats your question\n"
        f"- [bold]{solver_model}[/bold]: Solves the problem with detailed explanation\n"
        f"{mlx_text}\n"
        f"- [bold]{summarizer_model}[/bold]: Summarizes the solution to be concise\n\n"
        "Type [bold green]'exit'[/bold green] to quit or [bold green]'models'[/bold green] to change models.",
        title="Welcome",
        border_style="blue"
    ))

def display_models_table() -> None:
    """
    Display a table of available models.
    """
    model_table = Table(title="Available Ollama Models", show_header=True)
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Description")
    
    model_table.add_row("phi4", "Microsoft's phi-4 is a compact yet powerful instruction model")
    model_table.add_row("gemma", "Google's Gemma is a lightweight LLM for various tasks")
    model_table.add_row("qwen", "Qwen is a powerful language model for general reasoning")
    model_table.add_row("llama3", "Meta's Llama 3 is a versatile general-purpose model")
    model_table.add_row("mistral", "Mistral AI's model excels at instruction following")
    
    console.print(model_table)

def prompt_for_model_selection(current_models: dict) -> dict:
    """
    Prompt the user to select models for the workflow.
    
    Args:
        current_models: Dictionary with current model selections
        
    Returns:
        Updated dictionary with model selections
    """
    models = current_models.copy()
    available_models = ["phi4", "gemma", "qwen", "llama3", "mistral", "skip"]
    
    display_models_table()
    
    formatter_choice = Prompt.ask(
        "\nSelect formatter model", 
        choices=available_models, 
        default=models.get("formatter", "phi4")
    )
    
    if formatter_choice != "skip":
        models["formatter"] = formatter_choice
    
    solver_choice = Prompt.ask(
        "Select solver model", 
        choices=available_models, 
        default=models.get("solver", "qwen")
    )
    
    if solver_choice != "skip":
        models["solver"] = solver_choice
    
    summarizer_choice = Prompt.ask(
        "Select summarizer model", 
        choices=available_models, 
        default=models.get("summarizer", "phi4")
    )
    
    if summarizer_choice != "skip":
        models["summarizer"] = summarizer_choice
    
    use_mlx_choice = Prompt.ask(
        "Use MLX for alternative solution?", 
        choices=["y", "n", "skip"], 
        default="y" if models.get("use_mlx", True) else "n"
    )
    
    if use_mlx_choice != "skip":
        models["use_mlx"] = (use_mlx_choice == "y")
    
    return models

def display_results(primary_solution: str, alternative_solution: str, summary: str) -> None:
    """
    Display the results in a formatted table.
    
    Args:
        primary_solution: The primary solution
        alternative_solution: The alternative solution
        summary: The summarized solution
    """
    # Create a comparison table of solutions
    solution_table = Table(title="Math Solutions", show_header=True, header_style="bold magenta")
    solution_table.add_column("Source", style="cyan")
    solution_table.add_column("Solution")
    
    # Escape Rich markup syntax in solutions for display
    safe_primary = primary_solution.replace("[", "[[").replace("]", "]]")
    safe_alternative = alternative_solution.replace("[", "[[").replace("]", "]]")
    safe_summary = summary.replace("[", "[[").replace("]", "]]")
    
    # Truncate long solutions for display
    solution_table.add_row(
        "Primary Solution", 
        safe_primary[:500] + "..." if len(safe_primary) > 500 else safe_primary
    )
    solution_table.add_row(
        "Alternative Solution", 
        safe_alternative[:500] + "..." if len(safe_alternative) > 500 else safe_alternative
    )
    solution_table.add_row("Summary", safe_summary)
    
    # Display the results
    console.print("\n[bold green]Results:[/bold green]")
    console.print(solution_table)
