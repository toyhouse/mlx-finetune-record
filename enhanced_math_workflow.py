#!/usr/bin/env python3
"""
Enhanced Math Agent Workflow
---------------------------
An advanced agentic workflow that processes math questions through multiple LLMs:
1. Uses Ollama (phi4) to validate and format the question
2. Uses Ollama (qwen_deepseek) to solve and explain the problem in detail
3. Uses Ollama (phi4) again to summarize the solution to less than 512 tokens
4. Provides an interactive command-line interface

This workflow demonstrates how to chain multiple LLM backends in a single application.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
import json
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from rich.table import Table
import threading
import signal

# LangChain imports
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import our MLX model
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

console = Console()

class EnhancedMathWorkflow:
    """
    Orchestrates an enhanced math problem solving workflow using multiple LLMs in sequence.
    """
    
    def __init__(self, formatter_model="phi4", solver_model="qwen_deepseek", summarizer_model="phi4", use_mlx=True):
        """
        Initialize the workflow components.
        
        Args:
            formatter_model: Ollama model to use for question formatting (default: phi4)
            solver_model: Ollama model to use for solving the problem (default: qwen_deepseek)
            summarizer_model: Ollama model to use for summarizing the solution (default: phi4)
            use_mlx: Whether to use the MLX model for alternative solution (default: True)
        """
        self.formatter_model_name = formatter_model
        self.solver_model_name = solver_model
        self.summarizer_model_name = summarizer_model
        self.use_mlx = use_mlx
        
        self.setup_ollama_models()
        if use_mlx:
            self.setup_mlx_model()
        
    def setup_ollama_models(self):
        """Set up the Ollama models for question formatting, solving, and summarization."""
        console.print("[bold blue]Setting up Ollama models...[/bold blue]")
        
        # Initialize Ollama models
        try:
            self.formatter = Ollama(model=self.formatter_model_name)
            console.print(f"[green]✓ {self.formatter_model_name} loaded for question formatting[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading {self.formatter_model_name}: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to phi4 for formatting[/yellow]")
            self.formatter_model_name = "phi4"
            self.formatter = Ollama(model="phi4")
        
        try:
            self.solver = Ollama(model=self.solver_model_name)
            console.print(f"[green]✓ {self.solver_model_name} loaded for problem solving[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading {self.solver_model_name}: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to qwen_deepseek for solving[/yellow]")
            self.solver_model_name = "qwen_deepseek"
            self.solver = Ollama(model="qwen_deepseek")
        
        try:
            self.summarizer = Ollama(model=self.summarizer_model_name)
            console.print(f"[green]✓ {self.summarizer_model_name} loaded for summarization[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading {self.summarizer_model_name}: {str(e)}[/bold red]")
            console.print("[yellow]Falling back to phi4 for summarization[/yellow]")
            self.summarizer_model_name = "phi4"
            self.summarizer = Ollama(model="phi4")
    
    def setup_mlx_model(self):
        """Set up the MLX-based AceMath model."""
        console.print("[bold blue]Setting up MLX AceMath model...[/bold blue]")
        try:
            model_path = "nvidia/AceMath-7B-Instruct"
            self.math_model, self.tokenizer = load(model_path)
            console.print("[bold green]✓ MLX model loaded successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error loading MLX model: {str(e)}[/bold red]")
            sys.exit(1)
    
    def format_question(self, question: str) -> tuple:
        """
        Format and validate the math question using the formatter model.
        
        Args:
            question: The user's math question
            
        Returns:
            Tuple of (formatted_question, is_math, explanation)
        """
        console.print(f"[bold]Step 1:[/bold] Validating and formatting your question with {self.formatter_model_name}...")
        
        prompt = f"""
        I need you to format and validate this math question. If it's valid, rewrite it 
        clearly and precisely. If it's not a math question, explain why.
        
        QUESTION: {question}
        
        FORMAT:
        {{
            "is_math_question": true/false,
            "formatted_question": "the formatted question",
            "explanation": "brief explanation about the formatting or why it's not a math question"
        }}
        
        Return ONLY a valid JSON object.
        """
        
        try:
            response = self.formatter.invoke(prompt)
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    
                    if result.get("is_math_question", False):
                        console.print("[green]✓ Valid math question identified[/green]")
                        formatted = result.get('formatted_question', question)
                        explanation = result.get('explanation', '')
                        return formatted, True, explanation
                    else:
                        console.print("[yellow]! Not identified as a math question[/yellow]")
                        explanation = result.get('explanation', 'This does not appear to be a math question.')
                        return question, False, explanation
                except json.JSONDecodeError:
                    # If JSON parsing fails, just use the response text
                    console.print("[yellow]! Could not parse JSON response, using original question[/yellow]")
                    return question, True, "Using original question format."
            else:
                # Fallback for non-JSON responses
                console.print("[yellow]! Could not extract JSON, using formatter's full response[/yellow]")
                return question, True, response
                
        except Exception as e:
            console.print(f"[bold red]Error formatting question: {str(e)}[/bold red]")
            return question, True, f"Error during validation: {str(e)}"
    
    def solve_problem(self, problem: str, model_type="ollama", step_num=2, model_name=None):
        """
        Solve a math problem using the specified model.
        
        Args:
            problem: The formatted math problem to solve
            model_type: Type of model to use - "ollama" or "mlx" (default: "ollama")
            step_num: Step number for progress display (default: 2)
            model_name: Name of the model for display (default: None, uses class attribute)
            
        Returns:
            The solution to the math problem
        """
        if model_type == "ollama":
            if model_name is None:
                model_name = self.solver_model_name
                model = self.solver
            elif model_name == self.solver_model_name:
                model = self.solver
            elif model_name == self.formatter_model_name:
                model = self.formatter
            elif model_name == self.summarizer_model_name:
                model = self.summarizer
            else:
                console.print(f"[yellow]Model {model_name} not loaded, using {self.solver_model_name} instead[/yellow]")
                model_name = self.solver_model_name
                model = self.solver
                
            console.print(f"[bold]Step {step_num}:[/bold] Solving with {model_name} model...")
            
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
            
            try:
                # Create a timeout mechanism
                response = None
                timeout_seconds = 30  # Set timeout to 30 seconds
                timed_out = False
                
                # Use a thread and event to handle timeout
                def invoke_model():
                    nonlocal response
                    try:
                        response = model.invoke(prompt)
                    except Exception as e:
                        console.print(f"[bold red]Error during model invocation: {str(e)}[/bold red]")
                
                with Progress() as progress:
                    task = progress.add_task(f"[cyan]Processing with {model_name}...", total=100)
                    
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
                    
                    if timed_out:
                        console.print(f"[bold red]Operation timed out after {timeout_seconds} seconds[/bold red]")
                        # Fall back to a more reliable model
                        fallback_model_name = "phi4"  # Phi-4 is typically more reliable
                        console.print(f"[yellow]Falling back to {fallback_model_name} model...[/yellow]")
                        
                        if hasattr(self, 'formatter') and self.formatter_model_name == fallback_model_name:
                            fallback_model = self.formatter
                        else:
                            try:
                                fallback_model = Ollama(model=fallback_model_name)
                            except Exception as e:
                                console.print(f"[bold red]Error loading fallback model: {str(e)}[/bold red]")
                                return f"Error: Operation timed out and fallback model failed to load."
                        
                        task = progress.add_task(f"[cyan]Processing with fallback {fallback_model_name}...", total=None)
                        response = fallback_model.invoke(prompt)
                        progress.update(task, completed=100)
                    else:
                        progress.update(task, completed=100)
                
                if response:
                    console.print(f"[green]✓ Solution generated with {model_name if not timed_out else fallback_model_name}[/green]")
                    return response
                else:
                    return "Error: Failed to generate a response."
                
            except Exception as e:
                console.print(f"[bold red]Error solving with {model_name}: {str(e)}[/bold red]")
                return f"Error: {str(e)}"
                
        elif model_type == "mlx":
            console.print(f"[bold]Step {step_num}:[/bold] Getting solution with AceMath model...")
            
            try:
                # Check if MLX model is loaded
                if not hasattr(self, 'math_model') or not hasattr(self, 'tokenizer'):
                    console.print("[yellow]MLX model not loaded, attempting to load now...[/yellow]")
                    self.setup_mlx_model()
                
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
                
                console.print("[green]✓ Solution generated with AceMath[/green]")
                return response
                
            except Exception as e:
                console.print(f"[bold red]Error solving with AceMath: {str(e)}[/bold red]")
                return f"Error: {str(e)}"
        
        else:
            console.print(f"[bold red]Unknown model type: {model_type}[/bold red]")
            return f"Error: Unknown model type {model_type}"

    
    def summarize_solution(self, question: str, solution: str) -> str:
        """
        Summarize the solution to be less than 512 tokens using the selected summarizer model.
        
        Args:
            question: The original question
            solution: The detailed solution
            
        Returns:
            A concise summary of the solution
        """
        console.print(f"[bold]Step 4:[/bold] Summarizing solution with {self.summarizer_model_name}...")
        
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
        
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Summarizing...", total=None)
                summary = self.summarizer.invoke(prompt)
                progress.update(task, completed=100)
            
            # Estimate token count (rough approximation: ~4 chars per token)
            estimated_tokens = len(summary) / 4
            console.print(f"[green]✓ Summary generated with {self.summarizer_model_name} (~{int(estimated_tokens)} tokens)[/green]")
            return summary
            
        except Exception as e:
            console.print(f"[bold red]Error summarizing solution with {self.summarizer_model_name}: {str(e)}[/bold red]")
            return f"Error during summarization: {str(e)}"

    
    def run_interactive(self):
        """Run the interactive command-line interface."""
        console.print(Panel.fit(
            "[bold yellow]Enhanced Math Agent Workflow[/bold yellow]\n\n"
            "This agent uses multiple LLMs in sequence to process and solve math problems:\n"
            f"- [bold]{self.formatter_model_name}[/bold]: Validates and formats your question\n"
            f"- [bold]{self.solver_model_name}[/bold]: Solves the problem with detailed explanation\n"
            "- [bold]AceMath (MLX)[/bold]: Provides an alternative solution\n"
            f"- [bold]{self.summarizer_model_name}[/bold]: Summarizes the solution to be concise\n\n"
            "Type [bold green]'exit'[/bold green] to quit or [bold green]'models'[/bold green] to change models.",
            title="Welcome",
            border_style="blue"
        ))
        
        while True:
            question = Prompt.ask("\n[bold cyan]Ask a math question[/bold cyan] (or type 'models' to change models, 'exit' to quit)")
            
            if question.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            if question.lower() == "models":
                self.change_models()
                continue
            
            console.print("\n[bold]Processing your request...[/bold]")
            
            try:
                # Step 1: Format and validate the question using the formatter model
                formatted_question, is_math, explanation = self.format_question(question)
                
                if is_math:
                    # Step 2: Solve the math problem using the primary solver model
                    detailed_solution = self.solve_problem(formatted_question, "ollama", 2, self.solver_model_name)
                    
                    # Step 3: Get alternative solution with MLX if enabled
                    if self.use_mlx:
                        acemath_solution = self.solve_problem(formatted_question, "mlx", 3)
                    else:
                        # Get alternative solution with a different Ollama model if MLX is disabled
                        alt_model = "gemma" if self.solver_model_name != "gemma" else "phi4"
                        acemath_solution = self.solve_problem(formatted_question, "ollama", 3, alt_model)
                    
                    # Step 4: Summarize the solution using the selected summarizer model
                    summary = self.summarize_solution(formatted_question, detailed_solution)
                    
                    # Create a comparison table of solutions
                    solution_table = Table(title="Math Solutions", show_header=True, header_style="bold magenta")
                    solution_table.add_column("Source", style="cyan")
                    solution_table.add_column("Solution")
                    
                    # Escape Rich markup syntax in solutions for display
                    # Replace [ with [[ to escape Rich markup syntax
                    safe_detailed = detailed_solution.replace("[", "[[").replace("]", "]]")
                    safe_acemath = acemath_solution.replace("[", "[[").replace("]", "]]")
                    safe_summary = summary.replace("[", "[[").replace("]", "]]")
                    
                    # Truncate long solutions for display
                    solution_table.add_row("solver", 
                                         safe_detailed[:500] + "..." if len(safe_detailed) > 500 else safe_detailed)
                    solution_table.add_row("AceMath (alternate)", 
                                         safe_acemath[:500] + "..." if len(safe_acemath) > 500 else safe_acemath)
                    solution_table.add_row("summarizer", safe_summary)
                    
                    # Display the results
                    console.print("\n[bold green]Results:[/bold green]")
                    console.print(Panel(
                        f"[bold]Original Question:[/bold]\n{question}\n\n"
                        f"[bold]Formatted Question:[/bold]\n{formatted_question}\n\n"
                        f"[bold]Summary Solution:[/bold]\n{safe_summary}",
                        title="Math Problem Solution",
                        border_style="green"
                    ))
                    
                    # Ask if user wants to see detailed solutions
                    if Prompt.ask("\nShow detailed solutions?", choices=["y", "n"], default="n") == "y":
                        console.print(solution_table)
                        
                else:
                    # Not a math question
                    console.print("\n[bold yellow]Not a Math Question:[/bold yellow]")
                    console.print(Panel(
                        f"[bold]Your Question:[/bold]\n{question}\n\n"
                        f"[bold]Feedback:[/bold]\n{explanation}\n\n"
                        f"Please ask a mathematical question for me to solve.",
                        title="Non-Math Question",
                        border_style="yellow"
                    ))
                
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")


    def change_models(self):
        """Allow the user to change the models used for each step."""
        console.print("\n[bold blue]Change Models[/bold blue]")
        
        # Create a table of available models
        model_table = Table(title="Available Ollama Models", show_header=True)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Description")
        
        model_table.add_row("phi4", "Microsoft's phi-4 is a compact yet powerful instruction model")
        model_table.add_row("gemma", "Google's Gemma is a lightweight LLM for various tasks")
        model_table.add_row("qwen_deepseek", "Qwen-Deepseek is optimized for complex reasoning tasks")
        model_table.add_row("llama3", "Meta's Llama 3 is a versatile general-purpose model")
        model_table.add_row("mistral", "Mistral AI's model excels at instruction following")
        
        console.print(model_table)
        
        # Get user's model choices
        formatter_choice = Prompt.ask("\nSelect formatter model", 
                                      choices=["phi4", "gemma", "qwen_deepseek", "llama3", "mistral", "skip"], 
                                      default=self.formatter_model_name)
        
        if formatter_choice != "skip":
            self.formatter_model_name = formatter_choice
        
        solver_choice = Prompt.ask("Select solver model", 
                                   choices=["phi4", "gemma", "qwen_deepseek", "llama3", "mistral", "skip"], 
                                   default=self.solver_model_name)
        
        if solver_choice != "skip":
            self.solver_model_name = solver_choice
        
        summarizer_choice = Prompt.ask("Select summarizer model", 
                                       choices=["phi4", "gemma", "qwen_deepseek", "llama3", "mistral", "skip"], 
                                       default=self.summarizer_model_name)
        
        if summarizer_choice != "skip":
            self.summarizer_model_name = summarizer_choice
            
        # Ask if user wants to use MLX
        use_mlx_choice = Prompt.ask("Use MLX for alternative solution?", 
                                   choices=["y", "n", "skip"], 
                                   default="y" if self.use_mlx else "n")
        
        if use_mlx_choice != "skip":
            self.use_mlx = (use_mlx_choice == "y")
            if self.use_mlx and not hasattr(self, 'math_model'):
                self.setup_mlx_model()
        
        # Reload models
        console.print("\n[bold]Reloading models with new selections...[/bold]")
        self.setup_ollama_models()
        
        # Update welcome message
        console.print(Panel.fit(
            "[bold yellow]Enhanced Math Agent Workflow[/bold yellow]\n\n"
            "This agent uses multiple LLMs in sequence to process and solve math problems:\n"
            f"- [bold]{self.formatter_model_name}[/bold]: Validates and formats your question\n"
            f"- [bold]{self.solver_model_name}[/bold]: Solves the problem with detailed explanation\n"
            "- [bold]AceMath (MLX)[/bold]: Provides an alternative solution\n"
            f"- [bold]{self.summarizer_model_name}[/bold]: Summarizes the solution to be concise\n\n"
            "Type [bold green]'exit'[/bold green] to quit or [bold green]'models'[/bold green] to change models.",
            title="Models Updated",
            border_style="green"
        ))


if __name__ == "__main__":
    workflow = EnhancedMathWorkflow()
    workflow.run_interactive()
