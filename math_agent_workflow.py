#!/usr/bin/env python3
"""
Math Agent Workflow
------------------
An agentic workflow that processes math questions through multiple LLMs:
1. Uses Ollama to validate and format the question
2. Uses MLX-based AceMath model to solve the problem
3. Provides an interactive command-line interface

This workflow demonstrates how to combine multiple LLM backends in a single application.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress

# LangChain imports
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import our MLX model
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

console = Console()

class MathWorkflow:
    """
    Orchestrates the math problem solving workflow using multiple LLMs.
    """
    
    def __init__(self):
        """Initialize the workflow components."""
        self.setup_ollama_model()
        self.setup_mlx_model()
        
    def setup_ollama_model(self):
        """Set up the Ollama model for question formatting."""
        console.print("[bold blue]Setting up Ollama model...[/bold blue]")
        # Initialize Ollama model - phi4 should be good for instruction following
        self.ollama = Ollama(
            model="phi4"
        )
        console.print("[bold green]✓ Ollama model ready![/bold green]")
    
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
    
    def format_question(self, question: str) -> str:
        """
        Format and validate the math question using Ollama.
        
        Args:
            question: The user's math question
            
        Returns:
            A well-formatted version of the question
        """
        console.print("[bold]Step 1:[/bold] Validating and formatting your question...")
        
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
            response = self.ollama.invoke(prompt)
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
                console.print("[yellow]! Could not extract JSON, using Ollama's full response[/yellow]")
                return question, True, response
                
        except Exception as e:
            console.print(f"[bold red]Error formatting question: {str(e)}[/bold red]")
            return question, True, f"Error during validation: {str(e)}"
    
    def solve_math_problem(self, problem: str) -> str:
        """
        Solve a math problem using the MLX-based AceMath model.
        
        Args:
            problem: The formatted math problem to solve
            
        Returns:
            The solution to the math problem
        """
        console.print("[bold]Step 2:[/bold] Solving with AceMath model...")
        
        try:
            # Format prompt for instruction model
            formatted_prompt = f"[INST] {problem} [/INST]"
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=None)
                
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
            
            return response
            
        except Exception as e:
            console.print(f"[bold red]Error solving math problem: {str(e)}[/bold red]")
            return f"Error: {str(e)}"
    
    def run_interactive(self):
        """Run the interactive command-line interface."""
        console.print(Panel.fit(
            "[bold yellow]Math Agent Workflow[/bold yellow]\n\n"
            "This agent uses multiple LLMs to process and solve math problems:\n"
            "- [bold]Ollama[/bold]: Validates and formats your question\n"
            "- [bold]MLX AceMath[/bold]: Solves the mathematical problem\n\n"
            "Type [bold green]'exit'[/bold green] to quit.",
            title="Welcome",
            border_style="blue"
        ))
        
        while True:
            question = Prompt.ask("\n[bold cyan]Ask a math question[/bold cyan]")
            
            if question.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            console.print("\n[bold]Processing your request...[/bold]")
            
            try:
                # Step 1: Format and validate the question using Ollama
                formatted_question, is_math, explanation = self.format_question(question)
                
                if is_math:
                    # Step: 2: Solve the math problem using MLX
                    solution = self.solve_math_problem(formatted_question)
                    
                    # Display the results
                    console.print("\n[bold green]Results:[/bold green]")
                    console.print(Panel(
                        f"[bold]Original Question:[/bold]\n{question}\n\n"
                        f"[bold]Formatted Question:[/bold]\n{formatted_question}\n\n"
                        f"[bold]Solution:[/bold]\n{solution}",
                        title="Math Problem Solution",
                        border_style="green"
                    ))
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


if __name__ == "__main__":
    workflow = MathWorkflow()
    workflow.run_interactive()
