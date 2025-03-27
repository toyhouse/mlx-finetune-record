"""
Main entry point for the agentic math workflow.
"""

import argparse
from agentic_workflow.workflow import MathWorkflow

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agentic Math Workflow")
    
    # Model arguments
    parser.add_argument("--formatter_model", type=str, default="qwen:1.8b",
                        help="Model for formatting questions (default: qwen:1.8b)")
    parser.add_argument("--solver_model", type=str, default="qwen:1.8b",
                        help="Primary model for solving problems (default: qwen:1.8b)")
    parser.add_argument("--summarizer_model", type=str, default="deepseek-r1:1.5b",
                        help="Model for summarizing solutions (default: deepseek-r1:1.5b)")
    parser.add_argument("--use_mlx", action="store_true", default=True,
                        help="Enable MLX-based AceMath agent (default: True)")
    parser.add_argument("--no_mlx", action="store_false", dest="use_mlx",
                        help="Disable MLX-based AceMath agent")
    
    # Execution mode
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Run in interactive mode (default)")
    parser.add_argument("--question", type=str,
                        help="Process a single question and exit")
    parser.add_argument("--output", type=str,
                        help="Output file for results (when used with --question)")
    
    return parser.parse_args()

def main():
    """
    Main function to run the agentic math workflow.
    """
    args = parse_args()
    
    # Create the workflow with specified models
    workflow = MathWorkflow(
        formatter_model=args.formatter_model,
        solver_model=args.solver_model,
        summarizer_model=args.summarizer_model,
        use_mlx=args.use_mlx
    )
    
    # Process a single question if provided
    if args.question:
        from rich.console import Console
        from rich.markdown import Markdown
        
        console = Console()
        results = workflow.process_question(args.question)
        
        console.print(f"Question: {args.question}\n")
        console.print(f"Formatted question: {results['formatted_question']}\n")
        console.print(Markdown(f"**Primary Solution:**\n{results['primary_solution']}\n"))
        console.print(Markdown(f"**Summary:**\n{results['summary']}\n"))
        
        # Write to output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"# Question\n{args.question}\n\n")
                f.write(f"# Formatted Question\n{results['formatted_question']}\n\n")
                f.write(f"# Primary Solution\n{results['primary_solution']}\n\n")
                f.write(f"# Summary\n{results['summary']}\n")
    else:
        # Run the interactive interface
        workflow.run_interactive()

if __name__ == "__main__":
    main()
