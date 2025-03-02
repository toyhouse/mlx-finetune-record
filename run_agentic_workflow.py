#!/usr/bin/env python3
"""
Entry point script to run the agentic math workflow.
"""

from agentic_workflow import MathWorkflow

if __name__ == "__main__":
    # Create the workflow with default models
    # You can customize the models used in each step:
    # workflow = MathWorkflow(
    #     formatter_model="phi4",
    #     solver_model="qwen_deepseek",
    #     summarizer_model="phi4",
    #     use_mlx=True
    # )
    workflow = MathWorkflow()
    
    # Run the interactive interface
    workflow.run_interactive()
