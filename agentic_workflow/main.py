"""
Main entry point for the agentic math workflow.
"""

from .workflow import MathWorkflow

def main():
    """
    Main function to run the agentic math workflow.
    """
    # Create the workflow with default models
    workflow = MathWorkflow()
    
    # Run the interactive interface
    workflow.run_interactive()

if __name__ == "__main__":
    main()
