# Agentic Math Workflow

A multi-agent system for solving mathematical problems.

## How It Works

The Agentic Math Workflow processes math problems through a sequence of specialized agents:

1. **Formatter Agent**: Cleans and standardizes the input problem
2. **Solver Agent**: Applies mathematical reasoning to solve the problem
3. **AceMath Agent**: Provides alternative solutions (uses Indo Math Teacher model)
4. **Summarizer Agent**: Creates a concise explanation of the solution

Each agent is powered by a different language model optimized for its specific task, creating a more effective system than using a single model.

## How to Use

### Running the Interactive Interface

The simplest way to use the system is through the command-line interface:

```bash
python -m agentic_workflow.main
```

This will:
- Start an interactive session where you can type math problems
- Process each problem through all agents
- Display the formatted problem, solution steps, and summary
- Allow you to change models between questions

### Using in Your Code

You can also import and use the workflow programmatically:

```python
from agentic_workflow.workflow import MathWorkflow

# Create the workflow
workflow = MathWorkflow()

# Process a math question
results = workflow.process_question("Solve for x: 2x + 5 = 15")

# Access different parts of the result
print(f"Formatted question: {results['formatted_question']}")
print(f"Primary solution: {results['primary_solution']}")
print(f"Summary: {results['summary']}")
```

### Changing Models

You can configure which models to use for each agent:

```python
workflow = MathWorkflow(
    formatter_model="phi4",      # For input formatting
    solver_model="qwen_deepseek", # Primary problem solver  
    summarizer_model="phi4",     # For creating summaries
    use_mlx=True                 # Enable MLX-based AceMath agent
)
```

Or change models during an interactive session by selecting the "Change models" option.
