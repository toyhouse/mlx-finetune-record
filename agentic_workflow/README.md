# Agentic Math Workflow

A multi-agent system for solving mathematical problems, including integration with the Indo Math Teacher model.

## How It Works

The Agentic Math Workflow processes math problems through a sequence of specialized agents:

1. **Formatter Agent**: Cleans and standardizes the input problem
2. **Solver Agent**: Applies mathematical reasoning to solve the problem
3. **AceMath Agent**: Provides alternative solutions (uses MLX acceleration)
4. **Summarizer Agent**: Creates a concise explanation of the solution

Each agent is powered by a different language model optimized for its specific task, creating a more effective system than using a single model.

## The Indo Math Teacher Integration

The workflow can use the Indo Math Teacher model, which is specialized for teaching mathematics using the Gasing method in Bahasa Indonesia. This model:

- Provides step-by-step explanations in a conversational teaching style
- Uses visual and hands-on methods to explain concepts
- Excels in arithmetic operations, multiplication techniques, and problem-solving strategies
- Creates explanations focused on building mathematical understanding

The Indo Math Teacher model enhances the workflow by providing culturally relevant explanations using the Gasing method, which makes mathematics more accessible and engaging.

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

### Command-line Arguments

You can specify which models to use directly from the command line:

```bash
# With virtual environment activated:
python -m agentic_workflow.main \
    --formatter_model=qwen:1.8b \
    --solver_model=qwen:1.8b \
    --summarizer_model=deepseek-r1:1.5b \
    --use_mlx
```

You can also process a single question without entering interactive mode:

```bash
python -m agentic_workflow.main \
    --question="Calculate 3^3" \
    --output=results/solution.md
```

Available command-line options:
- `--formatter_model`: Model for formatting questions (default: qwen:1.8b)
- `--solver_model`: Primary model for solving problems (default: qwen:1.8b)
- `--summarizer_model`: Model for summarizing solutions (default: deepseek-r1:1.5b)
- `--use_mlx`: Enable MLX-based AceMath agent (default: True)
- `--no_mlx`: Disable MLX-based AceMath agent
- `--question`: Process a single question and exit
- `--output`: Output file for results (when used with --question)

### Automatic Model Pulling

When using the default models (`qwen:1.8b`, `deepseek-r1:1.5b`), the workflow will automatically attempt to pull these models using Ollama. This ensures you have the latest version of the model without manual intervention.

If a model cannot be pulled (e.g., no internet connection), the workflow will continue with a warning message.

### Troubleshooting

If you encounter issues with the Python command not being found, use the full path to your virtual environment's Python:

```bash
/path/to/venv/bin/python -m agentic_workflow.main \
    --formatter_model=qwen:1.8b \
    --solver_model=qwen:1.8b \
    --summarizer_model=deepseek-r1:1.5b
```

For example:
```bash
/Users/Henrykoo/Documents/mlx-finetune-record/.venv/bin/python -m agentic_workflow.main
```

If your question isn't recognized as a math question, try rephrasing it:
- Instead of "what is 8*8", use "Calculate 8*8" or "8*8=?"
- Instead of "3^4", use "Calculate 3^4" or "3 to the power of 4"

### Using in Your Code

You can also import and use the workflow programmatically:

```python
from agentic_workflow.workflow import MathWorkflow

# Create the workflow with default models
workflow = MathWorkflow()

# Or customize models
workflow = MathWorkflow(
    formatter_model="qwen:1.8b",            # For input formatting
    solver_model="qwen:1.8b",               # Primary problem solver  
    summarizer_model="deepseek-r1:1.5b",    # For creating summaries
    use_mlx=True                            # Enable MLX-based AceMath agent
)

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
    formatter_model="qwen:1.8b",            # For input formatting
    solver_model="qwen:1.8b",               # Primary problem solver  
    summarizer_model="deepseek-r1:1.5b",    # For creating summaries
    use_mlx=True                            # Enable MLX-based AceMath agent
)
```

Or change models during an interactive session by typing "models" at the prompt.

## Model Recommendations

- **Formatter**: `qwen:1.8b` or `phi3` work well for parsing and formatting questions
- **Solver**: `qwen:1.8b` for general math, `indo_math_teacher` for Indonesian math explanations
- **Summarizer**: `deepseek-r1:1.5b` or `llama2` for concise summaries

## Requirements

- Python 3.8 or higher
- MLX framework
- Ollama for local model inference
- Rich for interactive display

## Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama server:
   ```bash
   ollama serve
   ```

4. Run the workflow:
   ```bash
   python -m agentic_workflow.main
   ```
