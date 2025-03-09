# Solver Pattern

## Intent
The Solver Pattern encapsulates complex problem-solving logic within a dedicated component, providing a systematic approach to generate solutions with detailed explanations. It serves as the computational core that transforms well-defined problems into comprehensive solutions.

## Also Known As
- Resolution Engine
- Solution Generator
- Computational Core
- Problem Processor

## Motivation
In AI-driven systems that tackle domain-specific problems (such as mathematics), there's a need for a dedicated component that handles the actual problem-solving with clarity and detail. This component should:

1. Focus solely on solution generation with clear reasoning
2. Provide step-by-step explanations
3. Verify the correctness of solutions
4. Handle computational timeouts gracefully
5. Ensure reliability through fallback mechanisms

Consider a mathematical problem-solving system. After a problem has been validated and formatted, the system needs a specialized component that can apply domain knowledge, computational techniques, and reasoning to produce not just an answer, but a full explanation of how the answer was derived.

The Solver Pattern implements this core processing stage, ensuring that problems are solved methodically and explanations are comprehensive.

## Applicability
Use the Solver Pattern when:
- You need to separate complex problem-solving logic from input processing and output formatting
- Solutions require detailed, step-by-step explanations
- The problem domain requires specialized reasoning techniques
- You need to handle computational timeouts gracefully
- You want to encapsulate the core "thinking" component of your system
- Multiple fallback strategies may be needed for reliability

## Structure
![Solver Pattern Structure](https://www.example.com/solver_pattern.png)

## Participants
- **Client**: Initiates the solving process by providing a well-formed problem
- **SolverAgent**: Implements the solution generation logic
- **Model/LLM**: The underlying language model or processing engine that performs the actual computation
- **Workflow**: Coordinates the overall process and directs the solution to subsequent processing stages

## Collaborations
1. The **Client** (or upstream component like a Formatter) submits a well-formed problem to the **SolverAgent**
2. The **SolverAgent** constructs a prompt that incorporates the problem and solution criteria
3. The **SolverAgent** passes the prompt to the **Model/LLM**
4. The **Model/LLM** processes the prompt and returns a detailed solution
5. If the primary model fails or times out, the **SolverAgent** invokes a fallback model
6. The solution is returned to the **Workflow** for further handling

## Implementation
The Solver Pattern is implemented in the provided `SolverAgent` class. The key elements include:

### The Prompt Template
The prompt is the critical component that instructs the language model how to generate a solution:

```python
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
```

This prompt includes:
1. Clear task instructions
2. The problem statement
3. An explicit structure for the solution process
4. Guidance on explanation quality and verification

### Required Properties for Implementation
To implement the Solver Pattern effectively, the following properties are necessary:

1. **Clear Solution Structure**: The prompt must guide the model to produce solutions with a consistent, logical flow.

2. **Explicit Solution Criteria**: The implementation should specify what constitutes a good solution (step-by-step reasoning, verification, etc.).

3. **Timeout Handling**: As computational problems may be complex, the implementation must handle timeouts gracefully.

4. **Fallback Mechanisms**: When the primary solution approach fails, alternative strategies should be available.

5. **Model Selection Logic**: The pattern should incorporate knowledge about which models are best suited for which types of problems.

6. **Progress Monitoring**: For longer computations, the implementation should provide progress feedback.

7. **Verification Guidance**: The prompt should explicitly request verification steps to ensure solution correctness.

8. **Educational Value**: The solution should not only be correct but also instructive and explanatory.

## Sample Code
```python
def process(self, problem: str) -> str:
    """
    Solve a math problem with detailed explanation.
    
    Args:
        problem: The formatted math problem to solve
        
    Returns:
        The solution with explanation
    """
    console.print(f"[bold]Step 2:[/bold] Solving with {self.model_name} model...")
    
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
    
    response, timed_out = self.invoke_with_timeout(prompt)
    
    if timed_out:
        console.print(f"[bold red]Operation timed out[/bold red]")
        # Fall back to a more reliable model
        fallback_model_name = "phi4"  # Phi-4 is typically more reliable
        console.print(f"[yellow]Falling back to {fallback_model_name} model...[/yellow]")
        
        try:
            fallback_model = Ollama(model=fallback_model_name)
            with Progress() as progress:
                task = progress.add_task(f"[cyan]Processing with fallback {fallback_model_name}...", total=None)
                response = fallback_model.invoke(prompt)
                progress.update(task, completed=100)
            
            console.print(f"[green]✓ Solution generated with fallback model {fallback_model_name}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error with fallback model: {str(e)}[/bold red]")
            return f"Error: Operation timed out and fallback model failed."
    else:
        console.print(f"[green]✓ Solution generated with {self.model_name}[/green]")
        
    return response
```

## Known Uses
- Mathematical problem-solving systems
- Automated tutoring and educational platforms
- Scientific computing and analysis tools
- Expert systems in specialized domains
- Automated reasoning engines
- Proof generation in formal verification systems

## Related Patterns
- **Strategy Pattern**: The solver can implement different solution strategies based on problem type
- **Template Method**: The solving process follows a template with defined steps
- **Chain of Responsibility**: Multiple solvers can be chained to attempt different solution approaches
- **Command Pattern**: The problem and solution approach can be encapsulated as commands
- **Decorator**: Additional verification or explanation capabilities can be added through decoration

## Consequences
### Benefits
1. Centralizes complex solution logic in a dedicated component
2. Provides detailed, educational explanations with solutions
3. Handles computational challenges gracefully through timeouts and fallbacks
4. Separates the "thinking" process from input and output formatting
5. Allows for specialized reasoning adapted to the problem domain

### Liabilities
1. May require significant computational resources for complex problems
2. Solution quality depends heavily on prompt engineering and model capabilities
3. Timeouts may still occur even with fallback mechanisms
4. Can be challenging to debug why certain solutions are incorrect
5. May require domain-specific customization for different problem types
