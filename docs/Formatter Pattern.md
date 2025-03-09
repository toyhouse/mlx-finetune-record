# Formatter Pattern

## Intent
The Formatter Pattern validates, normalizes, and transforms input data into a standardized format suitable for downstream processing. It serves as a gatekeeper that ensures data meets specific criteria before further processing occurs.

## Also Known As
- Input Validator
- Data Normalizer
- Preprocessing Filter

## Motivation
In complex workflows, especially those involving natural language processing or mathematical problem-solving, raw input often requires preprocessing. This preprocessing ensures that:
1. The input is relevant to the task domain
2. The format is standardized for downstream processors
3. Invalid inputs are filtered out early in the pipeline

Consider an AI system designed to solve mathematical problems. Before attempting to solve a problem, the system must verify that:
- The input is indeed a mathematical question
- The question is structured clearly and unambiguously
- Any notation or formatting is standardized for the solver component

The Formatter Pattern implements this preprocessing stage, ensuring that only valid, well-formatted inputs proceed to more computationally expensive processing stages.

## Applicability
Use the Formatter Pattern when:
- Input quality varies significantly and requires normalization
- You need to validate that input belongs to a specific domain before processing
- Downstream components expect data in a standardized format
- Early filtering can prevent wasting resources on invalid inputs
- You want to separate input validation and normalization concerns from core processing logic

## Structure
![Formatter Pattern Structure](https://www.example.com/formatter_pattern.png)

## Participants
- **Client**: Initiates the formatting process by providing raw input data
- **FormatterAgent**: Implements the validation and formatting logic
- **Model/LLM**: The underlying language model or processing engine that performs the actual formatting
- **Workflow**: Coordinates the overall process and directs the formatted output to subsequent processing stages

## Collaborations
1. The **Client** submits raw input to the **FormatterAgent**
2. The **FormatterAgent** constructs a prompt that incorporates the input and instructions
3. The **FormatterAgent** passes the prompt to the **Model/LLM**
4. The **Model/LLM** processes the prompt and returns a response
5. The **FormatterAgent** parses the response into structured data
6. The processed data is returned to the **Workflow** for further handling

## Implementation
The Formatter Pattern is implemented in the provided `FormatterAgent` class. The key elements include:

### The Prompt Template
The prompt is the critical component that instructs the language model how to process the input:

```python
prompt = f"""
Your task is to determine if this is a math problem, and reformat it for better clarity if needed.

QUESTION: {question}

Respond strictly in the following format:
IS_MATH: [Yes/No]

FORMATTED_QUESTION: [The reformatted question, with better structure if needed]

EXPLANATION: [Brief explanation of your determination, and what changes you made to formatting if any]
"""
```

This prompt includes:
1. Clear task instructions
2. The input data
3. A strict output format specification

### Required Properties for Implementation
To implement the Formatter Pattern effectively, the following properties are necessary:

1. **Deterministic Output Structure**: The prompt must specify a precise output format that can be reliably parsed.

2. **Error Handling**: The implementation must handle failures gracefully, such as when:
   - The model fails to respond
   - The response doesn't follow the expected format
   - The response times out

3. **Default Behavior**: When validation fails, the system should have sensible defaults (e.g., proceeding with the original input).

4. **Clear Success/Failure Signals**: The formatter should provide clear signals about whether the input passed validation.

5. **Explanation Capability**: The formatter should explain its decisions, especially rejections, to help users understand the requirements.

6. **Timeout Mechanism**: As seen in the implementation, a timeout ensures the formatting step doesn't become a bottleneck.

## Sample Code
```python
def process(self, question: str) -> Tuple[str, bool, str]:
    """
    Format and validate a math question.
    
    Args:
        question: The original math question
        
    Returns:
        A tuple of (formatted_question, is_math, explanation)
    """
    console.print("[bold]Step 1:[/bold] Formatting and validating your question...")
    
    prompt = f"""
    Your task is to determine if this is a math problem, and reformat it for better clarity if needed.
    
    QUESTION: {question}
    
    Respond strictly in the following format:
    IS_MATH: [Yes/No]
    
    FORMATTED_QUESTION: [The reformatted question, with better structure if needed]
    
    EXPLANATION: [Brief explanation of your determination, and what changes you made to formatting if any]
    """
    
    try:
        response, timed_out = self.invoke_with_timeout(prompt)
        
        if timed_out:
            console.print("[yellow]Validation timed out, proceeding with original question[/yellow]")
            return question, True, "Validation timed out, proceeding as if this is a math question."

        # Parse the response
        is_math = False
        formatted_question = question
        explanation = "Could not parse the response properly."
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("IS_MATH:"):
                is_math = "yes" in line.lower()
            elif line.startswith("FORMATTED_QUESTION:"):
                formatted_question = line[len("FORMATTED_QUESTION:"):].strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line[len("EXPLANATION:"):].strip()
        
        if not formatted_question:
            formatted_question = question
        
        # Return the formatted results
        return formatted_question, is_math, explanation
        
    except Exception as e:
        # Handle exceptions gracefully
        return question, True, f"Error during validation: {str(e)}"
```

## Known Uses
- Input validation and normalization in natural language processing pipelines
- Data cleansing in ETL (Extract, Transform, Load) processes
- Query normalization in database systems
- Input filtering in expert systems or domain-specific problem solvers
- Standardizing notation in mathematical or scientific computing systems

## Related Patterns
- **Chain of Responsibility**: Similar to how formatters can be chained together to apply multiple transformations
- **Facade**: The formatter can act as a facade, hiding the complexity of the validation and normalization process
- **Decorator**: Formatting functionality can be added to existing objects through decoration
- **Adapter**: Transforms input from one format to another compatible format

## Consequences
### Benefits
1. Improves the quality of data entering the system
2. Centralizes input validation and normalization logic
3. Reduces errors in downstream processing
4. Provides clear feedback on input quality
5. Separates concerns between data preparation and processing

### Liabilities
1. Adds complexity and an additional processing step
2. Can become a bottleneck if not implemented efficiently
3. May require fine-tuning of prompts as input patterns evolve
4. Potential for false positives/negatives in validation
