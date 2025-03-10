---
language: 
  - en
tags:
  - ollama
  - gguf
  - language-model
  - small-model
  - english-teacher
license: apache-2.0
model-index:
  - name: Small Indo Teacher GGUF
    results:
      - task: 
          name: Language Teaching
          type: text-generation
        metrics:
          - type: Engagement
            value: High
            mode: qualitative

# Small Indo Teacher - GGUF Model

## Model Details
- **Base Model**: Qwen/Qwen1.5-1.8B-Chat
- **Format**: GGUF (Ollama-compatible)
- **Training**: Fine-tuned on video transcripts
- **Purpose**: English language teaching

## Usage with Ollama
```bash
ollama run small_indo_teacher
```

### Python Example
```python
import ollama

response = ollama.chat(model='small_indo_teacher', messages=[
    {
        'role': 'user',
        'content': 'Can you help me improve my English grammar?'
    }
])
print(response['message']['content'])
```

## Model Capabilities
- Provide clear language explanations
- Encourage language practice
- Offer constructive feedback
- Engage students interactively

## Limitations
- Best used for language learning and practice
- Performance may vary based on specific language contexts

## License
Apache 2.0