

python merge_jsonl.py ./json/calculator/train.jsonl ./json/general/train.jsonl ./json/test/train.jsonl

## Qwen2.5-Coder-7B-Instruct

### Fine Tuning

#### Full Fine Tune
```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
```

### Lora Fine Tune

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
```

### Testing the Adapters

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "what's 254-7?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "who is ada lovelace?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" --max-tokens 500 --adapter-path adapters --prompt "which number is bigger 3.9 or 3.11?"
```

### No Adapters

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "could you add 2665 to 1447?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "what's 254-7?"
```

```bash
python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
                --max-tokens 500 \
                --prompt "who is ada lovelace?"
```

## Qwen2.5-Coder-7B-Instruct

### Fine Tuning

#### Full Fine Tune
```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type full
```

### Lora Fine Tune

```bash
mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --train \
    --data "./jsonl/calculator-non-diverse" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
```


mlx_lm.lora \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --train \
    --data "./json/test" \
    --num-layers 4 \
    --learning-rate 1e-5 \
    --iters 100 \
    --fine-tune-type lora
    

python -m mlx_lm.generate --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
                --max-tokens 500 \
               --adapter-path adapters \
               --prompt "What is 990 * 75 + 12?"