from datasets import load_dataset
import json
import os

def convert_dataset_to_jsonl():
    """Convert HuggingFace dataset to JSONL format"""
    dataset_dir = os.path.join(os.getcwd(), "data", "s1k_prob")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("simplescaling/s1K-1.1")
    train_data = dataset["train"]
    
    # Print first example to verify structure
    print("\nExample data structure:")
    first_example = train_data[0]
    print("Type:", type(first_example))
    print("Keys:", train_data.features)
    print("Raw example:", first_example)
    print("JSON dump:", json.dumps(first_example, indent=2))
    
    # Create train.jsonl
    train_file = os.path.join(dataset_dir, "train.jsonl")
    print(f"\nConverting training set ({len(train_data)} examples)")
    
    # Split data into train and validation
    valid_size = len(train_data) // 5  # 20% for validation
    train_examples = train_data.select(range(len(train_data) - valid_size))  # First 80%
    valid_examples = train_data.select(range(len(train_data) - valid_size, len(train_data)))  # Last 20%
    
    print(f"Train size: {len(train_examples)}, Valid size: {len(valid_examples)}")
    
    def process_example(example):
        # Escape LaTeX backslashes and handle math expressions
        question = example['question'].replace('\\', '\\\\')
        solution = example['solution'].replace('\\', '\\\\')
        
        return {
            "text": question,
            "completion": solution
        }
    
    with open(train_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(train_examples, 1):
            entry = process_example(example)
            f.write(json.dumps(entry, ensure_ascii=False).strip() + "\n")
            if i % 100 == 0:
                print(f"Processed {i} examples...")
    
    # Create valid.jsonl
    valid_file = os.path.join(dataset_dir, "valid.jsonl")
    print(f"\nCreating validation set ({len(valid_examples)} examples)")
    
    with open(valid_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(valid_examples, 1):
            entry = process_example(example)
            f.write(json.dumps(entry, ensure_ascii=False).strip() + "\n")
            if i % 100 == 0:
                print(f"Processed {i} validation examples...")
    
    # Create empty test.jsonl as required by MLX
    test_file = os.path.join(dataset_dir, "test.jsonl")
    with open(test_file, "w", encoding="utf-8") as f:
        # Write at least one example to avoid empty dataset error
        if len(valid_examples) > 0:
            entry = process_example(valid_examples[0])
            f.write(json.dumps(entry, ensure_ascii=False).strip() + "\n")
    
    # Verify files
    train_size = os.path.getsize(train_file)
    valid_size = os.path.getsize(valid_file)
    test_size = os.path.getsize(test_file)
    
    print("\nDataset creation complete:")
    print(f"Train file: {train_file} ({train_size:,} bytes)")
    print(f"Valid file: {valid_file} ({valid_size:,} bytes)")
    print(f"Test file: {test_file} ({test_size:,} bytes)")
    
    # Validate JSON format
    print("\nValidating JSON format...")
    def validate_jsonl(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Error in {filepath}, line {i}: {e}")
                    return False
        return True
    
    if validate_jsonl(train_file) and validate_jsonl(valid_file) and validate_jsonl(test_file):
        print("All files are valid JSONL")
        print("\nFirst few examples from train:")
        with open(train_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"Example {i+1}:", line.strip())

if __name__ == "__main__":
    convert_dataset_to_jsonl()