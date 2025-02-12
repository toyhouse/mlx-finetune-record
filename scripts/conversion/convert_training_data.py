#!/usr/bin/env python3
import json
import os
import random

def convert_messages_to_text(messages):
    text = ""
    
    for msg in messages:
        if msg["from"] == "system":
            text += msg["value"] + "\n\n"
        elif msg["from"] == "human":
            text += "Human: " + msg["value"] + "\n"
        elif msg["from"] == "gpt":
            text += "Assistant: " + msg["value"] + "\n\n"
    
    return {
        "text": text
    }

def process_jsonl_file(input_file, train_file, valid_file, valid_ratio=0.1):
    # Read all data
    data = []
    with open(input_file, 'r') as f_in:
        for line in f_in:
            item = json.loads(line)
            converted = convert_messages_to_text(item["messages"])
            data.append(converted)
    
    # Shuffle and split
    random.shuffle(data)
    valid_size = int(len(data) * valid_ratio)
    valid_data = data[:valid_size]
    train_data = data[valid_size:]
    
    # Write train data
    with open(train_file, 'w') as f_out:
        for item in train_data:
            json.dump(item, f_out)
            f_out.write('\n')
    
    # Write validation data
    with open(valid_file, 'w') as f_out:
        for item in valid_data:
            json.dump(item, f_out)
            f_out.write('\n')

def main():
    input_dir = "./jsonl/GASING"
    output_dir = "./jsonl/GASING_converted"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert train.jsonl and create validation set
    if os.path.exists(os.path.join(input_dir, "train.jsonl")):
        process_jsonl_file(
            os.path.join(input_dir, "train.jsonl"),
            os.path.join(output_dir, "train.jsonl"),
            os.path.join(output_dir, "valid.jsonl")
        )
    
    # Convert test.jsonl if it exists
    if os.path.exists(os.path.join(input_dir, "test.jsonl")):
        process_jsonl_file(
            os.path.join(input_dir, "test.jsonl"),
            os.path.join(output_dir, "test.jsonl"),
            os.path.join(output_dir, "test_valid.jsonl")
        )

if __name__ == "__main__":
    main()
