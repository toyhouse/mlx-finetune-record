import json
import random
import argparse

def merge_jsonl(file1, file2, output_file):
    data = []
    count_f1 = 0
    count_f2 = 0

    # Read from the first file
    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                    count_f1 += 1
                except json.JSONDecodeError:
                    pass  # Skip malformed lines

    # Read from the second file
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                    count_f2 += 1
                except json.JSONDecodeError:
                    pass  # Skip malformed lines

    print(f"Read {count_f1} valid lines from {file1}")
    print(f"Read {count_f2} valid lines from {file2}")
    print(f"Total lines combined: {len(data)}")

    # Randomise
    random.shuffle(data)

    # Write to the output file
    with open(output_file, 'w', encoding='utf-8') as out:
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge two JSONL files and randomise their contents.")
    parser.add_argument('jsonl_file1', help="Path to the first JSONL file")
    parser.add_argument('jsonl_file2', help="Path to the second JSONL file")
    parser.add_argument('output_file', help="Path to the merged output JSONL file")
    args = parser.parse_args()

    merge_jsonl(args.jsonl_file1, args.jsonl_file2, args.output_file)
