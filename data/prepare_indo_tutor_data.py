import json
import random
import os

# Source files
source_files = [
    "/Users/Henrykoo/Documents/mlx-finetune-record/data/Instruktur Gasing/pengurangan/pengurangan_grouped.jsonl",
    "/Users/Henrykoo/Documents/mlx-finetune-record/data/Instruktur Gasing/penjumlahan/penjumlahan_grouped.jsonl",
    "/Users/Henrykoo/Documents/mlx-finetune-record/data/Instruktur Gasing/perkalian/perkalian_grouped.jsonl"
]

# Output files
train_file = "/Users/Henrykoo/Documents/mlx-finetune-record/data/indo_tutor/train.jsonl"
valid_file = "/Users/Henrykoo/Documents/mlx-finetune-record/data/indo_tutor/valid.jsonl"
test_file = "/Users/Henrykoo/Documents/mlx-finetune-record/data/indo_tutor/test.jsonl"

# Read all data from source files
all_data = []
for file_path in source_files:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "text" in record:
                            all_data.append(record)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {file_path}")
    else:
        print(f"Warning: File not found: {file_path}")

# Shuffle the data
random.seed(42)  # For reproducibility
random.shuffle(all_data)

# Split the data (80% train, 10% valid, 10% test)
total = len(all_data)
train_count = int(total * 0.8)
valid_count = int(total * 0.1)

train_data = all_data[:train_count]
valid_data = all_data[train_count:train_count + valid_count]
test_data = all_data[train_count + valid_count:]

# Write the data to the output files
def write_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for record in data:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

write_data(train_file, train_data)
write_data(valid_file, valid_data)
write_data(test_file, test_data)

print(f"Data split complete:")
print(f"Total records: {total}")
print(f"Train records: {len(train_data)}")
print(f"Validation records: {len(valid_data)}")
print(f"Test records: {len(test_data)}")
