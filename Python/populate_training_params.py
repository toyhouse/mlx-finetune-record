import subprocess
import sqlite3

# Connect to SQLite database
db_connection = sqlite3.connect('./data/Tabulated/promptinglog.db')
db_cursor = db_connection.cursor()

# Get the list of existing models
try:
    list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if list_result.returncode == 0:
        model_lines = list_result.stdout.split('\n')
        models = [line.split()[0] for line in model_lines if line.strip()]
    else:
        print(f"Error retrieving model list: {list_result.stderr.strip()}")
except Exception as e:
    print(f"An error occurred while listing models: {e}")

# Drop the training_params table if it exists
drop_table_query = '''
DROP TABLE IF EXISTS training_params;
'''
db_cursor.execute(drop_table_query)

# Create the training_params table if it doesn't exist
create_table_query = '''
CREATE TABLE IF NOT EXISTS training_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT UNIQUE,
    architecture TEXT,
    parameters TEXT,
    context_length INTEGER,
    embedding_length INTEGER,
    quantization TEXT,
    stop TEXT,
    system TEXT
);
'''
db_cursor.execute(create_table_query)

# Function to populate training_params table
def populate_training_params(models):
    for model in models:
        try:
            # Execute the ollama show command
            result = subprocess.run(["ollama", "show", model], capture_output=True, text=True)
            if result.returncode == 0:
                # Print the raw output for debugging
                print(f"Raw output for model {model}:\n{result.stdout}")
                # Extract parameters from the output
                lines = result.stdout.split('\n')
                params = {
                    'architecture': '',
                    'parameters': '',
                    'context length': '',
                    'embedding length': '',
                    'quantization': '',
                    'stop': [],
                    'system': ''
                }
                for i, line in enumerate(lines):
                    line = line.strip()
                    if 'architecture' in line:
                        params['architecture'] = line.split()[-1]
                    elif 'parameters' in line:
                        params['parameters'] = line.split()[-1]
                    elif 'context length' in line:
                        params['context length'] = line.split()[-1]
                    elif 'embedding length' in line:
                        params['embedding length'] = line.split()[-1]
                    elif 'quantization' in line:
                        params['quantization'] = line.split()[-1]
                    elif 'stop' in line:
                        params['stop'].append(line.split()[-1])
                    elif line.strip() == 'System':
                        # Capture system description
                        system_description = []
                        system_index = i + 1
                        while system_index < len(lines) and lines[system_index].strip() != '':
                            system_description.append(lines[system_index].strip())
                            system_index += 1
                        params['system'] = ' '.join(system_description)
                # Insert parameters into the database
                insert_query = '''
                INSERT INTO training_params (model_name, architecture, parameters, context_length, embedding_length, quantization, stop, system)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                db_cursor.execute(insert_query, (
                    model,
                    params['architecture'],
                    params['parameters'],
                    params['context length'],
                    params['embedding length'],
                    params['quantization'],
                    ', '.join(params['stop']),
                    params['system']
                ))
                db_connection.commit()
                print(f"Populated training_params for model: {model}")
            else:
                print(f"Error retrieving parameters for model {model}: {result.stderr.strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the population function
populate_training_params(models)

# Close the database connection
db_connection.close()
