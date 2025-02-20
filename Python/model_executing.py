import subprocess
import json
import logging
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(filename='./data/logs/model_testing_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a console handler to also output to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Connect to SQLite database (it will be created if it doesn't exist)
db_connection = sqlite3.connect('./data/Tabulated/promptinglog.db')
db_cursor = db_connection.cursor()

# Create a table for logging if it doesn't exist
create_table_query = '''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model_name TEXT,
    input TEXT,
    output TEXT
)
'''
db_cursor.execute(create_table_query)

# List of trained models to test
models = [
    "deepseek-r1:8b",
    "deepseek-r1:32b",
    "deepseek-r1:70b"
]

# Load test cases from JSON file
with open('data/Prompts/tirekicking.json', 'r') as file:
    test_cases = json.load(file)

# Function to log to SQLite database
def log_to_db(timestamp, model_name, input_text, output_text):
    insert_query = '''
    INSERT INTO logs (timestamp, model_name, input, output)
    VALUES (?, ?, ?, ?)
    '''
    db_cursor.execute(insert_query, (timestamp, model_name, input_text, output_text))
    db_connection.commit()

# Function to test models
def test_models(models, test_cases):
    for model in models:
        logging.info(f"Testing model: {model}")
        for test in test_cases:
            try:
                result = subprocess.run(["ollama", "run", model, test], capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    output = result.stdout.strip()
                    logging.info(f"Model: {model}, Input: {test}, Output: {output}")
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_to_db(timestamp, model, test, output)
                else:
                    error_message = result.stderr.strip()
                    logging.error(f"Model: {model}, Input: {test}, Error: {error_message}")
            except subprocess.TimeoutExpired:
                logging.warning(f"Model: {model}, Input: {test}, Status: Timeout expired.")

# Close the database connection when done
# db_connection.close()  # Uncomment this line when you want to close the connection

if __name__ == '__main__':
    test_models(models, test_cases)