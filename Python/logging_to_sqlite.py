import sqlite3

# Connect to SQLite database (it will be created if it doesn't exist)
db_connection = sqlite3.connect('./data/Tabulated/promptinglog.db')
db_cursor = db_connection.cursor()

# Create tables for logging if they don't exist
create_tables_query = '''
CREATE TABLE IF NOT EXISTS training_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model_name TEXT UNIQUE,
    base_model_path TEXT,
    data_path TEXT,
    batch_size INTEGER,
    training_iterations INTEGER,
    adapter_path TEXT,
    learning_rate REAL,
    save_interval INTEGER,
    steps_per_report INTEGER
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model_name TEXT,
    input TEXT,
    output TEXT
)
'''
db_cursor.executescript(create_tables_query)

def log_to_db(timestamp, model_name, input_text, output_text):
    insert_query = '''
    INSERT INTO logs (timestamp, model_name, input, output)
    VALUES (?, ?, ?, ?)
    '''
    db_cursor.execute(insert_query, (timestamp, model_name, input_text, output_text))
    db_connection.commit()

# Close the database connection when done
# db_connection.close()  # Uncomment this line when you want to close the connection
