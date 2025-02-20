# clean_logs.py
import sqlite3
import os
import shutil
from datetime import datetime

def clean_logs_table():
    db_path = "./data/Tabulated/promptinglog.db"
    backup_path = f"./data/Tabulated/promptinglog_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.db"
    shutil.copy(db_path, backup_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First count duplicates
        cursor.execute("""
            SELECT model_name, input, COUNT(*) 
            FROM logs 
            GROUP BY model_name, input 
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()
        
        if duplicates:
            print(f"Found {len(duplicates)} model/input pairs with duplicates")
            
            # Delete non-latest entries using explicit join
            delete_query = """
            DELETE FROM logs
            WHERE rowid NOT IN (
                SELECT MAX(rowid) 
                FROM logs 
                GROUP BY model_name, input
            )
            """
            
            cursor.execute(delete_query)
            conn.commit()
            
            print(f"Removed {cursor.rowcount} duplicate entries")
            
            # Verify results
            cursor.execute("SELECT COUNT(*) FROM logs")
            print(f"Total remaining records: {cursor.fetchone()[0]}")
        else:
            print("No duplicates found - database is already clean")

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    clean_logs_table()