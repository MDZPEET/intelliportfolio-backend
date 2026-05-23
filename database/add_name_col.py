import os
import psycopg2

def add_column():
    db_host = os.getenv("DATABASE_HOST", "localhost")
    print(f"Connecting to DB at {db_host}...")
    
    conn = psycopg2.connect(
        dbname="intelliport_db",
        user="admin",
        password="Heyrose05",
        host=db_host,
        port="5432"
    )
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE portfolios ADD COLUMN name VARCHAR(255) DEFAULT 'My Portfolio';")
        conn.commit()
        print("Column 'name' added successfully.")
    except Exception as e:
        print(f"Error (maybe column already exists): {e}")
        conn.rollback()
        
    cursor.close()
    conn.close()

if __name__ == "__main__":
    add_column()
