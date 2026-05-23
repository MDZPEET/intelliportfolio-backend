import psycopg2
import os

def run_sql_file(filename):
    try:
        conn = psycopg2.connect(
            dbname="intelliport_db",
            user="admin",
            password="Heyrose05",
            host="localhost",
            port="5432"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        with open(filename, 'r', encoding='utf-8') as file:
            sql_script = file.read()
            
        cursor.execute(sql_script)
        print("✅ Database schema updated successfully!")
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(current_dir, "CS-01 Database.session.sql")
    run_sql_file(sql_path)
