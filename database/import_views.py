# backend/import_views.py
import pandas as pd
import psycopg2

def import_stock_views():
    # 1. เชื่อมต่อฐานข้อมูล (ใช้รหัสผ่านตามที่คุณตั้งไว้)
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            database="intelliport_db",
            user="admin",
            password="Heyrose05",
            port="5432"
        )
        cur = conn.cursor()

        # 2. สร้างตารางรองรับข้อมูล (ถ้ายังไม่มี)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_views (
                ticker VARCHAR(10) PRIMARY KEY,
                last_price FLOAT,
                median_target FLOAT,
                mean_target FLOAT,
                expected_return FLOAT,
                variance FLOAT DEFAULT 0.02,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # ล้างข้อมูลของเดือนเก่าทิ้ง เพื่อรับข้อมูลเดือนใหม่
        cur.execute("TRUNCATE TABLE stock_views;")

        # 3. อ่านไฟล์ข้อมูล
        file_path = "views_data.xlsx"   # <--- แก้ตรงนี้
        print(f"กำลังอ่านไฟล์: {file_path}")
        df = pd.read_excel(file_path)

        # 4. วนลูปนำเข้าข้อมูล
        for index, row in df.iterrows():
            ticker = str(row['ชื่อย่อหลักทรัพย์']).strip().upper()
            last_price = float(row['ราคาล่าสุด'])
            median_target = float(row['ค่ามัธยฐาน *'])
            mean_target = float(row['ค่าเฉลี่ย *'])
            expected_return = float(row['ค่ามุมมอง'])
            
            # บันทึกลง PostgreSQL
            cur.execute("""
                INSERT INTO stock_views (ticker, last_price, median_target, mean_target, expected_return)
                VALUES (%s, %s, %s, %s, %s)
            """, (ticker, last_price, median_target, mean_target, expected_return))

        conn.commit()
        print(f"✅ นำเข้าข้อมูลมุมมองหุ้นสำเร็จจำนวน {len(df)} บริษัท")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    import_stock_views()