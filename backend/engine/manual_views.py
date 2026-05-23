# backend/engine/manual_views.py
import psycopg2
import os

class ManualViewProvider:
    
    @staticmethod
    def get_db_connection():
        try:
            db_host = os.getenv("DATABASE_HOST", "localhost")
            return psycopg2.connect(
                host=db_host,
                database="intelliport_db",
                user="admin",
                password="Heyrose05",
                port="5432"
            )
        except Exception as e:
            print(f"❌ DB Error in ManualViewProvider: {e}")
            return None

    @classmethod
    def get_view(cls, ticker):
        symbol = ticker.replace(".BK", "").upper()
        conn = cls.get_db_connection()
        if not conn:
            return 0.0 # คืนค่าเซฟตี้หาก DB พัง
        
        try:
            cur = conn.cursor()
            cur.execute("SELECT expected_return FROM stock_views WHERE ticker = %s", (symbol,))
            result = cur.fetchone()
            return result[0] if result else 0.0
        except Exception as e:
            print(f"Query Error for {symbol}: {e}")
            return 0.0
        finally:
            if conn:
                conn.close()

    @classmethod
    def get_all_views(cls, tickers):
        # ดึงข้อมูลรวดเดียวเพื่อความเร็วในการรัน Black-Litterman
        views = {}
        conn = cls.get_db_connection()
        if not conn:
            return {t: {"return_view": 0.0, "variance": 0.02} for t in tickers}
        
        try:
            cur = conn.cursor()
            cur.execute("SELECT ticker, expected_return, variance FROM stock_views")
            # แปลงเป็น Dictionary ให้ดึงค่าง่ายๆ
            db_views = {row[0]: {"return_view": row[1], "variance": row[2]} for row in cur.fetchall()}
            
            for t in tickers:
                symbol = t.replace(".BK", "").upper()
                views[t] = db_views.get(symbol, {"return_view": 0.0, "variance": 0.02})
            return views
        except Exception as e:
            print(f"Query Error in get_all_views: {e}")
            return {t: {"return_view": 0.0, "variance": 0.02} for t in tickers}
        finally:
            if conn:
                conn.close()