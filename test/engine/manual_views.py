# app/engine/manual_views.py

class ManualViewProvider:
    # อัปเดตข้อมูลตามตารางล่าสุด (ดึงค่าจากคอลัมน์ "ค่ามุมมอง")
    # ข้อมูลนี้สะท้อน Expected Return (Upside) ที่คำนวณจากราคาปัจจุบันและเป้าหมาย
    
    VIEWS_DATA = {
        "ADVANC": 0.01,   "AOT": 0.03,    "AWC": 0.04,    "BANPU": -0.01,  "BBL": -0.01,
        "BDMS": 0.20,     "BEM": 0.29,    "BH": -0.01,    "BJC": 0.11,     "BTS": 0.24,
        "CBG": 0.12,     "CCET": -0.01,   "CENTEL": 0.04, "COM7": 0.26,    "CPALL": 0.18,
        "CPF": 0.12,     "CPN": 0.05,    "CRC": 0.10,    "DELTA": -0.25,  "EGCO": 0.08,
        "GPSC": 0.08,    "GULF": 0.10,   "HMPRO": 0.06,  "IVL": -0.08,    "KBANK": -0.05,
        "KKP": -0.02,    "KTB": 0.02,    "KTC": 0.04,    "LH": -0.03,     "MINT": 0.30,
        "MTC": 0.27,     "OR": 0.19,     "OSP": 0.16,    "PTT": 0.05,     "PTTEP": -0.02,
        "PTTGC": -0.01,  "RATCH": 0.08,   "SAWAD": 0.11,  "SCB": -0.01,    "SCC": -0.01,
        "SCGP": 0.07,    "TCAP": -0.01,  "TIDLOR": 0.21,  "TISCO": -0.07,  "TLI": 0.24,
        "TOP": 0.00,     "TTB": -0.11,   "TU": 0.13,     "WHA": 0.12,
        "TRUE": 0.03     
    }

    @classmethod
    def get_view(cls, ticker):
        symbol = ticker.replace(".BK", "")
        # คืนค่ามุมมองตามชื่อหุ้น หากไม่พบให้คืนค่า 0.0
        return cls.VIEWS_DATA.get(symbol, 0.0)

    @classmethod
    def get_all_views(cls, tickers):
        # สร้างรูปแบบข้อมูลสำหรับ Black-Litterman Model
        # variance: 0.02 คือระดับความมั่นใจในข้อมูล (Confidence Level)
        return {t: {"return_view": cls.get_view(t), "variance": 0.02} for t in tickers}