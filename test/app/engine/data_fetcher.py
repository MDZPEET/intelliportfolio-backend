import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

class SETDataFetcher:
    @staticmethod
    def get_set50_tickers() -> List[str]:
        """
        ส่งคืนรายชื่อหุ้น SET50 ล่าสุดรอบ H1/2026 (Official List)
        Inclusion: CCET, CENTEL, SAWAD
        Exclusion: BCP, VGI
        """
        return [
            "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BH", "BJC", "BTS",
            "CBG", "CCET", "CENTEL", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EGCO",
            "GPSC", "GULF", "HMPRO", "IVL", "KBANK", "KKP", "KTB", "KTC", "LH", "MINT",
            "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH", "SAWAD", "SCB", "SCC",
            "SCGP", "TCAP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA"
        ]

    @staticmethod
    def get_market_caps(tickers: List[str]) -> pd.Series:
        """
        จำลองค่า Market Capitalization สำหรับคำนวณ Market Equilibrium ใน Black-Litterman
        หมายเหตุ: เนื่องจาก API ตลาดหลักทรัพย์มักบล็อกการดึงข้อมูลอัตโนมัติ 
        เราจึงใช้ค่าประมาณการตามสัดส่วนความใหญ่ของหุ้นในดัชนี (H1/2026)
        """
        # หน่วย: ล้านบาท (ตัวเลขประมาณการเพื่อใช้ถ่วงน้ำหนัก)
        base_caps = {
            "DELTA": 1800000, "PTT": 1000000, "AOT": 900000, "ADVANC": 850000, "GULF": 800000,
            "CPALL": 600000, "BDMS": 450000, "SCB": 380000, "KBANK": 350000, "BBL": 340000,
            "PTTEP": 500000, "SCC": 300000, "CPN": 280000, "TRUE": 250000, "MINT": 180000,
            "SAWAD": 50000, "CENTEL": 60000, "CCET": 45000 # หุ้นเข้าใหม่
        }
        
        # สำหรับหุ้นตัวอื่นที่ไม่ได้ระบุ ให้ใช้ค่าเฉลี่ย 100,000 ล้านบาท
        caps = {t: base_caps.get(t.replace(".BK", ""), 100000) for t in tickers}
        return pd.Series(caps)

    @staticmethod
    def get_official_beta(ticker: str):
        """
        (Deprecated) เดิมใช้ดึงจาก Settrade แต่ปัจจุบันถูกบล็อก 
        จึงคืนค่า None เพื่อให้ระบบไปใช้ค่าที่คำนวณจาก Yahoo Finance แทน
        """
        return None

    @staticmethod
    def get_iaa_consensus(ticker: str):
        """
        (Deprecated) เปลี่ยนไปใช้ ManualViewProvider ใน engine/manual_views.py แทน
        """
        return None


class YahooFinanceFetcher:
    @staticmethod
    def get_market_data_with_beta(tickers: List[str], period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        ดึงราคาหุ้นและคำนวณ Covariance Matrix พร้อมค่า Beta (Weekly 2Y)
        """
        # เพิ่ม .BK สำหรับหุ้นไทย และดึงดัชนีตลาด (^SET.BK)
        yf_tickers = [f"{t}.BK" for t in tickers]
        all_tickers = yf_tickers + ["^SET.BK"]
        
        print(f"📥 กำลังดึงข้อมูลราคา {len(tickers)} หุ้น จาก Yahoo Finance...")
        data = yf.download(all_tickers, period=period, interval="1wk", progress=False)
        
        if data.empty:
            raise ValueError("ไม่สามารถดึงข้อมูลจาก Yahoo Finance ได้")

        # สกัดเอาเฉพาะราคาปิด (Adj Close หรือ Close)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
        else:
            prices = data
            
        # คลีนชื่อคอลัมน์และจัดการค่าว่าง
        prices.columns = [c.replace(".BK", "") for c in prices.columns]
        prices = prices.ffill().bfill().dropna(axis=1)
        
        # คำนวณ Returns
        returns = prices.pct_change().dropna()
        
        # 1. คำนวณ Covariance Matrix (รายปี)
        cov_matrix = returns.drop(columns=["^SET"], errors='ignore').cov() * 52
        
        # 2. คำนวณค่า Beta รายตัวเทียบกับดัชนี ^SET
        betas = {}
        if "^SET" in returns.columns:
            market_var = returns["^SET"].var()
            for t in cov_matrix.columns:
                ticker_cov_market = returns[t].cov(returns["^SET"])
                betas[t] = ticker_cov_market / market_var if market_var != 0 else 1.0
        else:
            betas = {t: 1.0 for t in cov_matrix.columns}
            
        return cov_matrix, betas