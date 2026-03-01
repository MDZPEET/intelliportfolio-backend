import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class SETDataFetcher:
    # ใส่ Headers ให้ดูเหมือนคนเข้าเว็บผ่าน Browser มากที่สุด เพื่อลดโอกาสโดนบล็อก
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.settrade.com/"
    }

    @staticmethod
    def get_set50_tickers():
        """ดึงรายชื่อหุ้น SET50 ล่าสุด (ถ้าดึงไม่ผ่านจะใช้ Fallback 50 ตัว)"""
        try:
            url = "https://www.set.or.th/api/set/index/set50/composition"
            response = requests.get(url, headers=SETDataFetcher.HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [stock['symbol'].replace(".BK", "") for stock in data.get('composition', [])]
        except Exception as e:
            print(f"⚠️ API ตลาดหลักทรัพย์ไม่ตอบสนอง: {e}")
            pass
        
        print("⚠️ ใช้งานรายชื่อหุ้น SET50 สำรอง (Fallback 50 Tickers)")
        # แผนสำรอง: ใช้รายชื่อหุ้น SET50 ครบทั้ง 50 ตัว
        return [
            "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BH", "BJC", "BTS", 
            "CBG", "CCET", "CENTEL", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EGCO", 
            "GPSC", "GULF", "HMPRO", "IVL", "KBANK", "KKP", "KTB", "KTC", "LH", "MINT", 
            "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH", "SAWAD", "SCB", "SCC", 
            "SCGP", "TCAP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA"
        ]

    @staticmethod
    def get_official_beta(ticker):
        """ดึงค่า Beta จาก Key Statistics ของ Settrade"""
        try:
            symbol = ticker.replace(".BK", "")
            url = f"https://www.settrade.com/api/set/stock/{symbol}/key-statistic"
            response = requests.get(url, headers=SETDataFetcher.HEADERS, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('statistics', []):
                    if item.get('name') == 'Beta':
                        val = item.get('value')
                        return float(val) if val is not None else None
        except Exception: 
            pass
        return None

    @staticmethod
    def get_market_caps(tickers):
        """ดึง Market Cap ปัจจุบันจาก Yahoo Finance"""
        mc_dict = {}
        for t in tickers:
            symbol = t if t.endswith(".BK") else f"{t}.BK"
            try:
                ticker_obj = yf.Ticker(symbol)
                mc_dict[t.replace(".BK", "")] = ticker_obj.info.get('marketCap', 10000000000)
            except: 
                mc_dict[t.replace(".BK", "")] = 10000000000
        return pd.Series(mc_dict)

    @staticmethod
    def get_iaa_consensus(ticker):
        """ดึงมุมมองนักวิเคราะห์จาก Settrade"""
        symbol = ticker.replace(".BK", "")
        try:
            url = f"https://www.settrade.com/api/set/stock/{symbol}/iaa-consensus"
            response = requests.get(url, headers=SETDataFetcher.HEADERS, timeout=5)
            if response.status_code == 200:
                data = response.json()
                details = data.get('detail', {})
                last = details.get('lastPrice', 1)
                expected = (details.get('median', last) / last) - 1
                variance = ((details.get('high', last) - details.get('low', last)) / (last * 4)) ** 2
                return {"return_view": expected, "variance": variance if variance > 0 else 0.001}
        except Exception: 
            pass
        return None

class YahooFinanceFetcher:
    @staticmethod
    def get_market_data_with_beta(tickers: list):
        """ดึงข้อมูลราคา: 2 ปี Weekly สำหรับ Beta และ 1 ปี Daily สำหรับ Covariance"""
        # ป้องกันชื่อซ้ำซ้อน (เช่น ADVANC.BK.BK)
        clean_tickers = [t.replace(".BK", "") for t in tickers]
        yf_tickers = [f"{t}.BK" for t in clean_tickers] + ["^SET.BK"]
        
        # 1. คำนวณ Calculated Beta: ใช้ข้อมูลรายสัปดาห์ (Weekly) ย้อนหลัง 2 ปี
        hist_2y = yf.download(yf_tickers, period="2y", interval="1wk", progress=False)
        close_2y = hist_2y['Adj Close'] if 'Adj Close' in hist_2y else hist_2y['Close']
        close_2y.columns = [str(c).replace(".BK", "") for c in close_2y.columns]
        ret_2y = close_2y.pct_change().dropna()
        
        calc_betas = {}
        if "^SET" in ret_2y.columns:
            m_var = ret_2y["^SET"].var()
            for t in clean_tickers:
                if t in ret_2y.columns:
                    calc_betas[t] = ret_2y[t].cov(ret_2y["^SET"]) / m_var

        # 2. คำนวณ Covariance Matrix: ใช้ข้อมูลรายวัน (Daily) ย้อนหลัง 1 ปี (สำหรับจัดพอร์ต)
        valid_clean_tickers = [t for t in clean_tickers if t in ret_2y.columns]
        hist_1y = yf.download([f"{t}.BK" for t in valid_clean_tickers], period="1y", interval="1d", progress=False)
        close_1y = hist_1y['Adj Close'] if 'Adj Close' in hist_1y else hist_1y['Close']
        close_1y.columns = [str(c).replace(".BK", "") for c in close_1y.columns]
        cov_matrix = close_1y.pct_change().dropna().cov() * 252
        
        return cov_matrix, pd.Series(calc_betas)