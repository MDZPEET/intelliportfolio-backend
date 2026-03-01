import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class BacktestEngine:
    def run_backtest(self, weights_dict, start_date, end_date):
        if not weights_dict:
            return self._default_empty_result("ไม่มีหุ้นในพอร์ต")

        tickers_bk = [f"{t}.BK" if not t.endswith(".BK") else t for t in weights_dict.keys()]
        clean_requested = [t.replace(".BK", "") for t in tickers_bk]
        
        price_data = pd.DataFrame()
        
        try:
            # 1. พยายามดึงแบบกลุ่ม
            data = yf.download(tickers_bk, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    price_data = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
                else:
                    price_data = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]
            
            # 2. กรณีดึงแบบกลุ่มแล้วข้อมูลหาย พยายามดึงรายตัว
            price_data = price_data.dropna(axis=1, how='all')
            missing_after_batch = [t for t in clean_requested if t not in price_data.columns]
            
            if missing_after_batch:
                for t in missing_after_batch:
                    try:
                        single_data = yf.download(f"{t}.BK", start=start_date, end=end_date, progress=False)
                        if not single_data.empty:
                            s_price = single_data['Adj Close'] if 'Adj Close' in single_data else single_data['Close']
                            price_data[t] = s_price
                    except:
                        continue

            price_data.columns = [str(c).replace(".BK", "") for c in price_data.columns]
            price_data = price_data.dropna(axis=1, how='all').ffill().bfill()
            
            valid_tickers = list(price_data.columns)
            missing_tickers = [t for t in clean_requested if t not in valid_tickers]
            
            if not valid_tickers:
                return self._default_empty_result(f"ไม่พบประวัติราคาในช่วง {start_date} ถึง {end_date}")

            # 3. คำนวณ Portfolio Performance
            daily_returns = price_data.pct_change().dropna()
            valid_weights = np.array([weights_dict.get(t, 0) for t in valid_tickers])
            w_sum = np.sum(valid_weights)
            
            if w_sum <= 0:
                 return self._default_empty_result("น้ำหนักหุ้นรวมเป็นศูนย์")
                 
            valid_weights = valid_weights / w_sum 
            
            port_returns = daily_returns.dot(valid_weights)
            total_return = (1 + port_returns).prod() - 1
            port_volatility = port_returns.std() * np.sqrt(252)
            
            # 4. ดึง Benchmark (^SET.BK) และป้องกันปัญหา Error: Not a string or real number
            bm_data = yf.download("^SET.BK", start=start_date, end=end_date, progress=False)
            bm_total_return = 0.0
            
            if not bm_data.empty:
                # ดึงคอลัมน์ราคาปิด
                if 'Adj Close' in bm_data.columns:
                    bm_price = bm_data['Adj Close']
                else:
                    bm_price = bm_data['Close']
                
                # บังคับให้เป็น Series เสมอ (ป้องกันกรณีได้ DataFrame กลับมา)
                if isinstance(bm_price, pd.DataFrame):
                    bm_price = bm_price.iloc[:, 0]
                
                # คำนวณ และแปลงเป็นค่าตัวเลขตัวเดียว (item() หรือ float())
                if len(bm_price) > 1:
                    raw_return = (bm_price.iloc[-1] / bm_price.iloc[0]) - 1
                    # ถ้ายังเป็น Series ให้หยิบค่าแรกมา
                    bm_total_return = float(raw_return.iloc[0]) if hasattr(raw_return, "iloc") else float(raw_return)

            # ตรวจสอบค่าที่จะส่งกลับให้เป็น float เสมอ
            final_port_return = float(total_return.iloc[0]) if hasattr(total_return, "iloc") else float(total_return)
            final_port_vol = float(port_volatility.iloc[0]) if hasattr(port_volatility, "iloc") else float(port_volatility)

            result = {
                "portfolio_return": final_port_return,
                "benchmark_return": bm_total_return,
                "portfolio_volatility": final_port_vol
            }
            
            if missing_tickers:
                result["warning"] = f"เนื่องจากหุ้น {', '.join(missing_tickers)} ไม่มีข้อมูลในช่วงนี้ จึงคำนวณ Backtest เฉพาะหุ้นที่เหลือ ({', '.join(valid_tickers)})"
                
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc() # พิมพ์ Error ออกทาง Console เพื่อการ Debug
            return self._default_empty_result(f"ระบบมีปัญหา: {str(e)}")

    def _default_empty_result(self, msg: str):
        return {"portfolio_return": 0.0, "benchmark_return": 0.0, "portfolio_volatility": 0.0, "warning": msg}