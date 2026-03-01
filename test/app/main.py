from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Import Engines
from engine.data_fetcher import YahooFinanceFetcher, SETDataFetcher
from engine.core_optimizer import BlackLittermanEngine, GeneticPortfolioOptimizer
from analysis.backtester import BacktestEngine
# Import ระบบ Manual Views ที่เราสร้างขึ้นใหม่
from engine.manual_views import ManualViewProvider

app = FastAPI(
    title="Intelliportfolio Pro (H1/2026 Edition)", 
    description="Portfolio Optimization using Black-Litterman & GA with Manual View Management",
    version="7.0"
)

# ตั้งค่า CORS เพื่อให้ Frontend เชื่อมต่อได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    user_custom_views: Optional[Dict[str, float]] = None 
    target_beta: float = 0.8
    max_stocks: int = 5
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-01"

@app.post("/api/optimize")
def optimize_portfolio(req: OptimizeRequest):
    try:
        # 1. ดึงรายชื่อหุ้น SET50 (อ้างอิงรายชื่อปี 2026 จาก Fallback ใน data_fetcher)
        # รายชื่อจะประกอบด้วยหุ้นอย่าง ADVANC, AOT, CCET, CENTEL, SAWAD และอื่นๆ ตามเอกสาร
        tickers = SETDataFetcher.get_set50_tickers()
        
        # 2. ดึงข้อมูลราคาจาก Yahoo Finance และคำนวณ Beta/Covariance
        # ระบบใช้ Adjusted Close และสูตร Weekly 2Y เพื่อความแม่นยำสูงสุด
        cov_matrix, calc_betas = YahooFinanceFetcher.get_market_data_with_beta(tickers)
        
        # กรองเอาเฉพาะหุ้นที่มีข้อมูลราคาครบถ้วน
        valid_tickers = list(cov_matrix.columns)
        market_caps = SETDataFetcher.get_market_caps(valid_tickers)
        
        # 3. จัดการค่า Beta (ใช้ Calculated Beta ทั้งหมดเพื่อความเสถียร)
        # ค่าเหล่านี้จะสะท้อนความผันผวนจริง เช่น GULF (~1.69) หรือ BBL (~0.25)
        actual_betas_series = pd.Series({t: round(calc_betas.get(t, 1.0), 2) for t in valid_tickers})

        # 4. เตรียมมุมมองนักวิเคราะห์ (Views) - ระบบ Hybrid
        views_data = {}
        for t in valid_tickers:
            symbol = t.replace(".BK", "")
            
            # (A) ลำดับความสำคัญที่ 1: ใช้ค่าที่ผู้ใช้พิมพ์ส่งมาจากหน้าเว็บ (ถ้ามี)
            if req.user_custom_views and symbol in req.user_custom_views:
                views_data[t] = {
                    "return_view": req.user_custom_views[symbol], 
                    "variance": 0.01 # มั่นใจมากเพราะผู้ใช้ระบุเอง
                }
            # (B) ลำดับความสำคัญที่ 2: ใช้ค่าจากระบบ Manual (ที่คุณอัปเดตทุกเดือนใน manual_views.py)
            else:
                manual_return = ManualViewProvider.get_view(symbol)
                views_data[t] = {
                    "return_view": manual_return, 
                    "variance": 0.02 # ค่า Variance มาตรฐาน
                }
        
        # 5. ประมวลผล Black-Litterman เพื่อหา Posterior Expected Returns
        bl_engine = BlackLittermanEngine()
        adj_returns, updated_cov = bl_engine.calculate_posterior(market_caps, cov_matrix, views_data)
        
        # 6. ค้นหาพอร์ตที่เหมาะสมที่สุดด้วย Genetic Algorithm (GA)
        # กำหนดเป้าหมาย Beta (target_beta) และจำนวนหุ้นสูงสุด (max_stocks)
        ga_engine = GeneticPortfolioOptimizer()
        portfolio_df = ga_engine.run_optimization(
            tickers=valid_tickers, 
            bl_returns=adj_returns, 
            cov_matrix=updated_cov, 
            market_caps=market_caps,
            target_beta=req.target_beta, 
            max_stocks=req.max_stocks,
            actual_betas=actual_betas_series
        )
        
        # 7. กรองเฉพาะหุ้นที่มีน้ำหนักมากกว่า 1%
        final_portfolio = portfolio_df[portfolio_df['Weight'] > 0.01].copy().fillna(0.0)
        weights_dict = dict(zip(final_portfolio['Ticker'], final_portfolio['Weight']))
        
        # 8. ทำการทดสอบย้อนหลัง (Backtest) ตามช่วงวันที่ระบุ
        bt_engine = BacktestEngine()
        bt_results = bt_engine.run_backtest(weights_dict, req.start_date, req.end_date)
        
        # 9. ส่งผลลัพธ์กลับในรูปแบบ JSON
        return {
            "status": "success",
            "metadata": {
                "target_beta": req.target_beta,
                "view_source": "View_M1/2026",
                "backtest_period": f"{req.start_date} to {req.end_date}"
            },
            "portfolio": final_portfolio.to_dict(orient="records"),
            "backtest": bt_results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # รันเซิร์ฟเวอร์ที่ Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)