import random
import numpy as np
import pandas as pd
from typing import List, Dict
from deap import base, creator, tools, algorithms

class BlackLittermanEngine:
    @staticmethod
    def calculate_posterior(market_caps: pd.Series, cov_matrix: pd.DataFrame, views_data: Dict[str, Dict[str, float]]):
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        S = cov_matrix.values
        w_mkt = market_caps.values / market_caps.sum()
        pi = 2.5 * np.dot(S, w_mkt) 
        
        if not views_data: return pi, S
        p_list, q_list, omega_diag = [], [], []
        for t, data in views_data.items():
            if t in tickers:
                row = np.zeros(n); row[tickers.index(t)] = 1
                p_list.append(row); q_list.append(data['return_view'])
                omega_diag.append(data.get('variance', 0.05))
        
        if not q_list: return pi, S
        P, Q, tau = np.array(p_list), np.array(q_list), 0.05
        Omega = np.diag(omega_diag) 
        term1 = np.linalg.inv(np.linalg.inv(tau * S) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P))
        term2 = np.dot(np.linalg.inv(tau * S), pi) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q)
        return np.dot(term1, term2), S

class GeneticPortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.025):
        self.rf = risk_free_rate
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def run_optimization(self, tickers, bl_returns, cov_matrix, market_caps, target_beta=1.0, max_stocks=5, actual_betas=None):
        """
        ฟังก์ชันคำนวณสัดส่วนพอร์ตที่เหมาะสมที่สุดด้วย Genetic Algorithm
        :param max_stocks: จำนวนหุ้นสูงสุดในพอร์ต (รองรับขั้นต่ำ 3 ตัวตามที่คุณต้องการ)
        """
        random.seed(42)
        np.random.seed(42)
        
        cov = cov_matrix.values if hasattr(cov_matrix, "values") else cov_matrix
        asset_betas = actual_betas.values if hasattr(actual_betas, "values") else actual_betas

        toolbox = base.Toolbox()
        toolbox.register("attr", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=len(tickers))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(ind):
            w = np.array(ind); w = np.maximum(w, 0)
            
            # 🌟 ส่วนการกรองจำนวนหุ้นให้เหลือเท่ากับ max_stocks (เช่น 3, 5 หรือ 10)
            if np.count_nonzero(w) > max_stocks:
                # เรียงลำดับค่าน้ำหนักหุ้นและเก็บไว้เฉพาะตัวที่ติดอันดับ top ตามจำนวน max_stocks
                threshold = np.sort(w)[-max_stocks]
                w[w < threshold] = 0.0
            
            w_sum = np.sum(w)
            if w_sum <= 0: return -999.0,
            
            # คำนวณผลตอบแทนและความผันผวนของพอร์ต (เราไม่ได้ normalize ในระดับ ind เพื่อให้ penalty ทำงาน)
            w_normalized = w / w_sum
            p_ret = np.dot(w_normalized, bl_returns)
            p_vol = np.sqrt(np.dot(w_normalized.T, np.dot(cov, w_normalized)))
            if p_vol == 0: return -999.0,
            
            # Sharpe Ratio: วัดความคุ้มค่าของผลตอบแทนต่อความเสี่ยง
            sharpe = (p_ret - self.rf) / p_vol
            
            # Penalty: บทลงโทษค่าน้ำหนัก (ให้รวมกันได้ 100%) ตามเอกสารข้อ 2.1.6.2
            weight_penalty = 1000.0 * ((w_sum - 1.0) ** 2)
            
            # Penalty: บทลงโทษหากค่า Beta เบี่ยงเบนจากเป้าหมาย
            beta_penalty = abs(np.dot(w_normalized, asset_betas) - target_beta) * 10 
            
            # Fitness_penalized
            return (sharpe - weight_penalty - beta_penalty),

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # --- การรัน GA แบบ Manual พร้อมแสดงผลการวิวัฒนาการ (Logs) ---
        pop = toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        ngen = 150 # จำนวนรุ่นในการวิวัฒนาการ
        cxpb, mutpb = 0.7, 0.2

        print(f"\n🚀 เริ่มต้น GA (จำนวนหุ้นสูงสุด: {max_stocks}, เป้าหมาย Beta: {target_beta})")
        print("-" * 65)

        for gen in range(ngen):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
            hof.update(pop)
            
            # แสดงค่า Fitness ทุกๆ 10 รุ่นเพื่อติดตามการเรียนรู้ของ AI
            if gen % 10 == 0 or gen == ngen - 1:
                best_val = hof[0].fitness.values[0]
                print(f"🧬 Generation {gen:03d}: Best Fitness = {best_val:.4f}")

        print("-" * 65)
        print("✅ กระบวนการ Genetic Algorithm เสร็จสมบูรณ์!")

        best_w = np.array(hof[0])
        best_w = np.maximum(best_w, 0)
        
        # 🌟 บังคับให้เหลือเฉพาะหุ้นที่กำหนด (max_stocks) ในขั้นตอนสุดท้าย
        if np.count_nonzero(best_w) > max_stocks:
            threshold = np.sort(best_w)[-max_stocks]
            best_w[best_w < threshold] = 0.0

        if np.sum(best_w) > 0:
            best_w /= np.sum(best_w) # Normalize ครั้งสุดท้ายก่อนนำไปใช้จริง
        else:
            best_w = np.ones(len(tickers)) / len(tickers)
        
        # แสดงสถานะสุดท้ายของพอร์ตก่อนส่งกลับ
        final_ret = float(np.dot(best_w, bl_returns))
        final_vol = float(np.sqrt(np.dot(best_w.T, np.dot(cov, best_w))))
        final_beta = float(np.dot(best_w, asset_betas))
        print(f"📈 ผลตอบแทนคาดหวังรายปี: {final_ret:.2%}")
        print(f"🛡️ ค่าความเสี่ยงพอร์ต (Beta): {final_beta:.4f}")

        portfolio_df = pd.DataFrame({'Ticker': tickers, 'Weight': best_w, 'Beta': asset_betas})
        return portfolio_df, final_ret, final_vol