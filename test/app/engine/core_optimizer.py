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
        random.seed(42)
        np.random.seed(42)
        
        # ป้องกัน AttributeError
        cov = cov_matrix.values if hasattr(cov_matrix, "values") else cov_matrix
        asset_betas = actual_betas.values if hasattr(actual_betas, "values") else actual_betas

        toolbox = base.Toolbox()
        toolbox.register("attr", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=len(tickers))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(ind):
            w = np.array(ind); w = np.maximum(w, 0)
            if np.count_nonzero(w) > max_stocks:
                threshold = np.sort(w)[-max_stocks]
                w[w < threshold] = 0.0
            w_sum = np.sum(w)
            if w_sum <= 0: return -999.0,
            w /= w_sum
            for i in range(len(ind)): ind[i] = float(w[i])
            
            p_ret = np.dot(w, bl_returns)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if p_vol == 0: return -999.0,
            
            sharpe = (p_ret - self.rf) / p_vol
            penalty = abs(np.dot(w, asset_betas) - target_beta) * 10 
            return (sharpe - penalty),

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=150, halloffame=hof, verbose=False)
        
        best_w = np.array(hof[0])
        best_w = np.maximum(best_w, 0); best_w /= np.sum(best_w)
        return pd.DataFrame({'Ticker': tickers, 'Weight': best_w, 'Beta': asset_betas})