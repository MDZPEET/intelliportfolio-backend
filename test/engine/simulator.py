# app/engine/simulator.py
import numpy as np

class PortfolioSimulator:
    @staticmethod
    def calculate_duration(principal: float, goal: float, annual_return: float) -> float:
        if annual_return <= 0:
            raise ValueError("ผลตอบแทนต้องมากกว่า 0 เพื่อให้ถึงเป้าหมายได้")
        if principal >= goal:
            return 0.0
            
        duration = np.log(goal / principal) / np.log(1 + annual_return)
        return round(duration, 2)

    @staticmethod
    def calculate_required_principal(duration: float, goal: float, annual_return: float) -> float:
        if duration <= 0:
            raise ValueError("ระยะเวลาต้องมากกว่า 0 ปี")
            
        principal = goal / ((1 + annual_return) ** duration)
        return round(principal, 2)