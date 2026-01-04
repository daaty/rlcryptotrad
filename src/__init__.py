"""
Agente de Trading com Reinforcement Learning
Autor: Sistema RL Trading
Versão: 1.0.0
"""

__version__ = "1.0.0"

from .environment.trading_env import TradingEnv
from .data.data_collector import DataCollector
from .risk.risk_manager import RiskManager, PositionSizer
from .training.train import TradingTrainer
from .execution.executor import BinanceExecutor

__all__ = [
    'TradingEnv',
    'DataCollector',
    'RiskManager',
    'PositionSizer',
    'TradingTrainer',
    'BinanceExecutor'
]
