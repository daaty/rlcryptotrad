"""
Fixtures compartilhadas entre os testes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def minimal_config() -> dict:
    """Config mínimo válido para executor e risk_calculator."""
    return {
        "mode": "paper",
        "data": {
            "primary_symbol": "BTCUSDT",
            "symbols": ["BTC/USDT"],
        },
        "environment": {
            "position_size": 0.03,
            "leverage": 1.5,
            "initial_balance": 10_000,
        },
        "risk_management": {
            "kelly_fraction": 0.25,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_leverage": 3,
            "max_drawdown": 0.15,
            "max_total_exposure": 0.60,
            "correlation_threshold": 0.70,
        },
        "notifications": {"telegram": {"enabled": False}},
    }


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """
    DataFrame com 60 linhas e todas as colunas esperadas por validate_entry_quality
    e check_correlation. Preços sobem linearmente (sem tendência dramática).
    """
    n = 60
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":         close * 0.999,
        "high":         close * 1.002,
        "low":          close * 0.997,
        "close":        close,
        "volume":       np.random.uniform(1000, 2000, n),
        "RSI_14":       np.clip(np.random.uniform(0.4, 0.6, n), 0, 1),
        "SMA_20":       close * 1.001,
        "SMA_50":       close * 1.003,
        "BBL_20_2.0":   close * 0.98,
        "BBM_20_2.0":   close,
        "BBU_20_2.0":   close * 1.02,
        "BBB_20_2.0":   np.full(n, 4.0),
        "BBP_20_2.0":   np.full(n, 0.5),
        "MACD_12_26_9": np.random.randn(n) * 0.01,
        "MACDs_12_26_9": np.random.randn(n) * 0.01,
        "MACDh_12_26_9": np.random.randn(n) * 0.005,
        "EMA_9":        np.full(n, 1.001),   # ratio EMA9/close ≈ 1.0
        "EMA_21":       np.full(n, 1.002),   # ratio EMA21/close ≈ 1.0
        "ATR_14":       np.full(n, 2.0),
        "Volume_MA_20": np.random.uniform(0.8, 1.2, n),
        "timestamp":    np.arange(n) * 900_000,  # ms 15min
    })
    return df


@pytest.fixture()
def closed_trades_sample() -> list[dict]:
    """20 trades com 60% win rate e razão ganho/perda ~2."""
    import random
    random.seed(7)
    trades = []
    for i in range(20):
        win = random.random() < 0.6
        pnl = round(random.uniform(10, 20), 2) if win else -round(random.uniform(5, 10), 2)
        trades.append({
            "symbol":      "BTCUSDT",
            "realizedPnl": pnl,
            "side":        "LONG",
        })
    return trades
