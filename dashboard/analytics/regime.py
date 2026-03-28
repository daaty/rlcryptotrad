"""
Análise de regime de mercado — BULL, BEAR, SIDEWAYS.
Funções puras sem dependência de Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
    _TALIB_OK = True
except ImportError:
    _TALIB_OK = False

from dashboard.core.logging_setup import get_logger

logger = get_logger()


def detect_market_regime(df: pd.DataFrame) -> tuple[str, float]:
    """
    Detecta regime de mercado: 'BULL', 'BEAR' ou 'SIDEWAYS'.

    Usa SMA 20/50 crossover + ADX para força de tendência.
    Retorna (regime: str, adx_value: float).
    """
    try:
        if not _TALIB_OK or len(df) < 55:
            return 'UNKNOWN', 0.0

        close = df['close'].values.astype(float)
        high  = df['high'].values.astype(float)
        low_  = df['low'].values.astype(float)

        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        adx    = talib.ADX(high, low_, close, timeperiod=14)

        current_price = close[-1]
        current_sma20 = sma_20[-1]
        current_sma50 = sma_50[-1]
        current_adx   = adx[-1]

        if current_adx < 20:
            return 'SIDEWAYS', current_adx
        elif current_sma20 > current_sma50 and current_price > current_sma20:
            return 'BULL', current_adx
        elif current_sma20 < current_sma50 and current_price < current_sma20:
            return 'BEAR', current_adx
        else:
            return 'SIDEWAYS', current_adx

    except Exception as exc:
        logger.error(f"[REGIME] Erro ao detectar regime: {exc}")
        return 'UNKNOWN', 0.0


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calcula Average True Range (volatilidade). Retorna 0 em caso de erro."""
    try:
        if not _TALIB_OK or len(df) < period + 1:
            return 0.0
        atr = talib.ATR(
            df['high'].values.astype(float),
            df['low'].values.astype(float),
            df['close'].values.astype(float),
            timeperiod=period,
        )
        return float(atr[-1]) if len(atr) > 0 else 0.0
    except Exception:
        return 0.0


def calculate_correlation(df1: pd.DataFrame, df2: pd.DataFrame, period: int = 50) -> float:
    """Calcula correlação de Pearson dos retornos entre dois DataFrames."""
    try:
        r1 = df1['close'].pct_change().tail(period)
        r2 = df2['close'].pct_change().tail(period)
        return float(r1.corr(r2))
    except Exception:
        return 0.0
