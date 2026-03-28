"""
Cálculo centralizado de indicadores técnicos normalizados.
Fonte única de verdade — elimina duplicação entre websocket_manager e market_data.
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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores técnicos normalizados sobre um DataFrame OHLCV.

    Espera colunas: open, high, low, close, volume
    Adiciona (normalizadas pelo close):
        RSI_14, SMA_20, SMA_50, BBL/BBM/BBU/BBB/BBP_20_2.0,
        MACD_12_26_9, MACDs/MACDh, EMA_9, EMA_21, ATR_14, Volume_MA_20,
        open/high/low/close_return
    Preenche NaN com 0.
    """
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close'])

    if len(df) < 5 or not _TALIB_OK:
        df = df.fillna(0)
        return df

    close_arr  = df['close'].values.astype(float)
    high_arr   = df['high'].values.astype(float)
    low_arr    = df['low'].values.astype(float)
    volume_arr = df['volume'].values.astype(float)

    try:
        df['RSI_14']       = talib.RSI(close_arr, timeperiod=14) / 100.0
        df['SMA_20']       = talib.SMA(close_arr, timeperiod=20) / (close_arr + 1e-8)
        df['SMA_50']       = talib.SMA(close_arr, timeperiod=50) / (close_arr + 1e-8)

        upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20)
        df['BBL_20_2.0']   = lower  / (close_arr + 1e-8)
        df['BBM_20_2.0']   = middle / (close_arr + 1e-8)
        df['BBU_20_2.0']   = upper  / (close_arr + 1e-8)
        df['BBB_20_2.0']   = (upper - lower) / (middle + 1e-8)
        df['BBP_20_2.0']   = (close_arr - lower) / (upper - lower + 1e-8)

        macd, signal, hist = talib.MACD(close_arr)
        df['MACD_12_26_9']  = macd   / (close_arr + 1e-8)
        df['MACDs_12_26_9'] = signal / (close_arr + 1e-8)
        df['MACDh_12_26_9'] = hist   / (close_arr + 1e-8)

        df['EMA_9']        = talib.EMA(close_arr, timeperiod=9)   / (close_arr + 1e-8)
        df['EMA_21']       = talib.EMA(close_arr, timeperiod=21)  / (close_arr + 1e-8)
        df['ATR_14']       = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14) / (close_arr + 1e-8)

        vol_ma             = talib.SMA(volume_arr, timeperiod=20)
        df['Volume_MA_20'] = volume_arr / (vol_ma + 1e-8)

        df['open_return']  = df['open'].pct_change()
        df['high_return']  = df['high'].pct_change()
        df['low_return']   = df['low'].pct_change()
        df['close_return'] = df['close'].pct_change()

    except Exception as exc:
        logger.warning(f"[INDICATORS] Erro ao computar indicadores: {exc}")

    df = df.fillna(0)
    return df.reset_index(drop=True)
