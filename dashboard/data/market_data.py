"""
Coleta de dados de mercado — OHLCV multi-timeframe.
Prioridade: buffer WebSocket → REST (apenas se autorizado).
"""
from __future__ import annotations

import time
from collections import deque

import pandas as pd
import streamlit as st

from dashboard.core.config import KLINE_MAXLEN, load_config
from dashboard.core.logging_setup import get_logger
from dashboard.data.indicators import compute_indicators
from dashboard.data.websocket_manager import BinanceWebSocketManager

logger = get_logger()


def _get_ws_manager() -> BinanceWebSocketManager | None:
    # Lê do cache_resource (a fonte da verdade), não do session_state
    try:
        from dashboard.resources import get_ws_manager
        return get_ws_manager()
    except Exception:
        return st.session_state.get('ws_manager')  # fallback legacy


def collect_market_data(
    _client,
    symbol: str = 'BTCUSDT',
    interval: str = '15m',
    limit: int = 200,
) -> pd.DataFrame | None:
    """
    Coleta dados OHLCV + indicadores técnicos.

    Prioridade:
      1. Buffer WebSocket em memória (ZERO chamadas REST)
      2. REST API — somente se buffer vazio E usuário autorizou

    Parâmetro _client prefixado com _ para compatibilidade com @st.cache_data
    (Streamlit não hasheia parâmetros começando com _).
    """
    from dashboard.resources import is_banned_session, register_ban_session

    ws_mgr = _get_ws_manager()

    # ── 1. Tenta WebSocket buffer primeiro ────────────────────────────────
    if ws_mgr is not None:
        df = ws_mgr.get_klines_df(symbol, interval, limit=max(limit, 200))
        if df is not None and len(df) >= 5:
            logger.debug(f"[DATA-WS] {symbol}/{interval}: {len(df)} candles do buffer WS")
            return df

    # ── 2. Fallback REST (apenas se explicitamente autorizado) ────────────
    is_banned, remaining = is_banned_session()
    if is_banned:
        logger.warning(f"[DATA-REST] Ban ativo: {remaining:.0f}s restantes.")
        return None

    _rest_ok = (
        st.session_state.get('_rest_connected', False)
        or st.session_state.get('bot_running', False)
    )
    if not _rest_ok:
        logger.debug(f"[DATA-REST] REST desconectado — bloqueado para {symbol}/{interval}")
        return None

    try:
        from dashboard.core.ban_manager import touch_rest_rate
        touch_rest_rate()
        logger.info(f"[DATA-REST] REST call {symbol}/{interval}")
        klines = _client.futures_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore',
        ])
        df = compute_indicators(df)

        # Popula o buffer WS com estes dados para próximos ciclos
        if ws_mgr is not None:
            ws_mgr.kline_buffers.setdefault(symbol.upper(), {})
            buf = ws_mgr.kline_buffers[symbol.upper()].setdefault(
                interval, deque(maxlen=KLINE_MAXLEN)
            )
            for _, row in df.iterrows():
                buf.append({
                    'timestamp': int(row['timestamp']) if 'timestamp' in row else 0,
                    'open':   float(row['open']),
                    'high':   float(row['high']),
                    'low':    float(row['low']),
                    'close':  float(row['close']),
                    'volume': float(row['volume']),
                })
            logger.info(
                f"[DATA-REST] Buffer WS populado via REST: "
                f"{symbol}/{interval} ({len(buf)} candles)"
            )

        logger.info(f"[DATA-REST] {symbol} {interval}: {len(df)} candles")
        return df

    except Exception as exc:
        register_ban_session(str(exc), 'DATA')
        logger.error(f"[DATA-REST] Erro {symbol} {interval}: {exc}")
        return None


def collect_multi_timeframe_data(
    client,
    symbol: str = 'BTCUSDT',
) -> dict | None:
    """
    Coleta dados de múltiplos timeframes (15m, 1h, 4h).
    Usa buffer WS em memória — ZERO chamadas REST quando bootstrapped.

    Returns:
        dict {'15m': df, '1h': df, '4h': df} ou None.
    """
    try:
        config     = load_config()
        timeframes = config['data'].get('timeframes', {
            'tactical': '15m', 'operational': '1h', 'strategic': '4h',
        })
        data: dict = {}
        for _tf_name, tf_value in timeframes.items():
            df = collect_market_data(client, symbol=symbol, interval=tf_value, limit=200)
            if df is not None:
                data[tf_value] = df
        return data or None

    except Exception as exc:
        logger.error(f"[MULTI-TF] Erro: {exc}")
        return None


def get_klines(
    _client,
    symbol: str = 'BTCUSDT',
    interval: str = '15m',
    limit: int = 100,
) -> pd.DataFrame:
    """
    Retorna candles OHLCV para gráficos.
    Usa buffer WebSocket em memória — ZERO chamadas REST quando bootstrapped.
    """
    from dashboard.resources import is_banned_session, register_ban_session
    from dashboard.core.ban_manager import touch_rest_rate

    _empty = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ws_mgr = _get_ws_manager()

    # 1. Tenta WS buffer
    if ws_mgr is not None:
        df_ws = ws_mgr.get_klines_df(symbol, interval, limit=limit)
        if df_ws is not None and not df_ws.empty:
            if 'timestamp' in df_ws.columns and not pd.api.types.is_datetime64_any_dtype(df_ws['timestamp']):
                df_ws = df_ws.copy()
                df_ws['timestamp'] = pd.to_datetime(df_ws['timestamp'], unit='ms', errors='coerce')
            return df_ws

    # 2. Fallback REST
    is_banned, _ = is_banned_session()
    if is_banned:
        return _empty
    _rest_ok = (
        st.session_state.get('_rest_connected', False)
        or st.session_state.get('bot_running', False)
    )
    if not _rest_ok:
        return _empty
    try:
        touch_rest_rate()
        klines = _client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore',
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        register_ban_session(str(exc), 'KLINES')
        logger.error(f"[KLINES-REST] {exc}")
        return _empty
