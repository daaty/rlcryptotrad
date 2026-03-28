"""
Dados de conta Binance (balance, positions, trades).
Prioridade: WebSocket singleton (in-memory, live) → REST throttled via session_state.

Arquitetura de dados:
  1. ws_mgr._account_refresh_thread: chama REST a cada 60s em background → garante
     que user_data[balance|positions] tem no máximo 60s de idade.
  2. ws_mgr._handle_user_data: atualiza imediatamente quando ACCOUNT_UPDATE chega
     (após cada ordem executada).
  3. get_balance() / get_positions(): enriquecem markPrice do buffer kline (live)
     e recalculam unrealized_pnl em tempo real sem precisar de chamada REST.
  4. Aqui (account_data.py): lê do WS sem custo; fallback REST só se WS não iniciou.
"""
from __future__ import annotations

import time

import streamlit as st

from dashboard.core.logging_setup import get_logger

logger = get_logger()

_EMPTY_BALANCE   = {'total': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0,
                    'source': 'offline', 'error': None, 'age_secs': 0}
_EMPTY_POSITIONS = {'positions': [], 'source': 'offline', 'error': None, 'age_secs': 0}

# Throttle REST de fallback — só usado se WS não está disponível
_REST_THROTTLE = 120


def _can_call_rest(key: str) -> bool:
    """Retorna True se já passaram _REST_THROTTLE segundos desde a última chamada REST."""
    last = st.session_state.get(f'_rest_ts_{key}', 0.0)
    return (time.time() - last) >= _REST_THROTTLE


def _mark_rest_called(key: str) -> None:
    st.session_state[f'_rest_ts_{key}'] = time.time()


def _cache_rest(key: str, value) -> None:
    """Guarda último resultado REST no session_state para reutilizar durante throttle."""
    st.session_state[f'_rest_cache_{key}'] = value


def _get_rest_cache(key: str, default):
    return st.session_state.get(f'_rest_cache_{key}', default)


# ═══════════════════════════════════════════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════════════════════════════════════════

def get_account_balance_cached(_client) -> dict:
    """
    Retorna saldo da conta.
    1) Lê do WS singleton (in-memory, atualizado pelo background refresh a cada 60s).
    2) Se WS não tem dados ainda (primeiros 15s), faz REST apenas se throttle permitir.
    Retorna 'age_secs' para o UI mostrar frescor dos dados.
    """
    from dashboard.resources import get_ws_manager, is_banned_session, register_ban_session

    ws_mgr = get_ws_manager()
    if ws_mgr and ws_mgr.running:
        ws_balance = ws_mgr.get_balance()
        if ws_balance:
            return {**ws_balance, 'source': 'websocket'}

    # WS não tem dados ainda — fallback REST
    is_banned, remaining = is_banned_session()
    if is_banned:
        logger.warning(f"[BALANCE] Ban ativo: {remaining:.0f}s restantes")
        return _get_rest_cache('balance', {**_EMPTY_BALANCE, 'source': 'banned', 'error': 'IP banned'})

    if not _can_call_rest('balance'):
        return _get_rest_cache('balance', {**_EMPTY_BALANCE, 'source': 'rest_cached'})

    logger.debug("[BALANCE] WS sem dados — chamando REST (fallback inicial)")
    try:
        balance = _client.futures_account_balance()
        usdt    = next((b for b in balance if b['asset'] == 'USDT'), None)
        if usdt is None:
            result = {**_EMPTY_BALANCE, 'source': 'rest'}
        else:
            result = {
                'total':          float(usdt['balance']),
                'available':      float(usdt['availableBalance']),
                'unrealized_pnl': float(usdt.get('crossUnPnl', 0)),
                'source':         'rest',
                'error':          None,
                'age_secs':       0,
            }
        _mark_rest_called('balance')
        _cache_rest('balance', result)
        return result
    except Exception as exc:
        register_ban_session(str(exc), 'BALANCE')
        return {**_EMPTY_BALANCE, 'source': 'error', 'error': str(exc)}


def get_account_balance(client) -> dict:
    """Alias para compatibilidade."""
    return get_account_balance_cached(client)


# ═══════════════════════════════════════════════════════════════════════════
# POSITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_open_positions_cached(_client) -> dict:
    """
    Retorna posições abertas com markPrice e unRealizedProfit enriquecidos.
    1) Lê do WS singleton (atualizado a cada 60s pelo background refresh).
    2) Fallback REST se WS ainda não tem dados.
    """
    from dashboard.resources import get_ws_manager, is_banned_session, register_ban_session

    ws_mgr = get_ws_manager()
    if ws_mgr and ws_mgr.running:
        ws_positions = ws_mgr.get_positions()
        if ws_positions is not None:
            return {**ws_positions, 'source': 'websocket'}

    # WS sem dados ainda — fallback REST
    is_banned, remaining = is_banned_session()
    if is_banned:
        logger.warning(f"[POSITIONS] Ban ativo: {remaining:.0f}s restantes")
        return _get_rest_cache('positions', {**_EMPTY_POSITIONS, 'source': 'banned', 'error': 'IP banned'})

    if not _can_call_rest('positions'):
        return _get_rest_cache('positions', {**_EMPTY_POSITIONS, 'source': 'rest_cached'})

    logger.debug("[POSITIONS] WS sem dados — chamando REST (fallback inicial)")
    try:
        positions      = _client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        result = {'positions': open_positions, 'source': 'rest', 'error': None, 'age_secs': 0}
        _mark_rest_called('positions')
        _cache_rest('positions', result)
        return result
    except Exception as exc:
        register_ban_session(str(exc), 'POSITIONS')
        return {**_EMPTY_POSITIONS, 'source': 'error', 'error': str(exc)}


def get_open_positions(client) -> list[dict]:
    """Retorna lista de posições abertas (wrapper sem cache)."""
    result = get_open_positions_cached(client)
    if result.get('error'):
        logger.warning(f"[POSITIONS] Erro: {result['error']}")
        return []
    return result.get('positions', [])


# ═══════════════════════════════════════════════════════════════════════════
# TRADES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120)
def get_recent_trades(_client, symbol: str | None = None,
                      symbols: list[str] | None = None, limit: int = 10) -> list[dict]:
    """Retorna trades recentes com proteção anti-ban."""
    from dashboard.resources import is_banned_session, register_ban_session

    is_banned, remaining = is_banned_session()
    if is_banned:
        logger.warning(f"[TRADES] Ban ativo: {remaining:.0f}s restantes")
        return []

    try:
        if symbols:
            all_trades: list[dict] = []
            for sym in symbols:
                try:
                    trades = _client.futures_account_trades(symbol=sym, limit=limit)
                    all_trades.extend(trades)
                except Exception:
                    pass
            all_trades.sort(key=lambda x: x['time'], reverse=True)
            return all_trades[:limit] if limit else all_trades
        elif symbol:
            return _client.futures_account_trades(symbol=symbol, limit=limit)
        else:
            return []
    except Exception as exc:
        register_ban_session(str(exc), 'TRADES')
        return []
