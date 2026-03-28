"""
Singletons Streamlit — todos os @st.cache_resource centralizados aqui.
Também gerencia o session_state de ban e conexão REST.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from binance.client import Client
from dotenv import load_dotenv

from dashboard.core.config import load_config_raw, get_lstm_model_path
from dashboard.core.ban_manager import (
    read_ban_from_file, write_ban_to_file, clear_ban_file,
    rest_rate_ok, touch_rest_rate, register_ban_from_error,
    parse_ban_from_error,
)
from dashboard.core.logging_setup import get_logger
from dashboard.data.websocket_manager import BinanceWebSocketManager
from dashboard.trading.engine import TradingEngine

load_dotenv()
logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & CLIENT
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_config() -> dict:
    with open('config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_binance_client() -> Client:
    config = get_config()
    mode   = config.get('mode', 'testnet')
    # timeout generoso para testnet (latência alta) — evita premature ReadTimeout
    _req_params = {'timeout': 30}
    if mode == 'testnet':
        return Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True,
            requests_params=_req_params,
        )
    return Client(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_SECRET_KEY'),
        requests_params=_req_params,
    )


# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET + ENGINE SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ws_manager() -> BinanceWebSocketManager:
    """Cria BinanceWebSocketManager uma única vez por sessão de servidor."""
    cfg  = load_config_raw()
    mode = cfg.get('mode', 'testnet')
    _req_params = {'timeout': 30}
    if mode == 'testnet':
        _client = Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True,
            requests_params=_req_params,
        )
    else:
        _client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY'),
            requests_params=_req_params,
        )
    return BinanceWebSocketManager(_client)


@st.cache_resource
def get_trading_engine() -> TradingEngine:
    """Singleton TradingEngine — sobrevive a qualquer rerun/F5/reload."""
    return TradingEngine()


# ═══════════════════════════════════════════════════════════════════════════
# MODELOS & GESTÃO DE RISCO
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_models() -> dict:
    """Carrega LSTM V17.7 (RecurrentPPO 600k) — modelo principal."""
    models: dict = {'lstm_v17': None}
    config = get_config()
    lstm_path = get_lstm_model_path(config)
    try:
        logger.info(f"[MODELS] Carregando LSTM V17.7 de {lstm_path}...")
        from sb3_contrib import RecurrentPPO
        if Path(lstm_path).exists():
            models['lstm_v17'] = RecurrentPPO.load(lstm_path)
            logger.info(f"[MODELS] ✅ LSTM V17.7 carregado: {lstm_path}")
        else:
            logger.warning(f"[MODELS] ⚠️ Não encontrado: {lstm_path}")
    except Exception as exc:
        logger.error(f"[MODELS] ❌ Erro ao carregar LSTM V17.7: {exc}")
    return models


@st.cache_resource
def get_risk_manager():
    from src.risk.risk_manager import RiskManager
    return RiskManager()


@st.cache_resource
def get_trailing_stop_manager():
    from src.trading.advanced_risk import TrailingStopManager
    config      = get_config()
    risk_config = config.get('risk_management', {})
    activation  = risk_config.get('trailing_stop_activation', 0.03)
    distance    = risk_config.get('trailing_stop_distance', 0.015)
    return TrailingStopManager(activation_pct=activation, distance_pct=distance)


@st.cache_resource
def get_warmup_manager():
    from src.trading.advanced_risk import WarmupManager
    config           = get_config()
    risk_config      = config.get('risk_management', {})
    required_candles = risk_config.get('warm_up_candles', 50)
    return WarmupManager(required_candles=required_candles)


@st.cache_resource
def get_schedule_manager():
    from src.trading.advanced_risk import ScheduleManager
    return ScheduleManager()


# ═══════════════════════════════════════════════════════════════════════════
# BAN STATE — sincroniza arquivo ↔ session_state
# ═══════════════════════════════════════════════════════════════════════════

def restore_ban_to_session() -> None:
    """
    Restaura estado de ban do arquivo para session_state.
    Deve ser chamado no início de cada rerun para que a UI reflita o ban.
    """
    if 'ban_expires_at' not in st.session_state:
        banned, expires_at, banned_at = read_ban_from_file()
        if banned:
            st.session_state['ban_expires_at'] = expires_at
            st.session_state['last_ban_time']  = datetime.fromtimestamp(banned_at)
        else:
            # Garante que chaves inexistentes não causem KeyError na UI
            st.session_state.pop('ban_expires_at', None)
            st.session_state.pop('last_ban_time', None)


def is_banned_session() -> tuple[bool, float]:
    """
    Verifica ban via session_state (chamado após restore_ban_to_session).
    Retorna (banido: bool, segundos_restantes: float).
    """
    expires_at = st.session_state.get('ban_expires_at', 0)
    remaining  = expires_at - time.time()
    if remaining > 0:
        return True, remaining
    return False, 0.0


def register_ban_session(error_str: str, context: str = '') -> bool:
    """
    Detecta ban no error_str, persiste em arquivo e atualiza session_state.
    Retorna True se ban foi detectado.
    """
    ban_expires_at = parse_ban_from_error(error_str)
    if ban_expires_at is None:
        return False
    write_ban_to_file(ban_expires_at)
    st.session_state['ban_expires_at'] = ban_expires_at
    st.session_state['last_ban_time']  = datetime.now()
    register_ban_from_error(error_str, context)
    return True


def clear_ban_session() -> None:
    """Remove ban tanto do arquivo quanto do session_state."""
    clear_ban_file()
    st.session_state.pop('ban_expires_at', None)
    st.session_state.pop('last_ban_time', None)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS session_state bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def init_session_defaults() -> None:
    """Inicializa chaves de session_state com valores padrão."""
    if '_rest_connected' not in st.session_state:
        st.session_state['_rest_connected'] = False
    ws = get_ws_manager()
    st.session_state['ws_manager'] = ws  # compat legado
