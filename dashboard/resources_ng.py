"""
Singletons para NiceGUI — sem dependências do Streamlit.
Substitui resources.py quando usando main.py (NiceGUI).

Todas as funções get_*() são idempotentes (criam o objeto apenas uma vez).
Thread-safe através de _lock. Pode ser chamado de qualquer thread.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml
from binance.client import Client
from dotenv import load_dotenv

from dashboard.core.config import load_config_raw, get_lstm_model_path, get_vecnorm_path
from dashboard.core.ban_manager import (
    read_ban_from_file, write_ban_to_file, clear_ban_file,
    parse_ban_from_error, register_ban_from_error,
)
from dashboard.core.logging_setup import get_logger
from dashboard.data.websocket_manager import BinanceWebSocketManager
from dashboard.trading.engine import TradingEngine

load_dotenv()
logger = get_logger()

_lock = threading.Lock()

# ── Singletons ────────────────────────────────────────────────────────────────
_config: dict | None              = None
_client: Client | None            = None
_ws_manager: BinanceWebSocketManager | None = None
_engine: TradingEngine | None     = None
_models: dict | None              = None
_risk_mgr                         = None
_trail_mgr                        = None
_warmup_mgr                       = None
_schedule_mgr                     = None


def get_config() -> dict:
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                with open('config.yaml', encoding='utf-8') as f:
                    _config = yaml.safe_load(f)
    return _config


def reload_config() -> dict:
    """Force re-read config.yaml (usado após Promote Champion)."""
    global _config
    with _lock:
        with open('config.yaml', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
    return _config


def _make_client(config: dict) -> Client:
    mode = config.get('mode', 'testnet')
    req  = {'timeout': 30}
    if mode == 'testnet':
        return Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True, requests_params=req,
        )
    return Client(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_SECRET_KEY'),
        requests_params=req,
    )


def get_binance_client() -> Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _make_client(get_config())
    return _client


def get_ws_manager() -> BinanceWebSocketManager:
    global _ws_manager
    if _ws_manager is None:
        with _lock:
            if _ws_manager is None:
                cfg    = load_config_raw()
                client = _make_client(cfg)
                _ws_manager = BinanceWebSocketManager(client)
    return _ws_manager


def get_trading_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = TradingEngine()
    return _engine


def get_models() -> dict:
    global _models
    if _models is None:
        with _lock:
            if _models is None:
                models: dict = {'lstm_v17': None, 'vecnorm': None}
                config     = get_config()
                lstm_path  = get_lstm_model_path(config)
                try:
                    from sb3_contrib import RecurrentPPO
                    if Path(lstm_path).exists():
                        models['lstm_v17'] = RecurrentPPO.load(lstm_path)
                        logger.info(f"[MODELS] ✅ LSTM carregado: {lstm_path}")
                    else:
                        logger.warning(f"[MODELS] Não encontrado: {lstm_path}")
                except Exception as exc:
                    logger.error(f"[MODELS] Erro ao carregar LSTM: {exc}")
                # VecNormalize — obrigatório para V19 (normalização das observações)
                vecnorm_path = get_vecnorm_path(config)
                if vecnorm_path:
                    try:
                        import pickle
                        vn_path = Path(vecnorm_path)
                        if vn_path.exists():
                            with open(vn_path, 'rb') as _f:
                                models['vecnorm'] = pickle.load(_f)
                            logger.info(f"[MODELS] ✅ VecNormalize carregado: {vecnorm_path}")
                        else:
                            logger.warning(f"[MODELS] VecNormalize não encontrado: {vecnorm_path}")
                    except Exception as exc:
                        logger.error(f"[MODELS] Erro ao carregar VecNormalize: {exc}")
                _models = models
    return _models


def get_risk_manager():
    global _risk_mgr
    if _risk_mgr is None:
        with _lock:
            if _risk_mgr is None:
                from src.risk.risk_manager import RiskManager
                _risk_mgr = RiskManager()
    return _risk_mgr


def get_trailing_stop_manager():
    global _trail_mgr
    if _trail_mgr is None:
        with _lock:
            if _trail_mgr is None:
                from src.trading.advanced_risk import TrailingStopManager
                cfg = get_config()
                rm  = cfg.get('risk_management', {})
                _trail_mgr = TrailingStopManager(
                    activation_pct=rm.get('trailing_stop_activation', 0.03),
                    distance_pct=rm.get('trailing_stop_distance', 0.015),
                )
    return _trail_mgr


def get_warmup_manager():
    global _warmup_mgr
    if _warmup_mgr is None:
        with _lock:
            if _warmup_mgr is None:
                from src.trading.advanced_risk import WarmupManager
                cfg = get_config()
                rm  = cfg.get('risk_management', {})
                _warmup_mgr = WarmupManager(
                    required_candles=rm.get('warm_up_candles', 50),
                )
    return _warmup_mgr


def get_schedule_manager():
    global _schedule_mgr
    if _schedule_mgr is None:
        with _lock:
            if _schedule_mgr is None:
                from src.trading.advanced_risk import ScheduleManager
                _schedule_mgr = ScheduleManager()
    return _schedule_mgr


# ── Ban helpers (sem session_state) ──────────────────────────────────────────

import time as _time


def is_banned() -> tuple[bool, float]:
    """Retorna (banido, segundos_restantes). Lê direto do arquivo."""
    banned, expires_at, _ = read_ban_from_file()
    if not banned:
        return False, 0.0
    remaining = expires_at - _time.time()
    return (remaining > 0, max(0.0, remaining))


def get_ban_expires_at() -> float:
    """Retorna timestamp Unix da expiração do ban (0 se não banido)."""
    _, expires_at, _ = read_ban_from_file()
    return expires_at


def register_ban(error_str: str, context: str = '') -> bool:
    """Detecta e persiste ban do error_str. Retorna True se ban detectado."""
    ban_at = parse_ban_from_error(error_str)
    if ban_at is None:
        return False
    write_ban_to_file(ban_at)
    register_ban_from_error(error_str, context)
    return True


def clear_ban() -> None:
    clear_ban_file()


# ── Cached positions (evita REST a cada render) ───────────────────────────────

_positions_cache: list[dict] = []
_positions_ts: float = 0.0
_POSITIONS_TTL = 5.0   # segundos


def get_open_positions_cached() -> list[dict]:
    """Retorna posições abertas do WS manager (sem REST)."""
    global _positions_cache, _positions_ts
    now = _time.time()
    if now - _positions_ts < _POSITIONS_TTL:
        return _positions_cache
    ws = get_ws_manager()
    raw = ws.get_positions()
    if isinstance(raw, list):
        _positions_cache = raw
    elif isinstance(raw, dict):
        _positions_cache = raw.get('positions', [])
    else:
        _positions_cache = []
    _positions_ts = now
    return _positions_cache
