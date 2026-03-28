"""
Configuração global do projeto — carregamento do config.yaml e constantes.
"""
from __future__ import annotations

import yaml
from pathlib import Path

# ── Constantes de buffer WebSocket ────────────────────────────────────────────
KLINE_MAXLEN: int = 600           # candles mantidos em memória por símbolo/intervalo
INTERVALS_WS: list[str] = ['15m', '1h', '4h']  # TFs que o bot precisa
KLINE_LIMIT_BOOT: dict[str, int] = {'15m': 500, '1h': 200, '4h': 100}  # candles no bootstrap

# ── Caminhos de estado persistente ────────────────────────────────────────────
BAN_FILE = Path("logs/.ban_state.json")
REST_RATE_FILE = Path("logs/.last_rest_call")
REST_COOLDOWN_SECS: int = 90  # nunca faça REST calls com menos de 90s de intervalo

# ── Caminho padrão do modelo (pode ser sobrescrito via config.yaml) ────────────
DEFAULT_LSTM_PATH = "models/recurrent_ppo_v17_lstm_20260221_030417_600000_steps.zip"


def load_config_raw() -> dict:
    """
    Carrega config.yaml sem cache.
    Seguro para chamar antes da inicialização do Streamlit.
    """
    with open('config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    """
    Carrega config.yaml (sem cache — use @st.cache_resource em resources.py).
    """
    return load_config_raw()


def get_lstm_model_path(config: dict) -> str:
    """Retorna caminho do modelo LSTM do config.yaml, com fallback para o padrão.

    Prioridade de leitura:
      1. models.lstm_active.path  — definido pelo Champion/Challenger ao promover
      2. models.lstm_v17.path     — legado
      3. DEFAULT_LSTM_PATH        — hardcoded fallback
    """
    models_block = config.get('models', {})
    active = models_block.get('lstm_active', {}).get('path')
    if active:
        return active
    legacy = models_block.get('lstm_v17', {}).get('path')
    if legacy:
        return legacy
    return DEFAULT_LSTM_PATH


def get_vecnorm_path(config: dict) -> str | None:
    """Retorna caminho do VecNormalize .pkl para o modelo ativo, ou None."""
    return config.get('models', {}).get('lstm_active', {}).get('vecnorm_path')


def get_quantity_precision(config: dict, symbol: str) -> int:
    """
    Retorna número de casas decimais para quantidade de um símbolo.
    Lê de config.yaml['trading']['quantity_precision'] com fallback hardcoded.
    """
    hardcoded = {
        'BTCUSDT': 3,
        'ETHUSDT': 3,
        'BNBUSDT': 2,
        'SOLUSDT': 1,
        'ADAUSDT': 0,
        'DOTUSDT': 1,
        'MATICUSDT': 0,
    }
    from_config: dict = config.get('trading', {}).get('quantity_precision', {})
    precision = from_config.get(symbol, hardcoded.get(symbol, 3))
    return int(precision)
