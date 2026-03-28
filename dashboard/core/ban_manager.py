"""
Gerenciamento de estado de ban da API Binance.
Persiste em arquivo para sobreviver a reloads de página do Streamlit.
Esta camada é pura (sem dependência de st.*) — Streamlit lê/escreve
session_state em resources.py e nas helpers de UI.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from dashboard.core.config import BAN_FILE, REST_RATE_FILE, REST_COOLDOWN_SECS
from dashboard.core.logging_setup import get_logger

logger = get_logger()


# ── REST rate-limit ────────────────────────────────────────────────────────────

def rest_rate_ok() -> tuple[bool, float]:
    """
    Verifica cooldown mínimo entre chamadas REST (persiste via arquivo).
    Retorna (pode_chamar: bool, segundos_para_liberar: float).
    """
    try:
        if REST_RATE_FILE.exists():
            last_call = float(REST_RATE_FILE.read_text().strip())
            elapsed = time.time() - last_call
            if elapsed < REST_COOLDOWN_SECS:
                return False, REST_COOLDOWN_SECS - elapsed
    except Exception:
        pass
    return True, 0.0


def touch_rest_rate() -> None:
    """Registra timestamp da última chamada REST em arquivo."""
    try:
        REST_RATE_FILE.parent.mkdir(exist_ok=True)
        REST_RATE_FILE.write_text(str(time.time()))
    except Exception:
        pass


# ── Ban detection & persistence ───────────────────────────────────────────────

def read_ban_from_file() -> tuple[bool, float, float]:
    """
    Lê estado de ban do arquivo persistente.
    Retorna (banido: bool, ban_expires_at: float, banned_at: float).
    """
    try:
        if BAN_FILE.exists():
            data = json.loads(BAN_FILE.read_text())
            expires_at = float(data.get('ban_expires_at', 0))
            banned_at  = float(data.get('banned_at', 0))
            if expires_at > time.time():
                return True, expires_at, banned_at
            else:
                BAN_FILE.unlink(missing_ok=True)  # ban expirado
    except Exception:
        pass
    return False, 0.0, 0.0


def write_ban_to_file(ban_expires_at: float) -> None:
    """Salva estado de ban em arquivo para sobreviver a page reloads."""
    try:
        BAN_FILE.parent.mkdir(exist_ok=True)
        BAN_FILE.write_text(json.dumps({
            'ban_expires_at': ban_expires_at,
            'banned_at': time.time(),
        }))
    except Exception:
        pass


def clear_ban_file() -> None:
    """Remove arquivo de ban (chamado quando ban expira)."""
    try:
        BAN_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def is_banned_from_file() -> tuple[bool, float]:
    """
    Verifica se IP está banido consultando apenas o arquivo.
    Retorna (banido: bool, segundos_restantes: float).
    """
    banned, expires_at, _ = read_ban_from_file()
    if banned:
        return True, expires_at - time.time()
    return False, 0.0


def parse_ban_from_error(error_str: str) -> float | None:
    """
    Tenta extrair timestamp de expiração de ban da mensagem de erro Binance.
    Retorna ban_expires_at (epoch float) ou None se não for ban.
    """
    lower = error_str.lower()
    if 'banned' not in lower and '-1003' not in error_str:
        return None
    match = re.search(r'banned until (\d+)', error_str)
    if match:
        return int(match.group(1)) / 1000
    return time.time() + 600  # fallback: 10 min


def register_ban_from_error(error_str: str, context: str = '') -> bool:
    """
    Detecta e persiste ban a partir de mensagem de erro da Binance.
    Retorna True se ban foi detectado e registrado.
    NÃO toca em st.session_state — isso é feito em resources.py.
    """
    ban_expires_at = parse_ban_from_error(error_str)
    if ban_expires_at is None:
        return False

    write_ban_to_file(ban_expires_at)

    expires_str = datetime.fromtimestamp(ban_expires_at).strftime('%H:%M:%S')
    remaining_min = (ban_expires_at - time.time()) / 60
    tag = f"[{context}] " if context else ""
    logger.error(
        f"{tag}IP BANIDO até {expires_str} ({remaining_min:.1f} min restantes) — "
        "ban.json salvo, próximas chamadas REST bloqueadas automaticamente"
    )
    return True
