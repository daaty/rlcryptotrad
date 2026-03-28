"""
Configuração centralizada de logging — deve ser importado antes de qualquer outro módulo.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_logger_initialized = False
_logger: logging.Logger | None = None


def setup_logging() -> logging.Logger:
    """
    Inicializa handlers de arquivo (rotativo diariamente, até 10MB) e console.
    Idempotente: chamadas subsequentes retornam o mesmo logger.

    Arquivos gravados:
      logs/trading/YYYY-MM-DD.log  — rotativo 10MB, 7 arquivos
      logs/trading_decisions.log   — compat legado
    """
    global _logger_initialized, _logger
    if _logger_initialized and _logger is not None:
        return _logger

    # Diretórios
    log_dir = Path("logs/trading")
    log_dir.mkdir(parents=True, exist_ok=True)
    legacy_log = Path("logs/trading_decisions.log")

    # Reconfigura stdout para UTF-8 (Windows)
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        except Exception:
            pass

    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)

    # Handler 1: arquivo rotativo diário (10MB, guarda 7 dias)
    daily_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    rotating_handler = RotatingFileHandler(
        daily_path, maxBytes=10_000_000, backupCount=7, encoding='utf-8'
    )
    rotating_handler.setLevel(logging.DEBUG)
    rotating_handler.setFormatter(formatter)

    # Handler 2: arquivo legado (compat)
    legacy_handler = logging.FileHandler(legacy_log, encoding='utf-8')
    legacy_handler.setLevel(logging.INFO)
    legacy_handler.setFormatter(formatter)

    # Handler 3: console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        root.addHandler(rotating_handler)
        root.addHandler(legacy_handler)
        root.addHandler(console_handler)
    else:
        # Já inicializado — adiciona apenas o rotating se ainda não existe
        existing_files = {
            getattr(h, 'baseFilename', None)
            for h in root.handlers if isinstance(h, (logging.FileHandler, RotatingFileHandler))
        }
        if str(daily_path.resolve()) not in existing_files:
            root.addHandler(rotating_handler)

    _logger = logging.getLogger('trading_bot')
    _logger_initialized = True
    return _logger


def get_logger() -> logging.Logger:
    """Retorna o logger principal (sem reconfigurar)."""
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger
