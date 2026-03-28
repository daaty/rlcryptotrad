"""Módulo auxiliar para manter os símbolos selecionados no drawer."""
from __future__ import annotations

_selected: list[str] = []

ALL_SYMBOLS = [
    'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
    'DOGEUSDT','AVAXUSDT','DOTUSDT','LINKUSDT','MATICUSDT','LTCUSDT',
    'UNIUSDT','ATOMUSDT','FILUSDT','NEARUSDT','APTUSDT','ARBUSDT',
    'OPUSDT','INJUSDT',
]


def get_selected_symbols() -> list[str]:
    return list(_selected)


def set_selected_symbols(syms: list[str]) -> None:
    global _selected
    _selected = list(syms)
