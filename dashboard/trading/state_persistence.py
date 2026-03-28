"""
Persistência de estado do TradingEngine em disco.

Salva e restaura:
  - _tp1_done          set[str]           → lista JSON
  - _last_candle_ts    dict[str, int]      → dict JSON
  - lstm_states        dict[str, tuple]    → base64 numpy
  - trail_active_stops dict[str, dict]     → dict JSON (sem datetimes)

Formato: data/engine_state.json

Invocado ao final de cada tick (save) e no __init__ do engine (load).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/engine_state.json")


# ─── Numpy serialization ──────────────────────────────────────────────────────

def _ndarray_to_b64(arr: np.ndarray) -> dict:
    """Converte ndarray para dict serializável em JSON."""
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _b64_to_ndarray(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["data"])
    return np.frombuffer(raw, dtype=d["dtype"]).reshape(d["shape"])


def _serialize_lstm_states(lstm_states: Any) -> Any:
    """
    RecurrentPPO lstm_states é uma tupla de tuplas de ndarrays.
    Estrutura típica: ((h_actor, c_actor), (h_critic, c_critic))
    """
    if lstm_states is None:
        return None
    if isinstance(lstm_states, np.ndarray):
        return _ndarray_to_b64(lstm_states)
    if isinstance(lstm_states, (list, tuple)):
        return [_serialize_lstm_states(x) for x in lstm_states]
    return lstm_states


def _deserialize_lstm_states(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, dict) and data.get("__ndarray__"):
        return _b64_to_ndarray(data)
    if isinstance(data, list):
        result = [_deserialize_lstm_states(x) for x in data]
        # Reconstrói como tuple (formato original do RecurrentPPO)
        return tuple(result)
    return data


def _serialize_trail_stops(active_stops: dict) -> dict:
    """
    Serializa active_stops do TrailingStopManager.
    Remove objetos datetime (não serializáveis); opened_at fica como timestamp float.
    """
    result = {}
    for sym, d in active_stops.items():
        row = {}
        for k, v in d.items():
            if hasattr(v, "timestamp"):   # datetime
                row[k] = v.timestamp()
            elif v is None or isinstance(v, (int, float, bool, str)):
                row[k] = v
            else:
                row[k] = v  # fallback
        result[sym] = row
    return result


def _deserialize_trail_stops(data: dict) -> dict:
    """
    Restaura active_stops. 'opened_at' fica como timestamp float —
    TrailingStopManager usa datetime.now() apenas para logging, não é crítico.
    """
    from datetime import datetime
    result = {}
    for sym, d in data.items():
        row = dict(d)
        if "opened_at" in row and isinstance(row["opened_at"], float):
            try:
                row["opened_at"] = datetime.fromtimestamp(row["opened_at"])
            except Exception:
                row["opened_at"] = datetime.now()
        result[sym] = row
    return result


# ─── API pública ──────────────────────────────────────────────────────────────

def save_state(
    tp1_done: set,
    last_candle_ts: dict,
    lstm_states_map: dict,
    trail_active_stops: dict,
) -> None:
    """
    Persiste estado do engine em disco atomicamente.
    Escrita em .tmp e os.replace para evitar corrupção parcial.
    """
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at":        time.time(),
            "tp1_done":        sorted(tp1_done),
            "last_candle_ts":  last_candle_ts,
            "lstm_states":     {sym: _serialize_lstm_states(st)
                                 for sym, st in lstm_states_map.items()},
            "trail_stops":     _serialize_trail_stops(trail_active_stops),
        }
        tmp_path = STATE_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except Exception as exc:
        logger.warning(f"[STATE] Falha ao salvar estado: {exc}")


def load_state() -> dict | None:
    """
    Carrega estado salvo em disco.

    Returns:
        dict com chaves: tp1_done, last_candle_ts, lstm_states, trail_stops
        None se arquivo não existe ou está corrompido.
    """
    if not STATE_PATH.exists():
        logger.info("[STATE] Nenhum estado salvo encontrado — iniciando do zero")
        return None
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            payload = json.load(f)

        age = time.time() - payload.get("saved_at", 0)
        logger.info(f"[STATE] Estado carregado (idade: {age/3600:.1f}h)")

        return {
            "tp1_done":       set(payload.get("tp1_done", [])),
            "last_candle_ts": {k: int(v) for k, v in
                               payload.get("last_candle_ts", {}).items()},
            "lstm_states":    {sym: _deserialize_lstm_states(st)
                               for sym, st in payload.get("lstm_states", {}).items()},
            "trail_stops":    _deserialize_trail_stops(payload.get("trail_stops", {})),
        }
    except Exception as exc:
        logger.warning(f"[STATE] Falha ao carregar estado ({exc}) — iniciando do zero")
        return None
