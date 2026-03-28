"""
Testes para dashboard/trading/state_persistence.py.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import dashboard.trading.state_persistence as sp


@pytest.fixture(autouse=True)
def tmp_state_path(tmp_path):
    """Redireciona STATE_PATH para um diretório temporário em cada teste."""
    test_path = tmp_path / "engine_state.json"
    with patch.object(sp, "STATE_PATH", test_path):
        yield test_path


class TestSaveLoadState:

    def test_roundtrip_basic(self):
        """Salva e restaura tp1_done, last_candle_ts."""
        sp.save_state(
            tp1_done={"BTCUSDT", "ETHUSDT"},
            last_candle_ts={"BTCUSDT": 1700000000000},
            lstm_states_map={},
            trail_active_stops={},
        )
        result = sp.load_state()
        assert result is not None
        assert "BTCUSDT" in result["tp1_done"]
        assert "ETHUSDT" in result["tp1_done"]
        assert result["last_candle_ts"]["BTCUSDT"] == 1700000000000

    def test_roundtrip_numpy_arrays(self):
        """Serializa e restaura ndarrays (lstm_states)."""
        arr1 = np.random.randn(1, 64).astype(np.float32)
        arr2 = np.random.randn(1, 64).astype(np.float32)
        lstm = {"BTCUSDT": ((arr1, arr2),)}

        sp.save_state(
            tp1_done=set(),
            last_candle_ts={},
            lstm_states_map=lstm,
            trail_active_stops={},
        )
        result = sp.load_state()
        assert result is not None
        restored = result["lstm_states"]["BTCUSDT"]
        # Estrutura: tuple com tupla com dois arrays
        np.testing.assert_allclose(restored[0][0], arr1, rtol=1e-5)
        np.testing.assert_allclose(restored[0][1], arr2, rtol=1e-5)

    def test_roundtrip_trail_stops(self):
        """Serializa e restaura active_stops do TrailingStopManager."""
        from datetime import datetime
        stops = {
            "BTCUSDT": {
                "entry_price": 95000.0,
                "position_type": 1,
                "highest_mark": 96000.0,
                "stop_price": 94500.0,
                "activated": True,
                "opened_at": datetime.now(),
                "lowest_mark": 95000.0,
            }
        }
        sp.save_state(set(), {}, {}, stops)
        result = sp.load_state()
        assert result is not None
        btc_stop = result["trail_stops"]["BTCUSDT"]
        assert btc_stop["entry_price"] == pytest.approx(95000.0)
        assert btc_stop["activated"] is True

    def test_load_returns_none_when_no_file(self):
        """Se arquivo não existe → retorna None sem crash."""
        result = sp.load_state()
        assert result is None

    def test_load_returns_none_on_corrupt_file(self, tmp_state_path):
        """Arquivo corrompido → retorna None, não lança exceção."""
        tmp_state_path.write_text("{ invalid json }", encoding="utf-8")
        result = sp.load_state()
        assert result is None

    def test_atomic_write(self, tmp_state_path):
        """Arquivo .tmp não deve ficar após save com sucesso."""
        tmp_path = tmp_state_path.with_suffix(".tmp")
        sp.save_state({"X"}, {"A": 1}, {}, {})
        assert not tmp_path.exists()
        assert tmp_state_path.exists()
