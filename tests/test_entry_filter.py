"""
Testes para dashboard/trading/entry_filter.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dashboard.trading.entry_filter import validate_entry_quality


class TestValidateEntryQuality:

    def test_flat_always_passes(self, sample_df):
        """FLAT nunca é bloqueado independente das condições."""
        can, reason = validate_entry_quality(sample_df, "FLAT", 100.0, mode="strict")
        assert can is True

    def test_disabled_mode_always_passes(self, sample_df):
        """Modo disabled não bloqueia nada."""
        can, _ = validate_entry_quality(sample_df, "LONG", 100.0, mode="disabled")
        assert can is True
        can, _ = validate_entry_quality(sample_df, "SHORT", 100.0, mode="disabled")
        assert can is True

    def test_overbought_rsi_blocks_long_strict(self, sample_df):
        """RSI > 0.70 em modo strict deve bloquear LONG."""
        df = sample_df.copy()
        # Seta RSI alto (normalizado 0-1) nos últimos candles (penúltimo = índice -2)
        df.at[df.index[-2], "RSI_14"] = 0.75  # 75% → overbought
        can, reason = validate_entry_quality(df, "LONG", float(df["close"].iloc[-1]), mode="strict")
        assert can is False
        assert "RSI" in reason.upper() or "ob" in reason.lower()

    def test_oversold_rsi_blocks_short_strict(self, sample_df):
        """RSI < 0.30 em modo strict deve bloquear SHORT."""
        df = sample_df.copy()
        df.at[df.index[-2], "RSI_14"] = 0.25
        can, reason = validate_entry_quality(df, "SHORT", float(df["close"].iloc[-1]), mode="strict")
        assert can is False

    def test_normal_rsi_threshold_is_wider(self, sample_df):
        """RSI 0.75 (=75 normalizado) não bloqueia em modo normal (threshold RSI_ob=80).
        Usa close como current_price para evitar trigger do filtro de distância EMA."""
        df = sample_df.copy()
        df.at[df.index[-2], "RSI_14"] = 0.75
        # current_price = close do penúltimo candle — sem distância EMA
        current_price = float(df["close"].iloc[-2])
        can, reason = validate_entry_quality(df, "LONG", current_price, mode="normal")
        assert can is True, f"Esperava can=True mas bloqueou: {reason}"

    def test_few_rows_does_not_crash(self):
        """DataFrame com 2 linhas não deve lançar exceção."""
        tiny = pd.DataFrame({
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [500, 600],
            "RSI_14": [0.5, 0.5], "EMA_21": [100, 101],
            "Volume_MA_20": [1.0, 1.0],
        })
        can, reason = validate_entry_quality(tiny, "LONG", 102.0, mode="normal")
        assert isinstance(can, bool)

    def test_extreme_rsi_blocks_aggressive(self, sample_df):
        """RSI > 0.85 ainda bloqueia em modo aggressive."""
        df = sample_df.copy()
        df.at[df.index[-2], "RSI_14"] = 0.90
        can, _ = validate_entry_quality(df, "LONG", float(df["close"].iloc[-1]), mode="aggressive")
        assert can is False
