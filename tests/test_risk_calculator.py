"""
Testes para dashboard/analytics/risk_calculator.py (Kelly Criterion).
"""
from __future__ import annotations

import pytest
from dashboard.analytics.risk_calculator import kelly_position_size, _MIN_TRADES


class TestKellyPositionSize:

    def test_fallback_with_no_trades(self, minimal_config):
        """Sem histórico → usa fallback_pct fixo."""
        size = kelly_position_size(balance=10_000, closed_trades=[], fallback_pct=0.03)
        assert size == pytest.approx(300.0)

    def test_fallback_with_few_trades(self, minimal_config):
        """Histórico < MIN_TRADES → usa fallback."""
        trades = [{"realizedPnl": 10.0}] * (_MIN_TRADES - 1)
        size = kelly_position_size(balance=10_000, closed_trades=trades, fallback_pct=0.03)
        assert size == pytest.approx(300.0)

    def test_kelly_with_positive_edge(self, closed_trades_sample):
        """
        60% win rate, razão ganho/perda ~2:
        Kelly = (0.6 * 2 - 0.4) / 2 = 0.40 → frac 0.25 → 10% → clampado em 10%
        Resultado deve ser positivo e dentro dos limites.
        """
        size = kelly_position_size(
            balance=10_000,
            closed_trades=closed_trades_sample,
            kelly_fraction=0.25,
            fallback_pct=0.03,
        )
        assert size > 0
        assert size <= 10_000 * 0.10  # não pode ultrapassar MAX_POSITION_PCT

    def test_kelly_never_negative(self):
        """Com win_rate muito baixo → Kelly pode ficar negativo → clipa em 0, usa fallback."""
        all_losses = [{"realizedPnl": -10.0}] * 20
        size = kelly_position_size(balance=10_000, closed_trades=all_losses)
        assert size >= 0

    def test_kelly_min_floor(self):
        """Resultado sempre >= MIN_POSITION_PCT * balance."""
        from dashboard.analytics.risk_calculator import _MIN_POSITION_PCT
        trades = [{"realizedPnl": p} for p in [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 2]]
        size = kelly_position_size(balance=10_000, closed_trades=trades)
        assert size >= 10_000 * _MIN_POSITION_PCT

    def test_zero_balance_returns_zero(self):
        """Saldo zero → retorna zero sem crash."""
        size = kelly_position_size(balance=0, closed_trades=[])
        assert size == 0.0

    def test_all_wins_uses_fallback(self):
        """Série só de wins (sem losses) → sem razão P/L → fallback."""
        all_wins = [{"realizedPnl": 10.0}] * 20
        size = kelly_position_size(
            balance=10_000, closed_trades=all_wins, fallback_pct=0.03
        )
        assert size == pytest.approx(300.0)
