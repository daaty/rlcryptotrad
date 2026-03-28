"""
Cálculo de risco adaptativo — Kelly Criterion fracionário.

Substitui o position_size fixo do config.yaml por um tamanho de posição
calculado dinamicamente com base no histórico real de trades fechados.

Usado por dashboard/trading/executor.py.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# Limites de segurança
_MIN_POSITION_PCT = 0.01    # 1.0% mínimo — garante notional útil mesmo com saldo baixo
_MAX_POSITION_PCT = 0.10    # 10% máximo por posição
_MIN_TRADES = 10            # mínimo de trades para ativar Kelly


def kelly_position_size(
    balance: float,
    closed_trades: Sequence[dict],
    kelly_fraction: float = 0.25,
    fallback_pct: float = 0.03,
) -> float:
    """
    Calcula o tamanho ideal da posição usando Kelly Criterion fracionário,
    baseado no histórico real de trades fechados.

    Kelly % = (W × P - L) / P
    Onde:
        W = win_rate (fração de trades com PnL > 0)
        L = 1 - W
        P = avg_win / avg_loss (razão de ganho/perda)

    Aplica fração conservadora (kelly_fraction) e clipa entre limites.

    Args:
        balance:       Saldo disponível em USDT
        closed_trades: Lista de dicts com chave 'realizedPnl' (últimos N trades)
        kelly_fraction: Fração do Kelly completo a usar (default 25%)
        fallback_pct:  Position size fixo se histórico insuficiente

    Returns:
        Valor em USDT a usar na posição (já limitado por _MAX_POSITION_PCT)
    """
    if not closed_trades or len(closed_trades) < _MIN_TRADES:
        size = balance * fallback_pct
        logger.debug(
            f"[KELLY] Histórico insuficiente ({len(closed_trades)} trades) "
            f"→ fallback {fallback_pct:.1%} = ${size:.2f}"
        )
        return max(0.0, size)

    # Usa apenas os últimos 30 trades (janela deslizante)
    recent = list(closed_trades)[-30:]
    pnls   = [float(t.get('realizedPnl', 0)) for t in recent]

    wins  = [p for p in pnls if p > 0]
    loses = [p for p in pnls if p < 0]

    if not wins or not loses:
        # Série só de wins (ou só de losses) — usa fallback conservador
        size = balance * fallback_pct
        logger.debug(f"[KELLY] Sem diversidade win/loss → fallback ${size:.2f}")
        return max(0.0, size)

    win_rate  = len(wins) / len(recent)
    avg_win   = sum(wins)  / len(wins)
    avg_loss  = abs(sum(loses) / len(loses))

    if avg_loss == 0:
        return balance * fallback_pct

    profit_loss_ratio = avg_win / avg_loss
    loss_rate         = 1 - win_rate
    kelly_pct         = (win_rate * profit_loss_ratio - loss_rate) / profit_loss_ratio
    kelly_pct         = max(0.0, kelly_pct)  # nunca negativo

    fractional = kelly_pct * kelly_fraction
    clamped    = max(_MIN_POSITION_PCT, min(_MAX_POSITION_PCT, fractional))
    size       = balance * clamped

    logger.info(
        f"[KELLY] wr={win_rate:.1%} avg_win={avg_win:.2f} avg_loss={avg_loss:.2f} "
        f"kelly={kelly_pct:.1%} → frac={fractional:.1%} → clamp={clamped:.1%} = ${size:.2f}"
    )
    return size
