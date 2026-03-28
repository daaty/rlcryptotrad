"""
Métricas de performance de trading e dimensionamento dinâmico de posição.
Funções puras sem dependência de Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.core.logging_setup import get_logger

logger = get_logger()


def calculate_performance_metrics(trades: list[dict]) -> dict | None:
    """
    Calcula métricas avançadas de performance a partir de uma lista de trades.

    Args:
        trades: lista de dicts com chaves 'realizedPnl' e 'time' (ms).

    Returns:
        dict com métricas ou None se menos de 2 trades.
    """
    if not trades or len(trades) < 1:
        return None

    df = pd.DataFrame(trades)
    if 'realizedPnl' not in df.columns:
        return None
    df['realizedPnl'] = df['realizedPnl'].astype(float)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='ms')

    total_trades = len(df)
    wins         = len(df[df['realizedPnl'] > 0])
    losses       = len(df[df['realizedPnl'] < 0])
    win_rate     = wins / total_trades if total_trades > 0 else 0.0

    total_pnl = df['realizedPnl'].sum()
    avg_win   = df[df['realizedPnl'] > 0]['realizedPnl'].mean() if wins   > 0 else 0.0
    avg_loss  = df[df['realizedPnl'] < 0]['realizedPnl'].mean() if losses > 0 else 0.0

    returns = df['realizedPnl']
    sharpe_ratio = (
        (returns.mean() / returns.std()) * np.sqrt(365)
        if len(returns) > 1 and returns.std() > 0 else 0.0
    )

    gross_profit = df[df['realizedPnl'] > 0]['realizedPnl'].sum()
    gross_loss   = abs(df[df['realizedPnl'] < 0]['realizedPnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    df['cumulative_pnl'] = df['realizedPnl'].cumsum()
    df['running_max']    = df['cumulative_pnl'].cummax()
    df['drawdown']       = df['cumulative_pnl'] - df['running_max']
    max_drawdown         = float(df['drawdown'].min())

    recovery_factor = (total_pnl / abs(max_drawdown)) if max_drawdown < 0 else float('inf')
    expectancy      = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    return {
        'total_trades':    total_trades,
        'wins':            wins,
        'losses':          losses,
        'win_rate':        win_rate,
        'total_pnl':       total_pnl,
        'avg_trade_pnl':   float(total_pnl / total_trades) if total_trades > 0 else 0.0,
        'avg_win':         avg_win,
        'avg_loss':        avg_loss,
        'sharpe_ratio':    sharpe_ratio,
        'profit_factor':   profit_factor,
        'max_drawdown':    max_drawdown,
        'recovery_factor': recovery_factor,
        'expectancy':      expectancy,
    }


def calculate_position_size_dynamic(
    balance: float,
    base_size: float,
    volatility_atr: float,
    current_price: float,
    leverage: int,
    win_streak: int = 0,
    regime: str = 'SIDEWAYS',
    confidence: float = 1.0,
    risk_config: dict | None = None,
) -> float:
    """
    Calcula tamanho de posição dinâmico baseado em múltiplos fatores.

    Args:
        balance: saldo disponível (USDT)
        base_size: tamanho base (ex: 0.03 = 3%)
        volatility_atr: ATR normalizado (ATR/price)
        current_price: preço atual
        leverage: alavancagem
        win_streak: wins consecutivos (+) ou losses consecutivos (-)
        regime: 'BULL', 'BEAR' ou 'SIDEWAYS'
        confidence: nível de confiança do modelo [0, 1]
        risk_config: dicionário risk_management do config.yaml

    Returns:
        quantity: float — quantidade a operar
    """
    if risk_config is None:
        risk_config = {}

    try:
        # 1. Fator de volatilidade
        if volatility_atr > 0.02:
            volatility_factor = 0.7
        elif volatility_atr > 0.015:
            volatility_factor = 0.85
        else:
            volatility_factor = 1.0

        # 2. Fator de win streak
        if win_streak > 2:
            streak_factor = risk_config.get('max_win_streak_multiplier', 1.2)
        elif win_streak < -2:
            streak_factor = risk_config.get('min_win_streak_multiplier', 0.8)
        else:
            streak_factor = 1.0

        # 3. Fator de regime
        if regime == 'SIDEWAYS':
            regime_factor = 0.8
        elif regime in ('BULL', 'BEAR'):
            regime_factor = 1.1
        else:
            regime_factor = 1.0

        # 4. Fator de confiança
        confidence_factor = max(0.5, confidence)

        adjusted_size = base_size * volatility_factor * streak_factor * regime_factor * confidence_factor
        adjusted_size = max(0.01, min(0.05, adjusted_size))

        quantity = (balance * adjusted_size * leverage) / current_price

        logger.info(
            f"[POSITION_SIZE] Base:{base_size:.1%} Vol:{volatility_factor:.2f} "
            f"Streak:{streak_factor:.2f} Regime:{regime_factor:.2f} "
            f"Conf:{confidence_factor:.2f} → Final:{adjusted_size:.1%}"
        )
        return round(quantity, 3)

    except Exception as exc:
        logger.error(f"[POSITION_SIZE] Erro: {exc}")
        return 0.0
