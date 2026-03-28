"""
Verificação de correlação entre posições abertas.

Antes de abrir um novo trade, calcula a correlação de Pearson entre
os retornos do novo símbolo e cada símbolo que já tem posição aberta.
Bloqueia a entrada se qualquer correlação superar o threshold.

Usado por dashboard/trading/engine.py no _tick, antes de execute_trade.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_correlation(
    new_sym: str,
    open_syms: list[str],
    ws_mgr,
    threshold: float = 0.70,
    lookback: int = 50,
) -> tuple[bool, str]:
    """
    Verifica se o novo símbolo está correlacionado com posições abertas.

    Args:
        new_sym:   símbolo candidato (ex: 'ETHUSDT')
        open_syms: símbolos com posição aberta atualmente
        ws_mgr:    BinanceWebSocketManager (para buscar candles)
        threshold: correlação máxima permitida (default 0.70)
        lookback:  número de candles para calcular correlação (default 50)

    Returns:
        (can_enter: bool, reason: str)
        can_enter=True  → sem correlação problemática, pode abrir
        can_enter=False → correlação alta detectada, bloquear entrada
    """
    if not open_syms:
        return True, "sem posições abertas"

    # Obtém retornos do novo símbolo
    try:
        df_new = ws_mgr.get_klines_df(new_sym, '15m', limit=lookback + 5)
        if df_new is None or len(df_new) < lookback:
            # Dados insuficientes: não bloquear por falta de dados
            return True, f"dados insuficientes para {new_sym}"
        returns_new = np.diff(df_new['close'].values[-lookback:])
    except Exception as exc:
        logger.warning(f"[CORR] Erro ao obter dados de {new_sym}: {exc}")
        return True, f"erro ao calcular {new_sym}"

    # Compara com cada símbolo aberto
    for open_sym in open_syms:
        if open_sym == new_sym:
            continue
        try:
            df_open = ws_mgr.get_klines_df(open_sym, '15m', limit=lookback + 5)
            if df_open is None or len(df_open) < lookback:
                continue
            returns_open = np.diff(df_open['close'].values[-lookback:])

            # Alinha comprimentos
            min_len = min(len(returns_new), len(returns_open))
            r_new   = returns_new[-min_len:]
            r_open  = returns_open[-min_len:]

            # Pearson: evita divisão por zero (std == 0 quando candles freezam)
            std_new  = np.std(r_new)
            std_open = np.std(r_open)
            if std_new == 0 or std_open == 0:
                continue

            corr = float(np.corrcoef(r_new, r_open)[0, 1])
            abs_corr = abs(corr)

            logger.debug(f"[CORR] {new_sym} ↔ {open_sym}: corr={corr:.3f}")

            if abs_corr >= threshold:
                reason = (
                    f"correlação {new_sym}↔{open_sym} = {corr:+.2f} "
                    f">= {threshold:.0%} (bloqueado)"
                )
                logger.info(f"[CORR] ⛔ {reason}")
                return False, reason

        except Exception as exc:
            logger.warning(f"[CORR] Erro ao comparar {new_sym}/{open_sym}: {exc}")
            continue

    return True, "correlação OK"
