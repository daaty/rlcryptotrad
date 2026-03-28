"""
Filtro de qualidade de entrada — valida condições técnicas antes de abrir posição.

Modos disponíveis (config.yaml → entry_filter.mode):
  strict     — todos os filtros ativos, RSI 70/30, Vol 70%, direção alinhada
  normal     — filtros relaxados, RSI 80/20, Vol 30%, sem exigir direção
  aggressive — só bloqueia RSI extremo (85/15) e EMA muito distante (6%)
  disabled   — sem filtros, modelo opera livre

Camada pura sem dependência de Streamlit.
"""
from __future__ import annotations

import pandas as pd

from dashboard.core.logging_setup import get_logger

logger = get_logger()

# ── Parâmetros por modo ────────────────────────────────────────────────────────
# (rsi_ob, rsi_os, ema_max_pct, vol_min_ratio, body_min_ratio, check_direction)
_MODE_PARAMS: dict[str, tuple] = {
    #              RSI_OB  RSI_OS  EMA%    Vol%   Body%  Dir
    "strict":     (70,     30,     2.0,   0.70,  0.30,  True),
    "normal":     (80,     20,     4.0,   0.30,  0.15,  False),
    "aggressive": (85,     15,     6.0,   0.15,  0.00,  False),
    "disabled":   (100,     0,   999.0,   0.00,  0.00,  False),
}
_DEFAULT_MODE = "normal"


def validate_entry_quality(
    market_data: pd.DataFrame,
    decision: str,
    current_price: float,
    mode: str = _DEFAULT_MODE,
) -> tuple[bool, str]:
    """
    Valida se é um bom momento técnico para entrar na posição.

    Indicadores no DataFrame (compute_indicators):
        RSI_14       = talib.RSI / 100          → [0, 1]
        EMA_21       = talib.EMA / close        → ratio ≈ 1.0
        Volume_MA_20 = volume / vol_sma_20      → ratio (1.0 = media exata)

    Returns:
        (pode_entrar: bool, motivo_se_bloqueado: str)
    """
    try:
        if decision not in ('LONG', 'SHORT'):
            return True, ""   # FLAT sempre pode executar

        if len(market_data) < 3:
            return True, ""   # dados insuficientes, não bloquear

        params = _MODE_PARAMS.get(mode, _MODE_PARAMS[_DEFAULT_MODE])
        rsi_ob, rsi_os, ema_max_pct, vol_min_ratio, body_min_ratio, check_dir = params

        # Usa o penúltimo candle (último FECHADO) para os indicadores de volume/RSI/corpo.
        # O último candle (iloc[-1]) é o current live candle que acabou de abrir e tem
        # volume mínimo (causava bloqueio de 100% das entradas logo após o candle fechar).
        last_candle  = market_data.iloc[-2]
        candle_close = float(last_candle['close'])
        candle_open  = float(last_candle['open'])
        candle_high  = float(last_candle['high'])
        candle_low   = float(last_candle['low'])

        # De-normaliza RSI (armazenado ÷ 100)
        rsi = float(last_candle.get('RSI_14', 0.5)) * 100

        # EMA_21 = EMA / close  →  preço absoluto = ratio * close
        ema21_ratio = float(last_candle.get('EMA_21', 1.0))
        ema21_price = ema21_ratio * candle_close

        # Volume_MA_20 JÁ é o ratio volume/sma — não dividir novamente!
        vol_ratio = float(last_candle.get('Volume_MA_20', 1.0))

        candle_body  = abs(candle_close - candle_open)
        candle_range = candle_high - candle_low

        # ── FILTRO 1: RSI extremo ───────────────────────────────────────────
        if decision == 'LONG' and rsi > rsi_ob:
            return False, f"RSI sobrecomprado ({rsi:.1f} > {rsi_ob}) — aguardando correção"
        if decision == 'SHORT' and rsi < rsi_os:
            return False, f"RSI sobrevendido ({rsi:.1f} < {rsi_os}) — aguardando retração"

        # ── FILTRO 2: Distância da EMA21 ────────────────────────────────────
        if ema21_price > 0 and ema_max_pct < 999:
            distance_pct = abs(current_price - ema21_price) / ema21_price * 100
            if distance_pct > ema_max_pct:
                direction = "acima" if current_price > ema21_price else "abaixo"
                return False, (
                    f"Preço {distance_pct:.1f}% {direction} da EMA21 "
                    f"(máx {ema_max_pct:.0f}%) — momentum exaurido"
                )

        # ── FILTRO 3: Volume ─────────────────────────────────────────────────
        if vol_min_ratio > 0 and vol_ratio < vol_min_ratio:
            return False, (
                f"Volume fraco ({vol_ratio*100:.0f}% da média, mín {vol_min_ratio*100:.0f}%) "
                f"— falta confirmação"
            )

        # ── FILTRO 4: Corpo do candle ────────────────────────────────────────
        if body_min_ratio > 0 and candle_range > 0:
            body_ratio = candle_body / candle_range
            if body_ratio < body_min_ratio:
                return False, (
                    f"Candle de indecisão (corpo {body_ratio*100:.0f}% do range, "
                    f"mín {body_min_ratio*100:.0f}%) — mercado lateral"
                )

        # ── FILTRO 5: Direção do candle (só no modo strict) ──────────────────
        if check_dir:
            candle_bullish = candle_close > candle_open
            if decision == 'LONG' and not candle_bullish:
                return False, "Candle bearish — aguardando confirmação bullish"
            if decision == 'SHORT' and candle_bullish:
                return False, "Candle bullish — aguardando confirmação bearish"

        return True, ""

    except Exception as exc:
        logger.warning(f"[ENTRY_FILTER] Erro ao validar qualidade de entrada: {exc}")
        return True, ""   # failsafe: permite entrada em caso de erro

