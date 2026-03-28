"""Preparação de observações para o modelo LSTM — compatível V17.7 e V19.

VERSÕES:
  - V17.7 (modelo atual): OHLCV absoluto + np.clip(-100, 100) — SEM normalização
  - V19 (futuro):         normalize_ohlcv_v19() + np.clip(-10, 10)

⚠️  ATENÇÃO TRAIN/INFERENCE MISMATCH:
  O V17.7 foi treinado com `np.clip(-100, 100)` aplicado diretamente sobre
  preços absolutos (BTC close=$27 839 → 100). Usar normalize_ohlcv_v19() em
  live gera inputs completamente diferentes do treino → viés de ação (sempre LONG).

  prepare_observation() usa normalize=False por padrão (V17-compatível).
  Quando V19 for deployado, passar normalize=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.core.logging_setup import get_logger

logger = get_logger()

# ── Colunas do dataset de treino (20 features, sem timestamp) ────────────────
FEATURE_COLS_15M: list[str] = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI_14', 'SMA_20', 'SMA_50',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
    'EMA_9', 'EMA_21', 'ATR_14', 'Volume_MA_20',
]  # 20 features

# Índices das features no array de observação da TF maior
IDX_CLOSE:  int = 3
IDX_VOL:    int = 4
IDX_VOL_MA: int = 19
IDX_RSI:    int = 5
IDX_BBP:    int = 12
IDX_MACDH:  int = 15


def normalize_ohlcv_v19(window: np.ndarray) -> np.ndarray:
    """
    V19 FIX: converte OHLCV absolutos para valores relativos ANTES do clip.

    Replica exatamente TradingEnvV19LSTM._normalize_ohlcv() para garantir
    consistência train/inference (sem train/inference mismatch).

    Transformações:
      open  → (open / close - 1) × 100     [% desvio vs close]
      high  → (high / close - 1) × 100     [upper wick %]
      low   → (low  / close - 1) × 100     [lower wick %]
      close → (close / prev_close - 1) × 100 [retorno % do step]
      vol   → volume / vol_ma20             [ratio relativo]
      cols 5..19 (indicadores): sem mudança — já normalizados no CSV

    Args:
        window: (seq_len, n_features) com preços absolutos
    Returns:
        Array normalizado, mesma shape, dtype float32
    """
    w = window.copy().astype(np.float64)

    close = w[:, IDX_CLOSE].copy()
    close[close == 0] = 1.0

    w[:, 0] = (w[:, 0] / close - 1.0) * 100   # open  → % vs close
    w[:, 1] = (w[:, 1] / close - 1.0) * 100   # high  → upper wick %
    w[:, 2] = (w[:, 2] / close - 1.0) * 100   # low   → lower wick %

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    w[:, IDX_CLOSE] = (close / (prev_close + 1e-10) - 1.0) * 100  # retorno %

    vol_ma = w[:, IDX_VOL_MA].copy()
    vol_ma[vol_ma == 0] = 1.0
    w[:, IDX_VOL] = w[:, IDX_VOL] / (vol_ma + 1e-8)   # vol/vol_ma ratio

    return w.astype(np.float32)


def prepare_observation(
    market_data_15m: pd.DataFrame,
    multi_tf_data: dict | None = None,
    balance_norm: float = 1.0,
    position: float = 0.0,
    equity_norm: float = 1.0,
    normalize: bool = False,
) -> np.ndarray | None:
    """
    Prepara observação para o LSTM: shape (50, 31).

    Args:
        normalize: False = V17.7 (clip ±100, sem normalização OHLCV, padrão)
                   True  = V19   (normalize_ohlcv_v19 + clip ±10)

    Estrutura:
        (50, 20) 15m features   — mesma ordem de FEATURE_COLS_15M
        (50, 4)  1h context     — RSI, BBP, MACDh, close%diff
        (50, 4)  4h context     — RSI, BBP, MACDh, close%diff
        (50, 1)  balance_norm
        (50, 1)  position       (-1=short, 0=flat, 1=long)
        (50, 1)  equity_norm
        ─────────────────
        (50, 31) total
    """
    try:
        # === 15m: 20 features (50 candles mais recentes) =====================
        raw_15m = market_data_15m[FEATURE_COLS_15M].iloc[-50:].values.copy()  # (≤50, 20) — preços absolutos
        if len(raw_15m) < 50:
            pad = np.zeros((50 - len(raw_15m), 20))
            raw_15m = np.vstack([pad, raw_15m])

        if normalize:
            # V19: OHLCV → % relativos, clip ±10
            obs_15m = normalize_ohlcv_v19(raw_15m)
            clip_val = 10.0
            logger.info(f"[OBS] 15m shape: {obs_15m.shape} (V19 normalized, clip±10)")
        else:
            # V17/V18: preços absolutos, clip ±100 (replica treino do V17.7)
            obs_15m = raw_15m.astype(np.float32)
            clip_val = 100.0
            logger.info(f"[OBS] 15m shape: {obs_15m.shape} (V17 raw, clip±100)")

        # === 1h / 4h context: 4 features cada ================================
        ctx_1h = np.zeros((50, 4), dtype=np.float32)
        ctx_4h = np.zeros((50, 4), dtype=np.float32)

        if multi_tf_data is not None:
            df_1h = multi_tf_data.get('1h')
            df_4h = multi_tf_data.get('4h')

            if df_1h is not None and len(df_1h) > 0:
                arr_1h = df_1h[FEATURE_COLS_15M].values
                for i in range(50):
                    offset          = 49 - i
                    idx_from_end_1h = offset // 4 + 1
                    row             = max(0, len(arr_1h) - 1 - idx_from_end_1h)
                    price_15m       = float(raw_15m[i, IDX_CLOSE]) or 1.0  # preço absoluto para ratio
                    ctx_1h[i, 0]    = arr_1h[row, IDX_RSI]
                    ctx_1h[i, 1]    = arr_1h[row, IDX_BBP]
                    ctx_1h[i, 2]    = arr_1h[row, IDX_MACDH]
                    ctx_1h[i, 3]    = (arr_1h[row, IDX_CLOSE] / price_15m - 1) * 100

            if df_4h is not None and len(df_4h) > 0:
                arr_4h = df_4h[FEATURE_COLS_15M].values
                for i in range(50):
                    offset           = 49 - i
                    idx_from_end_4h  = offset // 16 + 1
                    row              = max(0, len(arr_4h) - 1 - idx_from_end_4h)
                    price_15m        = float(raw_15m[i, IDX_CLOSE]) or 1.0  # preço absoluto para ratio
                    ctx_4h[i, 0]     = arr_4h[row, IDX_RSI]
                    ctx_4h[i, 1]     = arr_4h[row, IDX_BBP]
                    ctx_4h[i, 2]     = arr_4h[row, IDX_MACDH]
                    ctx_4h[i, 3]     = (arr_4h[row, IDX_CLOSE] / price_15m - 1) * 100

        # === Portfolio: 3 colunas separadas ==================================
        balance_col  = np.full((50, 1), balance_norm)
        position_col = np.full((50, 1), float(position))
        equity_col   = np.full((50, 1), equity_norm)

        # === Concatenar: (50,20)+(50,4)+(50,4)+(50,1)+(50,1)+(50,1) = (50,31)
        obs = np.hstack([
            obs_15m, ctx_1h, ctx_4h,
            balance_col, position_col, equity_col,
        ]).astype(np.float32)

        obs = np.clip(obs, -clip_val, clip_val)

        # Diagnóstico: log stats da observação para detectar viés de input
        _ohlcv_mean = float(obs[:, :5].mean())
        _ohlcv_std  = float(obs[:, :5].std())
        _rsi_last   = float(obs[-1, IDX_RSI])
        logger.info(
            f"[OBS] Shape: {obs.shape} clip=±{clip_val:.0f} | "
            f"ohlcv mean={_ohlcv_mean:.2f} std={_ohlcv_std:.2f} | "
            f"RSI_last={_rsi_last:.1f} | pos={position:.0f}"
        )
        return obs

    except Exception as exc:
        logger.error(f"[OBS] Erro ao preparar observação: {exc}")
        import traceback
        traceback.print_exc()
        return None


def apply_vecnorm(vec_norm, obs: np.ndarray) -> np.ndarray:
    """
    Aplica estatísticas de VecNormalize a uma observação manual (V19).

    RecurrentPPO / VecNormalize salva obs_rms no espaço FLAT do ambiente:
      shape = (n_features,) se o obs space é 1-D
      shape = (seq_len * n_features,) = (1550,) se o obs é flattened antes de entrar no vec env

    Estratégia:
      - Se mean.shape[0] == obs.shape[-1]   → aplica por linha (broadcast)
      - Se mean.shape[0] == obs.size        → flatten, normaliza, reshape
      - Caso contrário                      → skip (fallback sem alteração)
    """
    mean = np.asarray(vec_norm.obs_rms.mean, dtype=np.float64).flatten()
    var  = np.asarray(vec_norm.obs_rms.var,  dtype=np.float64).flatten()
    eps  = getattr(vec_norm, 'epsilon', 1e-8)
    clip = getattr(vec_norm, 'clip_obs', 10.0)

    if mean.shape[0] == obs.shape[-1]:
        # (31,) stats — aplica broadcast linha a linha sobre (50, 31)
        normed = (obs.astype(np.float64) - mean) / np.sqrt(var + eps)
        return np.clip(normed, -clip, clip).astype(np.float32)

    if mean.shape[0] == obs.size:
        # (1550,) stats — flatten, normaliza, reshape de volta
        obs_flat = obs.flatten().astype(np.float64)
        normed   = (obs_flat - mean) / np.sqrt(var + eps)
        return np.clip(normed, -clip, clip).reshape(obs.shape).astype(np.float32)

    logger.warning(
        f"[VECNORM] Shape não reconhecido: obs={obs.shape} (size={obs.size}) "
        f"vs vecnorm n_features={mean.shape[0]} — skipping normalization"
    )
    return obs.astype(np.float32)


def lstm_predict(
    model,
    obs: np.ndarray,
    lstm_states,
    episode_start: np.ndarray,
) -> tuple[float, str, object]:
    """
    Faz predição com LSTM V17.7 (RecurrentPPO), mantendo estado oculto.

    Args:
        model: RecurrentPPO model
        obs: np.array shape (50, 31)
        lstm_states: hidden states anteriores (None para início de episódio)
        episode_start: np.array bool

    Returns:
        (action_value: float, final_action: str, new_lstm_states)
    """
    obs_batched = obs[np.newaxis]  # (1, 50, 31)
    action, new_lstm_states = model.predict(
        obs_batched,
        state=lstm_states,
        episode_start=episode_start,
        deterministic=True,
    )
    action_value = float(np.squeeze(action))
    if action_value < -0.1:
        final_action = "SHORT"
    elif action_value > 0.1:
        final_action = "LONG"
    else:
        final_action = "FLAT"
    logger.info(f"[LSTM] action={action_value:.3f} → {final_action}")
    return action_value, final_action, new_lstm_states
