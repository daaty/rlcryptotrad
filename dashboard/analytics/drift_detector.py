"""
Drift Detector — detecta mudança na distribuição das features de mercado
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compara a distribuição atual das features V19 (janela live de klines)
contra o baseline computado a partir dos dados de treinamento.

Features monitoradas (as mesmas que o ambiente V19 vê):
  0  close_ret%    retorno % do candle vs anterior
  1  open%         (open/close - 1) * 100 → corpo do candle
  2  high%         (high/close - 1) * 100 → upper wick
  3  low%          (low/close  - 1) * 100 → lower wick  (negativo)
  4  vol_ratio     volume / Volume_MA_20   → spike detector
  5  RSI_14        indicador normalizado (0-1 no CSV)
  6  BBP           percentile Bollinger

Baseline:
  Calculado em generate_feature_baseline.py a partir dos CSVs de treino.
  Salvo em data/feature_baseline.json como {mean: [...], std: [...]}.

Drift signal:
  Z-score = |live_mean - train_mean| / train_std
  ⚠️  Warning quando Z > 2.0
  🚨  Alert   quando Z > 3.5
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

BASELINE_PATH = Path("data") / "feature_baseline.json"

FEATURE_NAMES = [
    "close_ret%",
    "open%",
    "high%",
    "low%",
    "vol_ratio",
    "RSI_14",
    "BBP",
]

DRIFT_WARN   = 2.0   # Z-score para warning
DRIFT_ALERT  = 3.5   # Z-score para alerta crítico


# ── Baseline ────────────────────────────────────────────────────────────────

def load_baseline() -> Optional[dict]:
    """Carrega o baseline de features. Retorna None se não existir."""
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return None


def save_baseline(baseline: dict):
    """Salva baseline em data/feature_baseline.json."""
    Path("data").mkdir(exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding='utf-8')


# ── Extração de features do buffer de klines ────────────────────────────────

def extract_features_from_klines(df) -> Optional[np.ndarray]:
    """
    Extrai as 7 features monitoradas de um DataFrame de klines.

    Espera colunas: open, high, low, close, volume, RSI_14, BBP_20_2.0
    (ou Volume_MA_20 para o vol_ratio).

    Args:
        df: pd.DataFrame com pelo menos 50 linhas, colunas {open,high,low,close,volume,...}
    Returns:
        (n, 7) ndarray ou None se dados insuficientes/faltando colunas.
    """
    if df is None or len(df) < 30:
        return None

    try:
        close = df['close'].values.astype(float)
        open_ = df['open'].values.astype(float)
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        vol   = df['volume'].values.astype(float)

        prev_close      = np.roll(close, 1)
        prev_close[0]   = close[0]
        close_ret       = (close / (prev_close + 1e-10) - 1.0) * 100
        open_pct        = (open_ / (close + 1e-8) - 1.0) * 100
        high_pct        = (high / (close + 1e-8) - 1.0) * 100
        low_pct         = (low  / (close + 1e-8) - 1.0) * 100

        # vol_ratio = vol / vol_ma20
        vol_ma   = _rolling_mean(vol, 20)
        vol_ratio = vol / (vol_ma + 1e-8)

        # RSI (coluna pode chamar RSI_14 ou rsi)
        rsi = _get_col(df, ['RSI_14', 'rsi'])
        if rsi is None:
            rsi = np.full(len(close), 0.5)

        # BBP (coluna pode chamar BBP_20_2.0 ou bbp)
        bbp = _get_col(df, ['BBP_20_2.0', 'BBP', 'bbp'])
        if bbp is None:
            bbp = np.full(len(close), 0.5)

        features = np.stack([
            close_ret, open_pct, high_pct, low_pct, vol_ratio,
            rsi, bbp,
        ], axis=1)  # (n, 7)

        # Remover NaN e infinitos
        mask = np.isfinite(features).all(axis=1)
        features = features[mask]

        return features if len(features) >= 20 else None

    except Exception:
        return None


def _get_col(df, names: list) -> Optional[np.ndarray]:
    """Retorna a primeira coluna que existir no DataFrame."""
    for name in names:
        if name in df.columns:
            return df[name].values.astype(float)
    return None


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean sem pandas."""
    result = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    result[window - 1:] = (cs[window:] - cs[:-window]) / window
    # Preencher NaN iniciais com média do período disponível
    for i in range(window - 1):
        result[i] = arr[:i + 1].mean()
    return result


# ── Cálculo de desvio ────────────────────────────────────────────────────────

class DriftResult:
    __slots__ = ['feature', 'live_mean', 'live_std', 'train_mean', 'train_std', 'z_score', 'status']

    def __init__(self, feature: str, live_mean: float, live_std: float,
                 train_mean: float, train_std: float):
        self.feature    = feature
        self.live_mean  = live_mean
        self.live_std   = live_std
        self.train_mean = train_mean
        self.train_std  = train_std
        self.z_score    = abs(live_mean - train_mean) / max(train_std, 1e-9)

        if self.z_score >= DRIFT_ALERT:
            self.status = 'ALERT'
        elif self.z_score >= DRIFT_WARN:
            self.status = 'WARN'
        else:
            self.status = 'OK'


def compute_drift(live_features: np.ndarray, baseline: dict) -> list[DriftResult]:
    """
    Compara as features live com o baseline de treinamento.

    Args:
        live_features: (n, 7) ndarray com features recentes
        baseline: {'mean': [...], 'std': [...], ...}
    Returns:
        Lista de DriftResult (uma entrada por feature).
    """
    train_mean = np.array(baseline['mean'])
    train_std  = np.array(baseline['std'])

    live_mean = np.nanmean(live_features, axis=0)
    live_std  = np.nanstd(live_features, axis=0)

    results = []
    n_feats = min(len(FEATURE_NAMES), len(train_mean), live_features.shape[1])
    for i in range(n_feats):
        results.append(DriftResult(
            feature    = FEATURE_NAMES[i],
            live_mean  = float(live_mean[i]),
            live_std   = float(live_std[i]),
            train_mean = float(train_mean[i]),
            train_std  = float(train_std[i]),
        ))

    return results


def overall_drift_status(results: list[DriftResult]) -> str:
    """Retorna o status global: OK / WARN / ALERT."""
    statuses = [r.status for r in results]
    if 'ALERT' in statuses:
        return 'ALERT'
    if 'WARN' in statuses:
        return 'WARN'
    return 'OK'


# ── Geração de baseline a partir de CSVs ────────────────────────────────────

def compute_baseline_from_csvs(csv_paths: list[str]) -> Optional[dict]:
    """
    Computa o baseline de features a partir de uma lista de CSVs de treinamento.
    Usa apenas o CSV do 15m de cada par (que é o principal para decisão).

    Args:
        csv_paths: caminhos dos CSVs 15m de treinamento
    Returns:
        Baseline dict com mean, std, n_samples, source_files, generated_at
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    all_features = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            feats = extract_features_from_klines(df)
            if feats is not None:
                all_features.append(feats)
                print(f"  ✅ {Path(path).name}: {len(feats):,} amostras")
            else:
                print(f"  ⚠️  {Path(path).name}: features insuficientes")
        except Exception as exc:
            print(f"  ❌ {Path(path).name}: {exc}")

    if not all_features:
        return None

    combined = np.concatenate(all_features, axis=0)
    n_feats = combined.shape[1]

    return {
        'mean'          : np.nanmean(combined, axis=0)[:n_feats].tolist(),
        'std'           : np.nanstd(combined,  axis=0)[:n_feats].tolist(),
        'n_samples'     : int(len(combined)),
        'n_features'    : n_feats,
        'feature_names' : FEATURE_NAMES[:n_feats],
        'source_files'  : [str(Path(p).name) for p in csv_paths],
        'generated_at'  : __import__('datetime').datetime.now().isoformat(timespec='seconds'),
    }
