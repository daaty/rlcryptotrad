"""
Recalcula indicadores dos CSVs existentes com normalização relativa ao close.
Não precisa baixar novamente da Binance — usa os full_*.csv já existentes.

Normalização: RELATIVA AO CLOSE por linha (scale-invariant)
  - RSI: / 100
  - SMA, EMA, BBL/BBM/BBU: / close
  - MACD, Signal, Hist:    / close
  - ATR:                   / close
  - Volume_MA_20:          / volume
  - OHLCV: mantido bruto
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
from datetime import datetime


def recalculate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Recalcula todos os indicadores com normalização relativa ao close."""
    df = df_raw.copy()
    
    # OHLCV bruto (sem normalização)
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    
    # RSI → 0-1
    df['RSI_14'] = talib.RSI(close, timeperiod=14) / 100.0
    
    # SMA → ratio relativo ao close (~1.0 ± %)
    df['SMA_20'] = talib.SMA(close, timeperiod=20) / (close + 1e-8)
    df['SMA_50'] = talib.SMA(close, timeperiod=50) / (close + 1e-8)
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df['BBL_20_2.0'] = lower  / (close + 1e-8)
    df['BBM_20_2.0'] = middle / (close + 1e-8)
    df['BBU_20_2.0'] = upper  / (close + 1e-8)
    df['BBB_20_2.0'] = (upper - lower) / (middle + 1e-8)
    df['BBP_20_2.0'] = (close - lower) / (upper - lower + 1e-8)
    
    # MACD → ratio relativo ao close (~0.0001 a 0.005)
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_12_26_9']  = macd   / (close + 1e-8)
    df['MACDs_12_26_9'] = signal / (close + 1e-8)
    df['MACDh_12_26_9'] = hist   / (close + 1e-8)
    
    # EMA → ratio relativo ao close (~1.0 ± %)
    df['EMA_9']  = talib.EMA(close, timeperiod=9)  / (close + 1e-8)
    df['EMA_21'] = talib.EMA(close, timeperiod=21) / (close + 1e-8)
    
    # ATR → volatilidade relativa (~0.005 - 0.03)
    df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-8)
    
    # Volume MA: volume relativo à média (~0.1-5x, sem outliers)
    # volume/vol_ma > 1 = volume acima da média (entrada possível)
    vol_ma = talib.SMA(volume, timeperiod=20)
    df['Volume_MA_20'] = volume / (vol_ma + 1e-8)
    
    # Remover NaN (primeiras linhas com indicadores incompletos)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    return df


def process_timeframe(full_path: str, split_ratio: float = 0.8):
    """Processa um timeframe: recalcula indicadores + salva train/test."""
    full_path = Path(full_path)
    if not full_path.exists():
        print(f"  ❌ Não encontrado: {full_path}")
        return None
    
    print(f"\n  📂 Carregando: {full_path.name} ... ", end='')
    df_raw = pd.read_csv(full_path)
    print(f"{len(df_raw):,} candles")
    
    # Verificar colunas base
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df_raw.columns for c in required):
        print(f"  ❌ Colunas faltando: {required}")
        return None
    
    # Recalcular
    print(f"  🔧 Recalculando indicadores (normalização relativa ao close)...")
    df_new = recalculate(df_raw)
    
    # Verificar range dos indicadores
    print(f"  ✅ {len(df_new):,} candles após dropar NaN")
    print(f"     RSI range:    {df_new['RSI_14'].min():.3f} - {df_new['RSI_14'].max():.3f}  (esperado: 0-1)")
    print(f"     SMA_20 range: {df_new['SMA_20'].min():.4f} - {df_new['SMA_20'].max():.4f}  (esperado: ~1.0)")
    print(f"     MACD range:   {df_new['MACD_12_26_9'].min():.6f} - {df_new['MACD_12_26_9'].max():.6f}  (esperado: ~0)")
    print(f"     ATR range:    {df_new['ATR_14'].min():.6f} - {df_new['ATR_14'].max():.6f}  (esperado: 0.003-0.05)")
    print(f"     VolMA range:  {df_new['Volume_MA_20'].min():.3f} - {df_new['Volume_MA_20'].max():.3f}  (esperado: 0.2-5.0)")
    
    # Split train/test (mantém mesma proporção)
    split_idx = int(len(df_new) * split_ratio)
    df_train = df_new[:split_idx]
    df_test  = df_new[split_idx:]
    
    # Extrair data do nome original para manter compatibilidade de nomes
    # ex: full_btcusdt_36m_15m_20260125.csv → date = 20260125
    stem = full_path.stem  # full_btcusdt_36m_15m_20260125
    parts = stem.split('_')
    date_tag = parts[-1]   # 20260125
    
    # Reconstruir nomes
    base = '_'.join(parts[1:])  # btcusdt_36m_15m_20260125
    train_path = full_path.parent / f"train_{base}.csv"
    test_path  = full_path.parent / f"test_{base}.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path,   index=False)
    
    print(f"  💾 Train: {train_path.name} ({len(df_train):,} candles)")
    print(f"  💾 Test:  {test_path.name}  ({len(df_test):,} candles)")
    
    return {'train': str(train_path), 'test': str(test_path)}


def main():
    print("\n" + "="*70)
    print("🔧 RECALCULANDO INDICADORES - Normalização Relativa ao Close")
    print("   Correção de train/inference mismatch para V17.7")
    print("="*70)
    
    data_dir = Path('data')
    
    # Procurar os 3 full CSVs mais recentes por timeframe
    full_files = {
        '15m': None,
        '1h':  None,
        '4h':  None,
    }
    
    for f in sorted(data_dir.glob('full_btcusdt_36m_*.csv')):
        for tf in ['15m', '1h', '4h']:
            if f'_{tf}_' in f.name:
                # Pegar o mais recente (sorted coloca em ordem, pegamos o último)
                full_files[tf] = f
    
    print("\n📂 Arquivos encontrados:")
    for tf, f in full_files.items():
        print(f"  {tf}: {f.name if f else 'NÃO ENCONTRADO'}")
    
    success = {}
    for tf, full_path in full_files.items():
        if full_path is None:
            print(f"\n❌ {tf}: arquivo full não encontrado! Execute collect_multi_timeframe.py primeiro.")
            continue
        
        print(f"\n{'─'*60}")
        print(f"PROCESSANDO {tf}")
        result = process_timeframe(str(full_path))
        if result:
            success[tf] = result
    
    print("\n" + "="*70)
    print("✅ CONCLUÍDO!")
    print(f"   {len(success)}/{len(full_files)} timeframes processados")
    print("\nPróximos passos:")
    print("  1. Treinar V17.7: python train_recurrent_ppo_v17.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
