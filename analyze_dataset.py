import pandas as pd
import numpy as np

df = pd.read_csv('data/train_btcusdt_12m_20260105.csv')

print(f"📊 ANÁLISE DO DATASET")
print(f"=" * 60)
print(f"\n1. SHAPE: {df.shape}")
print(f"   Candles: {len(df):,}")
print(f"   Features: {len(df.columns)}")

print(f"\n2. COLUNAS:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n3. PRIMEIRAS 3 LINHAS:")
print(df.head(3))

print(f"\n4. ÚLTIMAS 3 LINHAS:")
print(df.tail(3))

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\n5. PERÍODO:")
    print(f"   Início: {df['timestamp'].min()}")
    print(f"   Fim: {df['timestamp'].max()}")
    print(f"   Duração: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")

if 'close' in df.columns:
    print(f"\n6. PREÇO (Close):")
    print(f"   Min: ${df['close'].min():,.2f}")
    print(f"   Max: ${df['close'].max():,.2f}")
    print(f"   Variação: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Volatilidade
    returns = df['close'].pct_change().dropna()
    print(f"\n7. VOLATILIDADE (Returns):")
    print(f"   Média: {returns.mean():.4f}")
    print(f"   Std: {returns.std():.4f}")
    print(f"   Min: {returns.min():.4f}")
    print(f"   Max: {returns.max():.4f}")

print(f"\n8. PROBLEMA IDENTIFICADO:")
print(f"   ⚠️  Episódio completo = {len(df):,} steps")
print(f"   ⚠️  Com 200k timesteps = apenas ~7 episódios")
print(f"   ⚠️  Poucos resets = exploração insuficiente")
