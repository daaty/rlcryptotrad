import pandas as pd
import numpy as np

# Carregar dataset completo
df = pd.read_csv('data/train_btcusdt_12m_20260105.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calcular volatilidade rolling (50 candles = ~12.5h)
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(50).std()

# Criar 3 níveis de dificuldade
# FÁCIL: Top 30% volatilidade (movimentos claros, fácil lucrar)
high_vol_threshold = df['volatility'].quantile(0.70)
df_easy = df[df['volatility'] >= high_vol_threshold].copy()

# MÉDIO: 30-60% volatilidade
med_vol_threshold = df['volatility'].quantile(0.40)
df_medium = df[(df['volatility'] >= med_vol_threshold) & (df['volatility'] < high_vol_threshold)].copy()

# DIFÍCIL: Dataset completo
df_hard = df.copy()

# Salvar
df_easy.drop(['returns', 'volatility'], axis=1, inplace=True)
df_medium.drop(['returns', 'volatility'], axis=1, inplace=True)
df_hard.drop(['returns', 'volatility'], axis=1, inplace=True)

df_easy.to_csv('data/curriculum_easy_btcusdt.csv', index=False)
df_medium.to_csv('data/curriculum_medium_btcusdt.csv', index=False)
df_hard.to_csv('data/curriculum_hard_btcusdt.csv', index=False)

print(f"[CURRICULUM LEARNING] Datasets criados:")
print(f"  EASY:   {len(df_easy):,} candles (alta volatilidade)")
print(f"  MEDIUM: {len(df_medium):,} candles (volatilidade média)")
print(f"  HARD:   {len(df_hard):,} candles (dataset completo)")
print(f"\nPlano de treinamento:")
print(f"  1. Treinar 200k steps em EASY (aprende a lucrar em trends claros)")
print(f"  2. Fine-tune 200k steps em MEDIUM (adapta para mercado normal)")
print(f"  3. Fine-tune 200k steps em HARD (generaliza para todas condições)")
