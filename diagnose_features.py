"""
Diagnóstico: Comparar features do dataset vs modelo
"""
import pandas as pd
import numpy as np
from stable_baselines3 import SAC

print("=" * 70)
print("DIAGNÓSTICO: MISMATCH DE FEATURES")
print("=" * 70)

# Dataset
df = pd.read_csv('data/train_btcusdt_36m_20260109.csv')
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n1. DATASET (train_btcusdt_36m_20260109.csv):")
print(f"   Total colunas: {len(df.columns)}")
print(f"   Colunas numéricas: {len(numeric_cols)}")
print(f"   Colunas: {list(df.columns)}")

# Modelo
print(f"\n2. MODELO (sac_v8_100000_steps.zip):")
model = SAC.load('models/sac_v8_100000_steps.zip')
obs_space = model.observation_space
print(f"   Observation space shape: {obs_space.shape}")
print(f"   Window size: {obs_space.shape[0]}")
print(f"   Features esperadas: {obs_space.shape[1]}")

# Análise
expected_features = obs_space.shape[1]
dataset_features = len(numeric_cols)
portfolio_features = 3  # balance, position, equity

print(f"\n3. ANÁLISE:")
print(f"   Dataset tem: {dataset_features} features")
print(f"   Portfolio: {portfolio_features} features (balance, position, equity)")
print(f"   Total: {dataset_features + portfolio_features}")
print(f"   Modelo espera: {expected_features}")

difference = expected_features - (dataset_features + portfolio_features)
print(f"\n   DIFERENÇA: {difference}")

if difference > 0:
    print(f"   ❌ FALTAM {difference} features!")
    print(f"      Modelo foi treinado com MAIS features do que o dataset tem.")
elif difference < 0:
    print(f"   ❌ SOBRAM {abs(difference)} features!")
    print(f"      Dataset tem MAIS features do que o modelo foi treinado.")
else:
    print(f"   ✅ Features batem!")

print("\n" + "=" * 70)
print("POSSÍVEIS CAUSAS:")
print("=" * 70)
print("1. Modelo treinado com dataset DIFERENTE (mais/menos indicadores)")
print("2. Sentiment features ativadas no treino mas não no backtest (ou vice-versa)")
print("3. Dataset de treino tinha features extras que foram removidas")
print("4. Config de treino vs backtest está diferente")
