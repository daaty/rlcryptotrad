"""
Teste: Verifica shape do observation space do ambiente de backtest
"""
import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnv

print("Carregando dataset...")
df = pd.read_csv('data/train_btcusdt_36m_20260109.csv')
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Pegar apenas colunas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

print(f"Dataset: {len(df_numeric)} candles, {len(df_numeric.columns)} features")

print("\nCriando ambiente (mesma config do backtest)...")
env = TradingEnv(
    df=df_numeric,
    initial_balance=10000,
    commission=0.0004,
    slippage=0.0005,
    leverage=1.5,
    position_size=0.05,
    window_size=50,
    max_episode_steps=4000,
    random_start=False,
    persist_balance=False,
    use_sharpe_reward=True,
    enable_indicator_shaping=True
)

print(f"\nObservation space shape: {env.observation_space.shape}")
print(f"  Window size: {env.observation_space.shape[0]}")
print(f"  Total features: {env.observation_space.shape[1]}")

# Testar observação real
print("\nTestando get_observation()...")
obs, info = env.reset()
print(f"  Observation shape real: {obs.shape}")
print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}")

print("\nCarregando modelo...")
from stable_baselines3 import SAC
model = SAC.load('models/sac_v8_100000_steps.zip')
print(f"  Modelo espera: {model.observation_space.shape}")

if obs.shape == model.observation_space.shape:
    print("\n✅ SHAPES BATEM! Ambiente e modelo compatíveis.")
else:
    print(f"\n❌ MISMATCH!")
    print(f"   Ambiente gera: {obs.shape}")
    print(f"   Modelo espera: {model.observation_space.shape}")
