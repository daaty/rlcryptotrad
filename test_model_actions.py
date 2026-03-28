"""Testa se o modelo TD3 está retornando ações variadas."""
from stable_baselines3 import TD3
import numpy as np
import pandas as pd

print("Carregando modelo...")
model = TD3.load('models/base_btcusdt_final.zip')

print("\n1. Testando com observação aleatória fixa:")
obs = np.random.randn(50, 19).astype(np.float32)
for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    print(f"  Action {i+1}: {action[0]:.6f}")

print("\n2. Testando com dados reais:")
df = pd.read_csv('data/train_btcusdt_12m_20260105.csv')

# Pegar 10 janelas aleatórias
for i in range(10):
    idx = np.random.randint(50, len(df) - 100)
    
    # Montar observação com features reais
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI_14', 'SMA_20', 'SMA_50',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
    ]
    
    market_data = df[feature_cols].iloc[idx:idx+50].values  # (50, 16)
    
    # Adicionar features de portfólio (flat, sem posição)
    portfolio = np.array([
        [1.0, 0.0, 1.0]  # balance_norm, position, equity_norm
    ] * 50)
    
    obs = np.concatenate([market_data, portfolio], axis=1).astype(np.float32)  # (50, 19)
    
    action, _ = model.predict(obs, deterministic=True)
    action_value = action[0]
    
    if action_value < -0.33:
        decision = "SHORT"
    elif action_value > 0.33:
        decision = "LONG"
    else:
        decision = "FLAT"
    
    price = df.iloc[idx + 49]['close']
    rsi = df.iloc[idx + 49]['RSI_14']
    
    print(f"  Janela {i+1}: Price=${price:.2f}, RSI={rsi:.2f} -> Action={action_value:.6f} ({decision})")

print("\nSe todas as actions forem iguais, o modelo convergiu para política constante.")
