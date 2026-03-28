"""
Teste CRÍTICO: Modelo predizendo com observações REAIS do ambiente
"""
import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnv
from stable_baselines3 import SAC

print("=" * 70)
print("TESTE: MODELO COM OBSERVAÇÕES REAIS DO AMBIENTE")
print("=" * 70)

# Carregar dataset
df = pd.read_csv('data/train_btcusdt_36m_20260109.csv')
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

# Criar ambiente
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

# Carregar modelo
print("\nCarregando modelo...")
model = SAC.load('models/sac_v8_100000_steps.zip')

# Resetar ambiente
obs, info = env.reset()
print(f"\nObservação inicial: shape={obs.shape}")

# Fazer 100 previsões com o ambiente
print("\nFazendo 100 previsões com ambiente REAL:")
print("=" * 70)

actions = []
positions = []

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0])
    actions.append(action_value)
    
    # Classificar
    if action_value < -0.1:
        position = "Short"
        pos_code = -1
    elif action_value > 0.1:
        position = "Long"
        pos_code = 1
    else:
        position = "Flat"
        pos_code = 0
    
    positions.append(pos_code)
    
    if i < 10:
        print(f"Step {i+1:3d}: action={action_value:+.4f} → {position:5s} | Balance: ${env.balance:,.2f}")
    
    # Step no ambiente
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        print(f"\nEpisódio terminou no step {i+1}")
        break

# Estatísticas
actions = np.array(actions)
positions = np.array(positions)

long_count = np.sum(positions == 1)
short_count = np.sum(positions == -1)
flat_count = np.sum(positions == 0)
total = len(positions)

print("\n" + "=" * 70)
print(f"ESTATÍSTICAS ({total} steps):")
print(f"  Long:  {long_count:3d} ({long_count/total*100:.1f}%)")
print(f"  Short: {short_count:3d} ({short_count/total*100:.1f}%)")
print(f"  Flat:  {flat_count:3d} ({flat_count/total*100:.1f}%)")
print(f"\n  Action média: {actions.mean():+.4f}")
print(f"  Action std:   {actions.std():.4f}")
print(f"  Action min:   {actions.min():+.4f}")
print(f"  Action max:   {actions.max():+.4f}")

# DIAGNÓSTICO
print("\n" + "=" * 70)
print("DIAGNÓSTICO:")
if short_count > total * 0.7:
    print(f"❌ MODELO FAZ {short_count/total*100:.0f}% SHORT!")
    print("   Confirmado: Modelo tem bias extremo para short.")
elif long_count > total * 0.7:
    print(f"❌ MODELO FAZ {long_count/total*100:.0f}% LONG!")
elif flat_count > total * 0.7:
    print(f"⚠️  MODELO FAZ {flat_count/total*100:.0f}% FLAT!")
    print("   Modelo paralisado.")
else:
    print("✅ Modelo está balanceado nos primeiros 100 steps.")
    print(f"   Long: {long_count/total*100:.1f}%, Short: {short_count/total*100:.1f}%, Flat: {flat_count/total*100:.1f}%")
