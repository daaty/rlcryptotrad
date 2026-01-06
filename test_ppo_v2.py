"""
Teste rapido do modelo PPO v2 (800k steps)
"""

import pandas as pd
from stable_baselines3 import PPO
from src.environment.trading_env import TradingEnv

print("="*60)
print("TESTANDO PPO v2 (800k steps)")
print("="*60)

# 1. Carrega modelo
print("\n1. Carregando modelo...")
try:
    model = PPO.load("models/best_ppo_v2/best_model.zip")
    print("OK! Modelo carregado com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar: {e}")
    exit(1)

# 2. Carrega dados de validação
print("\n2. Carregando dados de validacao...")
df = pd.read_csv('data/val_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"OK! {len(df)} candles carregados")

# 3. Cria ambiente
print("\n3. Criando ambiente de teste...")
env = TradingEnv(
    df=df,
    initial_balance=10000,
    position_size=0.1,
    leverage=3,
    commission=0.0004
)
print("OK! Ambiente criado")

# 4. Roda episódio de teste
print("\n4. Executando episodio de teste...")
obs, _ = env.reset()
done = False
total_reward = 0
step_count = 0

while not done and step_count < 100:  # Limita a 100 steps para teste rápido
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    done = terminated or truncated

print(f"OK! Teste concluido!")

# 5. Resultados
print("\n" + "="*60)
print("RESULTADOS DO TESTE:")
print("="*60)
print(f"Balance inicial: $10,000.00")
print(f"Balance final: ${info['balance']:,.2f}")
print(f"P&L Total: ${info['total_pnl']:,.2f}")
print(f"Trades: {info['trades']}")
print(f"Wins: {info['wins']}")
print(f"Losses: {info['losses']}")
if info['trades'] > 0:
    win_rate = (info['wins'] / info['trades']) * 100
    print(f"Win Rate: {win_rate:.1f}%")
print(f"Total Reward: {total_reward:.2f}")
print(f"Steps executados: {step_count}")

# 6. Avaliação
print("\n" + "="*60)
if info['total_pnl'] > 0:
    print("OK! MODELO FUNCIONAL! P&L positivo!")
elif info['trades'] > 0:
    print("MODELO OK mas precisa melhorar (P&L negativo)")
else:
    print("MODELO precisa mais treinamento (sem trades)")

print("="*60)
