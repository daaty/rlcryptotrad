"""
🎮 Demo Simples - Paper Trading com Ensemble
Sem LLM (apenas modelos RL)
"""

import logging
import numpy as np
import yaml
from pathlib import Path

from src.models.ensemble_model import EnsembleModel
from src.environment.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("="*60)
print("🎮 DEMO: ENSEMBLE PPO + TD3 PAPER TRADING")
print("="*60)

# Carrega config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 1. Carrega modelos individuais
print("\n🤖 Carregando modelos RL...")
from stable_baselines3 import PPO, TD3

models = {
    'ppo': PPO.load('models/ensemble/ppo/ppo_final.zip'),
    'td3': TD3.load('models/ensemble/td3/td3_final.zip')
}
print(f"✅ {len(models)} modelos carregados")

# 2. Cria ensemble
print("\n🎯 Criando ensemble...")
ensemble = EnsembleModel(
    models=models,
    strategy=config['ensemble']['strategy'],
    weights=config['ensemble']['weights']
)
print(f"   Estratégia: {ensemble.strategy.value}")

# 2. Cria environment com dados de teste
print("\n📊 Criando environment de teste...")
env = TradingEnv(
    data_path='data/test_data.csv',
    config=config['environment']
)
print(f"✅ Environment criado")

# 3. Simula 10 episódios
print("\n🎲 Simulando 10 episódios de trading...")
print("="*60)

total_rewards = []
total_trades = []

for episode in range(10):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        # Ensemble vota
        action, voting_info = ensemble.predict(obs)
        
        # Executa
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        steps += 1
    
    total_rewards.append(episode_reward)
    total_trades.append(env.trades)
    
    action_names = ['FLAT', 'LONG', 'SHORT']
    print(f"Ep {episode+1:2d}: Reward={episode_reward:7.2f} | "
          f"Steps={steps:3d} | Trades={env.trades:2d} | "
          f"Final Balance=${env.balance:.2f} | "
          f"Win Rate={env.wins/(env.trades if env.trades > 0 else 1):.1%}")

print("\n" + "="*60)
print("📊 ESTATÍSTICAS FINAIS")
print("="*60)
print(f"Reward Médio:  {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print(f"Melhor Reward: {np.max(total_rewards):.2f}")
print(f"Pior Reward:   {np.min(total_rewards):.2f}")
print(f"Trades Médios: {np.mean(total_trades):.1f}")
print(f"Balance Final: ${env.balance:.2f}")

print("\n✅ Demo concluída!")
print("\n💡 Próximos passos:")
print("1. Adicionar análise de sentimento (configurar OpenAI no .env)")
print("2. Conectar à Binance real (adicionar API keys)")
print("3. Aumentar timesteps de treinamento para melhorar performance")
