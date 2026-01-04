"""
Teste rápido do sistema de voting com desempate
"""
import numpy as np
from stable_baselines3 import PPO, TD3
from src.models.ensemble_model import EnsembleModel
import logging

logging.basicConfig(level=logging.INFO)

# Carrega modelos
print("Carregando modelos...")
ppo = PPO.load("models/ensemble/ppo/ppo_final.zip")
td3 = TD3.load("models/ensemble/td3/td3_final.zip")

models = {
    'ppo': ppo,
    'td3': td3
}

# Cria ensemble
ensemble = EnsembleModel(
    models=models,
    strategy='weighted',
    weights={'ppo': 0.5, 'td3': 0.5}
)

# Cria observação fake (50, 23)
obs = np.random.randn(50, 23).astype(np.float32)

print("\nTestando predição...")
action, info = ensemble.predict(obs)

print(f"\nResultado:")
print(f"  Votos: {info['votes']}")
print(f"  Confianças: {info['confidences']}")
print(f"  Ação final: {action}")
print(f"  Acordo: {info['agreement']:.1%}")
print(f"  Estratégia: {info['strategy']}")

# Testa 10 vezes
print("\n\nTestando 10 vezes:")
for i in range(10):
    obs = np.random.randn(50, 23).astype(np.float32)
    action, info = ensemble.predict(obs)
    print(f"  {i+1}. PPO={info['votes']['ppo']}, TD3={info['votes']['td3']} -> Final={action}")
