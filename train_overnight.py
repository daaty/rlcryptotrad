"""
TREINAMENTO NOTURNO - Otimizado para completar em 6-8h
===============================================

Configuracao:
- 1.5M timesteps (mais rapido que 2M)
- Hyperparameters agressivos (mais exploração)
- Treinamento completo PPO + TD3
- Checkpoints apenas ao final
"""

import yaml
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.environment.trading_env import TradingEnv

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_env():
    """Cria ambiente com dados completos"""
    df = pd.read_csv('data/train_data_6m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    env = TradingEnv(
        df=df,
        initial_balance=10000,
        position_size=0.1,
        leverage=3,
        commission=0.0004
    )
    return DummyVecEnv([lambda: env])

def train_overnight(timesteps=1500000):
    """Treinamento otimizado para noite"""
    print("="*60)
    print("TREINAMENTO NOTURNO INICIADO")
    print(f"Timesteps: {timesteps:,}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    config = load_config()
    
    # DirectML
    device = "cpu"
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"\nUsando DirectML: {torch_directml.device_name(0)}")
    except:
        print("\nUsando CPU")
    
    # 1. PPO
    print("\n" + "="*60)
    print("TREINANDO PPO...")
    print("="*60)
    
    env = create_env()
    
    ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Mais agressivo
        n_steps=4096,
        batch_size=256,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.05,  # MUITO mais exploracao
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./logs/tensorboard_night/"
    )
    
    ppo.learn(total_timesteps=timesteps)
    ppo.save("models/ppo_night.zip")
    print("\nPPO SALVO: models/ppo_night.zip")
    
    # 2. TD3
    print("\n" + "="*60)
    print("TREINANDO TD3...")
    print("="*60)
    
    env = create_env()
    
    td3 = TD3(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Mais agressivo
        buffer_size=500000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        verbose=1,
        device=device,
        tensorboard_log="./logs/tensorboard_night/"
    )
    
    td3.learn(total_timesteps=timesteps)
    td3.save("models/td3_night.zip")
    print("\nTD3 SALVO: models/td3_night.zip")
    
    # 3. Teste rapido
    print("\n" + "="*60)
    print("TESTE RAPIDO DOS MODELOS")
    print("="*60)
    
    df_test = pd.read_csv('data/val_data.csv')
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    
    env_test = TradingEnv(
        df=df_test,
        initial_balance=10000,
        position_size=0.1,
        leverage=3,
        commission=0.0004
    )
    
    for name, model in [("PPO", ppo), ("TD3", td3)]:
        obs, _ = env_test.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"\n{name}:")
        print(f"  Balance: ${info['balance']:,.2f}")
        print(f"  P&L: ${info['total_pnl']:,.2f}")
        print(f"  Trades: {info['trades']}")
        print(f"  Win Rate: {(info['wins']/info['trades']*100) if info['trades'] > 0 else 0:.1f}%")
    
    # 4. Relatorio
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUIDO!")
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nModelos salvos:")
    print("  - models/ppo_night.zip")
    print("  - models/td3_night.zip")
    print("\nProximo: Testar no dashboard!")
    print("="*60)

if __name__ == "__main__":
    train_overnight()
