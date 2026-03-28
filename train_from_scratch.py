"""
Treina SAC do ZERO com reward refinado e ent_coef=0.5
Teste rápido: 100k steps para comparar com 1.3M checkpoint
"""

import yaml
import torch
import torch_directml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from src.environment.trading_env import TradingEnv
import numpy as np
import pandas as pd


def main():
    # DirectML GPU
    dml_device = torch_directml.device()
    print(f"🎮 DirectML Device: {dml_device}")
    
    # Carrega config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Dataset
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    print("=" * 60)
    print("🆕 TREINO SAC DO ZERO - Reward Refinado")
    print("=" * 60)
    print(f"📊 Dataset: {data_path}")
    print(f"🎯 Target: 100k steps (teste rápido)")
    print(f"⚙️  Configuração:")
    print(f"   - ent_coef: 0.5 (refinamento)")
    print(f"   - Reward: Penalty 2x maior (-0.02)")
    print(f"   - Action noise: 40%")
    print(f"   - log_std_init: -1.0")
    print(f"   - buffer_size: 200k")
    print("=" * 60 + "\n")
    
    # Ambiente
    env = DummyVecEnv([lambda: TradingEnv(
        data_path=data_path,
        config=config['environment'],
        random_start=True,
        persist_balance=False,
        use_sharpe_reward=False,
        use_hybrid_reward=False
    )])
    
    # Action noise: Moderado
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.4 * np.ones(n_actions)  # 40%
    )
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=[256, 256],
        log_std_init=-1.0,  # σ ≈ 0.37
    )
    
    print("🏗️  Criando modelo SAC do zero...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        ent_coef=0.5,  # 0.5 (não 0.7)
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=dml_device,
        tensorboard_log="./logs/sac_scratch_test/"
    )
    
    print(f"✅ Modelo criado!")
    print(f"   - ent_coef: {model.ent_coef}")
    print(f"   - learning_rate: {model.learning_rate}")
    print(f"   - action_noise: 40%")
    print(f"   - SDE enabled: {model.use_sde}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path='./models/',
        name_prefix='sac_scratch_test',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=False,
        render=False,
        n_eval_episodes=5
    )
    
    # TREINO: 100k steps (teste rápido)
    print("\n" + "=" * 60)
    print("🚀 Iniciando treinamento: 0 → 100k steps")
    print("   Checkpoints a cada 50k")
    print("   Tempo estimado: ~50min")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=100_000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    model.save("models/sac_scratch_test_100k")
    print("\n✅ Treinamento completo!")
    print("   Total steps: 100,000")
    print("   Modelo salvo: models/sac_scratch_test_100k.zip")
    print("\n📊 Teste com:")
    print("   python backtest.py models/sac_scratch_test_100k.zip data/train_btcusdt_36m_20260109.csv")


if __name__ == '__main__':
    main()
