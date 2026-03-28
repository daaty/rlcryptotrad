"""
Treina SAC do ZERO com reward refinado e ent_coef=0.5
Foco: QUALIDADE de trades (winrate alto) sobre quantidade
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
    print("🔄 FINE-TUNING SAC 1.3M - Reward Suavizado")
    print("=" * 60)
    print(f"📊 Dataset: {data_path}")
    print(f"📦 Checkpoint base: sac_scratch_checkpoint_13_1300000.zip")
    print(f"🎯 Objetivo: Winrate 15-25% (transição suave)")
    print(f"⚙️  Ajustes aplicados:")
    print(f"   - ent_coef: 0.7 → 0.5 (menos exploração)")
    print(f"   - learning_rate: 3e-4 → 1e-4 (fine-tuning)")
    print(f"   - Penalty 2x maior: -0.02 (vs -0.01) [SUAVE]")
    print(f"   - Bonus reduzido: 0.015 (vs 0.02)")
    print(f"   - Penalidade mal timing: -0.01 (vs -0.02) [SUAVE]")
    print("=" * 60 + "\n")
    
    # Ambiente (DEVE usar mesmo config do treino original!)
    env = DummyVecEnv([lambda: TradingEnv(
        data_path=data_path,
        config=config['environment'],
        random_start=True,
        persist_balance=False,
        use_sharpe_reward=False,
        use_hybrid_reward=False
    )])
    
    # 🔥 CARREGA CHECKPOINT 1.3M
    checkpoint_path = "models/sac_scratch_checkpoint_13_1300000.zip"
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    model = SAC.load(
        checkpoint_path,
        env=env,
        device=dml_device,
        print_system_info=True
    )
    
    # AJUSTA HYPERPARAMETERS corretamente para fine-tuning
    # ent_coef: Precisa criar novo tensor
    if isinstance(model.ent_coef, torch.Tensor):
        model.ent_coef = torch.tensor([0.5]).to(dml_device)
    else:
        model.ent_coef = 0.5
    
    # learning_rate: Ajusta nos optimizers
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = 1e-4
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = 1e-4
    
    print(f"\n✅ Checkpoint carregado e ajustado!")
    print(f"   - Base: 1.3M steps treinados")
    print(f"   - ent_coef ajustado: 0.7 → 0.5")
    print(f"   - learning_rate ajustado: 3e-4 → 1e-4 (fine-tuning)")
    print(f"   - Replay buffer: {model.replay_buffer.size()} samples")
    print(f"   - Novo reward: Penalty 2x maior (transição suave)")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # Salva a cada 50k (teremos 2 checkpoints em 100k)
        save_path='./models/',
        name_prefix='sac_finetuned',
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
    
    # FINE-TUNING: 100k steps (teste rápido ~1h)
    print("\n" + "=" * 60)
    print("🚀 Iniciando fine-tuning: 1.3M → 1.4M (100k steps)")
    print("   Checkpoints a cada 50k")
    print("   Tempo estimado: ~1h")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=100_000,  # Teste rápido
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False  # NÃO reseta contador (continua de 1.3M)
    )
    
    model.save("models/sac_finetuned_1400k")
    print("\n✅ Fine-tuning completo!")
    print("   Total steps: 1,400,000 (1.3M base + 100k fine-tuning)")
    print("   Modelo salvo: models/sac_finetuned_1400k.zip")
    print("\n📊 Teste com:")
    print("   python backtest.py models/sac_finetuned_1400k.zip data/train_btcusdt_36m_20260109.csv")


if __name__ == '__main__':
    main()
