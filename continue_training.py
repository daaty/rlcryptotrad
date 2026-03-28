"""
Continua treinamento SAC a partir do checkpoint 1.3M
Preserva replay buffer, optimizer state e learning schedule.
"""

import yaml
import torch
import torch_directml  # AMD GPU via DirectML
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from src.environment.trading_env import TradingEnv
import numpy as np
import pandas as pd


def make_env(config):
    """Cria ambiente de treino."""
    df = pd.read_csv(config['data_path'])
    
    def _init():
        return TradingEnv(
            df=df,
            config=config['environment'],
            random_start=True,
            persist_balance=False,  # Cada episódio recomeça com balance limpo
            use_sharpe_reward=False,  # Delta equity puro (como estava)
            use_hybrid_reward=False
        )
    return _init


def main():
    # Carrega configuração
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Path do dataset
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    print("=" * 60)
    print("🔄 CONTINUANDO TREINAMENTO SAC - Checkpoint 1.3M")
    print("=" * 60)
    print(f"📊 Dataset: {data_path}")
    print(f"🎯 Target: 1.3M → 3M steps (1.7M adicional)")
    print(f"💰 Commission: {config['environment']['commission']}")
    print(f"📈 Current winrate: ~20% → Target: 40-50%")
    print("=" * 60)
    
    # Path do dataset
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    # 🔥 CONFIGURA DIRECTML ANTES DE TUDO
    dml_device = torch_directml.device()
    print(f"\n🎮 DirectML Device: {dml_device}")
    
    # Cria ambiente
    env = DummyVecEnv([lambda: TradingEnv(
        data_path=data_path,
        config=config['environment'],
        random_start=True,
        persist_balance=False,
        use_sharpe_reward=False,
        use_hybrid_reward=False
    )])
    
    # 🔥 CARREGA MODELO DO CHECKPOINT 1.3M
    checkpoint_path = "models/sac_scratch_checkpoint_13_1300000.zip"
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    # Carrega modelo com DirectML diretamente
    model = SAC.load(
        checkpoint_path,
        env=env,
        device=dml_device,  # Usa DirectML desde o load
        print_system_info=True
    )
    
    print("\n✅ Checkpoint loaded successfully!")
    print(f"   - Replay buffer size: {model.replay_buffer.size()}")
    print(f"   - Actor learning rate: {model.actor.optimizer.param_groups[0]['lr']}")
    print(f"   - Entropy coef: {model.ent_coef}")
    print(f"   - Steps trained: 1,300,000")
    
    # Callbacks: Salva a cada 100k, eval a cada 5k
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./models/',
        name_prefix='sac_continue',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=False,  # IMPORTANTE: SDE precisa de noise
        render=False,
        n_eval_episodes=5
    )
    
    # 🚀 CONTINUA TREINAMENTO: 1.3M → 3M (mais 1.7M steps)
    print("\n" + "=" * 60)
    print("🚀 Starting training from 1.3M → 3M steps")
    print("   Next checkpoint: 1.4M (in 100k steps)")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=1_700_000,  # 1.3M + 1.7M = 3M total
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False  # CRÍTICO: Não reseta contador!
    )
    
    # Salva modelo final
    model.save("models/sac_final_3M")
    print("\n✅ Training complete! Final model saved: models/sac_final_3M.zip")
    print(f"   Total steps: 3,000,000")
    print(f"   Checkpoints: sac_continue_*.zip (1.4M, 1.5M, ..., 3M)")


if __name__ == '__main__':
    main()
