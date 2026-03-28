"""
Continua treinamento do checkpoint 100k com nova config
Modelo está aprendendo 8x mais rápido que o antigo!
"""

import yaml
import torch
import torch_directml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environment.trading_env import TradingEnv


def main():
    # DirectML GPU
    dml_device = torch_directml.device()
    print(f"🎮 DirectML Device: {dml_device}")
    
    # Carrega config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    print("=" * 60)
    print("🚀 CONTINUANDO TREINO - 100k → 500k")
    print("=" * 60)
    print(f"📊 Base: 100k steps (8.18% winrate)")
    print(f"🎯 Target: 500k total (400k novos)")
    print(f"📈 Taxa aprendizado: 8x mais rápido!")
    print(f"📊 Projeção: 200k→16%, 300k→24%, 500k→40%")
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
    
    # CARREGA CHECKPOINT 100k
    checkpoint_path = "models/sac_scratch_test_100k.zip"
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    model = SAC.load(
        checkpoint_path,
        env=env,
        device=dml_device,
        print_system_info=True
    )
    
    print(f"\n✅ Checkpoint carregado!")
    print(f"   - Steps treinados: 100,000")
    print(f"   - Winrate atual: 8.18%")
    print(f"   - ent_coef: {model.ent_coef}")
    print(f"   - Replay buffer: {model.replay_buffer.size()} samples")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # A cada 100k
        save_path='./models/',
        name_prefix='sac_continue_new',
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
    
    # CONTINUA: 100k → 500k (mais 400k steps)
    print("\n" + "=" * 60)
    print("🚀 Iniciando: 100k → 500k (400k steps)")
    print("   Checkpoints: 200k, 300k, 400k, 500k")
    print("   Tempo estimado: ~3h")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=400_000,  # 100k base + 400k = 500k total
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False  # Continua contagem
    )
    
    model.save("models/sac_final_500k")
    print("\n✅ Treinamento completo!")
    print("   Total steps: 500,000")
    print("   Modelo salvo: models/sac_final_500k.zip")
    print("\n📊 Teste checkpoints:")
    print("   python backtest.py models/sac_continue_new_200000_steps.zip data/train_btcusdt_36m_20260109.csv")
    print("   python backtest.py models/sac_continue_new_300000_steps.zip data/train_btcusdt_36m_20260109.csv")
    print("   python backtest.py models/sac_final_500k.zip data/train_btcusdt_36m_20260109.csv")


if __name__ == '__main__':
    main()
