"""
DIAGNOSTICO V2: Simula comportamento exato do modelo SAC
"""

import sys
sys.path.append('src')
sys.path.append('callbacks')

import torch
import torch_directml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from environment.trading_env import TradingEnv
import numpy as np

# Fix encoding for Windows terminal
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def test_sac_model_liquidation():
    """Simula 10 steps com modelo SAC randômico."""
    
    # DirectML
    dml_device = torch_directml.device()
    
    def make_env():
        return TradingEnv(
            data_path="data/train_btcusdt_36m_20260109.csv",
            initial_balance=10000,
            commission=0.0004,
            slippage=0.0005,
            leverage=3,
            position_size=0.1,
            window_size=50,
            max_episode_steps=5000,
            random_start=True,
            persist_balance=False,
            use_sharpe_reward=False,
            use_hybrid_reward=False,
            maintenance_margin_rate=0.005,
            liquidation_threshold=0.10,
            enable_indicator_shaping=True
        )
    
    env = DummyVecEnv([make_env])
    
    # Action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.4 * np.ones(n_actions)
    )
    
    # Criar modelo SAC
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0
    )
    
    target_entropy = -0.5
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy=target_entropy,
        action_noise=action_noise,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=dml_device,
        tensorboard_log="./logs/diagnosis/"
    )
    
    print("TESTE COM MODELO SAC REAL (RANDOMICO)\n")
    print("="*80)
    
    # Reset
    obs = env.reset()
    
    for step in range(20):
        # Predict action (modelo randômico no início)
        action, _states = model.predict(obs, deterministic=False)
        
        # Unwrap environment
        unwrapped_env = env.envs[0]
        
        print(f"\nSTEP {step + 1}")
        print(f"   Action: {float(action[0][0]):.4f}")
        print(f"   Balance ANTES: ${unwrapped_env.balance:.2f}")
        print(f"   Position ANTES: {unwrapped_env.position}")
        
        # Execute
        obs, reward, done, info = env.step(action)
        
        print(f"   Balance DEPOIS: ${unwrapped_env.balance:.2f}")
        print(f"   Equity: ${unwrapped_env.equity:.2f}")
        print(f"   Position: {unwrapped_env.position}")
        print(f"   Trades: {info[0]['trades']}")
        print(f"   Liquidations: {info[0]['liquidations']}")
        print(f"   Reward: {reward[0]:.6f}")
        print(f"   Done: {done[0]}")
        
        if info[0]['liquidations'] > 0:
            print(f"\n   [X] LIQUIDACAO DETECTADA!")
            print(f"\n   DETALHES:")
            print(f"      current_step: {unwrapped_env.current_step}")
            print(f"      episode_length: {unwrapped_env.episode_length}")
            print(f"      position_size usado: {unwrapped_env.current_position_size * 100:.1f}%")
            print(f"      position_value: ${abs(unwrapped_env.position_value):.2f}")
            
            # Pega preço atual
            current_price = unwrapped_env.df.loc[unwrapped_env.current_step - 1, 'close']
            prev_price = unwrapped_env.df.loc[unwrapped_env.current_step - 2, 'close']
            print(f"      Preço anterior: ${prev_price:.2f}")
            print(f"      Preço atual: ${current_price:.2f}")
            print(f"      Variação: {((current_price - prev_price) / prev_price * 100):.2f}%")
            
            break
        
        if done[0]:
            print(f"\n   [OK] Episodio terminou (step {step + 1})")
            obs = env.reset()
            
            if step < 5:
                print(f"   [!] Episodio MUITO curto! Possivel problema")

if __name__ == "__main__":
    test_sac_model_liquidation()
