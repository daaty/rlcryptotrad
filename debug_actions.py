"""
DEBUG: Analisa distribuicao de acoes do modelo SAC
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

def analyze_actions():
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
    
    # Criar modelo
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0
    )
    
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
        target_entropy=-0.5,
        action_noise=action_noise,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=dml_device
    )
    
    print("="*80)
    print("ANALISE DE DISTRIBUICAO DE ACOES")
    print("="*80)
    
    obs = env.reset()
    
    actions_list = []
    action_types = {'flat': 0, 'long': 0, 'short': 0}
    
    print("\nColetando 1000 acoes do modelo...\n")
    
    for i in range(1000):
        # Predict com deterministic=False (modo treino)
        action, _states = model.predict(obs, deterministic=False)
        actions_list.append(float(action[0][0]))
        
        # Classificar acao
        if action[0][0] < -0.1:
            action_types['short'] += 1
        elif action[0][0] > 0.1:
            action_types['long'] += 1
        else:
            action_types['flat'] += 1
        
        # Step
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            obs = env.reset()
    
    actions_array = np.array(actions_list)
    
    print("ESTATISTICAS:")
    print(f"  Media: {actions_array.mean():.4f}")
    print(f"  Std: {actions_array.std():.4f}")
    print(f"  Min: {actions_array.min():.4f}")
    print(f"  Max: {actions_array.max():.4f}")
    print(f"  Mediana: {np.median(actions_array):.4f}")
    
    print(f"\nDISTRIBUICAO:")
    print(f"  Flat (zona neutra): {action_types['flat']} ({action_types['flat']/10:.1f}%)")
    print(f"  Long (> 0.1): {action_types['long']} ({action_types['long']/10:.1f}%)")
    print(f"  Short (< -0.1): {action_types['short']} ({action_types['short']/10:.1f}%)")
    
    print(f"\nHISTOGRAMA:")
    bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
    hist, _ = np.histogram(actions_array, bins=bins)
    print(f"  [-1.0, -0.5): {hist[0]} ({hist[0]/10:.1f}%)")
    print(f"  [-0.5, -0.1): {hist[1]} ({hist[1]/10:.1f}%)")
    print(f"  [-0.1, 0.1): {hist[2]} ({hist[2]/10:.1f}%) <- ZONA NEUTRA")
    print(f"  [0.1, 0.5): {hist[3]} ({hist[3]/10:.1f}%)")
    print(f"  [0.5, 1.0]: {hist[4]} ({hist[4]/10:.1f}%)")
    
    print("\n" + "="*80)
    print("DIAGNOSTICO:")
    if action_types['flat'] > 800:
        print("  [X] PROBLEMA: >80% das acoes na zona neutra!")
        print("  Causa provavel: action_noise muito alto OU zona neutra muito larga")
        print("\n  SOLUCOES:")
        print("    1. Reduzir action_noise: 0.4 -> 0.2")
        print("    2. Reduzir zona neutra: 0.1 -> 0.05")
        print("    3. Desabilitar use_sde temporariamente")
    elif action_types['flat'] > 500:
        print("  [!] ATENCAO: >50% das acoes na zona neutra")
        print("  Modelo ainda explorando, pode melhorar com treino")
    else:
        print("  [OK] Distribuicao de acoes parece normal")
    
    print("="*80)

if __name__ == "__main__":
    analyze_actions()
