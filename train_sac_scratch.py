"""
TREINO SAC DO ZERO - SEM TRANSFER LEARNING
Configuração agressiva para forçar exploração e trades
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from src.environment.trading_env import TradingEnv
import torch
import subprocess


class CheckpointCallback(BaseCallback):
    """Callback que salva modelo a cada N steps e executa backtest automático."""
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        experiment_name: str,
        train_data_path: str,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.experiment_name = experiment_name
        self.train_data_path = train_data_path
        self.checkpoint_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.checkpoint_count += 1
            checkpoint_path = f"{self.save_path}/sac_{self.experiment_name}_checkpoint_{self.checkpoint_count}_{self.n_calls}.zip"
            
            print(f"\n{'='*70}")
            print(f"💾 CHECKPOINT {self.checkpoint_count} - {self.n_calls:,} steps")
            print(f"{'='*70}")
            
            # Salvar modelo
            self.model.save(checkpoint_path)
            print(f"✅ Modelo salvo: {checkpoint_path}")
            
            # Executar backtest automático
            print(f"\n🔬 Executando backtest automático...")
            try:
                result = subprocess.run(
                    ['python', 'backtest.py', checkpoint_path, self.train_data_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    if "Return:" in output and "Winrate:" in output:
                        for line in output.split('\n'):
                            if any(metric in line for metric in ["Return:", "Winrate:", "Profit Factor:", "Sharpe:", "Total Trades:"]):
                                print(f"  {line.strip()}")
                    else:
                        print("  ✅ Backtest concluído")
                else:
                    print(f"  ⚠️ Backtest falhou")
            except Exception as e:
                print(f"  ⚠️ Erro no backtest")
            
            print(f"{'='*70}\n")
        
        return True


def train_sac_from_scratch():
    """Treina SAC do zero com configurações agressivas para exploração."""
    
    # Detectar GPU
    try:
        import torch_directml
        device = torch_directml.device()
        print("[OK] GPU AMD via DirectML")
    except ImportError:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEVICE] {device}")
    
    # Carregar config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Dados
    train_data = 'data/train_btcusdt_36m_20260109.csv'
    df = pd.read_csv(train_data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n[DATA] {len(df):,} candles carregados")
    
    # Criar ambiente com DELTA EQUITY puro (sem Sharpe)
    env_config = config['environment']
    env = TradingEnv(
        df=df,
        initial_balance=env_config['initial_balance'],
        commission=env_config['commission'],
        slippage=env_config.get('slippage', 0.0005),
        leverage=env_config['leverage'],
        position_size=env_config['position_size'],
        window_size=env_config['window_size'],
        max_episode_steps=5000,
        random_start=True,
        persist_balance=False,  # DESABILITADO: Começa do zero a cada episódio
        use_sharpe_reward=False,  # DELTA EQUITY PURO
        use_hybrid_reward=False
    )
    env = DummyVecEnv([lambda: env])
    
    # Policy kwargs com SDE e log_std_init alto para máxima exploração
    policy_kwargs = dict(
        net_arch=[256, 256],  # Menor que TD3 (400, 300) - aprende mais rápido
        activation_fn=torch.nn.ReLU,
        log_std_init=-1.0  # Começa com desvio padrão ~0.37 (alto ruído inicial)
    )
    
    # Action noise ALTO para forçar exploração
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.5 * np.ones(n_actions)  # 50% de noise (vs 30% antes)
    )
    
    print("\n[SAC] Configuração AGRESSIVA para exploração:")
    print("  net_arch: [256, 256] (menor = aprende rápido)")
    print("  ent_coef: 0.7 FIXO (máxima exploração constante)")
    print("  use_sde: True + log_std_init=-1.0 (ruído alto)")
    print("  action_noise: 50% (forçar exploração)")
    print("  learning_rate: 3e-4 (mais rápido)")
    print("  reward: DELTA EQUITY PURO (lucro direto)")
    
    # Criar modelo SAC DO ZERO
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,  # Mais rápido que 1e-4
        buffer_size=200000,  # 200k (cabe na memória: ~700MB vs 1.77GB)
        learning_starts=5000,  # Começa cedo (5k vs 10k)
        batch_size=256,
        tau=0.005,
        gamma=0.99,  # Menor gamma (mais imediatista)
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.7,  # FIXO ALTO - máxima exploração constante
        use_sde=True,  # SDE ativo com log_std_init alto
        action_noise=action_noise,
        verbose=1,
        device=device
    )
    
    # Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=100000,  # 100k steps (mais frequente)
        save_path='models',
        experiment_name='scratch',
        train_data_path=train_data,
        verbose=1
    )
    
    # Eval callback para mostrar reward médio
    eval_env = TradingEnv(
        df=df,
        initial_balance=env_config['initial_balance'],
        commission=env_config['commission'],
        slippage=env_config.get('slippage', 0.0005),
        leverage=env_config['leverage'],
        position_size=env_config['position_size'],
        window_size=env_config['window_size'],
        max_episode_steps=5000,
        random_start=True,
        persist_balance=False,
        use_sharpe_reward=False,
        use_hybrid_reward=False
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=5000,  # Avalia a cada 5k steps
        n_eval_episodes=3,  # 3 episódios de teste
        log_path=None,  # Sem salvar logs (economiza disco)
        best_model_save_path=None,  # Sem salvar melhor modelo
        deterministic=False,  # Avalia com exploração
        verbose=1
    )
    
    # Combinar callbacks
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    
    # Treinar
    timesteps = 1000000  # 1M steps (mais rápido que 1.5M)
    
    print(f"\n⏳ Treinando SAC DO ZERO ({timesteps/1e6:.1f}M steps)...")
    print(f"💾 Checkpoint a cada 100k steps")
    print(f"📊 Avaliação de reward a cada 5k steps")
    print(f"Tempo estimado: 3-4 horas")
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_cb,
        progress_bar=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    # Salvar modelo final
    final_path = 'models/sac_scratch_final.zip'
    model.save(final_path)
    
    print(f"\n✅ TREINAMENTO COMPLETO!")
    print(f"Tempo: {duration:.2f} horas")
    print(f"Modelo: {final_path}")
    print(f"\nBacktest:")
    print(f"  python backtest.py {final_path} {train_data}")


if __name__ == "__main__":
    train_sac_from_scratch()
