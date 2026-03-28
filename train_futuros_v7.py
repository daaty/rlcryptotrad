"""
Train Futuros V7 - CORREÇÃO ANTI-BUY&HOLD
========================================
Mudanças críticas em relação a V6:
1. ✅ Penalidade por holding prolongado (>100 steps)
2. ✅ Bonificação por realizar lucros
3. ✅ Indicator shaping melhorado (6 técnicas)
4. ✅ Treino do ZERO (não continuar V6 viciado)

Meta: Modelo que faz 50-500 trades com winrate 25-40%
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from src.environment.trading_env import TradingEnv
from src.callbacks.tensorboard_callback import TensorboardCallback
from src.callbacks.checkpoint_callback_directml import CheckpointCallbackDirectML
import torch

def main():
    print("="*64)
    print("TREINO FUTUROS V7 - ANTI-BUY&HOLD")
    print("="*64)
    print("CORREÇÕES:")
    print("  ✓ Penalidade por holding >100 steps")
    print("  ✓ Bonificação por lucros realizados")
    print("  ✓ 6 técnicas de análise técnica")
    print("  ✓ Treinamento do ZERO (sem viés V6)")
    print("="*64)
    
    # ===== CONFIGURAÇÃO =====
    DATA_PATH = "data/train_btcusdt_36m_20260109.csv"
    TOTAL_TIMESTEPS = 2_000_000  # 2M steps (V7: mais steps pra convergir com decaimento)
    SAVE_FREQ = 100_000  # Checkpoint a cada 100k
    
    # ===== CARREGAR DADOS =====
    print("\n[1/5] Carregando dados...")
    df = pd.read_csv(DATA_PATH)
    
    # Remover timestamp (não-numérico)
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    
    # Pegar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    print(f"  ✓ {len(df)} candles carregados")
    print(f"  ✓ {len(numeric_cols)} features: {list(numeric_cols[:10])}...")
    
    # ===== CRIAR AMBIENTE (CONFIGS V7 - ANTI-BUY&HOLD) =====
    print("\n[2/5] Criando ambiente V7...")
    
    def make_env():
        return TradingEnv(
            df=df,
            initial_balance=10000,           # $10k inicial
            commission=0.0004,                # 0.04% (Binance taker)
            slippage=0.0005,                  # 0.05%
            leverage=1.5,                     # 1.5x (seguro)
            position_size=0.05,               # 5% base (action limita a 2.5%)
            window_size=50,                   # 50 candles
            max_episode_steps=2000,           # 2000 steps/episódio
            random_start=True,                # ✅ CRITICAL: Start aleatório
            persist_balance=True,             # ✅ Balance persiste
            use_sharpe_reward=True,           # ✅ Sharpe Ratio
            enable_indicator_shaping=True,    # ✅ 6 técnicas ativas
            maintenance_margin_rate=0.005,
            liquidation_threshold=0.10
        )
    
    # Vetorizar ambiente (1 env)
    env = DummyVecEnv([make_env])
    
    print("  ✓ Ambiente criado com:")
    print("    - Penalidade holding >100 steps")
    print("    - Bonificação lucros >0.5%")
    print("    - 6 indicadores técnicos ativos")
    
    # ===== CRIAR MODELO SAC V7 (DO ZERO!) =====
    print("\n[3/5] Criando modelo SAC V7...")
    
    # Detecta device: DirectML (AMD), CUDA (NVIDIA) ou CPU
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            device_name = 'DirectML (AMD GPU)'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device_name = 'CUDA (NVIDIA)' if torch.cuda.is_available() else 'CPU'
    except ImportError:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_name = 'CUDA (NVIDIA)' if torch.cuda.is_available() else 'CPU'
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,               # LR padrão
        buffer_size=100_000,              # Replay buffer 100k
        learning_starts=1000,             # Começa aprender após 1k steps
        batch_size=256,                   # Batch maior (mais estável)
        tau=0.005,                        # Soft update target networks
        gamma=0.99,                       # Discount factor
        train_freq=1,                     # Treina a cada step
        gradient_steps=1,                 # 1 gradient step por env step
        ent_coef='auto',                  # Entropia automática
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,                    # Sem SDE (mais simples)
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        tensorboard_log="./logs/sac_v7",
        verbose=1,
        device=device
    )
    
    print(f"  ✓ Modelo SAC criado (device: {device_name})")
    print(f"  ✓ Treinando do ZERO (sem viés V6)")
    
    # ===== CALLBACKS =====
    print("\n[4/5] Configurando callbacks...")
    
    # Checkpoint callback (DirectML-compatible)
    checkpoint_callback = CheckpointCallbackDirectML(
        save_freq=SAVE_FREQ,
        save_path='./models/',
        name_prefix='sac_v7',
        save_replay_buffer=True,
        verbose=1
    )
    
    # TensorBoard callback customizado
    tensorboard_callback = TensorboardCallback()
    
    callback_list = CallbackList([checkpoint_callback, tensorboard_callback])
    
    print(f"  ✓ Salvando checkpoints a cada {SAVE_FREQ:,} steps")
    print(f"  ✓ TensorBoard: logs/sac_v7")
    
    # ===== TREINAR =====
    print("\n[5/5] Iniciando treinamento...")
    print(f"  Target: {TOTAL_TIMESTEPS:,} steps")
    print(f"  Checkpoints: a cada {SAVE_FREQ:,} steps (20 checkpoints)")
    print(f"  Tempo estimado: ~{TOTAL_TIMESTEPS / 1000 / 60:.0f}h (depende da GPU)")
    print("\n  V7 FEATURES:")
    print("    - Combos reduzidos (0.04 max) para evitar viés")
    print("    - Decaimento gradual do shaping após 500k steps")
    print("    - 100% → 0% entre 500k e 2M steps")
    print("="*64)
    print("\n🚀 TREINANDO...\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            log_interval=10,
            progress_bar=True
        )
        
        # Salvar modelo final
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_path = f'models/sac_v7_final_{timestamp}.zip'
        model.save(final_path)
        
        print("\n" + "="*64)
        print("✅ TREINAMENTO COMPLETO!")
        print("="*64)
        print(f"Modelo final salvo: {final_path}")
        print(f"Total steps: {TOTAL_TIMESTEPS:,}")
        print("\nPróximos passos:")
        print("  1. Rodar backtest: python backtest.py {final_path} data/train_btcusdt_36m_20260109.csv")
        print("  2. Verificar métricas:")
        print("     - Trades: esperado 200-500")
        print("     - Win rate: esperado 25-40%")
        print("     - Return: esperado +5-15%")
        print("  3. Se ainda fizer 1 trade, ajustar penalidades")
        
    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        interrupted_path = f'models/sac_v7_interrupted_{timestamp}.zip'
        model.save(interrupted_path)
        print(f"Modelo parcial salvo: {interrupted_path}")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        
        # Tentar salvar modelo antes de crashar
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            error_path = f'models/sac_v7_error_{timestamp}.zip'
            model.save(error_path)
            print(f"Modelo salvo antes do crash: {error_path}")
        except:
            print("Não foi possível salvar modelo")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
