"""
Script de treinamento V12 - REWARD SIMPLES E SIMÉTRICO (BALANCEADO)

CORREÇÕES V12 (vs V11):
- Anti-overtrading MODERADO (3 trades/24h max, -0.03 por extra) - era 1/-0.10
- Flip-flop penalty SUAVE (-0.02 se Long→Short em <50 steps) - era -0.05/<100
- Stop-loss -7% (equilibrado) - era -8%
- REMOVIDOS: Bônus por sair de Short/Long (causavam ASSIMETRIA!)
- Mantém: Punição aleatório, bônus lucro, bônus holding
- Episodes 4000 steps (contexto longo)
- ent_coef=0.05 FIXO (MAIS exploração vs 0.03) - CRITICAL!

OBJETIVO V12:
- Long/Short: 40-50% CADA (SIMÉTRICO!)
- Trades: 500-1000 totais (permitir mais que V11)
- Win rate: 25-35% (realista)
- Profit factor: >1.2
- Flat: 20-40% (esperar mas não paralisar)

PRINCÍPIO: SIMPLICIDADE = SIMETRIA = BALANCE
"""

import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
import pandas as pd
from datetime import datetime

from src.environment.trading_env import TradingEnv
from src.callbacks.checkpoint_callback_directml import CheckpointCallbackDirectML
from src.callbacks.tensorboard_callback import TensorboardCallback

# ===== CONFIGURAÇÃO =====
DATA_PATH = 'data/train_btcusdt_36m_20260109.csv'
MODELS_DIR = './models'
LOGS_DIR = './logs'

# Parâmetros de treinamento
TOTAL_TIMESTEPS = 2_000_000
SAVE_FREQ = 50_000  # V12: Save a cada 50k (validação rápida Long/Short)

# Criar diretórios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("="*70)
print("TREINAMENTO SAC V12 - REWARD SIMPLES E SIMÉTRICO")
print("="*70)

# ===== CARREGAR DADOS =====
print("\n📊 Carregando dados...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset carregado: {len(df)} candles ({len(df)/96:.1f} dias)")
print(f"  Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# ===== CRIAR AMBIENTE V8 =====
print("\n🏗️ Criando ambiente V8...")

def make_env():
    """Factory para criar ambiente vetorizado."""
    env = TradingEnv(
        data_path=DATA_PATH,
        initial_balance=10000,
        commission=0.0004,  # 0.04% Binance
        slippage=0.0005,    # 0.05%
        leverage=1.5,       # Baixo (seguro)
        position_size=0.05, # 5% por trade
        window_size=50,
        max_episode_steps=4000,  # V11: AUMENTADO 3500→4000 (mata churn de vez)
        random_start=True,
        persist_balance=True,
        use_sharpe_reward=True,
        enable_indicator_shaping=True  # V11: Apenas punição aleatório
    )
    return env

env = DummyVecEnv([make_env])

print("✓ Ambiente V8 criado:")
print(f"  Observation shape: {env.observation_space.shape}")
print(f"  Action shape: {env.action_space.shape}")
print(f"  Episodes: ~{TOTAL_TIMESTEPS // 2000} (2000 steps cada)")

# ===== DETECTAR GPU (DirectML para AMD) =====
print("\n🎮 Detectando acelerador...")
device = 'cpu'
device_name = 'CPU'

try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        device_name = 'DirectML (AMD GPU)'
        print(f"✓ DirectML disponível!")
        print(f"  Device: {device}")
except ImportError:
    print("⚠️ torch-directml não instalado, usando CPU")
except Exception as e:
    print(f"⚠️ Erro ao inicializar DirectML: {e}")
    print("  Fallback para CPU")

print(f"✓ Usando: {device_name}")

# ===== CRIAR MODELO SAC =====
print("\n🤖 Criando modelo SAC V12...")

model = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef=0.05,  # V12: AUMENTADO 0.03→0.05 (MAIS exploração - CRÍTICO!)
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU
    ),
    verbose=1,
    device=device,
    tensorboard_log=LOGS_DIR
)

print("✓ Modelo SAC V11 criado:")
print(f"  Policy: MLP [256, 256]")
print(f"  Learning rate: 3e-4")
print(f"  Buffer: 100k")
print(f"  ent_coef: 0.03 (FIXO - MENOS exploração que V10!)")
print(f"  Device: {device_name}")

# ===== CALLBACKS =====
print("\n📋 Configurando callbacks...")

# Checkpoint callback (DirectML-safe)
checkpoint_callback = CheckpointCallbackDirectML(
    save_freq=SAVE_FREQ,
    save_path=MODELS_DIR,
    name_prefix='sac_v8',
    save_replay_buffer=True,
    verbose=1
)

# TensorBoard callback (atualiza shaping decay)
tensorboard_callback = TensorboardCallback(verbose=1)

callbacks = CallbackList([checkpoint_callback, tensorboard_callback])

print("✓ Callbacks configurados:")
print(f"  Checkpoints: a cada {SAVE_FREQ:,} steps")
print(f"  Diretório: {MODELS_DIR}")
print(f"  TensorBoard: {LOGS_DIR}")

# ===== TREINAR =====
print("\n" + "="*70)
print("🚀 INICIANDO TREINAMENTO V11 - ANTI-OVERTRADING EXTREMO")
print("="*70)
print(f"Total steps: {TOTAL_TIMESTEPS:,}")
print(f"Save frequency: {SAVE_FREQ:,}")
print(f"Checkpoints esperados: {TOTAL_TIMESTEPS // SAVE_FREQ}")
print(f"\n🎯 V11 MUDANÇAS EXTREMAS:")
print(f"  - Máximo: 1 trade/24h (-0.10 por extra) [BRUTAL]")
print(f"  - Flip-flop: -0.05 (<100 steps) [AMPLIADO]")
print(f"  - Stop-loss: -8% (não corta winners)")
print(f"  - REMOVIDOS: TODOS combos/bônus direcionais")
print(f"  - Episodes: 4000 steps (anti-churn extremo)")
print(f"  - ent_coef: 0.03 FIXO (menos exploração)")
print(f"\n🎯 EXPECTATIVA:")
print(f"  - Trades: 200-500 totais (vs 1,120 em V9)")
print(f"  - Win rate: 30-40% (mais qualidade)")
print(f"  - Flat: 30-50% (finalmente espera!)")
print("\n⏱️ Tempo estimado: ~24-28h (AMD DirectML)")
print("="*70 + "\n")

start_time = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"⏱️ Tempo total: {training_time}")
    print(f"📊 FPS médio: {TOTAL_TIMESTEPS / training_time.total_seconds():.1f}")
    
    # Salvar modelo final
    final_path = f"{MODELS_DIR}/sac_v8_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n💾 Salvando modelo final: {final_path}.zip")
    
    # Salvar com torch.no_grad() (DirectML-safe)
    with torch.no_grad():
        try:
            model.save(final_path)
            print("✓ Modelo final salvo com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro ao salvar modelo final: {e}")
            print("  (Checkpoints já estão salvos)")
    
    print("\n" + "="*70)
    print("PRÓXIMOS PASSOS:")
    print("="*70)
    print("1. Backtest checkpoint 100k:")
    print(f"   python backtest.py {MODELS_DIR}/sac_v8_100000_steps.zip {DATA_PATH}")
    print("\n2. Verificar se faz LONG E SHORT (não só um):")
    print("   - Espera: 30-40% long, 20-30% short, 30-50% flat")
    print("   - Win rate: 30-40%")
    print("   - Trades: 200-800")
    print("\n3. Se checkpoint 100k balanceado, testar 2M:")
    print(f"   python backtest.py {MODELS_DIR}/sac_v8_2000000_steps.zip {DATA_PATH}")
    print("="*70 + "\n")

except KeyboardInterrupt:
    print("\n⚠️ Treinamento interrompido pelo usuário")
    print("  Checkpoints já salvos continuam disponíveis")
except Exception as e:
    print(f"\n❌ ERRO durante treinamento: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
    print("\n✓ Ambiente fechado")
