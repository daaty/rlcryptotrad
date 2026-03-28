"""
SAC V14 - BASEADO NO V8/V12 QUE FUNCIONOU
==========================================

APRENDIZADO DOS TESTES:
- V6 500k: Melhor modelo (20% win, 43%/43% balance) MAS colapsa aos 600k+
- V8 600k: 18% win, balanceado, mas overtrading (-10%)
- V13: FALHOU completamente (win rate obs quebrou tudo)

ESTRATÉGIA V14:
- Usar ambiente/config do V8 que FUNCIONOU
- Buffer 100k (vs V6 200k) - menos catastrophic forgetting
- Episodes 4000 steps (vs V6 2000) - contexto maior
- ent_coef 0.05 (vs V6 0.1) - exploração moderada
- Network [256, 256] (vs V13 [256, 256, 128]) - mais simples
- Adicionar balanceamento Long/Short (V14 novo)

META:
- Win Rate: 20-25%
- Return: Positivo (+2% a +5%)
- Trades: 600-1000
- Long/Short: 40-50% CADA (BALANCEADO!)
"""

import os
import sys
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from datetime import datetime
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# DirectML setup
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch_directml
dml_device = torch_directml.device()

from environment.trading_env import TradingEnv
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor

# Configuração
DATA_PATH = 'data/train_btcusdt_36m_20260109.csv'
MODELS_DIR = './models'
LOGS_DIR = './logs'
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 5_000  # V14: Save a cada 5k para monitoramento fino

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("\n" + "="*80)
print("🚀 SAC V14 - CONFIG V8 + BALANCEAMENTO")
print("="*80)

print("\n📋 DIFERENÇAS vs V6 (que colapsou):")
print("   • Buffer: 200k → 100k (MENOS catastrophic forgetting)")
print("   • Episodes: 2000 → 4000 steps (MAIS contexto)")
print("   • ent_coef: 0.1 → 0.05 (exploração MODERADA)")
print("   • Network: [256,256] (SIMPLES)")
print("   • Balanceamento: ATIVO (penaliza >80% uma direção)")

print("\n📋 DIFERENÇAS vs V13 (que falhou):")
print("   • SEM win rate nas observações")
print("   • SEM stop_risk indicator")
print("   • SEM multiplicadores de reward")
print("   • AMBIENTE V8 PURO + balanceamento")

print("\n🎯 METAS V14:")
print("   • Win Rate: 20-25%")
print("   • Return: +2% a +5%")
print("   • Trades: 600-1000")
print("   • Long/Short: 40-50% CADA")

print("\n" + "="*80)

response = input("\nIniciar treino V14? (s/n): ").strip().lower()
if response != 's':
    print("❌ Cancelado.")
    sys.exit(0)

# ===== CRIAR AMBIENTE V8 =====
print("\n🏗️ Criando ambiente V8...")

def make_env():
    """Factory para criar ambiente V8."""
    env = TradingEnv(
        data_path=DATA_PATH,
        initial_balance=10000,
        commission=0.0004,
        slippage=0.0005,
        leverage=1.5,
        position_size=0.05,  # 5% por trade (V8)
        window_size=50,
        max_episode_steps=4000,  # V8: 4000 steps (vs V6: 2000)
        random_start=True,
        persist_balance=True,
        use_sharpe_reward=True,
        enable_indicator_shaping=True
    )
    return env

env = DummyVecEnv([make_env])

print("✅ Ambiente V8 criado!")
print(f"   🔑 max_episode_steps: 4000 (V8 - contexto longo)")
print(f"   🔑 leverage: 1.5x")
print(f"   🔑 position_size: 5% (V8)")
print(f"   🔑 buffer: 100k (V8 - MENOR que V6)")

# ===== CRIAR MODELO SAC V14 =====
print("\n🤖 Criando modelo SAC V14...")

# Action noise (V8 style)
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.15 * np.ones(n_actions)  # V8: 15% noise (moderado)
)

model = SAC(
    "MlpPolicy",
    env,
    
    # === CONFIGS V8 (QUE FUNCIONARAM) ===
    learning_rate=3e-4,
    buffer_size=100000,          # V8: 100k (vs V6: 200k) - CRÍTICO!
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    use_sde=True,
    
    ent_coef=0.05,               # V8: 0.05 (vs V6: 0.1, vs V13: 0.15)
    target_entropy='auto',
    learning_starts=1000,        # V8: 1k (vs V13: 2k)
    
    action_noise=action_noise,
    
    # Network V8: [256, 256] - SIMPLES!
    policy_kwargs=dict(
        net_arch=[256, 256],     # V8: [256, 256] (vs V13: [256, 256, 128])
        activation_fn=torch.nn.ReLU
    ),
    
    tensorboard_log="./logs/sac_v14",
    verbose=1,
    device=dml_device
)

print("✅ Modelo SAC V14 criado!")
print(f"   • ent_coef: 0.05 (V8 - exploração MODERADA)")
print(f"   • buffer_size: 100k (V8 - MENOR que V6)")
print(f"   • net_arch: [256, 256] (V8 - SIMPLES)")
print(f"   • action_noise: 15% (V8)")
print(f"   • episodes: ~4000 steps cada")

# ===== CALLBACKS =====
print("\n🎮 Configurando callbacks...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Métricas
metrics_callback = TradingMetricsCallback(verbose=1)

# 2. Monitor de liquidações
liquidation_monitor = LiquidationMonitor(
    max_liquidations=5,
    check_freq=5000,  # V14: Check a cada 5k (antes 10k)
    verbose=1
)

# 3. Monitor de decaimento
decay_monitor = PerformanceDecayMonitor(
    min_winrate=0.05,
    patience=5,
    verbose=1
)

# 4. Checkpoints (SEM replay buffer - economiza espaço)
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODELS_DIR,
    name_prefix=f"sac_v14_{timestamp}",
    save_replay_buffer=False,  # Desabilitado - 200 checkpoints seria 100GB+
    save_vecnormalize=True,
    verbose=1
)

callbacks = CallbackList([
    metrics_callback,
    liquidation_monitor,
    decay_monitor,
    checkpoint_callback
])

print("✅ Callbacks configurados!")

# ===== TREINAR =====
print("\n" + "="*80)
print("🚀 INICIANDO TREINO V14 - 1M STEPS")
print("="*80)
print(f"Timestamp: {timestamp}")
print(f"TensorBoard: tensorboard --logdir=./logs/sac_v14/")
print(f"⏱️ Tempo estimado: ~8-10h (AMD GPU)")
print(f"\n📊 Checkpoints esperados (a cada 5k):")
print(f"   5k, 10k, 15k, 20k, ..., 1M (200 checkpoints!)")
print(f"\n⚠️ IMPORTANTE: Monitorar win rate e balanceamento!")
print(f"   python backtest_stochastic.py models/sac_v14_{timestamp}_XXXXX_steps.zip data/train_btcusdt_36m_20260109.csv")
print("="*80 + "\n")

start_time = datetime.now()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = datetime.now() - start_time
    
    print("\n" + "="*80)
    print("✅ TREINAMENTO V14 CONCLUÍDO!")
    print("="*80)
    print(f"⏱️ Tempo total: {training_time}")
    
    # Salvar modelo final
    final_path = f"{MODELS_DIR}/sac_v14_final_{timestamp}"
    print(f"\n💾 Salvando modelo final: {final_path}.zip")
    
    with torch.no_grad():
        model.save(final_path)
        print("✅ Modelo final salvo!")
    
    print("\n" + "="*80)
    print("PRÓXIMOS PASSOS:")
    print("="*80)
    print("1. Validar checkpoints:")
    print(f"   python validate_v14_checkpoints.py")
    print("\n2. Comparar com V6 500k:")
    print(f"   - V6: -0.96% return, 20.21% win, 43%/43% balance")
    print(f"   - V14: Esperado melhor em TODOS os aspectos!")
    print("="*80 + "\n")

except KeyboardInterrupt:
    print("\n⚠️ Treinamento interrompido")
    print(f"   Checkpoints salvos até: {model.num_timesteps} steps")
    
    # Salvar parcial
    partial_path = f"{MODELS_DIR}/sac_v14_partial_{timestamp}_{model.num_timesteps}steps"
    print(f"\n💾 Salvando modelo parcial: {partial_path}.zip")
    
    with torch.no_grad():
        model.save(partial_path)
    
    print("✅ Modelo parcial salvo!")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()
    print("\n✓ Ambiente fechado")
