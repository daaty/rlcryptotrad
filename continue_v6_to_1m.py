"""
CONTINUAR TREINO V6: 500k → 1M steps
=====================================

MUDANÇAS:
- target_entropy: -1.0 → -0.5 (MAIS LIBERDADE!)
- Continua de: models/sac_futuros_v6_final_20260112_012926.zip
- Total: +500k steps (500k→1M)
- Mantém TUDO igual: reward, stop-loss, leverage, etc.

Exploration: SAC já decai automaticamente via entropy tuning!
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import pandas as pd

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from environment.trading_env import TradingEnv
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor


print("=" * 80)
print("🚀 CONTINUAR TREINO V6: 500k → 1M STEPS")
print("=" * 80)
print()

# ===== CONFIGURAÇÃO =====
DATA_PATH = 'data/train_btcusdt_36m_20260109.csv'
MODEL_PATH = 'models/sac_futuros_v6_final_20260112_012926.zip'  # Carrega modelo final
ADDITIONAL_STEPS = 500_000  # +500k steps (total = 1M)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"./logs/sac_futuros_v6/continue_to_1m_{TIMESTAMP}"

print(f"📂 Carregando modelo: {MODEL_PATH}")
print(f"📊 Dataset: {DATA_PATH}")
print(f"🎯 Steps adicionais: {ADDITIONAL_STEPS:,}")
print(f"📝 Log dir: {LOG_DIR}")
print()

# ===== VERIFICAR SE MODELO EXISTE =====
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERRO: Modelo não encontrado!")
    print(f"   Esperado: {MODEL_PATH}")
    print()
    print("💡 Modelos disponíveis:")
    if os.path.exists('models'):
        models = [f for f in os.listdir('models') if f.startswith('sac_futuros_v6')]
        for m in sorted(models):
            print(f"   - {m}")
    sys.exit(1)

# ===== CRIAR AMBIENTE (MESMAS CONFIGURAÇÕES V6) =====
print("🏗️ Criando ambiente V6 (configurações originais)...")

df = pd.read_csv(DATA_PATH)
print(f"   📊 Dataset carregado: {len(df):,} candles")

env = TradingEnv(
    df=df,
    initial_balance=10000,
    commission=0.0004,
    slippage=0.0005,
    leverage=1.5,              # V6: Reduzido
    position_size=0.1,         # Base 10% (limitado a 5% no step)
    window_size=50,
    max_episode_steps=2000,    # V6: Episódios mais curtos
    random_start=True,
    persist_balance=True,
    use_sharpe_reward=True,    # V6: Sharpe ativo
    use_hybrid_reward=False,
    enable_indicator_shaping=True,  # V6: Indicator shaping ativo
    maintenance_margin_rate=0.005,
    liquidation_threshold=0.10
)

print("✅ Ambiente criado!")
print(f"   • Leverage: {env.leverage}x")
print(f"   • Stop-loss: -5% (forçado)")
print(f"   • Position size: max 5%")
print(f"   • Sharpe reward: {env.use_sharpe_reward}")
print(f"   • Indicator shaping: {env.enable_indicator_shaping}")
print()

# ===== CARREGAR MODELO V6 =====
print("🔄 Carregando modelo V6 existente...")

try:
    # CRÍTICO: Carrega para CPU primeiro (evita erro de device)
    print("   📥 Carregando para CPU primeiro...")
    model = SAC.load(
        MODEL_PATH,
        env=env,
        device='cpu'
    )
    
    # Depois move para DirectML (AMD GPU)
    print("   🔄 Movendo para DirectML (AMD GPU)...")
    model.set_parameters(
        model.get_parameters(),
        exact_match=True,
        device='privateuseone'
    )
    
    print("✅ Modelo V6 carregado com sucesso!")
    print(f"   📦 De: {MODEL_PATH}")
    print(f"   🖥️ Device: {model.device}")
    
    # Configura novo logger para o treino continuado
    print(f"   📝 Configurando logger: {LOG_DIR}")
    from stable_baselines3.common.logger import configure
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    print()
except Exception as e:
    print(f"❌ ERRO ao carregar modelo: {e}")
    print()
    print("💡 Tentando carregar apenas em CPU...")
    try:
        model = SAC.load(MODEL_PATH, env=env, device='cpu')
        print("✅ Modelo carregado em CPU!")
        print("⚠️ Treino será LENTO (sem GPU)")
        print()
    except Exception as e2:
        print(f"❌ ERRO mesmo em CPU: {e2}")
        sys.exit(1)

# ===== ATUALIZAR TARGET_ENTROPY =====
print("🎯 Atualizando target_entropy...")
print(f"   Antes: {model.target_entropy}")

# Calcula novo target_entropy baseado em action_dim
action_dim = env.action_space.shape[0]
new_target_entropy = -0.5 * action_dim  # -0.5 para dim=1 → -0.5

# Atualiza target_entropy
model.target_entropy = new_target_entropy

# Converte para tensor no device correto
if isinstance(model.target_entropy, (int, float)):
    model.target_entropy = torch.tensor(
        [model.target_entropy],
        dtype=torch.float32
    ).to(model.device)

print(f"   Depois: {model.target_entropy}")
print("   ✅ Mais liberdade para explorar!")
print()

# ===== CONFIGURAR ACTION NOISE (V6 original: 20%) =====
print("🎲 Configurando action noise (V6: 20%)...")

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)  # 20% (V6 original)
)

# Atualiza noise no model
model.action_noise = action_noise

print("✅ Action noise configurado!")
print(f"   • Sigma: 0.2 (20%)")
print()

# ===== CALLBACKS =====
print("🎮 Configurando callbacks...")

# 1. Trading Metrics (TensorBoard)
# NOTA: TradingMetricsCallback usa o logger do modelo automaticamente
metrics_callback = TradingMetricsCallback(
    verbose=0
)

# 2. Liquidation Monitor (max 5, não deveria ativar!)
liquidation_monitor = LiquidationMonitor(
    max_liquidations=5,
    verbose=1
)

# 3. Performance Decay Monitor
performance_monitor = PerformanceDecayMonitor(
    min_winrate=0.05,
    patience=5,
    verbose=1
)

# 4. Checkpoint Callback (salva a cada 100k)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path='./models/',
    name_prefix=f'sac_futuros_v6_continue',
    save_replay_buffer=False,
    save_vecnormalize=False,
    verbose=1
)

callback_list = CallbackList([
    metrics_callback,
    liquidation_monitor,
    performance_monitor,
    checkpoint_callback
])

print("✅ Callbacks configurados!")
print("   1. TradingMetricsCallback")
print("   2. LiquidationMonitor (max: 5)")
print("   3. PerformanceDecayMonitor (min winrate: 5%)")
print("   4. CheckpointCallback (freq: 100k)")
print()

# ===== INFORMAÇÕES PRÉ-TREINO =====
print("=" * 80)
print("🚀 INICIANDO CONTINUAÇÃO DO TREINO V6")
print("=" * 80)
print()
print(f"Timestamp: {TIMESTAMP}")
print()
print(f"📊 Configuração:")
print(f"   • Steps anteriores: ~500k")
print(f"   • Steps adicionais: {ADDITIONAL_STEPS:,}")
print(f"   • Total esperado: ~1M steps")
print(f"   • Checkpoints: 600k, 700k, 800k, 900k, 1M")
print()
print(f"🎯 Mudanças:")
print(f"   • target_entropy: -1.0 → -0.5 (MAIS LIBERDADE!)")
print(f"   • Exploration: SAC decai automaticamente (entropy tuning)")
print()
print(f"🔒 Mantido do V6:")
print(f"   • Stop-loss forçado: -5%")
print(f"   • Leverage: 1.5x")
print(f"   • Position size: max 5%")
print(f"   • Sharpe reward: True")
print(f"   • Indicator shaping: True")
print(f"   • ent_coef: {model.ent_coef} (fixo)")
print()
print(f"📝 Logs no TensorBoard:")
print(f"   tensorboard --logdir={LOG_DIR}")
print()
print(f"⏱️ Tempo estimado: ~4-5h (AMD GPU)")
print()
print("=" * 80)
print()

# ===== TREINO =====
try:
    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        callback=callback_list,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False  # CRÍTICO: Não reseta contador!
    )
    
    print()
    print("=" * 80)
    print("✅ CONTINUAÇÃO DO TREINO V6 CONCLUÍDA!")
    print("=" * 80)
    print()
    
    # Salva modelo final
    final_model_path = f"models/sac_futuros_v6_1m_{TIMESTAMP}.zip"
    model.save(final_model_path)
    print(f"💾 Modelo final salvo: {final_model_path}")
    print()
    
    print("📊 Próximos passos:")
    print("   1. Backtest dos checkpoints:")
    print("      - 500k (já existe)")
    print("      - 600k, 700k, 800k, 900k")
    print("      - 1M (final)")
    print()
    print("   2. Comparar performance:")
    print("      - Qual checkpoint tem melhor winrate?")
    print("      - Estabilidade melhorou?")
    print()
    print("   3. Se winrate >30%:")
    print("      - Testar em testnet Binance")
    print("      - Validar com dados out-of-sample")
    print()
    
except KeyboardInterrupt:
    print()
    print("⚠️ Treino interrompido pelo usuário!")
    print()
    
    # Salva checkpoint intermediário
    interrupt_model_path = f"models/sac_futuros_v6_interrupted_{TIMESTAMP}.zip"
    model.save(interrupt_model_path)
    print(f"💾 Checkpoint salvo: {interrupt_model_path}")
    print()
    
except Exception as e:
    print()
    print(f"❌ ERRO durante treino: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    print("🔚 Fechando ambiente...")
    env.close()
    print("✅ Ambiente fechado.")
    print()
