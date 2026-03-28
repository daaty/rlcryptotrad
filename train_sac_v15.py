"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🚀 TREINAR SAC V15 - BTC/USDT 🚀                         ║
║                                                                              ║
║  📋 ESTRATÉGIA V15: AMBIENTE BALANCEADO E LIMPO                               ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  🎯 OBJETIVO: Win rate 18-22% com reward structure balanceada                ║
║                                                                              ║
║  🔧 MUDANÇAS V15 (baseadas em ANALISE_AMBIENTE_V14.md):                      ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  ✅ Bônus winners AUMENTADOS 4x: +0.02/+0.04 (era 0.005/0.01)                ║
║  ✅ Bônus cortar loss IGUAL a lucro: +0.05 (balanceado!)                     ║
║  ✅ Penalidades ADIADAS: começam em -4% (era -3%)                            ║
║  ✅ Penalidade flat AUMENTADA 100x: -0.01 (era -0.0001)                      ║
║  ✅ Indicator shaping DESABILITADO (RSI contratrend removido)                ║
║  ✅ Episodes REDUZIDOS: 2000 steps (era 4000) - Sharpe mais estável          ║
║                                                                              ║
║  📊 CONFIGS V8 (mantidas - funcionaram melhor que V6):                       ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  - buffer_size=100k (evita catastrophic forgetting do V6 700k)               ║
║  - ent_coef=0.05 (exploração moderada)                                      ║
║  - net_arch=[256,256] (simples e efetivo)                                   ║
║  - learning_rate=3e-4                                                       ║
║  - action_noise=15% (exploração via ruído)                                  ║
║                                                                              ║
║  🎯 TARGET: Win rate 20%+, Return +2-5%, Balance 40-50% Long/Short           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

# ============= IMPORTS ESTÁVEIS =============
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# ============= IMPORTS LOCAIS =============
from src.environment.trading_env import TradingEnv
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor

# ============= V15: HIPERPARÂMETROS FINAIS =============
TOTAL_TIMESTEPS = 1_000_000  # 1M steps total
SAVE_FREQ = 5_000  # Checkpoint a cada 5k (200 total) - monitoramento fino
CHECK_FREQ = 5_000  # Verifica estatísticas a cada 5k

# V15: Ambiente balanceado (baseado em análise completa)
ENV_CONFIG = {
    'data_path': 'data/train_btcusdt_36m_20260109.csv',
    'window_size': 50,
    'max_episode_steps': 2000,  # V15: REDUZIDO de 4000 para 2000 (V6 length)
    'leverage': 1.5,
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,  # 5%
    'use_sharpe_reward': True,  # Sharpe Ratio como base
    'enable_indicator_shaping': False,  # V15: DESABILITADO! (RSI contratrend removido)
}

# V8: SAC configs (melhores que V6)
SAC_CONFIG = {
    'learning_rate': 3e-4,
    'buffer_size': 100_000,  # 100k (menor que V6 200k - evita forgetting)
    'learning_starts': 1000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 0.05,  # V8: exploração moderada (V6 era 0.1)
    'target_update_interval': 1,
    'use_sde': False,  # SDE desabilitado
}

# V13: Network architecture (simples e efetivo)
NETWORK_CONFIG = {
    'net_arch': [256, 256],  # V8: mais simples que V13 [256,256,128]
    'activation_fn': torch.nn.ReLU,  # V14: ReLU (DirectML compatível!)
}

def make_env():
    """Factory para criar ambiente vectorizado"""
    def _init():
        env = TradingEnv(**ENV_CONFIG)
        return env
    return _init

def main():
    # Timestamp único para este treino
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO SAC V15 - BTC/USDT")
    print("="*80)
    print(f"📅 Timestamp: {timestamp}")
    print(f"🎯 Target: 1M steps com checkpoints a cada 5k")
    print(f"💾 Modelo: models/sac_v15_{timestamp}_XXXXX_steps.zip")
    print(f"📊 Episodes: {ENV_CONFIG['max_episode_steps']} steps (V15: reduzido de 4000)")
    print(f"🎨 Indicator shaping: {'HABILITADO' if ENV_CONFIG['enable_indicator_shaping'] else 'DESABILITADO (V15 fix!)'}")
    print("="*80 + "\n")
    
    # === V15: ANÁLISE DE CORREÇÕES ===
    print("🔧 CORREÇÕES APLICADAS V15:")
    print("  ✅ Bônus winners: +0.02/+0.04 (4x maior)")
    print("  ✅ Bônus loss = lucro: +0.05 (balanceado)")
    print("  ✅ Penalidades adiadas: -4% (não -3%)")
    print("  ✅ Penalidade flat: -0.01 (100x maior)")
    print("  ✅ Indicator shaping: DESABILITADO")
    print("  ✅ Episodes: 2000 steps (Sharpe estável)")
    print("="*80 + "\n")
    
    # Verificar se GPU está disponível (DirectML ou CUDA)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"
    
    print(f"🖥️  Device: {device_name}")
    print(f"🎯 Política: Continuous SAC (3 ações contínuas: Long/Short/Flat)\n")
    
    # === CRIAR AMBIENTE ===
    print("📁 Criando ambiente...")
    env = DummyVecEnv([make_env()])
    print(f"✅ Ambiente criado: {env.num_envs} env(s)\n")
    
    # Obter dimensões
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"📐 Obs shape: {obs_shape}")
    print(f"📐 Action shape: {act_shape}\n")
    
    # === ACTION NOISE (V6: 15% - balanceado) ===
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.15 * np.ones(n_actions)  # 15% de exploração via ruído
    )
    print("🎲 Action noise: NormalActionNoise (sigma=0.15)\n")
    
    # === CRIAR MODELO SAC ===
    print("🏗️  Criando modelo SAC V15...")
    print(f"   - Learning rate: {SAC_CONFIG['learning_rate']}")
    print(f"   - Buffer size: {SAC_CONFIG['buffer_size']:,}")
    print(f"   - Batch size: {SAC_CONFIG['batch_size']}")
    print(f"   - Ent coef: {SAC_CONFIG['ent_coef']} (V8: moderado)")
    print(f"   - Network: {NETWORK_CONFIG['net_arch']}")
    print(f"   - Activation: ReLU (DirectML OK!)\n")
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=SAC_CONFIG['learning_rate'],
        buffer_size=SAC_CONFIG['buffer_size'],
        learning_starts=SAC_CONFIG['learning_starts'],
        batch_size=SAC_CONFIG['batch_size'],
        tau=SAC_CONFIG['tau'],
        gamma=SAC_CONFIG['gamma'],
        train_freq=SAC_CONFIG['train_freq'],
        gradient_steps=SAC_CONFIG['gradient_steps'],
        action_noise=action_noise,
        ent_coef=SAC_CONFIG['ent_coef'],
        target_update_interval=SAC_CONFIG['target_update_interval'],
        use_sde=SAC_CONFIG['use_sde'],
        policy_kwargs={
            'net_arch': NETWORK_CONFIG['net_arch'],
            'activation_fn': NETWORK_CONFIG['activation_fn'],
        },
        verbose=1,
        device=device,
        tensorboard_log=f"./tensorboard/sac_v15_{timestamp}/",
    )
    
    print("✅ Modelo criado!\n")
    
    # === CALLBACKS ===
    print("📊 Configurando callbacks...")
    
    # 1. Métricas básicas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # 2. Monitor de liquidações com estatísticas agregadas (V14 enhanced)
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=5,
        check_freq=CHECK_FREQ,
        verbose=1
    )
    
    # 3. Monitor de decaimento
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,
        patience=5,
        verbose=1
    )
    
    # 4. Checkpoints (sem replay buffer - economiza espaço)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"./models/",
        name_prefix=f"sac_v15_{timestamp}",
        save_replay_buffer=False,  # V14: desabilitado (economiza 100GB+ de espaço)
        save_vecnormalize=True,
        verbose=1
    )
    
    print(f"   ✅ TradingMetricsCallback: métricas básicas")
    print(f"   ✅ LiquidationMonitor: verifica a cada {CHECK_FREQ:,} steps")
    print(f"   ✅ PerformanceDecayMonitor: detecta colapso")
    print(f"   ✅ CheckpointCallback: salva a cada {SAVE_FREQ:,} steps")
    print(f"   ✅ Replay buffer: NÃO será salvo (economiza espaço)\n")
    
    # === TREINAR ===
    print("="*80)
    print("🎓 INICIANDO TREINO...")
    print("="*80)
    print(f"⏱️  Duração esperada: ~10-20h (AMD DirectML)")
    print(f"📈 Acompanhe no TensorBoard: tensorboard --logdir=./tensorboard/\n")
    print("💡 V15 Hypothesis: Win rate deve subir para 18-22% com reward balanceada")
    print("💡 Monitorar: (1) Win rate trend, (2) Long/Short balance, (3) Trade count")
    print("="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[metrics_callback, liquidation_monitor, decay_monitor, checkpoint_callback],
            log_interval=10,
            progress_bar=True,
        )
        
        print("\n" + "="*80)
        print("✅ TREINO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        # Salvar modelo final
        final_model_path = f"./models/sac_v15_{timestamp}_final.zip"
        model.save(final_model_path)
        print(f"💾 Modelo final salvo: {final_model_path}")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("  1. Executar backtest em checkpoints chave (200k, 400k, 600k)")
        print("  2. Comparar com V6 500k baseline (20.21% win rate)")
        print("  3. Validar hipótese de 18-22% win rate")
        print("  4. Se sucesso, testar em out-of-sample data")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo usuário!")
        print("💾 Último checkpoint foi salvo automaticamente.\n")
    except Exception as e:
        print(f"\n❌ ERRO durante treino: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
