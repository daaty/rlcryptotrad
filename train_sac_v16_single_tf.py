"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TREINO SAC V16 - SINGLE TIMEFRAME (15m)                ║
║                         TESTE DE HIPÓTESE: Multi-TF é o problema?           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Mesmo config do V16, MAS usando APENAS 15m para comparação:                ║
║    - Se funcionar: problema É o multi-timeframe                             ║
║    - Se colapsar: problema é outra coisa                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import glob

# ============= IMPORTS ESTÁVEIS =============
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ============= IMPORTS LOCAIS =============
from src.environment.trading_env import TradingEnv  # V15 single-timeframe
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor

# ============= V16 CONFIGS (IDÊNTICOS) =============
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 5_000
CHECK_FREQ = 5_000

ENV_CONFIG = {
    'window_size': 50,
    'max_episode_steps': 2000,
    'leverage': 1.0,  # V16: conservador
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,
    'use_sharpe_reward': True,
    'enable_indicator_shaping': False,
    'random_start': True,
    'persist_balance': False,  # V16: independente
    'liquidation_threshold': 0.30,  # V16: margem 3x maior
}

SAC_CONFIG = {
    'learning_rate': 3e-4,
    'buffer_size': 300_000,  # V16: 3x maior
    'learning_starts': 10_000,  # V16: 10x mais exploração
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 0.2,  # V16: exploração alta
    'target_update_interval': 1,
    'use_sde': False,
}

# V16: Network igual ao multi-TF
NETWORK_CONFIG = {
    'net_arch': [512, 512, 256],
    'activation_fn': torch.nn.ReLU,
}

def make_env(data_path):
    """Factory para TradingEnv single-timeframe."""
    def _init():
        env = TradingEnv(
            data_path=data_path,
            **ENV_CONFIG
        )
        return env
    return _init

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🧪 TESTE: SAC V16 - SINGLE TIMEFRAME (15m)")
    print("="*80)
    print(f"📅 Timestamp: {timestamp}")
    print(f"🎯 Target: 100k steps (teste rápido)")
    print(f"💾 Modelo: models/sac_v16_single_15m_{timestamp}_XXXXX_steps.zip")
    print("="*80 + "\n")
    
    # Encontrar arquivo 15m
    print("🔍 Procurando dados de 15m...")
    data_dir = Path('data')
    files_15m = sorted(data_dir.glob('train_btcusdt_*_15m_*.csv'), reverse=True)
    
    if not files_15m:
        print("❌ ERRO: Arquivo 15m não encontrado!")
        sys.exit(1)
    
    data_path = str(files_15m[0])
    print(f"✅ Dados: {Path(data_path).name}\n")
    
    # Detectar device
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = f"DirectML ({device})"
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = device
    
    print(f"🖥️  Device: {device_name}")
    print(f"🎯 Política: Continuous SAC\n")
    
    # Criar ambiente
    print("📁 Criando ambiente SINGLE timeframe (15m)...")
    env = DummyVecEnv([make_env(data_path)])
    print(f"✅ Ambiente criado: {env.num_envs} env(s)\n")
    
    obs_shape = env.observation_space.shape
    print(f"📐 Obs shape: {obs_shape} (SINGLE: ~3x MENOR que multi-TF!)\n")
    
    # Criar modelo
    print("🏗️  Criando modelo SAC V16 (single-TF)...")
    print(f"   - Learning rate: {SAC_CONFIG['learning_rate']}")
    print(f"   - Buffer size: {SAC_CONFIG['buffer_size']:,}")
    print(f"   - Ent coef: {SAC_CONFIG['ent_coef']}")
    print(f"   - Network: {NETWORK_CONFIG['net_arch']}\n")
    
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
        ent_coef=SAC_CONFIG['ent_coef'],
        target_update_interval=SAC_CONFIG['target_update_interval'],
        use_sde=SAC_CONFIG['use_sde'],
        policy_kwargs={
            'net_arch': NETWORK_CONFIG['net_arch'],
            'activation_fn': NETWORK_CONFIG['activation_fn'],
        },
        verbose=1,
        device=device,
        tensorboard_log=f"./tensorboard/sac_v16_single_15m_{timestamp}/",
    )
    
    print(f"Using {device} device")
    print("✅ Modelo criado!\n")
    
    # Callbacks
    print("📊 Configurando callbacks...")
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=1000,
        check_freq=CHECK_FREQ,
        verbose=1
    )
    
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,
        patience=5,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"./models/",
        name_prefix=f"sac_v16_single_15m_{timestamp}",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    
    print(f"   ✅ TradingMetricsCallback")
    print(f"   ✅ LiquidationMonitor")
    print(f"   ✅ PerformanceDecayMonitor")
    print(f"   ✅ CheckpointCallback\n")
    
    # Treinar
    print("="*80)
    print("🎓 INICIANDO TESTE SINGLE-TIMEFRAME...")
    print("="*80)
    print(f"⏱️  Duração estimada: ~1-2h")
    print(f"📈 Comparar com multi-TF:")
    print(f"   → Multi-TF: colapsa após 10k (2-4 trades)")
    print(f"   → Single-TF: deveria manter 50-200 trades")
    print("="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=100_000,  # Teste rápido: 100k
            callback=[metrics_callback, liquidation_monitor, decay_monitor, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo usuário!")
        print("💾 Último checkpoint salvo automaticamente.\n")
    
    # Salvar modelo final
    final_model_path = f"./models/sac_v16_single_15m_{timestamp}_final.zip"
    model.save(final_model_path)
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO!")
    print("="*80)
    print(f"💾 Modelo final: {final_model_path}")
    print(f"\n📊 ANÁLISE:")
    print(f"   Se manteve 50-200 trades: Multi-TF É o problema ❌")
    print(f"   Se colapsou para 2-4 trades: Problema é outro ⚠️")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
