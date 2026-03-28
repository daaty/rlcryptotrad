"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  🚀 TREINAR SAC V16 - MULTI-TIMEFRAME 🚀                     ║
║                                                                              ║
║  📋 ESTRATÉGIA V16: V15 + ANÁLISE MULTI-TEMPORAL                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  🎯 OBJETIVO: Melhorar decisões com contexto de 3 timeframes                 ║
║                                                                              ║
║  🔧 NOVIDADES V16:                                                           ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  ✨ Multi-timeframe: 15m (tático) + 1h (operacional) + 4h (estratégico)    ║
║  ✨ Observation space expandido: ~3x mais informação                        ║
║  ✨ Decisões baseadas em múltiplas escalas temporais                        ║
║                                                                              ║
║  📊 MANTÉM CONFIGS V15 (comprovadas):                                        ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  ✅ Bônus winners: +0.02/+0.04                                               ║
║  ✅ Bônus cortar loss: +0.05                                                 ║
║  ✅ Penalidades adiadas: -4%                                                 ║
║  ✅ Penalidade flat: -0.01                                                   ║
║  ✅ Indicator shaping: DESABILITADO                                          ║
║  ✅ Episodes: 2000 steps                                                     ║
║  ✅ SAC configs: buffer=100k, ent_coef=0.05, net_arch=[256,256]             ║
║                                                                              ║
║  🎯 HIPÓTESE: Multi-timeframe → Win rate 22-25%+ (vs 18-22% do V15)         ║
║                                                                              ║
║  📁 DADOS NECESSÁRIOS:                                                       ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Execute primeiro: python collect_multi_timeframe.py                        ║
║  Gera:                                                                       ║
║    - data/train_btcusdt_36m_15m_YYYYMMDD.csv                                ║
║    - data/train_btcusdt_36m_1h_YYYYMMDD.csv                                 ║
║    - data/train_btcusdt_36m_4h_YYYYMMDD.csv                                 ║
║                                                                              ║
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
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# ============= IMPORTS LOCAIS =============
from src.environment.trading_env_multi_tf import TradingEnvMultiTF
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor

# ============= V16: HIPERPARÂMETROS (IGUAIS V15) =============
TOTAL_TIMESTEPS = 1_000_000  # 1M steps total
SAVE_FREQ = 5_000  # Checkpoint a cada 5k
CHECK_FREQ = 5_000  # Verifica estatísticas a cada 5k

# V16: NOVO - Caminhos para múltiplos timeframes
# NOTA: Atualize as datas conforme os arquivos gerados por collect_multi_timeframe.py
def find_latest_data_files():
    """Encontra os arquivos de dados mais recentes para cada timeframe."""
    data_dir = Path('data')
    
    files_15m = sorted(data_dir.glob('train_btcusdt_*_15m_*.csv'), reverse=True)
    files_1h = sorted(data_dir.glob('train_btcusdt_*_1h_*.csv'), reverse=True)
    files_4h = sorted(data_dir.glob('train_btcusdt_*_4h_*.csv'), reverse=True)
    
    if not files_15m or not files_1h or not files_4h:
        print("❌ ERRO: Arquivos de dados não encontrados!")
        print("\n📋 Execute primeiro: python collect_multi_timeframe.py")
        print("   Isso irá baixar dados de 15m, 1h e 4h automaticamente.\n")
        sys.exit(1)
    
    data_paths = {
        '15m': str(files_15m[0]),
        '1h': str(files_1h[0]),
        '4h': str(files_4h[0])
    }
    
    return data_paths

# V16: Ambiente multi-timeframe (MESMAS configs V15)
ENV_CONFIG = {
    'window_size': 50,
    'max_episode_steps': 2000,
    'leverage': 1.0,  # V16: REDUZIDO 1.5 → 1.0 para training estável
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,  # 5%
    'use_sharpe_reward': False,  # V16: DESABILITADO - causava colapso após learning_starts
    'enable_indicator_shaping': False,
    'random_start': True,
    'persist_balance': False,  # V16: Episódios independentes
    'liquidation_threshold': 0.30,  # V16: AUMENTADO 0.10 → 0.30
}

# V15: SAC configs (MANTIDOS)
SAC_CONFIG = {
    'learning_rate': 3e-4,
    'buffer_size': 300_000,  # V16: AUMENTADO 100k → 300k (dilui experiências ruins)
    'learning_starts': 10_000,  # V16: AUMENTADO 1k → 10k (mais exploração inicial)
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 0.2,  # V16: AUMENTADO 0.15 → 0.2 (mais exploração contínua)
    'target_update_interval': 1,
    'use_sde': False,
}

# V16: Network architecture EXPANDIDA (obs space é 1450, 3x maior!)
NETWORK_CONFIG = {
    'net_arch': [512, 512, 256],  # Maior capacidade para multi-timeframe
    'activation_fn': torch.nn.ReLU,
}

def make_env(data_paths):
    """Factory para criar ambiente vectorizado multi-timeframe."""
    def _init():
        env = TradingEnvMultiTF(
            data_paths=data_paths,
            **ENV_CONFIG
        )
        return env
    return _init

def main():
    # Timestamp único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO SAC V16 - MULTI-TIMEFRAME")
    print("="*80)
    print(f"📅 Timestamp: {timestamp}")
    print(f"🎯 Target: 1M steps com checkpoints a cada 5k")
    print(f"💾 Modelo: models/sac_v16_multi_tf_{timestamp}_XXXXX_steps.zip")
    print("="*80 + "\n")
    
    # Encontrar arquivos de dados
    print("🔍 Procurando arquivos de dados multi-timeframe...")
    data_paths = find_latest_data_files()
    
    print("\n📂 DADOS ENCONTRADOS:")
    for tf, path in data_paths.items():
        print(f"   {tf:>3}: {Path(path).name}")
    print()
    
    # === V16: RESUMO DE MUDANÇAS ===
    print("🆕 NOVIDADES V16:")
    print("  ✨ Multi-timeframe: 15m + 1h + 4h")
    print("  ✨ Obs space: ~3x maior (mais contexto)")
    print("  ✨ Decisões com visão macro+micro")
    print("\n📊 MANTÉM V15:")
    print("  ✅ Reward structure balanceada")
    print("  ✅ Bônus/penalidades testadas")
    print("  ✅ SAC configs otimizadas")
    print("  ✅ Network architecture [256,256]")
    print("="*80 + "\n")
    
    # Verificar GPU (DirectML > CUDA > CPU)
    device = "cpu"
    device_name = "CPU"
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = f"DirectML ({device})"
    except Exception:
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
    
    print(f"🖥️  Device: {device_name}")
    print(f"🎯 Política: Continuous SAC (3 ações contínuas)\n")
    
    # === CRIAR AMBIENTE ===
    print("📁 Criando ambiente multi-timeframe...")
    env = DummyVecEnv([make_env(data_paths)])
    print(f"✅ Ambiente criado: {env.num_envs} env(s)\n")
    
    # Obter dimensões
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"📐 Obs shape: {obs_shape} (MULTI-TIMEFRAME: 3x maior que V15!)")
    print(f"📐 Action shape: {act_shape}\n")
    
    # === ACTION NOISE ===
    # V16: REMOVIDO - SAC já tem exploração via entropy, action noise pode causar conflito
    print("🎲 Action noise: NONE (SAC usa entropy coef=0.2)\n")
    
    # === CRIAR MODELO SAC ===
    print("🏗️  Criando modelo SAC V16...")
    print(f"   - Learning rate: {SAC_CONFIG['learning_rate']}")
    print(f"   - Buffer size: {SAC_CONFIG['buffer_size']:,}")
    print(f"   - Batch size: {SAC_CONFIG['batch_size']}")
    print(f"   - Ent coef: {SAC_CONFIG['ent_coef']}")
    print(f"   - Network: {NETWORK_CONFIG['net_arch']}")
    print(f"   - Activation: ReLU\n")
    
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
        tensorboard_log=f"./tensorboard/sac_v16_multi_tf_{timestamp}/",
    )
    
    print("✅ Modelo criado!\n")
    
    # === CALLBACKS ===
    print("📊 Configurando callbacks...")
    
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=1000,  # V16: Aumentado para permitir mais exploração
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
        name_prefix=f"sac_v16_multi_tf_{timestamp}",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    
    print(f"   ✅ TradingMetricsCallback")
    print(f"   ✅ LiquidationMonitor (check={CHECK_FREQ:,})")
    print(f"   ✅ PerformanceDecayMonitor")
    print(f"   ✅ CheckpointCallback (save={SAVE_FREQ:,})")
    print(f"   ✅ Replay buffer: NÃO salvo\n")
    
    # === TREINAR ===
    print("="*80)
    print("🎓 INICIANDO TREINO MULTI-TIMEFRAME...")
    print("="*80)
    print(f"⏱️  Duração: ~10-20h (AMD DirectML)")
    print(f"📈 TensorBoard: tensorboard --logdir=./tensorboard/\n")
    print("💡 HIPÓTESE V16:")
    print("   → Multi-timeframe captura padrões em múltiplas escalas")
    print("   → Win rate target: 22-25%+ (vs 18-22% V15)")
    print("   → Melhor timing de entrada/saída")
    print("   → Menos falsos sinais (contexto macro filtra ruído)")
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
        final_model_path = f"./models/sac_v16_multi_tf_{timestamp}_final.zip"
        model.save(final_model_path)
        print(f"💾 Modelo final salvo: {final_model_path}")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("  1. Executar backtest nos checkpoints")
        print("  2. Comparar V16 vs V15 (single vs multi-timeframe)")
        print("  3. Analisar se contexto macro melhorou decisões")
        print("  4. Validar em out-of-sample data")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo usuário!")
        print("💾 Último checkpoint salvo automaticamente.\n")
    except Exception as e:
        print(f"\n❌ ERRO durante treino: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
