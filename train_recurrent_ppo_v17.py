"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             🧠 TREINAR RECURRENT PPO V17 - LSTM MULTI-TIMEFRAME              ║
║                                                                              ║
║  📋 ESTRATÉGIA V17-LSTM: Multi-Timeframe com Memória Temporal               ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  🎯 OBJETIVO: LSTM aprende dependências temporais que MLP não captura        ║
║                                                                              ║
║  🔧 ARQUITETURA:                                                             ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  ✨ RecurrentPPO (SB3-Contrib)                                              ║
║  ✨ MlpLstmPolicy com 2 camadas LSTM (256 neurons cada)                     ║
║  ✨ Observations sequenciais: (50, 29)                                      ║
║  ✨ LSTM mantém memória de curto prazo entre steps                          ║
║                                                                              ║
║  🆚 DIFERENÇAS VS V16.3 (SAC + MLP):                                        ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  V16.3 SAC:                                                                  ║
║    - MLP feedforward [512, 512, 256]                                        ║
║    - Flatten observations → 1450 features                                   ║
║    - SEM memória temporal                                                   ║
║    - Win rate: 42% treino, 30% teste                                        ║
║    - Overtrading: 784 trades/2000 steps                                     ║
║                                                                              ║
║  V17 RecurrentPPO:                                                           ║
║    - LSTM [256, 256] + MLP [256, 256]                                       ║
║    - Sequences (50, 29) preservam estrutura temporal                        ║
║    - Memória de curto prazo (hidden states)                                 ║
║    - Target: 35%+ win rate teste, <400 trades                               ║
║                                                                              ║
║  📁 REQUER:                                                                  ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  pip install stable-baselines3-contrib                                      ║
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
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("❌ ERRO: stable-baselines3-contrib não instalado!")
    print("   Instale com: pip install sb3-contrib")
    sys.exit(1)

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ============= IMPORTS LOCAIS =============
from src.environment.trading_env_multi_tf_lstm import TradingEnvMultiTFLSTM
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor, ValueLossDivergenceMonitor

# ============= V17-LSTM: HIPERPARÂMETROS =============
TOTAL_TIMESTEPS = 1_500_000  # 1.5M steps (LSTM mais lento que SAC)
SAVE_FREQ = 10_000  # Checkpoint a cada 10k (menos frequente)
CHECK_FREQ = 10_000

# Encontrar dados
def find_latest_data_files():
    """Encontra os arquivos de dados mais recentes."""
    data_dir = Path('data')
    
    files_15m = sorted(data_dir.glob('train_btcusdt_*_15m_*.csv'), reverse=True)
    files_1h = sorted(data_dir.glob('train_btcusdt_*_1h_*.csv'), reverse=True)
    files_4h = sorted(data_dir.glob('train_btcusdt_*_4h_*.csv'), reverse=True)
    
    if not files_15m or not files_1h or not files_4h:
        print("❌ ERRO: Dados não encontrados!")
        print("   Execute: python collect_multi_timeframe.py")
        sys.exit(1)
    
    return {
        '15m': str(files_15m[0]),
        '1h': str(files_1h[0]),
        '4h': str(files_4h[0])
    }

# V17-LSTM: Configuração do ambiente (IGUAL V16.3)
ENV_CONFIG = {
    'window_size': 50,
    'max_episode_steps': 2000,
    'leverage': 1.0,
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,
    'use_sharpe_reward': False,
    'enable_indicator_shaping': False,
    'random_start': True,
    'persist_balance': False,
    'liquidation_threshold': 0.30,
}

# V17.7 REAL MULTI-TF: Obs (50,31) com 1h/4h reais + anti-critic-overfitting mantido
PPO_CONFIG = {
    'learning_rate': 2e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 4,              # V17.6: Reduzido 10→4 - MENOS overfitting do critic!
    'gamma': 0.95,              # V17.6: Reduzido 0.99→0.95 - Returns mais difíceis = vantagens MAIORES!
    'gae_lambda': 0.9,          # V17.6: Reduzido 0.95→0.9 - Menos bias nas vantagens
    'clip_range': 0.2,
    'ent_coef': 0.03,
    'vf_coef': 0.1,             # V17.6: Reduzido drásticamente 0.5→0.1 - Critic NÃO domina!
    'max_grad_norm': 0.5,
    # target_kl removido: LSTM cria KL artificial nas bordas de rollout → early stopping no step 0
    # KL spikes (0.96) se auto-curam como visto no V17.6 — o clip_range=0.2 já protege
}

# V17-LSTM: LSTM architecture config
LSTM_CONFIG = {
    'lstm_hidden_size': 256,  # Tamanho dos hidden states LSTM
    'n_lstm_layers': 2,       # 2 camadas LSTM empilhadas
    'net_arch': [256, 256],   # MLP após LSTM (dict=256, pi=256, vf=256)
    'activation_fn': torch.nn.ReLU,
    'ortho_init': False,      # IMPORTANTE: False para LSTM
}

def make_env(data_paths):
    """Factory para criar ambiente LSTM."""
    def _init():
        env = TradingEnvMultiTFLSTM(
            data_paths=data_paths,
            **ENV_CONFIG
        )
        return env
    return _init

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🔧 INICIANDO TREINO RECURRENT PPO V17.6 - FIX REAL")
    print("="*80)
    print(f"📅 Timestamp: {timestamp}")
    print(f"🎯 Target: 1.5M steps com checkpoints a cada 10k")
    print(f"💾 Modelo: models/recurrent_ppo_v17_lstm_{timestamp}_XXXXX_steps.zip")
    print(f"⚠️  LSTM requer CPU (DirectML não suporta)")
    print("="*80)
    print("✨ V17.6 FIX REAL - CAUSA RAIZ ENCONTRADA:")
    print("  🔍 DIAGNÓSTICO: Critic overfitting colapsando advantages!")
    print("  ❌ explained_variance=0.98 → V(s)≈Returns → A(s,a)≈0")
    print("  ❌ A(s,a)≈0 → policy_grad≈0 → policy congela (approx_kl~0)")
    print("  ❌ Distribuição muda → VF errada → value_loss explode!")
    print("  ✅ FIX: gamma=0.95 (vantagens maiores)")
    print("  ✅ FIX: n_epochs=4 (menos overfitting critic)")
    print("  ✅ FIX: vf_coef=0.1 (critic fraco, advantages preservadas!)")
    print("  ✅ SEM reward scaling (V17.4/5 foi na direção errada)")
    print("="*80 + "\n")
    
    # Encontrar dados
    print("🔍 Procurando dados multi-timeframe...")
    data_paths = find_latest_data_files()    
    print("\n⚠️  MONITORAMENTO ATIVO:")
    print("  • Se value_loss > 2500: Early stop (divergence)")
    print("  • Se trades < 50/ep por 5 rollouts: Warning")
    print("  • Se clip_fraction < 0.001 por 10 rollouts: Warning")
    print("")    
    print("\n📂 DADOS ENCONTRADOS:")
    for tf, path in data_paths.items():
        print(f"   {tf:>3}: {Path(path).name}")
    
    # Resumo
    print("\n" + "="*80)
    print("\n🆕 V17.6 - CAUSA RAIZ IDENTIFICADA:")
    print("  ❌ V17.3: explained_variance=0.98 → advantages≈0 → policy freeze")
    print("  ❌ V17.4/5: reward scaling aumentou VF target → divergiu imediato")
    print("  💡 FIX REAL: Impedir critic de ficar perfeito demais")
    print("  • gamma=0.95: horizonte curto → returns imprevisíveis → vantagens GRANDES")
    print("  • n_epochs=4: menos passos de gradient no critic (não overfit)")
    print("  • vf_coef=0.1: critic nunca domina (advantages persistem!)")
    print("\n🎯 TARGETS V17.6:")
    print("  • Trades: 300-500/ep (nem overtrading nem freeze)")
    print("  • Win Rate: 35%+ (estabilizar)")
    print("  • Value Loss: <1000 (não divergir)")
    print("  • Return: Positivo")
    print("="*80 + "\n")
    
    # Device - LSTM REQUIRES CPU (DirectML não suporta LSTM!)
    device = "cpu"
    device_name = "CPU"
    
    # Tentar usar CUDA se disponível (mas não DirectML!)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
    
    print(f"🖥️  Device: {device_name}")
    print(f"⚠️  NOTA: LSTM requer CPU ou CUDA. DirectML não suportado.")
    print(f"🎯 Política: RecurrentPPO com MlpLstmPolicy\n")
    
    # Criar ambiente
    print("📁 Criando ambiente LSTM...")
    env = DummyVecEnv([make_env(data_paths)])
    print(f"✅ Ambiente criado: {env.num_envs} env(s)\n")
    
    # Dimensões
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"📐 Obs shape: {obs_shape} (SEQUENTIAL: 50 timesteps × 29 features)")
    print(f"📐 Action shape: {act_shape}\n")
    
    # Criar modelo RecurrentPPO
    print("🏗️  Criando modelo RecurrentPPO...")
    print(f"   - Learning rate: {PPO_CONFIG['learning_rate']}")
    print(f"   - N steps: {PPO_CONFIG['n_steps']}")
    print(f"   - Batch size: {PPO_CONFIG['batch_size']}")
    print(f"   - N epochs: {PPO_CONFIG['n_epochs']} (⬇️ 10→4 - menos critic overfitting!)")
    print(f"   - Gamma: {PPO_CONFIG['gamma']} (⬇️ 0.99→0.95 - vantagens maiores!)")
    print(f"   - GAE lambda: {PPO_CONFIG['gae_lambda']}")
    print(f"   - LSTM hidden: {LSTM_CONFIG['lstm_hidden_size']}")
    print(f"   - LSTM layers: {LSTM_CONFIG['n_lstm_layers']}")
    print(f"   - MLP net arch: {LSTM_CONFIG['net_arch']}")
    print(f"   - Ent coef: {PPO_CONFIG['ent_coef']}")
    print(f"   - VF coef: {PPO_CONFIG['vf_coef']} (⬇️ 0.5→0.1 - critic FRACO, advantages preservadas!)")
    print(f"   - Max grad norm: {PPO_CONFIG['max_grad_norm']}\n")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=PPO_CONFIG['learning_rate'],
        n_steps=PPO_CONFIG['n_steps'],
        batch_size=PPO_CONFIG['batch_size'],
        n_epochs=PPO_CONFIG['n_epochs'],
        gamma=PPO_CONFIG['gamma'],
        gae_lambda=PPO_CONFIG['gae_lambda'],
        clip_range=PPO_CONFIG['clip_range'],
        ent_coef=PPO_CONFIG['ent_coef'],
        vf_coef=PPO_CONFIG['vf_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm'],
        policy_kwargs={
            'lstm_hidden_size': LSTM_CONFIG['lstm_hidden_size'],
            'n_lstm_layers': LSTM_CONFIG['n_lstm_layers'],
            'net_arch': LSTM_CONFIG['net_arch'],
            'activation_fn': LSTM_CONFIG['activation_fn'],
            'ortho_init': LSTM_CONFIG['ortho_init'],
        },
        verbose=1,
        device=device,
        tensorboard_log=f"./tensorboard/recurrent_ppo_v17_lstm_{timestamp}/",
    )
    
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
    
    # V17.2 NOVO: Monitor de divergência de value_loss
    value_loss_monitor = ValueLossDivergenceMonitor(
        max_value_loss=2500,  # Para se value_loss > 2500
        patience=3,           # Por 3 rollouts consecutivos
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=f"./models/",
        name_prefix=f"recurrent_ppo_v17_lstm_{timestamp}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    callbacks = [
        metrics_callback,
        liquidation_monitor,
        decay_monitor,
        value_loss_monitor,  # V17.2 NOVO
        checkpoint_callback,
    ]
    
    print("   ✅ TradingMetricsCallback")
    print("   ✅ LiquidationMonitor")
    print("   ✅ PerformanceDecayMonitor")
    print("   ✅ ValueLossDivergenceMonitor (V17.2 NOVO)")
    print("   ✅ CheckpointCallback")
    print()
    
    # Treinar
    print("="*80)
    print("🧠 INICIANDO TREINO LSTM...")
    print("="*80)
    if device == "cpu":
        print(f"⏱️  Duração estimada: ~60-80h (CPU - LSTM é lento!)")
    else:
        print(f"⏱️  Duração estimada: ~30-40h (CUDA GPU)")
    print(f"📈 TensorBoard: tensorboard --logdir=./tensorboard/\n")
    
    print("💡 HIPÓTESE V17-LSTM:")
    print("   → LSTM captura dependências temporais")
    print("   → Aprende QUANDO segurar vs fechar posições")
    print("   → Memória ajuda a evitar flip-flops")
    print("   → Menos overtrading que V16.3 SAC")
    print("   → Win rate estável treino→teste (menos overfitting)")
    print("="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Salvar modelo final
        final_model_path = f"./models/recurrent_ppo_v17_lstm_{timestamp}_final.zip"
        model.save(final_model_path)
        
        print("\n" + "="*80)
        print("✅ TREINO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"💾 Modelo final salvo: {final_model_path}")
        print(f"📊 TensorBoard: tensorboard --logdir=./tensorboard/recurrent_ppo_v17_lstm_{timestamp}/")
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("   1. Avaliar TensorBoard (win rate, trades)")
        print("   2. Backtest no modelo 500k e 1M")
        print("   3. Comparar com V16.3 SAC")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo usuário")
        interrupted_path = f"./models/recurrent_ppo_v17_lstm_{timestamp}_interrupted.zip"
        model.save(interrupted_path)
        print(f"💾 Modelo parcial salvo: {interrupted_path}")
    
    except Exception as e:
        print(f"\n❌ ERRO durante treino: {e}")
        error_path = f"./models/recurrent_ppo_v17_lstm_{timestamp}_error.zip"
        try:
            model.save(error_path)
            print(f"💾 Modelo parcial salvo: {error_path}")
        except:
            print("❌ Não foi possível salvar modelo")
        raise

if __name__ == "__main__":
    main()
