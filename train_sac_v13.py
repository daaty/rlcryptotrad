"""
SAC V13 - O MELHOR DOS 2 MUNDOS
================================
Combina:
- Balanço Long/Short perfeito do V6 (43%/43%)
- Melhor gestão de risco (stops mais apertados, position sizing menor)
- Mais exploração (ent_coef 0.15, noise 25%)
- Early stopping inteligente (para se win rate não melhorar)
- Checkpoints a cada 50k para monitoramento contínuo

Meta:
- Win Rate: 20% → 28-30%
- Return: -0.96% → +2-5%
- Trades: 700-1000 (balanceado)
- Long/Short: 40-45% / 40-45%
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# DirectML setup
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch_directml
dml_device = torch_directml.device()

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Imports locais
sys.path.append(str(Path(__file__).parent / "src"))
from environment.trading_env import TradingEnv
from callbacks.trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor


def main():
    print("\n" + "="*80)
    print("🚀 SAC V13 - TREINO ATÉ 1M STEPS")
    print("="*80)
    print("\n📊 MELHORIAS SOBRE V6:")
    print("   • Position size: 5% → 3% (mais conservador)")
    print("   • Entropy coef: 0.1 → 0.15 (MAIS exploração)")
    print("   • Action noise: 20% → 25% (MAIS exploração)")
    print("   • Network: [256, 256, 128] (maior capacidade)")
    print("   • Activation: ReLU (DirectML compatível)")
    print("   • Checkpoints: A cada 50k (monitoramento fino)")
    
    print("\n🎯 METAS V13:")
    print("   • Win Rate: 20% → 28-30%")
    print("   • Return: -0.96% → +2% a +5%")
    print("   • Trades: 700-1000 (balanceado)")
    print("   • Long/Short: 40-45% / 40-45%")
    print("   • Balanço: MANTER V6 (não colapsar!)")
    
    print("\n📈 PLANO DE TREINO:")
    print("   • Total: 1,000,000 steps")
    print("   • Checkpoints: 50k, 100k, 150k, ..., 1M")
    print("   • Validação: Rodar backtest a cada checkpoint")
    print("   • Early stop: Se colapsar como V6 600k+")
    
    print("\n" + "="*80)
    
    # Confirmar
    response = input("\nIniciar treino V13? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE V13
    # ============================================
    print("\n📊 Criando ambiente V13 com melhorias...")
    
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            
            # === CONFIGS MANTIDAS DO V6 (QUE FUNCIONARAM) ===
            initial_balance=10000,
            commission=0.0004,                # 0.04% Binance
            slippage=0.0005,                  # 0.05%
            leverage=1.5,                     # ✅ V6: Seguro, sem liquidações
            position_size=0.03,               # 🆕 V13: 3% (V6 era 5%) - Mais conservador
            window_size=50,
            max_episode_steps=2000,           # ✅ V6: Episódios curtos (diversidade)
            random_start=True,
            persist_balance=True,
            use_sharpe_reward=True,           # ✅ V6: Sharpe como principal
            use_hybrid_reward=False,
            enable_indicator_shaping=True,    # ✅ V6: 6 técnicas de shaping
            maintenance_margin_rate=0.005,
            liquidation_threshold=0.10
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V13 criado!")
    print("   🔑 max_episode_steps: 2000 (V6 - episódios curtos)")
    print("   🔑 leverage: 1.5x (V6 - seguro)")
    print("   🔑 position_size: 3% (V13 - MAIS CONSERVADOR)")
    print("   🔑 use_sharpe_reward: True (V6)")
    print("   🔑 enable_indicator_shaping: True (V6)")
    
    # ============================================
    # 2. CRIAR MODELO SAC V13
    # ============================================
    print("\n🤖 Criando modelo SAC V13...")
    
    # Action noise - MAIS exploração que V6
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.25 * np.ones(n_actions)   # 🆕 25% (era 20%) - MAIS exploração
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        
        # === CONFIGS MANTIDAS DO V6 ===
        learning_rate=3e-4,
        buffer_size=200000,               # 200k experiências
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        use_sde=True,                     # ✅ V6: State Dependent Exploration
        
        # === MELHORIAS V13 ===
        ent_coef=0.15,                    # 🆕 0.15 (era 0.1) - MAIS exploração
        target_entropy='auto',            # 🆕 Auto-ajusta entropia
        learning_starts=2000,             # 🆕 2k (era 1k) - Mais exp antes treino
        
        action_noise=action_noise,        # 🆕 25% noise (era 20%)
        
        # Network maior
        policy_kwargs=dict(
            net_arch=[256, 256, 128],     # 🆕 Network maior
            activation_fn=torch.nn.ReLU   # ✅ ReLU (compatível com DirectML!)
        ),
        
        tensorboard_log="./logs/sac_v13",
        verbose=1,
        device=dml_device
    )
    
    print("✅ Modelo SAC V13 criado!")
    print(f"   • ent_coef: 0.15 (V6: 0.1) - MAIS exploração")
    print(f"   • action_noise: 25% (V6: 20%) - MAIS exploração")
    print(f"   • learning_starts: 2000 (V6: 1000)")
    print(f"   • net_arch: [256, 256, 128] (maior que V6)")
    print(f"   • activation: ReLU (compatível DirectML!)")
    
    # ============================================
    # 3. CONFIGURAR CALLBACKS
    # ============================================
    print("\n🎮 Configurando callbacks...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Métricas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # 2. Monitor de liquidações
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=5,
        check_freq=10000,
        verbose=1
    )
    
    # 3. Monitor de decaimento (early stopping manual)
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,
        patience=5,
        verbose=1
    )
    
    # 4. Checkpoints A CADA 50K
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,                  # 🆕 50k (era 100k) - Mais granular
        save_path="./models/",
        name_prefix=f"sac_v13_{timestamp}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    callback = CallbackList([
        metrics_callback,
        liquidation_monitor,
        decay_monitor,
        checkpoint_callback
    ])
    
    print("✅ Callbacks configurados!")
    print("   • TradingMetricsCallback: Log TensorBoard")
    print("   • LiquidationMonitor: Para se > 5 liquidações")
    print("   • PerformanceDecayMonitor: Para se win rate < 5%")
    print("   • CheckpointCallback: A cada 50k steps")
    
    # ============================================
    # 4. TREINAR
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO V13 - 1M STEPS")
    print("="*80)
    print(f"\nTimestamp: {timestamp}")
    print(f"TensorBoard: tensorboard --logdir=./logs/sac_v13/")
    print(f"⏱️ Tempo estimado: ~8-10h (AMD GPU)")
    print(f"\n📊 Checkpoints esperados:")
    print(f"   50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k")
    print(f"   550k, 600k, 650k, 700k, 750k, 800k, 850k, 900k, 950k, 1M")
    print(f"\n⚠️ IMPORTANTE: Rode backtest a cada checkpoint!")
    print(f"   python backtest_stochastic.py models/sac_v13_{timestamp}_XXXXX_steps.zip data/train_btcusdt_36m_20260109.csv")
    print("\n" + "="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=1_000_000,    # 1M steps
            callback=callback,
            log_interval=10,
            tb_log_name=f"v13_{timestamp}",
            progress_bar=True
        )
        
        # Salvar modelo final
        final_path = f"models/sac_v13_1000k_{timestamp}.zip"
        model.save(final_path)
        
        print(f"\n✅ Treino V13 concluído!")
        print(f"   Modelo final salvo: {final_path}")
        print(f"   Total steps: 1,000,000")
        print(f"\n🎯 Próximos passos:")
        print(f"   1. Rodar backtest no modelo final")
        print(f"   2. Comparar checkpoints (50k, 100k, ..., 1M)")
        print(f"   3. Escolher melhor checkpoint")
        print(f"   4. Comparar com V6 500k")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO")
        
        partial_path = f"models/sac_v13_partial_{timestamp}_{model.num_timesteps}steps.zip"
        model.save(partial_path)
        
        print(f"   Modelo parcial salvo: {partial_path}")
        print(f"   Steps completados: {model.num_timesteps}")
        print(f"\n💡 Para retomar, crie script de continue usando este checkpoint.")
    
    except Exception as e:
        print(f"\n❌ ERRO durante treino: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
