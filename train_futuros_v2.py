"""
🚀 TREINO SAC DO ZERO - AMBIENTE OTIMIZADO V2
========================================

Características:
- Ambiente com simulação de Futuros Binance (liquidação realista)
- Reward shaping com indicadores (SMA, RSI, MACD)
- Position size dinâmico (action contínuo)
- Normalização robusta (Z-Score + clipping)
- Penalidades progressivas de alavancagem
- Logging completo no TensorBoard

Hiperparâmetros:
- ent_coef='auto' com target_entropy CUSTOMIZADO (-0.5)
- Learning rate: 3e-4
- Buffer: 200k
- SDE: True
- Action noise: 40%
"""

import sys
sys.path.append('src')
sys.path.append('callbacks')

import yaml
import torch
import torch_directml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from environment.trading_env import TradingEnv
from trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor
import numpy as np
from datetime import datetime


def main():
    # DirectML GPU (AMD)
    dml_device = torch_directml.device()
    print(f"🎮 DirectML Device: {dml_device}")
    
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    print("\n" + "="*80)
    print("🚀 TREINO SAC - AMBIENTE OTIMIZADO V2 (FUTUROS BINANCE)")
    print("="*80)
    print("\n📊 AMBIENTE:")
    print("  ✅ Simulação Futuros Binance (liquidação, margin call)")
    print("  ✅ Reward shaping com indicadores (SMA50, RSI, MACD)")
    print("  ✅ Position size dinâmico (action contínuo)")
    print("  ✅ Normalização robusta (Z-Score + clipping)")
    print("  ✅ Penalidades progressivas de alavancagem (1%, 5%, 8%)")
    
    print("\n🎯 HIPERPARÂMETROS:")
    print("  • ent_coef: 'auto' com target_entropy=-0.5 (HÍBRIDO)")
    print("  • learning_rate: 3e-4")
    print("  • buffer_size: 200k")
    print("  • leverage: 3x")
    print("  • position_size: 10% (base, varia com action)")
    print("  • action_noise: 40% NormalActionNoise")
    print("  • use_sde: True (State Dependent Exploration)")
    
    print("\n🎮 CALLBACKS ATIVOS:")
    print("  1. TradingMetricsCallback (logging TensorBoard)")
    print("  2. LiquidationMonitor (para se >50 liquidações)")
    print("  3. PerformanceDecayMonitor (para se winrate <5% por 5 episódios)")
    print("  4. CheckpointCallback (salva a cada 50k steps)")
    
    print("\n🎯 META:")
    print("  • Winrate: 13% → 30-40% (indicadores + position dinâmico)")
    print("  • Liquidações: <10 em 500k steps")
    print("  • Sharpe Ratio: >2.0")
    print("="*80 + "\n")
    
    # Confirmar antes de iniciar
    response = input("Iniciar treino? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treino cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE OTIMIZADO
    # ============================================
    print("\n📊 Criando ambiente...")
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            initial_balance=10000,
            commission=0.0004,  # Binance taker
            slippage=0.0005,
            leverage=3,
            position_size=0.1,  # Base 10%
            window_size=50,
            max_episode_steps=5000,
            random_start=True,
            persist_balance=False,  # Cada episódio independente
            use_sharpe_reward=False,  # Delta equity puro
            use_hybrid_reward=False,
            # FUTUROS BINANCE
            maintenance_margin_rate=0.005,
            liquidation_threshold=0.10,
            enable_indicator_shaping=True  # CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente criado!")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    
    # ============================================
    # 2. CONFIGURAR ACTION NOISE (40%)
    # ============================================
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.4 * np.ones(n_actions)  # 40% de noise
    )
    
    print(f"\n🎲 Action Noise: 40% NormalActionNoise")
    
    # ============================================
    # 3. CRIAR MODELO SAC COM ENT_COEF='AUTO'
    # ============================================
    print("\n🤖 Criando modelo SAC...")
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0  # σ inicial ≈ 0.37
    )
    
    # TARGET ENTROPY CUSTOMIZADO
    # Default SAC: -dim(action_space) = -1.0 (muito negativo)
    # Customizado: -0.5 (força entropia maior, evita colapso)
    target_entropy = -0.5
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        # ===== HÍBRIDO: AUTO + TARGET CUSTOMIZADO =====
        ent_coef='auto',  # SAC ajusta automaticamente
        target_entropy=target_entropy,  # MAS força target maior
        # ==============================================
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=dml_device,
        tensorboard_log="./logs/sac_futuros_v2/"
    )
    
    print("✅ Modelo criado!")
    print(f"   ent_coef: 'auto' (ajuste dinâmico)")
    print(f"   target_entropy: {target_entropy} (customizado, evita colapso)")
    print(f"   learning_rate: {model.learning_rate}")
    print(f"   buffer_size: {model.buffer_size}")
    print(f"   use_sde: {model.use_sde}")
    
    # ============================================
    # 4. CONFIGURAR CALLBACKS
    # ============================================
    print("\n🎮 Configurando callbacks...")
    
    # 4.1 TradingMetrics (logging TensorBoard)
    metrics_cb = TradingMetricsCallback(verbose=1)
    
    # 4.2 LiquidationMonitor (para se muitas liquidações)
    liq_monitor = LiquidationMonitor(
        max_liquidations=50,  # Para se >50
        check_freq=10000,
        verbose=1
    )
    
    # 4.3 PerformanceDecayMonitor (detecta overfitting/collapse)
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,  # 5% mínimo
        patience=5,  # 5 episódios ruins consecutivos
        verbose=1
    )
    
    # 4.4 CheckpointCallback (salva a cada 50k)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path='./models/',
        name_prefix='sac_futuros_v2',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Combinar todos
    callback = CallbackList([
        metrics_cb,
        liq_monitor,
        decay_monitor,
        checkpoint_cb
    ])
    
    print("✅ Callbacks configurados!")
    print("   1. TradingMetricsCallback")
    print("   2. LiquidationMonitor (max: 50)")
    print("   3. PerformanceDecayMonitor (min winrate: 5%)")
    print("   4. CheckpointCallback (freq: 50k)")
    
    # ============================================
    # 5. INICIAR TREINO
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO - 500k STEPS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📊 Métricas sendo logadas no TensorBoard:")
    print("   • episode/win_rate")
    print("   • episode/liquidations")
    print("   • episode/sharpe_ratio")
    print("   • episode/profit_factor")
    print("   • rollout/entropy (monitora colapso)")
    
    print("\n💡 Para visualizar TensorBoard:")
    print("   tensorboard --logdir=./logs/sac_futuros_v2/")
    print("\n⏱️ Tempo estimado: ~4-5h (AMD GPU)")
    print("\n" + "="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=500_000,
            callback=callback,
            log_interval=10,
            tb_log_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        print("\n" + "="*80)
        print("✅ TREINO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        # Salvar modelo final
        final_path = f"models/sac_futuros_v2_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(final_path)
        print(f"\n💾 Modelo final salvo: {final_path}")
        
        # Estatísticas finais
        print("\n📊 ESTATÍSTICAS FINAIS:")
        if hasattr(env.envs[0], 'liquidations'):
            print(f"   Liquidations totais: {env.envs[0].liquidations}")
        if hasattr(model, 'ent_coef'):
            if isinstance(model.ent_coef, torch.Tensor):
                print(f"   ent_coef final: {model.ent_coef.item():.6f}")
            else:
                print(f"   ent_coef final: {model.ent_coef}")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("   1. Rodar backtest: python backtest.py " + final_path + " data/train_btcusdt_36m_20260109.csv")
        print("   2. Analisar TensorBoard: tensorboard --logdir=./logs/sac_futuros_v2/")
        print("   3. Se winrate >25%, testar em testnet Binance")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO PELO USUÁRIO")
        # Salvar modelo parcial
        partial_path = f"models/sac_futuros_v2_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(partial_path)
        print(f"💾 Modelo parcial salvo: {partial_path}")
        
    except Exception as e:
        print(f"\n\n❌ ERRO NO TREINO: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Tentar salvar modelo
        try:
            error_path = f"models/sac_futuros_v2_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            model.save(error_path)
            print(f"💾 Modelo salvo antes do erro: {error_path}")
        except:
            pass
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
