"""
🚀 TREINO SAC V3 - SISTEMA DE PUNIÇÃO PROGRESSIVA
========================================

MUDANÇAS vs V2:
1. target_entropy: -0.5 → -1.5 (FORÇA ENTROPIA ALTA POR MAIS TEMPO)
2. ent_coef: 'auto' → 0.15 FIXO (EVITA COLAPSO)
3. leverage: 3x → 2x (MENOS RISCO)
4. liquidation_threshold: 0.10 → 0.15 (LIQUIDA MAIS CEDO)
5. LiquidationMonitor: max 50 → max 30 (MAIS RESTRITIVO)
6. Penalidade liquidação: -1.0 → -5.0 (TRAUMA FINAL)

🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA (ANTI-LIQUIDAÇÃO):
Ensina o modelo a SAIR de posições perdedoras ANTES da liquidação:
  • Perda 1-3%:   -0.005 (alerta gentil)
  • Perda 3-5%:   -0.02  (atenção)
  • Perda 5-8%:   -0.08  (PERIGO! Saia!)
  • Perda 8-12%:  -0.20  (CRÍTICO!)
  • Perda 12-15%: -0.50  (CATASTRÓFICO!)
  • Perda >15%:   -5.0   (LIQUIDAÇÃO = TRAUMA)

Objetivo: Reduzir liquidações drasticamente via "caminho da dor"
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
    print("🚀 TREINO SAC V3 - SISTEMA DE PUNIÇÃO PROGRESSIVA")
    print("="*80)
    print("\n📊 AMBIENTE:")
    print("  ✅ Simulação Futuros Binance (liquidação, margin call)")
    print("  ✅ Reward shaping com indicadores (SMA50, RSI, MACD)")
    print("  ✅ Position size dinâmico (action contínuo)")
    print("  ✅ Normalização robusta (Z-Score + clipping)")
    print("  🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA (5 níveis)")
    print("     └─ Pune posições perdedoras ANTES da liquidação!")
    
    print("\n🎯 HIPERPARÂMETROS (CORRIGIDOS):")
    print("  • ent_coef: 0.15 FIXO (evita colapso, antes: 'auto')")
    print("  • target_entropy: -1.5 (força entropia alta)")
    print("  • leverage: 2x (reduzido de 3x, menos risco)")
    print("  • liquidation_threshold: 15% (aumentado de 10%)")
    print("  • learning_rate: 3e-4")
    print("  • buffer_size: 200k")
    print("  • position_size: 10% (base, varia com action)")
    print("  • action_noise: 30% NormalActionNoise (reduzido de 40%)")
    print("  • use_sde: True (State Dependent Exploration)")
    
    print("\n🆕 ESCALA DE PUNIÇÕES (Caminho da Dor):")
    print("  • 1-3% loss:   -0.005 (🟡 Alerta)")
    print("  • 3-5% loss:   -0.02  (🟠 Atenção)")
    print("  • 5-8% loss:   -0.08  (🔴 PERIGO!)")
    print("  • 8-12% loss:  -0.20  (🚨 CRÍTICO!)")
    print("  • 12-15% loss: -0.50  (☠️ CATASTRÓFICO!)")
    print("  • >15% loss:   -5.0   (💀 LIQUIDAÇÃO)")
    
    print("\n🎮 CALLBACKS ATIVOS:")
    print("  1. TradingMetricsCallback (logging TensorBoard)")
    print("  2. LiquidationMonitor (para se >30 liquidações)")
    print("  3. PerformanceDecayMonitor (para se winrate <5% por 5 episódios)")
    print("  4. CheckpointCallback (salva a cada 50k steps)")
    
    print("\n🎯 META:")
    print("  • Winrate: 13% → 25-35% (conservador, mas estável)")
    print("  • Liquidações: <10 em 500k steps (DRÁSTICA REDUÇÃO!)")
    print("  • Entropy: Mantido >0.1 (exploração contínua)")
    print("="*80 + "\n")
    
    # Confirmar antes de iniciar
    response = input("Iniciar treino V3 (sistema anti-liquidação)? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treino cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE OTIMIZADO V3 + PUNIÇÃO PROGRESSIVA
    # ============================================
    print("\n📊 Criando ambiente V3 (com sistema anti-liquidação)...")
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            initial_balance=10000,
            commission=0.0004,  # Binance taker
            slippage=0.0005,
            leverage=2,  # REDUZIDO: 3x → 2x
            position_size=0.1,  # Base 10%
            window_size=50,
            max_episode_steps=5000,
            random_start=True,
            persist_balance=False,  # Cada episódio independente
            use_sharpe_reward=False,  # Delta equity puro
            use_hybrid_reward=False,
            # FUTUROS BINANCE V3
            maintenance_margin_rate=0.01,  # AUMENTADO: 0.5% → 1% (2x leverage)
            liquidation_threshold=0.15,  # AUMENTADO: 10% → 15%
            enable_indicator_shaping=True  # CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V3 criado (SISTEMA ANTI-LIQUIDAÇÃO ATIVO)!")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Leverage: 2x (REDUZIDO)")
    print(f"   Liquidation threshold: 15% (AUMENTADO)")
    print(f"   🆕 Punição progressiva: 5 níveis (alerta → trauma)")
    
    # ============================================
    # 2. CONFIGURAR ACTION NOISE (30%)
    # ============================================
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions)  # REDUZIDO: 40% → 30%
    )
    
    print(f"\n🎲 Action Noise: 30% NormalActionNoise (REDUZIDO)")
    
    # ============================================
    # 3. CRIAR MODELO SAC COM ENT_COEF FIXO
    # ============================================
    print("\n🤖 Criando modelo SAC V3...")
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0  # σ inicial ≈ 0.37
    )
    
    # ENT_COEF FIXO (NÃO AUTO!)
    # Valor 0.15 = meio termo entre exploração e exploitation
    ent_coef_fixed = 0.15
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,  # IGUAL V2 (funciona!)
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef=ent_coef_fixed,  # FIXO! Não 'auto'
        target_update_interval=1,
        target_entropy=-1.5,  # AUMENTADO: -0.5 → -1.5 (força entropia alta)
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,  # IGUAL V2
        stats_window_size=100,
        tensorboard_log="./logs/sac_futuros_v3/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
        device=dml_device,
        _init_setup_model=True,
    )
    
    print("✅ Modelo V3 criado!")
    print(f"   ent_coef: {ent_coef_fixed} FIXO (NÃO auto)")
    print(f"   learning_rate: 0.0003")
    print(f"   buffer_size: 200000")
    print(f"   use_sde: True")
    
    # ============================================
    # 4. CONFIGURAR CALLBACKS V3
    # ============================================
    print("\n🎮 Configurando callbacks V3...")
    
    # Callback de métricas customizadas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # Monitor de liquidações (MUITO MAIS RESTRITIVO)
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=10,  # DRASTICAMENTE REDUZIDO: 30 → 10 (sistema anti-liquidação!)
        check_freq=10000,
        verbose=1
    )
    
    # Monitor de performance decaída
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,  # 5%
        patience=5,
        verbose=1
    )
    
    # Checkpoint a cada 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="sac_futuros_v3",
        save_replay_buffer=True,  # IGUAL V2
        save_vecnormalize=True,  # IGUAL V2
        verbose=1
    )
    
    # Combinar callbacks
    callback = CallbackList([
        metrics_callback,
        liquidation_monitor,
        decay_monitor,
        checkpoint_callback
    ])
    
    print("✅ Callbacks V3 configurados!")
    print("   1. TradingMetricsCallback")
    print("   2. LiquidationMonitor (max: 10, DRASTICAMENTE REDUZIDO!)")
    print("   3. PerformanceDecayMonitor (min winrate: 5%)")
    print("   4. CheckpointCallback (freq: 50k)")
    
    # ============================================
    # 5. INICIAR TREINO V3
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO V3 - 500k STEPS")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTimestamp: {timestamp}")
    
    print("\n📊 Métricas sendo logadas no TensorBoard:")
    print("   • episode/win_rate")
    print("   • episode/liquidations")
    print("   • episode/sharpe_ratio")
    print("   • episode/profit_factor")
    print("   • train/ent_coef (FIXO em 0.15)")
    
    print("\n💡 Para visualizar TensorBoard:")
    print("   tensorboard --logdir=./logs/sac_futuros_v3/")
    
    print("\n⏱️ Tempo estimado: ~4-5h (AMD GPU)")
    print("\n" + "="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=500_000,
            callback=callback,
            log_interval=10,
            tb_log_name=f"run_{timestamp}",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Salvar modelo final
        final_path = f"models/sac_futuros_v3_final_{timestamp}.zip"
        model.save(final_path)
        print(f"\n✅ Treino V3 concluído!")
        print(f"   Modelo salvo: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO PELO USUÁRIO")
        
        # Salvar progresso parcial
        partial_path = f"models/sac_futuros_v3_partial_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(partial_path)
            print(f"   Modelo parcial salvo: {partial_path}")
        except Exception as e:
            print(f"   ❌ Erro ao salvar modelo parcial: {e}")
            
    except Exception as e:
        print(f"\n\n❌ ERRO DURANTE O TREINO:")
        print(f"   {type(e).__name__}: {e}")
        
        # Tentar salvar progresso
        error_path = f"models/sac_futuros_v3_error_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(error_path)
            print(f"   Modelo de erro salvo: {error_path}")
        except:
            print(f"   ❌ Não foi possível salvar modelo de erro")
        
        raise
    
    finally:
        env.close()
        print("\n🔚 Ambiente fechado.")


if __name__ == "__main__":
    main()
