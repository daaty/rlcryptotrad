"""
🚀 TREINO SAC V5 - CORREÇÃO: ENTROPY CONTROLADO + PUNIÇÕES EQUILIBRADAS
========================================

MUDANÇAS CRÍTICAS vs V4 (que paralisou o modelo):
1. 🔑 target_entropy: -1.5 → -1.0 (CHAVE! Menos exploração caótica)
2. Punições MODERADAS nos níveis 4-5 (0.25, 0.60 vs 0.40, 1.0)
3. Dor por tempo REDUZIDA (não paralisa: 0.005, 0.015 vs 0.01, 0.03)
4. Trauma liquidação: -10.0 → -5.0 (forte mas não trava aprendizado)
5. 🆕 RECOMPENSA POR HOLDING EM LUCRO (+0.005 se >2%, +0.01 se >5%)
6. 🆕 RECOMPENSA POR REALIZAR LUCRO (+0.05 se >3%, +0.08 se >5%)

📊 ANÁLISE V3 vs V4:
V3 (baseline):
  • 11 liquidações em 20k steps
  • 1005 trades, 19.9% winrate
  • Taxa liquidação: 1.1%
  
V4 (FALHOU - punições fortes demais):
  • 12 liquidações em 20k steps (+9%)
  • 805 trades (-20%), 9.07% winrate (-54%!)
  • Modelo PARALISADO por medo

🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA V3 (EQUILIBRADO):
  • Perda 1-3%:   -0.005             (🟡 Alerta)
  • Perda 3-5%:   -0.02              (🟠 Atenção)
  • Perda 5-8%:   -0.08              (🔴 PERIGO!)
  • Perda 8-12%:  -0.25 -0.005/step  (🚨 Crítico - moderado)
  • Perda 12-15%: -0.60 -0.015/step  (☠️ Catastrófico - moderado)
  • Perda >15%:   -5.0               (💀 Liquidação)
  
🎁 RECOMPENSAS (NOVO V5):
  • Holding em lucro >2%: +0.005 (deixa winner correr)
  • Holding em lucro >5%: +0.01  (winner grande!)
  • Realizar lucro >3%:   +0.05  (bom profit-taking)
  • Realizar lucro >5%:   +0.08  (excelente!)
  • Cortar loss -8/-12%:  +0.10  (bom stop-loss)
  • Cortar loss < -12%:   +0.15  (salvou!)

Objetivo V5: <5 liquidações em 500k steps, 25-30% winrate
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
    print("🚀 TREINO SAC V5 - ENTROPY CONTROLADO + SISTEMA EQUILIBRADO")
    print("="*80)
    print("\n📊 AMBIENTE:")
    print("  ✅ Simulação Futuros Binance (liquidação, margin call)")
    print("  ✅ Reward shaping com indicadores (SMA50, RSI, MACD)")
    print("  ✅ Position size dinâmico (action contínuo)")
    print("  ✅ Normalização robusta (Z-Score + clipping)")
    print("  🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA V3 (EQUILIBRADO)")
    print("     └─ Punições moderadas (não paralisa)")
    print("     └─ Recompensas por holding EM lucro!")
    print("     └─ Recompensas por realizar lucros!")
    
    print("\n🎯 HIPERPARÂMETROS V5 (CORRIGIDOS):")
    print("  • ent_coef: 0.15 FIXO (evita colapso)")
    print("  🔑 target_entropy: -1.0 (REDUZIDO! Era -1.5, menos caos)")
    print("  • leverage: 2x (menos risco)")
    print("  • liquidation_threshold: 15%")
    print("  • learning_rate: 3e-4")
    print("  • buffer_size: 200k")
    print("  • position_size: 10% (base, varia com action)")
    print("  • action_noise: 30% NormalActionNoise")
    print("  • use_sde: True (State Dependent Exploration)")
    
    print("\n🆕 ESCALA DE PUNIÇÕES V3 (EQUILIBRADO):")
    print("  • 1-3% loss:   -0.005              (🟡 Alerta)")
    print("  • 3-5% loss:   -0.02               (🟠 Atenção)")
    print("  • 5-8% loss:   -0.08               (🔴 PERIGO!)")
    print("  • 8-12% loss:  -0.25 -0.005/step   (🚨 Moderado)")
    print("  • 12-15% loss: -0.60 -0.015/step   (☠️ Severo)")
    print("  • >15% loss:   -5.0                (💀 Trauma)")
    
    print("\n🎁 RECOMPENSAS V5 (NOVO!):")
    print("  • Holding lucro >2%: +0.005 (deixa winner correr)")
    print("  • Holding lucro >5%: +0.01  (winner grande!)")
    print("  • Realizar >3%:      +0.05  (bom profit)")
    print("  • Realizar >5%:      +0.08  (excelente!)")
    print("  • Cortar -8/-12%:    +0.10  (stop-loss)")
    print("  • Cortar < -12%:     +0.15  (salvou!)")
    
    print("\n🎮 CALLBACKS ATIVOS:")
    print("  1. TradingMetricsCallback (logging TensorBoard)")
    print("  2. LiquidationMonitor (para se >10 liquidações)")
    print("  3. PerformanceDecayMonitor (para se winrate <5% por 5 episódios)")
    print("  4. CheckpointCallback (salva a cada 50k steps)")
    
    print("\n🎯 META V5:")
    print("  • Winrate: 9% → 25-30% (recuperar de V4)")
    print("  • Liquidações: <5 em 500k steps")
    print("  • Taxa de liquidação: <0.5%")
    print("  • Entropy: Controlado (target -1.0, não -1.5)")
    
    print("\n📊 POR QUE V4 FALHOU:")
    print("  ❌ target_entropy -1.5 = EXPLORAÇÃO CAÓTICA")
    print("     └─ Modelo ignora punições, age aleatório")
    print("  ❌ Punições muito fortes (-0.40, -1.0, -10.0)")
    print("     └─ Modelo fica com MEDO de agir")
    print("  ❌ Resultado: -20% trades, -54% winrate, +9% liquidações")
    
    print("\n✅ CORREÇÃO V5:")
    print("  ✅ target_entropy -1.0 = EXPLORAÇÃO CONTROLADA")
    print("     └─ Modelo aprende com punições")
    print("  ✅ Punições moderadas (-0.25, -0.60, -5.0)")
    print("     └─ Modelo não paralisa")
    print("  ✅ Recompensas por lucro")
    print("     └─ Incentiva deixar winners correrem E realizar")
    print("="*80 + "\n")
    
    # Confirmar antes de iniciar
    response = input("Iniciar treino V5 (ENTROPY CONTROLADO)? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treino cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE V5 (SISTEMA EQUILIBRADO)
    # ============================================
    print("\n📊 Criando ambiente V5 (sistema equilibrado)...")
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            initial_balance=10000,
            commission=0.0004,  # Binance taker
            slippage=0.0005,
            leverage=2,
            position_size=0.1,  # Base 10%
            window_size=50,
            max_episode_steps=5000,
            random_start=True,
            persist_balance=False,  # Cada episódio independente
            use_sharpe_reward=False,  # Delta equity puro
            use_hybrid_reward=False,
            # FUTUROS BINANCE V5
            maintenance_margin_rate=0.01,  # 1% (2x leverage)
            liquidation_threshold=0.15,  # 15%
            enable_indicator_shaping=True  # CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V5 criado (SISTEMA EQUILIBRADO)!")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Leverage: 2x")
    print(f"   Liquidation threshold: 15%")
    print(f"   🆕 Punições V3: Moderadas (não paralisa)")
    print(f"   🎁 Recompensas: Holding lucro + Realizar lucro")
    
    # ============================================
    # 2. CONFIGURAR ACTION NOISE (30%)
    # ============================================
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions)
    )
    
    print(f"\n🎲 Action Noise: 30% NormalActionNoise")
    
    # ============================================
    # 3. CRIAR MODELO SAC V5 (ENTROPY CONTROLADO!)
    # ============================================
    print("\n🤖 Criando modelo SAC V5...")
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0
    )
    
    # ENT_COEF FIXO
    ent_coef_fixed = 0.15
    
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
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef=ent_coef_fixed,
        target_update_interval=1,
        target_entropy=-1.0,  # 🔑 CHAVE! -1.5 → -1.0 (exploração controlada)
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        stats_window_size=100,
        tensorboard_log="./logs/sac_futuros_v5/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
        device=dml_device,
        _init_setup_model=True,
    )
    
    print("✅ Modelo V5 criado!")
    print(f"   ent_coef: {ent_coef_fixed} FIXO")
    print(f"   🔑 target_entropy: -1.0 (CONTROLADO!)")
    print(f"   learning_rate: 0.0003")
    print(f"   buffer_size: 200000")
    print(f"   use_sde: True")
    
    # ============================================
    # 4. CONFIGURAR CALLBACKS V5
    # ============================================
    print("\n🎮 Configurando callbacks V5...")
    
    # Callback de métricas customizadas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # Monitor de liquidações
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=10,
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
        name_prefix="sac_futuros_v5",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    # Combinar callbacks
    callback = CallbackList([
        metrics_callback,
        liquidation_monitor,
        decay_monitor,
        checkpoint_callback
    ])
    
    print("✅ Callbacks V5 configurados!")
    print("   1. TradingMetricsCallback")
    print("   2. LiquidationMonitor (max: 10)")
    print("   3. PerformanceDecayMonitor (min winrate: 5%)")
    print("   4. CheckpointCallback (freq: 50k)")
    
    # ============================================
    # 5. INICIAR TREINO V5
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO V5 - 500k STEPS")
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
    print("   tensorboard --logdir=./logs/sac_futuros_v5/")
    
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
        final_path = f"models/sac_futuros_v5_final_{timestamp}.zip"
        model.save(final_path)
        print(f"\n✅ Treino V5 concluído!")
        print(f"   Modelo salvo: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO PELO USUÁRIO")
        
        # Salvar progresso parcial
        partial_path = f"models/sac_futuros_v5_partial_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(partial_path)
            print(f"   Modelo parcial salvo: {partial_path}")
        except Exception as e:
            print(f"   ❌ Erro ao salvar modelo parcial: {e}")
            
    except Exception as e:
        print(f"\n\n❌ ERRO DURANTE O TREINO:")
        print(f"   {type(e).__name__}: {e}")
        
        # Tentar salvar progresso
        error_path = f"models/sac_futuros_v5_error_{timestamp}_{model.num_timesteps}steps.zip"
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
