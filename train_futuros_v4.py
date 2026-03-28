"""
🚀 TREINO SAC V4 - SISTEMA DE PUNIÇÃO PROGRESSIVA APERFEIÇOADO
========================================

MUDANÇAS vs V3:
1. Punição NÍVEL 4 (8-12%): -0.20 → -0.40 (2x mais forte) + penalidade por TEMPO (-1%/step)
2. Punição NÍVEL 5 (12-15%): -0.50 → -1.0 (2x mais forte) + penalidade por TEMPO (-3%/step)
3. Trauma de liquidação: -5.0 → -10.0 (2x mais forte, punição extrema)
4. 🆕 RECOMPENSA POR SAIR: +10% se fechar posição em nível 4, +20% se nível 5
5. LiquidationMonitor: max 10 (MANTIDO, já muito restritivo)

🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA V2 (APERFEIÇOADO):
Ensina o modelo a SAIR de posições perdedoras ANTES da liquidação:
  • Perda 1-3%:   -0.005             (🟡 Alerta)
  • Perda 3-5%:   -0.02              (🟠 Atenção)
  • Perda 5-8%:   -0.08              (🔴 PERIGO!)
  • Perda 8-12%:  -0.40 -0.01/step   (🚨 CRÍTICO! 2x + dor/tempo)
  • Perda 12-15%: -1.0  -0.03/step   (☠️ CATASTRÓFICO! 2x + agonia)
  • Perda >15%:   -10.0              (💀 LIQUIDAÇÃO EXTREMA!)
  
🎁 RECOMPENSAS POR SAIR (NOVO):
  • Fechar posição em nível 4 (loss -8% a -12%): +0.10 (coragem de cortar loss!)
  • Fechar posição em nível 5 (loss < -12%): +0.20 (salvou o que resta!)

Objetivo: Reduzir liquidações a <5 em 500k steps (0.5% de tolerância)
Resultado V3: 11 liquidações em ~20k steps (1.1% de taxa) = BOM MAS PODE MELHORAR!
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
    print("🚀 TREINO SAC V4 - SISTEMA DE PUNIÇÃO PROGRESSIVA V2 (APERFEIÇOADO)")
    print("="*80)
    print("\n📊 AMBIENTE:")
    print("  ✅ Simulação Futuros Binance (liquidação, margin call)")
    print("  ✅ Reward shaping com indicadores (SMA50, RSI, MACD)")
    print("  ✅ Position size dinâmico (action contínuo)")
    print("  ✅ Normalização robusta (Z-Score + clipping)")
    print("  🆕 SISTEMA DE PUNIÇÃO PROGRESSIVA V2 (5 níveis APERFEIÇOADOS)")
    print("     └─ Punições 2x mais fortes nos níveis críticos!")
    print("     └─ Penalidade POR TEMPO em posições perdedoras!")
    print("     └─ RECOMPENSA por SAIR antes da catástrofe!")
    
    print("\n🎯 HIPERPARÂMETROS:")
    print("  • ent_coef: 0.15 FIXO (evita colapso)")
    print("  • target_entropy: -1.5 (entropia alta, exploração)")
    print("  • leverage: 2x (menos risco)")
    print("  • liquidation_threshold: 15%")
    print("  • learning_rate: 3e-4")
    print("  • buffer_size: 200k")
    print("  • position_size: 10% (base, varia com action)")
    print("  • action_noise: 30% NormalActionNoise")
    print("  • use_sde: True (State Dependent Exploration)")
    
    print("\n🆕 ESCALA DE PUNIÇÕES V2 (Caminho da Dor APERFEIÇOADO):")
    print("  • 1-3% loss:   -0.005              (🟡 Alerta)")
    print("  • 3-5% loss:   -0.02               (🟠 Atenção)")
    print("  • 5-8% loss:   -0.08               (🔴 PERIGO!)")
    print("  • 8-12% loss:  -0.40 -0.01/step    (🚨 CRÍTICO! 2x + dor)")
    print("  • 12-15% loss: -1.0  -0.03/step    (☠️ CATASTRÓFICO! 2x + agonia)")
    print("  • >15% loss:   -10.0               (💀 LIQUIDAÇÃO 2x TRAUMA!)")
    
    print("\n🎁 RECOMPENSAS POR SAIR (NOVO V4):")
    print("  • Fechar em nível 4 (-8% a -12%): +0.10 (bom stop-loss!)")
    print("  • Fechar em nível 5 (< -12%):     +0.20 (salvou tudo!)")
    
    print("\n🎮 CALLBACKS ATIVOS:")
    print("  1. TradingMetricsCallback (logging TensorBoard)")
    print("  2. LiquidationMonitor (para se >10 liquidações)")
    print("  3. PerformanceDecayMonitor (para se winrate <5% por 5 episódios)")
    print("  4. CheckpointCallback (salva a cada 50k steps)")
    
    print("\n🎯 META V4:")
    print("  • Winrate: 19.9% → 28-35% (melhoria contínua)")
    print("  • Liquidações: <5 em 500k steps (vs 11 em 20k no V3)")
    print("  • Taxa de liquidação: <0.5% (vs 1.1% no V3)")
    print("  • Entropy: Mantido >0.1 (exploração contínua)")
    print("\n📊 RESULTADO V3 (baseline):")
    print("  • 11 liquidações em ~20k steps")
    print("  • 1005 trades totais (19.9% winrate)")
    print("  • Taxa de liquidação: 1.1% (11/1005)")
    print("  • ✅ BOM! Mas pode melhorar com punições mais fortes")
    print("="*80 + "\n")
    
    # Confirmar antes de iniciar
    response = input("Iniciar treino V4 (sistema anti-liquidação APERFEIÇOADO)? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treino cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE V4 + PUNIÇÃO PROGRESSIVA V2
    # ============================================
    print("\n📊 Criando ambiente V4 (com sistema anti-liquidação V2)...")
    
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
            # FUTUROS BINANCE V4
            maintenance_margin_rate=0.01,  # 1% (2x leverage)
            liquidation_threshold=0.15,  # 15%
            enable_indicator_shaping=True  # CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V4 criado (SISTEMA ANTI-LIQUIDAÇÃO V2 ATIVO)!")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Leverage: 2x")
    print(f"   Liquidation threshold: 15%")
    print(f"   🆕 Punições V2: Níveis 4-5 dobrados + dor/tempo")
    print(f"   🎁 Recompensa por sair: +10% (nível 4) / +20% (nível 5)")
    
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
    # 3. CRIAR MODELO SAC V4
    # ============================================
    print("\n🤖 Criando modelo SAC V4...")
    
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
        target_entropy=-1.5,
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        stats_window_size=100,
        tensorboard_log="./logs/sac_futuros_v4/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
        device=dml_device,
        _init_setup_model=True,
    )
    
    print("✅ Modelo V4 criado!")
    print(f"   ent_coef: {ent_coef_fixed} FIXO")
    print(f"   learning_rate: 0.0003")
    print(f"   buffer_size: 200000")
    print(f"   use_sde: True")
    
    # ============================================
    # 4. CONFIGURAR CALLBACKS V4
    # ============================================
    print("\n🎮 Configurando callbacks V4...")
    
    # Callback de métricas customizadas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # Monitor de liquidações (MANTÉM 10, já muito restritivo)
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
        name_prefix="sac_futuros_v4",
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
    
    print("✅ Callbacks V4 configurados!")
    print("   1. TradingMetricsCallback")
    print("   2. LiquidationMonitor (max: 10)")
    print("   3. PerformanceDecayMonitor (min winrate: 5%)")
    print("   4. CheckpointCallback (freq: 50k)")
    
    # ============================================
    # 5. INICIAR TREINO V4
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO V4 - 500k STEPS")
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
    print("   tensorboard --logdir=./logs/sac_futuros_v4/")
    
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
        final_path = f"models/sac_futuros_v4_final_{timestamp}.zip"
        model.save(final_path)
        print(f"\n✅ Treino V4 concluído!")
        print(f"   Modelo salvo: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO PELO USUÁRIO")
        
        # Salvar progresso parcial
        partial_path = f"models/sac_futuros_v4_partial_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(partial_path)
            print(f"   Modelo parcial salvo: {partial_path}")
        except Exception as e:
            print(f"   ❌ Erro ao salvar modelo parcial: {e}")
            
    except Exception as e:
        print(f"\n\n❌ ERRO DURANTE O TREINO:")
        print(f"   {type(e).__name__}: {e}")
        
        # Tentar salvar progresso
        error_path = f"models/sac_futuros_v4_error_{timestamp}_{model.num_timesteps}steps.zip"
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
