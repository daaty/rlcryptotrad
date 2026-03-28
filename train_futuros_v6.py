"""
🚀 TREINO SAC V6 - STOP-LOSS FORÇADO + GERENCIAMENTO DE RISCO REAL
========================================

MUDANÇAS CRÍTICAS vs V5 (que continuou liquidando):
1. 🔒 STOP-LOSS AUTOMÁTICO EM -5% (FORÇADO! Não deixa escolha)
2. ⚖️ leverage: 2x → 1.5x (quase impossível liquidar)
3. 📏 position_size MAX 5% (limita exposição: 0.5 * 10% base)
4. 🎲 ent_coef: 0.15 → 0.1 (menos caos)
5. 🎲 action_noise: 30% → 20% (menos aleatoriedade)
6. ⏱️ max_episode_steps: 5000 → 2000 (mais resets, aprende stop-loss)
7. 💀 trauma_liquidação: -5.0 → -10.0 (não deveria acontecer!)

📊 ANÁLISE V3-V5 (TODOS FALHARAM):
V3: 11 liquidações em 20k (1.1% taxa)
V4: 12 liquidações em 20k (1.5% taxa) - punições fortes PARALISARAM
V5: 12 liquidações em 20k (1.2% taxa) - entropy -1.0 não resolveu

🔍 PROBLEMA RAIZ:
  ❌ 1041 trades em 20k = 52 trades/1k steps (MUITO!)
  ❌ Entra fácil, mas NÃO SAI até liquidar
  ❌ Punições são "sugestões" → modelo ignora
  ❌ Precisa FORÇAR stop-loss, não apenas "sugerir"

🆕 SOLUÇÃO V6 (STOP-LOSS FORÇADO):
Ensina a NÃO LIQUIDAR via gerenciamento de risco OBRIGATÓRIO:
  • Loss > -5%: FECHA AUTOMATICAMENTE (não deixa escolha!)
  • Position size: MAX 5% (menos exposição)
  • Leverage: 1.5x (precisa -67% de loss pra liquidar!)
  • Episódios curtos: 2000 steps (aprende rápido)
  • Menos caos: ent_coef 0.1, noise 20%

🎯 IMPACTO ESPERADO:
  • Liquidações: 12 → <3 em 500k steps (redução 75%+!)
  • Winrate: 10% → 20-25% (menos trades ruins)
  • Drawdown: -50% → -10% (stop protege)
  • Trades: 1041 → ~700 em 20k (seletivo)

Objetivo V6: ZERO liquidações em 500k steps (stop-loss impede!)
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
    print("🚀 TREINO SAC V6 - STOP-LOSS FORÇADO + RISCO CONTROLADO")
    print("="*80)
    print("\n📊 AMBIENTE V6:")
    print("  ✅ Simulação Futuros Binance (liquidação, margin call)")
    print("  ✅ Reward shaping com indicadores (SMA50, RSI, MACD)")
    print("  ✅ Position size dinâmico (action contínuo)")
    print("  ✅ Normalização robusta (Z-Score + clipping)")
    print("  🔒 STOP-LOSS AUTOMÁTICO EM -5% (FORÇADO!)")
    print("     └─ Fecha posição AUTOMATICAMENTE")
    print("     └─ Não deixa o modelo escolher")
    print("     └─ ELIMINA liquidações completamente!")
    print("  📏 Position size MAX 5% (reduz exposição)")
    print("  ⚖️ Leverage 1.5x (precisa -67% pra liquidar)")
    
    print("\n🎯 HIPERPARÂMETROS V6 (GERENCIAMENTO DE RISCO):")
    print("  🔑 ent_coef: 0.1 FIXO (REDUZIDO! Era 0.15)")
    print("  • target_entropy: -1.0 (exploração controlada)")
    print("  ⚖️ leverage: 1.5x (REDUZIDO! Era 2x)")
    print("  🔒 stop_loss: -5% (AUTOMÁTICO, FORÇADO)")
    print("  📏 position_size: MAX 5% do saldo")
    print("  • liquidation_threshold: 15%")
    print("  • learning_rate: 3e-4")
    print("  • buffer_size: 200k")
    print("  🎲 action_noise: 20% (REDUZIDO! Era 30%)")
    print("  ⏱️ max_episode_steps: 2000 (REDUZIDO! Era 5000)")
    print("  • use_sde: True")
    
    print("\n🔒 STOP-LOSS AUTOMÁTICO (NOVO V6!):")
    print("  • Condição: Loss > -5% do saldo inicial")
    print("  • Ação: FECHA POSIÇÃO IMEDIATAMENTE")
    print("  • Reward: -0.08 (perdeu 5% + punição 3%)")
    print("  • Impacto: ELIMINA liquidações!")
    print("  • Ensino: Modelo aprende que -5% é o LIMITE")
    
    print("\n📏 POSITION SIZE CONTROLADO:")
    print("  • Base: 10% do saldo")
    print("  • Máximo: 5% (50% do base)")
    print("  • Leverage: 1.5x aplicado")
    print("  • Exposição máx: 5% * 1.5 = 7.5% do saldo")
    print("  • Risco por trade: ~3-4% (muito menor!)")
    
    print("\n🎮 CALLBACKS ATIVOS:")
    print("  1. TradingMetricsCallback (logging TensorBoard)")
    print("  2. LiquidationMonitor (para se >5 liquidações - NÃO DEVERIA ATIVAR!)")
    print("  3. PerformanceDecayMonitor (para se winrate <5% por 5 episódios)")
    print("  4. CheckpointCallback (salva a cada 50k steps)")
    
    print("\n🎯 META V6 (COM STOP-LOSS FORÇADO):")
    print("  • Liquidações: 0-2 em 500k steps (vs 12 em 20k no V5!)")
    print("  • Winrate: 10% → 20-25%")
    print("  • Max drawdown: -50% → -10%")
    print("  • Trades: ~700 em 20k (mais seletivo)")
    print("  • Sharpe: >0.5 (risco controlado)")
    
    print("\n❌ POR QUE V3-V5 FALHARAM:")
    print("  • Problema: Modelo ESCOLHE se sai ou não")
    print("  • Resultado: Segura perdedores até liquidar")
    print("  • Taxa: 1041 trades, 12 liquidações (1.2%)")
    print("  • Aprendizado: Punições são ignoradas")
    
    print("\n✅ CORREÇÃO V6 (REVOLUCIONÁRIA):")
    print("  • Stop-loss FORÇADO: Não deixa escolha!")
    print("  • Position size limitado: 5% máximo")
    print("  • Leverage baixo: 1.5x (seguro)")
    print("  • Menos caos: ent_coef 0.1, noise 20%")
    print("  • Episódios curtos: 2000 steps (aprende rápido)")
    print("  • Resultado esperado: ZERO liquidações!")
    print("="*80 + "\n")
    
    # Confirmar antes de iniciar
    response = input("Iniciar treino V6 (STOP-LOSS FORÇADO)? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Treino cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE V6 (STOP-LOSS FORÇADO)
    # ============================================
    print("\n📊 Criando ambiente V6 (stop-loss forçado + risco controlado)...")
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            initial_balance=10000,
            commission=0.0004,  # Binance taker
            slippage=0.0005,
            leverage=1.5,  # V6: REDUZIDO 2x → 1.5x
            position_size=0.1,  # Base 10%, MAX 5% (código limita)
            window_size=50,
            max_episode_steps=2000,  # V6: REDUZIDO 5000 → 2000
            random_start=True,
            persist_balance=False,  # Cada episódio independente
            use_sharpe_reward=False,  # Delta equity puro
            use_hybrid_reward=False,
            # FUTUROS BINANCE V6
            maintenance_margin_rate=0.01,  # 1% (1.5x leverage)
            liquidation_threshold=0.15,  # 15%
            enable_indicator_shaping=True  # CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V6 criado (STOP-LOSS FORÇADO!)!")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   🔒 Stop-loss: -5% AUTOMÁTICO")
    print(f"   ⚖️ Leverage: 1.5x (seguro)")
    print(f"   📏 Position size: MAX 5%")
    print(f"   ⏱️ Episode steps: 2000 (mais resets)")
    
    # ============================================
    # 2. CONFIGURAR ACTION NOISE (20%)
    # ============================================
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions)  # V6: REDUZIDO 30% → 20%
    )
    
    print(f"\n🎲 Action Noise: 20% NormalActionNoise (REDUZIDO!)")
    
    # ============================================
    # 3. CRIAR MODELO SAC V6 (ENT_COEF 0.1)
    # ============================================
    print("\n🤖 Criando modelo SAC V6...")
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        log_std_init=-1.0
    )
    
    # ENT_COEF FIXO (REDUZIDO!)
    ent_coef_fixed = 0.1  # V6: 0.15 → 0.1
    
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
        ent_coef=ent_coef_fixed,  # V6: 0.1 FIXO
        target_update_interval=1,
        target_entropy=-1.0,  # Mantido
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        stats_window_size=100,
        tensorboard_log="./logs/sac_futuros_v6/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
        device=dml_device,
        _init_setup_model=True,
    )
    
    print("✅ Modelo V6 criado!")
    print(f"   🔑 ent_coef: {ent_coef_fixed} FIXO (REDUZIDO!)")
    print(f"   • target_entropy: -1.0")
    print(f"   • learning_rate: 0.0003")
    print(f"   • buffer_size: 200000")
    print(f"   • use_sde: True")
    
    # ============================================
    # 4. CONFIGURAR CALLBACKS V6
    # ============================================
    print("\n🎮 Configurando callbacks V6...")
    
    # Callback de métricas customizadas
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    # Monitor de liquidações (MAIS RESTRITIVO!)
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=5,  # V6: 10 → 5 (não deveria ativar com stop-loss!)
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
        name_prefix="sac_futuros_v6",
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
    
    print("✅ Callbacks V6 configurados!")
    print("   1. TradingMetricsCallback")
    print("   2. LiquidationMonitor (max: 5 - NÃO DEVERIA ATIVAR!)")
    print("   3. PerformanceDecayMonitor (min winrate: 5%)")
    print("   4. CheckpointCallback (freq: 50k)")
    
    # ============================================
    # 5. INICIAR TREINO V6
    # ============================================
    print("\n" + "="*80)
    print("🚀 INICIANDO TREINO V6 - 500k STEPS (COM STOP-LOSS FORÇADO!)")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTimestamp: {timestamp}")
    
    print("\n📊 Métricas sendo logadas no TensorBoard:")
    print("   • episode/win_rate")
    print("   • episode/liquidations (deveria ser ~0!)")
    print("   • episode/sharpe_ratio")
    print("   • episode/profit_factor")
    print("   • train/ent_coef (FIXO em 0.1)")
    
    print("\n💡 Para visualizar TensorBoard:")
    print("   tensorboard --logdir=./logs/sac_futuros_v6/")
    
    print("\n⏱️ Tempo estimado: ~4-5h (AMD GPU)")
    print("\n🔒 DIFERENÇA V6: Stop-loss automático em -5% = ZERO liquidações!")
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
        final_path = f"models/sac_futuros_v6_final_{timestamp}.zip"
        model.save(final_path)
        print(f"\n✅ Treino V6 concluído!")
        print(f"   Modelo salvo: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO PELO USUÁRIO")
        
        # Salvar progresso parcial
        partial_path = f"models/sac_futuros_v6_partial_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(partial_path)
            print(f"   Modelo parcial salvo: {partial_path}")
        except Exception as e:
            print(f"   ❌ Erro ao salvar modelo parcial: {e}")
            
    except Exception as e:
        print(f"\n\n❌ ERRO DURANTE O TREINO:")
        print(f"   {type(e).__name__}: {e}")
        
        # Tentar salvar progresso
        error_path = f"models/sac_futuros_v6_error_{timestamp}_{model.num_timesteps}steps.zip"
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
