"""
🔄 CONTINUAÇÃO DO TREINAMENTO V6
=================================

Este script continua o treinamento do modelo V6 existente
sem perder o progresso já alcançado.

MODELO BASE: sac_futuros_v6_final_20260112_012926.zip
STATUS ATUAL: 500k steps, Win Rate 20.21%, Return -0.96%
META: 800k steps total (+300k novos)

IMPORTANTE: Mantém TODAS as configurações vencedoras do V6!
"""

import sys
sys.path.append('src')
sys.path.append('callbacks')

import torch
import torch_directml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.trading_env import TradingEnv
from trading_metrics import TradingMetricsCallback, LiquidationMonitor, PerformanceDecayMonitor
from datetime import datetime
from pathlib import Path


def main():
    # DirectML GPU (AMD)
    dml_device = torch_directml.device()
    print(f"🎮 DirectML Device: {dml_device}")
    
    data_path = "data/train_btcusdt_36m_20260109.csv"
    base_model_path = "models/sac_futuros_v6_final_20260112_012926.zip"  # 500k steps
    
    print("\n" + "="*80)
    print("🔄 CONTINUAÇÃO TREINO V6 - MANTENDO CONFIGS VENCEDORAS")
    print("="*80)
    
    # Verificar se modelo existe
    if not Path(base_model_path).exists():
        print(f"\n❌ ERRO: Modelo base não encontrado!")
        print(f"   Procurando: {base_model_path}")
        return
    
    print(f"\n📊 Modelo base: {base_model_path}")
    print(f"   Status: 500k steps, Win Rate 20.21%, Return -0.96%")
    print(f"   🏆 MELHOR MODELO ATUAL!")
    
    print("\n🎯 PLANO DE CONTINUAÇÃO:")
    print("   • Steps adicionais: 500,000")
    print("   • Total esperado: 1,000,000 steps")
    print("   • Checkpoints: a cada 100k")
    print("   • Meta Win Rate: 23-25%")
    print("   • Meta Return: +2% a +5%")
    
    print("\n✅ CONFIGURAÇÕES V6 ORIGINAIS (500k) MANTIDAS:")
    print("   • ent_coef: 0.1 FIXO (do modelo original)")
    print("   • leverage: 1.5x")
    print("   • position_size: MAX 5%")
    print("   • stop_loss: -5% automático")
    print("   • episode_steps: 2000")
    print("   • action_noise: 20%")
    print("   • learning_rate: 3e-4 (original)")
    print("   • buffer_size: 200k (original)")
    print("\n⚠️ IMPORTANTE: NÃO modificando nenhum parâmetro!")
    print("   Continuando com EXATAMENTE as mesmas configs do treino original.")
    print("="*80 + "\n")
    
    # Confirmar
    response = input("Continuar treinamento V6? (s/n): ").strip().lower()
    if response != 's':
        print("❌ Cancelado pelo usuário.")
        return
    
    # ============================================
    # 1. CRIAR AMBIENTE V6 ORIGINAL (CRÍTICO!)
    # ============================================
    print("\n📊 Criando ambiente V6 ORIGINAL (configs de 500k)...")
    
    def make_env():
        return TradingEnv(
            data_path=data_path,
            initial_balance=10000,
            commission=0.0004,
            slippage=0.0005,
            leverage=1.5,  # V6 ORIGINAL
            position_size=0.05,  # V6: 5% base (action limita max 5%)
            window_size=50,
            max_episode_steps=2000,  # V6 ORIGINAL! (não 4000!)
            random_start=True,
            persist_balance=True,  # V6: Balance persiste
            use_sharpe_reward=True,  # V6: Usa Sharpe Ratio como principal
            use_hybrid_reward=False,
            maintenance_margin_rate=0.005,  # 0.5% (1.5x leverage)
            liquidation_threshold=0.10,  # 10%
            enable_indicator_shaping=True  # V6 CRÍTICO!
        )
    
    env = DummyVecEnv([make_env])
    
    print("✅ Ambiente V6 ORIGINAL criado!")
    print(f"   🔑 max_episode_steps: 2000 (V6 ORIGINAL, não 4000!)")
    print(f"   🔑 leverage: 1.5x (V6)")
    print(f"   🔑 use_sharpe_reward: False (V6)")
    print(f"   🔑 enable_indicator_shaping: True (V6 CRÍTICO!)")
    
    # ============================================
    # 2. CARREGAR MODELO EXISTENTE
    # ============================================
    print("\n🤖 Carregando modelo V6 existente...")
    
    try:
        # Carregar modelo SEM modificar parâmetros (mantém originais do V6!)
        model = SAC.load(
            base_model_path,
            env=env,
            device=dml_device
        )
        
        # CRÍTICO: Reconfigurar TensorBoard após load (senão logs ficam zerados!)
        model.tensorboard_log = "./logs/sac_futuros_v6/"
        
        print("✅ Modelo V6 carregado com sucesso!")
        print(f"   Steps já treinados: {model.num_timesteps}")
        print(f"   Replay buffer: {model.replay_buffer.size()} experiências")
        print(f"\n🔍 Configs mantidas do V6 original:")
        print(f"   • ent_coef: {model.ent_coef}")
        print(f"   • learning_rate: {model.learning_rate}")
        print(f"   • buffer_size: {model.buffer_size}")
        print(f"   • batch_size: {model.batch_size}")
        
    except Exception as e:
        print(f"\n❌ ERRO ao carregar modelo: {e}")
        return
    
    # ============================================
    # 3. CONFIGURAR CALLBACKS
    # ============================================
    print("\n🎮 Configurando callbacks...")
    
    metrics_callback = TradingMetricsCallback(verbose=1)
    
    liquidation_monitor = LiquidationMonitor(
        max_liquidations=5,
        check_freq=10000,
        verbose=1
    )
    
    decay_monitor = PerformanceDecayMonitor(
        min_winrate=0.05,
        patience=5,
        verbose=1
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # A cada 100k (treino longo)
        save_path="./models/",
        name_prefix=f"sac_futuros_v6_continue",
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
    
    # ============================================
    # 4. CONTINUAR TREINO
    # ============================================
    print("\n" + "="*80)
    print("🚀 CONTINUANDO TREINO V6 - +500k STEPS (500k → 1M)")
    print("="*80)
    print(f"\nTimestamp: {timestamp}")
    print(f"TensorBoard: tensorboard --logdir=./logs/sac_futuros_v6/")
    print(f"⏱️ Tempo estimado: ~4-5h (AMD GPU)")
    print(f"📊 Checkpoints: 600k, 700k, 800k, 900k, 1M")
    print("\n" + "="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=500_000,  # +500k steps
            callback=callback,
            log_interval=10,
            tb_log_name=f"continue_{timestamp}",
            reset_num_timesteps=False,  # CRÍTICO! Mantém contador
            progress_bar=True
        )
        
        # Salvar modelo final
        final_path = f"models/sac_futuros_v6_1000k_{timestamp}.zip"
        model.save(final_path)
        
        print(f"\n✅ Continuação concluída!")
        print(f"   Modelo salvo: {final_path}")
        print(f"   Total steps: ~1,000,000")
        print(f"\n🎯 Próximo passo: Rodar backtest_stochastic.py nos checkpoints!")
        print(f"\n📊 Checkpoints: 600k, 700k, 800k, 900k, 1M")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ TREINO INTERROMPIDO")
        
        partial_path = f"models/sac_futuros_v6_partial_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(partial_path)
            print(f"   Progresso salvo: {partial_path}")
        except Exception as e:
            print(f"   ❌ Erro ao salvar: {e}")
            
    except Exception as e:
        print(f"\n\n❌ ERRO DURANTE TREINO:")
        print(f"   {type(e).__name__}: {e}")
        
        error_path = f"models/sac_futuros_v6_error_{timestamp}_{model.num_timesteps}steps.zip"
        try:
            model.save(error_path)
            print(f"   Modelo salvo: {error_path}")
        except:
            print(f"   ❌ Não foi possível salvar")
        
        raise
    
    finally:
        env.close()
        print("\n🔚 Ambiente fechado.")


if __name__ == "__main__":
    main()
