"""
🎯 Retreinar Modelos com Reward Function Aprimorada
====================================================

Este script retreina os modelos PPO e TD3 com a nova função de reward
que penaliza FLAT e recompensa trading ativo inteligente.

Melhorias na reward function:
1. Penaliza ficar FLAT durante oportunidades claras
2. Recompensa holding de posições lucrativas
3. Penaliza manter posições perdedoras
4. Considera momentum e volatilidade
5. Penaliza overtrading
"""

import yaml
import torch
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.environment.trading_env import TradingEnv
from src.data.data_collector import DataCollector
from src.models.ensemble_model import EnsembleModel

# Load environment variables
load_dotenv()

def load_config():
    """Carrega configuração"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_training_env(config):
    """Cria ambiente de treinamento com TODOS os dados históricos salvos (6 MESES!)"""
    print("📊 Carregando dados históricos salvos...")
    
    # Usa os 6 MESES de dados históricos que já temos!
    import pandas as pd
    df = pd.read_csv('data/train_data_6m.csv')
    
    # Converte timestamp para datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Dados carregados: {len(df)} candles (~{len(df)*15/60/24:.0f} dias)")
    print(f"   📅 Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    # Cria ambiente
    env = TradingEnv(
        df=df,
        initial_balance=config['environment']['initial_balance'],
        position_size=config['environment']['position_size'],
        leverage=config['environment']['leverage'],
        commission=config['environment']['commission']
    )
    
    return DummyVecEnv([lambda: env])

def train_model(model_class, env, config, model_name):
    """Treina um modelo específico"""
    print(f"\n🚀 Treinando {model_name}...")
    
    # Device setup - try DirectML first
    device = "cpu"
    try:
        import torch_directml
        dml = torch_directml.device()
        device = dml
        print(f"✅ Usando DirectML: {torch_directml.device_name(0)}")
    except:
        print("⚠️ DirectML não disponível, usando CPU")
    
    # Hyperparameters OTIMIZADOS para modelos TOP
    if model_name == "PPO":
        model = model_class(
            "MlpPolicy",
            env,
            learning_rate=3e-4,  # Learning rate adaptativo
            n_steps=4096,  # Aumentado para melhor coleta de experiência
            batch_size=256,  # Maior batch = gradientes mais estáveis
            n_epochs=15,  # Mais épocas por update
            gamma=0.995,  # Maior discount = foco em recompensas futuras
            gae_lambda=0.98,  # GAE mais alto para melhor credit assignment
            clip_range=0.2,
            ent_coef=0.02,  # Mais exploração inicial
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
            tensorboard_log=f"./logs/tensorboard_{model_name}_v2/"
        )
    else:  # TD3
        model = model_class(
            "MlpPolicy",
            env,
            learning_rate=3e-4,  # Learning rate mais alto
            buffer_size=500000,  # Buffer MAIOR = mais experiência diversa
            learning_starts=10000,  # Warm-up maior para buffer robusto
            batch_size=256,  # Batch maior = updates mais estáveis
            tau=0.005,
            gamma=0.995,  # Maior discount para visão de longo prazo
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            verbose=1,
            device=device,
            tensorboard_log=f"./logs/tensorboard_{model_name}_v2/"
        )
    
    # Callbacks OTIMIZADOS (mantém arquivos pequenos!)
    # Checkpoint reduzido: só 5 checkpoints durante todo treinamento
    checkpoint_callback = CheckpointCallback(
        save_freq=400000,  # Checkpoint a cada 400k steps (5 checkpoints total)
        save_path=f'./models/checkpoints_{model_name.lower()}_v2/',
        name_prefix=f'{model_name.lower()}_model',
        verbose=1
    )
    
    # EvalCallback: Salva APENAS o MELHOR modelo baseado em performance
    eval_env = env  # Usa mesmo env para avaliação
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/best_{model_name.lower()}_v2/',
        log_path=f'./logs/eval_{model_name.lower()}_v2/',
        eval_freq=25000,  # Avalia a cada 25k steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Treinamento
    total_timesteps = config['training']['total_timesteps']
    print(f"⏳ Treinando por {total_timesteps:,} timesteps...")
    print(f"   📦 Checkpoints: 5 arquivos (~75 MB total)")
    print(f"   🏆 Melhor modelo: salvo automaticamente (~15 MB)")
    print(f"   💾 Espaço total estimado: ~150 MB por modelo")
    print(f"   ⏰ Tempo estimado: {total_timesteps/200000 * 30:.0f}-{total_timesteps/200000 * 60:.0f} minutos\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Salva modelo final
    model_path = f"models/{model_name.lower()}_model_v2.zip"
    model.save(model_path)
    print(f"✅ {model_name} salvo em: {model_path}")
    
    return model

def evaluate_models(ppo_model, td3_model, config):
    """Avalia os modelos retreinados usando dados de VALIDAÇÃO"""
    print("\n📊 Avaliando modelos retreinados...")
    
    # Usa dados de validação separados
    import pandas as pd
    df_test = pd.read_csv('data/val_data.csv')
    
    # Converte timestamp para datetime
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    
    print(f"📊 Dados de teste: {len(df_test)} candles")
    
    env_test = TradingEnv(
        df=df_test,
        initial_balance=config['environment']['initial_balance'],
        position_size=config['environment']['position_size'],
        leverage=config['environment']['leverage'],
        commission=config['environment']['commission']
    )
    
    results = {}
    
    for name, model in [("PPO v2", ppo_model), ("TD3 v2", td3_model)]:
        obs, _ = env_test.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward
            done = terminated or truncated
        
        results[name] = {
            'total_reward': total_reward,
            'final_balance': info['balance'],
            'total_pnl': info['total_pnl'],
            'trades': info['trades'],
            'wins': info['wins'],
            'losses': info['losses'],
            'win_rate': (info['wins'] / info['trades'] * 100) if info['trades'] > 0 else 0
        }
        
        print(f"\n{'='*50}")
        print(f"📈 {name} - Resultados:")
        print(f"  💰 Balance Final: ${results[name]['final_balance']:,.2f}")
        print(f"  📊 Total P&L: ${results[name]['total_pnl']:,.2f}")
        print(f"  🎯 Trades: {results[name]['trades']}")
        print(f"  ✅ Wins: {results[name]['wins']}")
        print(f"  ❌ Losses: {results[name]['losses']}")
        print(f"  📈 Win Rate: {results[name]['win_rate']:.1f}%")
        print(f"  🏆 Total Reward: {results[name]['total_reward']:,.2f}")
    
    return results

def create_ensemble_v2(config):
    """Cria ensemble com novos modelos"""
    print("\n🎯 Criando Ensemble v2...")
    
    # Try DirectML
    use_directml = False
    try:
        import torch_directml
        use_directml = True
    except:
        pass
    
    ensemble = EnsembleModel(
        model_paths={
            'ppo': 'models/ppo_model_v2.zip',
            'td3': 'models/td3_model_v2.zip'
        },
        confidence_threshold=config['ensemble'].get('confidence_threshold', 0.7),
        use_directml=use_directml
    )
    
    # Salva configuração do ensemble
    ensemble_config = {
        'version': 2,
        'created_at': datetime.now().isoformat(),
        'reward_improvements': [
            'Penaliza FLAT durante oportunidades',
            'Recompensa holding lucrativo',
            'Penaliza posições perdedoras',
            'Considera momentum e volatilidade',
            'Previne overtrading'
        ]
    }
    
    with open('models/ensemble_v2_config.yaml', 'w') as f:
        yaml.dump(ensemble_config, f)
    
    print("✅ Ensemble v2 criado!")
    return ensemble

def main():
    """Pipeline completo de retreinamento"""
    print("="*60)
    print("🚀 RETREINAMENTO COM REWARD FUNCTION APRIMORADA")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Carrega configuração
    config = load_config()
    
    # 2. Cria ambiente de treinamento
    env = create_training_env(config)
    
    # 3. Treina PPO
    ppo_model = train_model(PPO, env, config, "PPO")
    
    # 4. Treina TD3 (recria ambiente para limpar estado)
    env = create_training_env(config)
    td3_model = train_model(TD3, env, config, "TD3")
    
    # 5. Avalia modelos
    results = evaluate_models(ppo_model, td3_model, config)
    
    # 6. Cria ensemble
    ensemble = create_ensemble_v2(config)
    
    # 7. Salva relatório
    report = {
        'training_date': datetime.now().isoformat(),
        'reward_improvements': [
            'Penaliza FLAT durante oportunidades claras',
            'Recompensa holding de posições lucrativas',
            'Penaliza manter posições perdedoras',
            'Considera momentum (MACD) e volatilidade',
            'Penaliza overtrading (>5% trade frequency)'
        ],
        'results': results,
        'hyperparameters': {
            'learning_rate': '3e-5',
            'batch_size': 128,
            'total_timesteps': config['training']['total_timesteps'],
            'entropy_coefficient': 0.01
        }
    }
    
    with open('logs/retrain_report_v2.yaml', 'w') as f:
        yaml.dump(report, f)
    
    print("\n" + "="*60)
    print("✅ RETREINAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Arquivos criados:")
    print("  - models/ppo_model_v2.zip")
    print("  - models/td3_model_v2.zip")
    print("  - models/ensemble_v2_config.yaml")
    print("  - logs/retrain_report_v2.yaml")
    print("\n🚀 Próximo passo: Testar ensemble_v2 no dashboard!")

if __name__ == "__main__":
    main()
