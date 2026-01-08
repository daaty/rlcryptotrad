"""
Teste do NOVO sistema de recompensa
Valida que reward incentiva ação vs inatividade
"""

import yaml
import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnv

def test_reward_system():
    """Testa se novo reward incentiva trades vs FLAT"""
    
    print("="*70)
    print("🧪 TESTANDO NOVO SISTEMA DE RECOMPENSA")
    print("="*70)
    
    # Carregar config e dados
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv('data/train_btcusdt_12m_20260105.csv').head(1000)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Criar ambiente
    env_config = config['environment']
    env = TradingEnv(
        df=df,
        initial_balance=env_config['initial_balance'],
        commission=env_config['commission'],
        slippage=env_config.get('slippage', 0.0005),
        leverage=env_config['leverage'],
        position_size=env_config['position_size'],
        window_size=env_config['window_size']
    )
    
    # Reset
    obs, info = env.reset()
    
    # Testar 100 steps com diferentes ações
    flat_rewards = []
    long_rewards = []
    short_rewards = []
    
    print("\n🔬 Simulando 100 steps com 3 estratégias...")
    
    for i in range(100):
        # Reset para mesmo estado
        obs, info = env.reset()
        
        # FLAT (ação 0)
        obs_flat, reward_flat, done, trunc, info = env.step(0)
        flat_rewards.append(reward_flat)
        
        # Reset novamente
        obs, info = env.reset()
        
        # LONG (ação 1)
        obs_long, reward_long, done, trunc, info = env.step(1)
        long_rewards.append(reward_long)
        
        # Reset novamente
        obs, info = env.reset()
        
        # SHORT (ação -1 ou 2)
        obs_short, reward_short, done, trunc, info = env.step(2)
        short_rewards.append(reward_short)
    
    # Análise
    print("\n📊 RESULTADOS (100 steps):")
    print("-" * 70)
    
    avg_flat = np.mean(flat_rewards)
    avg_long = np.mean(long_rewards)
    avg_short = np.mean(short_rewards)
    
    print(f"FLAT  (não fazer nada): Média = ${avg_flat:+.2f}")
    print(f"LONG  (comprar):        Média = ${avg_long:+.2f}")
    print(f"SHORT (vender):         Média = ${avg_short:+.2f}")
    
    print("\n🎯 ANÁLISE:")
    
    # Verificar se FLAT é menos recompensado
    if avg_flat < avg_long or avg_flat < avg_short:
        print("✅ CORRETO: FLAT é menos recompensado que trades ativos")
        print(f"   Diferença LONG vs FLAT: ${avg_long - avg_flat:+.2f}")
        print(f"   Diferença SHORT vs FLAT: ${avg_short - avg_flat:+.2f}")
    else:
        print("❌ PROBLEMA: FLAT ainda é mais recompensado!")
        print("   Ajustar penalidades no TradingEnv")
    
    # Verificar magnitude das penalidades
    min_flat = min(flat_rewards)
    max_flat = max(flat_rewards)
    
    print(f"\n📉 Range de rewards FLAT:")
    print(f"   Mínimo: ${min_flat:.2f}")
    print(f"   Máximo: ${max_flat:.2f}")
    
    if abs(min_flat) > 100:  # Penalidades >= $100
        print("✅ CORRETO: Penalidades significativas por FLAT (-$100+)")
    else:
        print("⚠️  AVISO: Penalidades ainda pequenas")
    
    # Verificar variação
    std_flat = np.std(flat_rewards)
    std_long = np.std(long_rewards)
    std_short = np.std(short_rewards)
    
    print(f"\n📊 Variação (desvio padrão):")
    print(f"   FLAT:  ${std_flat:.2f}")
    print(f"   LONG:  ${std_long:.2f}")
    print(f"   SHORT: ${std_short:.2f}")
    
    if std_long > std_flat and std_short > std_flat:
        print("✅ CORRETO: Trades têm maior variação (risco/recompensa)")
    else:
        print("⚠️  AVISO: FLAT deveria ter menor variação")
    
    # Conclusão
    print("\n" + "="*70)
    if avg_flat < min(avg_long, avg_short) and abs(min_flat) > 100:
        print("✅ SISTEMA DE RECOMPENSA VALIDADO!")
        print("   O modelo vai preferir fazer trades a ficar FLAT")
        print("\n🚀 PRONTO PARA TREINAR!")
    else:
        print("⚠️  SISTEMA PRECISA DE AJUSTES")
        print("   Revisar penalidades no trading_env.py")
    print("="*70)


if __name__ == "__main__":
    test_reward_system()
