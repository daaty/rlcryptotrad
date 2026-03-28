"""
🔬 DIAGNÓSTICO: Por que liquidação instantânea?
"""

import sys
sys.path.append('src')

from environment.trading_env import TradingEnv
import numpy as np

def test_immediate_liquidation():
    """Simula exatamente o que acontece no treino."""
    
    env = TradingEnv(
        data_path="data/train_btcusdt_36m_20260109.csv",
        initial_balance=10000,
        commission=0.0004,
        slippage=0.0005,
        leverage=3,
        position_size=0.1,
        window_size=50,
        max_episode_steps=5000,
        random_start=True,
        persist_balance=False,
        use_sharpe_reward=False,
        use_hybrid_reward=False,
        maintenance_margin_rate=0.005,
        liquidation_threshold=0.10,
        enable_indicator_shaping=True
    )
    
    print("🔬 TESTE DE LIQUIDAÇÃO INSTANTÂNEA\n")
    print("="*80)
    
    # Testa 10 episódios
    for episode in range(10):
        obs, info = env.reset()
        
        print(f"\n📊 EPISÓDIO {episode}")
        print(f"   Start step: {env.current_step}")
        print(f"   Initial balance: ${info['balance']:.2f}")
        print(f"   Initial equity: ${info['equity']:.2f}")
        
        # Step 1: Modelo abre Long 100% (pior caso)
        action = np.array([1.0])  # Long com position_size = 100% * 0.1 = 10% do balance
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n   ➡️ ACTION: Long 100% (action=1.0)")
        print(f"   📈 Step 1 result:")
        print(f"      Balance: ${info['balance']:.2f}")
        print(f"      Equity: ${info['equity']:.2f}")
        print(f"      Position: {info['position']}")
        print(f"      Trades: {info['trades']}")
        print(f"      Liquidations: {info['liquidations']}")
        print(f"      Reward: {reward:.6f}")
        print(f"      Terminated: {terminated} | Truncated: {truncated}")
        
        if info['liquidations'] > 0:
            print(f"   ❌ LIQUIDADO NO STEP 1!")
            
            # Calcula o que aconteceu
            print(f"\n   🔍 ANÁLISE:")
            print(f"      Position size: {env.current_position_size * 100:.1f}% do base (0.1)")
            print(f"      Leverage: {env.leverage}x")
            print(f"      Position USDT: ${env.balance * env.current_position_size * env.leverage:.2f}")
            print(f"      Maintenance margin: {env.maintenance_margin_rate * 100:.2f}%")
            print(f"      Liquidation threshold: {env.liquidation_threshold * 100:.0f}% loss")
            
            break
        
        # Step 2: Flat para fechar
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n   📈 Step 2 (Flat para fechar):")
        print(f"      Balance: ${info['balance']:.2f}")
        print(f"      Equity: ${info['equity']:.2f}")
        print(f"      Trades: {info['trades']}")
        print(f"      Win Rate: {info['win_rate']:.1%}")
        
        if episode == 0 and info['liquidations'] == 0:
            print("\n   ✅ SEM LIQUIDAÇÃO! Ambiente OK")
            break

if __name__ == "__main__":
    test_immediate_liquidation()
