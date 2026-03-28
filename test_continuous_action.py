"""
Testa o novo action space contínuo:
- Valores diferentes de action devem resultar em position_sizes diferentes
- SAC pode agora otimizar gradiente suave ao invés de discretização brutal
"""

import sys
sys.path.append('src')

from environment.trading_env import TradingEnv
import numpy as np

def test_continuous_action():
    print("="*60)
    print("TESTE: ACTION SPACE CONTÍNUO (POSITION SIZE DINÂMICO)")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        leverage=3,
        position_size=0.1,  # Base: 10%
        enable_indicator_shaping=False,
        random_start=False
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente inicializado")
    print(f"Balance: ${info['balance']:.2f}")
    print(f"Position size base: {env.position_size*100}%")
    
    # Testa diferentes intensidades de ação
    test_cases = [
        (1.0, "Long 100%"),
        (0.8, "Long 80%"),
        (0.5, "Long 50%"),
        (0.2, "Long 20%"),
        (0.05, "Flat (zona neutra)"),
        (-0.2, "Short 20%"),
        (-0.5, "Short 50%"),
        (-0.8, "Short 80%"),
        (-1.0, "Short 100%"),
    ]
    
    print("\n📊 Testando diferentes intensidades de ação:")
    print("-" * 60)
    
    for action_value, description in test_cases:
        # Reset para testar cada ação independentemente
        env.reset()
        
        action = np.array([action_value])
        obs, reward, terminated, truncated, info = env.step(action)
        
        position_usdt = abs(env.position_value) if env.position != 0 else 0
        expected_size = min(abs(action_value), 1.0) * env.position_size if abs(action_value) > 0.1 else 0
        
        print(f"\nAction: {action_value:+.2f} ({description})")
        print(f"  Position: {['Flat', 'Long', 'Short'][env.position + 1]}")
        print(f"  Position Size usado: {env.current_position_size*100:.1f}% (esperado: {expected_size*100:.1f}%)")
        print(f"  Exposição: ${position_usdt:.2f} USDT")
        
        if abs(action_value) > 0.1:
            # Calcula exposição esperada
            expected_usdt = env.balance * expected_size * env.leverage
            diff = abs(position_usdt - expected_usdt)
            
            if diff < 1:  # Tolerância de $1
                print(f"  ✅ Correto! (exposição ${expected_usdt:.2f})")
            else:
                print(f"  ❌ ERRO! Esperado ${expected_usdt:.2f}, obteve ${position_usdt:.2f}")
    
    print("\n" + "="*60)
    print("📈 VANTAGENS DO ACTION CONTÍNUO:")
    print("="*60)
    print("1. SAC pode otimizar gradiente suave (0.8 → 0.9 tem efeito)")
    print("2. Modelo aprende a reduzir exposição em incerteza")
    print("3. Action 0.3 = Long 30% (menor risco que 100%)")
    print("4. Elimina 'zonas mortas' da discretização")
    print("="*60)

if __name__ == "__main__":
    test_continuous_action()
