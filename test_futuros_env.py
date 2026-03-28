"""
Script de teste para validar:
1. Simulação de liquidação (margin call)
2. Reward shaping com indicadores (EMA, RSI, MACD)
"""

import sys
sys.path.append('src')

from environment.trading_env import TradingEnv
import numpy as np

def test_liquidation():
    """Testa se liquidação está funcionando corretamente."""
    print("="*60)
    print("TESTE 1: SIMULAÇÃO DE LIQUIDAÇÃO")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        leverage=3,
        position_size=0.5,  # 50% do balance (muito agressivo, facilita liquidação)
        maintenance_margin_rate=0.005,
        liquidation_threshold=0.10,
        enable_indicator_shaping=True,
        random_start=False
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente inicializado")
    print(f"Balance inicial: ${info['balance']:.2f}")
    print(f"Leverage: {env.leverage}x")
    print(f"Position size: {env.position_size*100}%")
    
    # Simula trade Long em ponto ruim
    print("\n📊 Simulando Long position...")
    for i in range(100):
        action = np.array([1.0])  # Long (sempre)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info['liquidations'] > 0:
            print(f"\n❌ LIQUIDAÇÃO DETECTADA no step {i}!")
            print(f"Equity final: ${info['equity']:.2f}")
            print(f"Perda total: ${10000 - info['equity']:.2f}")
            print(f"Liquidations: {info['liquidations']}")
            break
        
        if i % 20 == 0:
            print(f"Step {i}: Equity ${info['equity']:.2f} | Position: {info['position']} | PnL: ${info['total_pnl']:.2f}")
        
        if terminated or truncated:
            print(f"\n✅ Episódio terminou no step {i}")
            print(f"Liquidations: {info['liquidations']}")
            break
    
    print("\n" + "="*60)

def test_indicator_reward():
    """Testa reward shaping com indicadores."""
    print("\n" + "="*60)
    print("TESTE 2: REWARD SHAPING COM INDICADORES")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        leverage=3,
        position_size=0.1,
        enable_indicator_shaping=True,  # ATIVA indicadores
        random_start=False
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente com indicadores ativado")
    
    # Testa diferentes ações e verifica reward
    actions = [
        (np.array([1.0]), "Long"),
        (np.array([-1.0]), "Short"),
        (np.array([0.0]), "Flat")
    ]
    
    print("\n📊 Testando reward com diferentes ações...")
    for action, name in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nAção: {name:6s} | Reward: {reward:+.6f} | Position: {info['position']}")
        
        # Pega indicadores do candle atual
        current_row = env.df.iloc[env.current_step - 1]
        print(f"  Close: ${current_row['close']:.2f}")
        print(f"  SMA50: ${current_row['SMA_50']:.2f} (Trend: {'UP' if current_row['close'] > current_row['SMA_50'] else 'DOWN'})")
        print(f"  RSI:   {current_row['RSI_14']:.2f} ({'Overbought' if current_row['RSI_14'] > 70 else 'Oversold' if current_row['RSI_14'] < 30 else 'Neutral'})")
        
        if 'MACD_12_26_9' in current_row:
            macd_diff = current_row['MACD_12_26_9'] - current_row['MACDs_12_26_9']
            print(f"  MACD:  {macd_diff:+.2f} (Momentum: {'Bullish' if macd_diff > 0 else 'Bearish'})")
        
        if terminated or truncated:
            break
    
    print("\n" + "="*60)

def test_comparison():
    """Compara performance COM e SEM indicadores."""
    print("\n" + "="*60)
    print("TESTE 3: COMPARAÇÃO COM/SEM INDICADORES")
    print("="*60)
    
    configs = [
        (False, "SEM indicadores"),
        (True, "COM indicadores")
    ]
    
    for enable_indicators, label in configs:
        print(f"\n🔍 Testando {label}...")
        
        env = TradingEnv(
            data_path='data/train_btcusdt_36m_20260109.csv',
            initial_balance=10000,
            leverage=3,
            position_size=0.1,
            enable_indicator_shaping=enable_indicators,
            random_start=False,
            max_episode_steps=500
        )
        
        obs, info = env.reset()
        total_reward = 0
        
        # Simula 500 steps com ações aleatórias
        for i in range(500):
            action = np.random.uniform(-1, 1, size=(1,))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Total Reward: {total_reward:+.4f}")
        print(f"  Final Equity: ${info['equity']:.2f}")
        print(f"  Win Rate: {info['win_rate']:.2%}")
        print(f"  Trades: {info['trades']}")
        print(f"  Liquidations: {info['liquidations']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("\n🚀 TESTANDO AMBIENTE DE FUTUROS BINANCE\n")
    
    test_liquidation()
    test_indicator_reward()
    test_comparison()
    
    print("\n✅ TODOS OS TESTES CONCLUÍDOS!\n")
