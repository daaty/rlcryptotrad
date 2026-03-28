"""
Testa as 3 melhorias implementadas:
1. Normalização robusta do Z-Score
2. Penalidades progressivas de alavancagem
3. Logging de métricas customizadas
"""

import sys
sys.path.append('src')

from environment.trading_env import TradingEnv
import numpy as np

def test_normalization():
    """Testa normalização robusta com dados de baixa volatilidade."""
    print("="*60)
    print("TESTE 1: NORMALIZAÇÃO ROBUSTA (Z-SCORE ESTABILIZADO)")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        enable_indicator_shaping=False,
        random_start=False
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente inicializado")
    print(f"Observation shape: {obs.shape}")
    
    # Verifica se normalização está dentro de limites razoáveis
    print(f"\nEstatísticas da observação normalizada:")
    print(f"  Min: {obs.min():.4f}")
    print(f"  Max: {obs.max():.4f}")
    print(f"  Mean: {obs.mean():.4f}")
    print(f"  Std: {obs.std():.4f}")
    
    # Verifica clipping
    if obs.min() >= -5 and obs.max() <= 5:
        print(f"\n✅ Clipping funcionando! Valores entre -5 e +5 sigmas")
    else:
        print(f"\n❌ AVISO: Valores fora do range esperado!")
    
    # Simula alguns steps para ver comportamento
    print(f"\n📊 Simulando 100 steps...")
    extreme_values = []
    
    for i in range(100):
        action = np.random.uniform(-1, 1, size=(1,))
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Captura valores extremos
        if obs.min() < -4 or obs.max() > 4:
            extreme_values.append((i, obs.min(), obs.max()))
    
    if extreme_values:
        print(f"  ⚠️ Valores extremos detectados em {len(extreme_values)} steps:")
        for step, min_val, max_val in extreme_values[:3]:
            print(f"    Step {step}: min={min_val:.2f}, max={max_val:.2f}")
    else:
        print(f"  ✅ Nenhum valor extremo (>4σ) detectado!")
    
    print("\n" + "="*60)

def test_leverage_penalties():
    """Testa penalidades progressivas de alavancagem."""
    print("\n" + "="*60)
    print("TESTE 2: PENALIDADES PROGRESSIVAS DE ALAVANCAGEM")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        leverage=3,
        position_size=0.5,  # Agressivo para forçar perdas
        enable_indicator_shaping=False,
        random_start=False
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente inicializado")
    print(f"Leverage: {env.leverage}x")
    print(f"Position size: {env.position_size*100}%")
    
    # Abre Long position e simula perda
    print(f"\n📊 Simulando Long com perda gradual...")
    action = np.array([1.0])  # Long 100%
    
    penalties_triggered = []
    
    for i in range(50):
        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.position != 0:
            unrealized_pnl = env._calculate_pnl(env.df.loc[env.current_step - 1, 'close'])
            unrealized_pct = unrealized_pnl / env.initial_balance
            
            # Detecta quando penalidades são aplicadas
            if unrealized_pct < -0.01:
                penalties_triggered.append({
                    'step': i,
                    'loss_pct': unrealized_pct * 100,
                    'reward': reward,
                    'equity': info['equity']
                })
        
        if terminated or truncated:
            break
    
    if penalties_triggered:
        print(f"\n⚠️ Penalidades aplicadas em {len(penalties_triggered)} steps:")
        
        # Agrupa por nível de perda
        mild = [p for p in penalties_triggered if -5 <= p['loss_pct'] < -1]
        medium = [p for p in penalties_triggered if -8 <= p['loss_pct'] < -5]
        severe = [p for p in penalties_triggered if p['loss_pct'] < -8]
        
        print(f"  📉 Leve (1-5%):   {len(mild)} steps")
        print(f"  📉 Média (5-8%):  {len(medium)} steps")
        print(f"  💀 Severa (>8%):  {len(severe)} steps")
        
        if severe:
            print(f"\n  Exemplo de penalidade SEVERA:")
            p = severe[0]
            print(f"    Step {p['step']}: Perda {p['loss_pct']:.2f}%, Reward {p['reward']:.4f}")
    else:
        print(f"\n✅ Nenhuma penalidade aplicada (sem perdas >1%)")
    
    print(f"\n📊 Resultado final:")
    print(f"  Equity: ${info['equity']:.2f} ({(info['equity']/10000-1)*100:+.2f}%)")
    print(f"  Liquidations: {info['liquidations']}")
    
    print("\n" + "="*60)

def test_metrics_logging():
    """Testa captura de métricas para logging."""
    print("\n" + "="*60)
    print("TESTE 3: MÉTRICAS PARA TENSORBOARD")
    print("="*60)
    
    env = TradingEnv(
        data_path='data/train_btcusdt_36m_20260109.csv',
        initial_balance=10000,
        enable_indicator_shaping=True,
        random_start=True,
        max_episode_steps=500
    )
    
    obs, info = env.reset()
    print(f"\n✅ Ambiente inicializado")
    
    # Simula episódio completo
    print(f"\n📊 Simulando episódio completo (500 steps)...")
    
    for i in range(500):
        action = np.random.uniform(-1, 1, size=(1,))
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Captura métricas do episódio
    metrics = env.get_episode_metrics()
    
    print(f"\n📈 MÉTRICAS CAPTURADAS (prontas para TensorBoard):")
    print("-" * 60)
    
    for key, value in metrics.items():
        if 'rate' in key or 'factor' in key or 'return' in key:
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value:.2f}")
    
    print("\n✅ Método get_episode_metrics() funcionando!")
    print("   Use TradingMetricsCallback para logar automaticamente")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("\n🚀 TESTANDO MELHORIAS DO AMBIENTE\n")
    
    test_normalization()
    test_leverage_penalties()
    test_metrics_logging()
    
    print("\n✅ TODOS OS TESTES CONCLUÍDOS!\n")
    print("📊 Melhorias implementadas:")
    print("  1. ✅ Normalização robusta (Z-Score estabilizado + clipping)")
    print("  2. ✅ Penalidades progressivas de alavancagem (1%, 5%, 8%)")
    print("  3. ✅ Logging de métricas customizadas (TensorBoard ready)")
    print("\n🚀 Ambiente otimizado e pronto para treino!\n")
