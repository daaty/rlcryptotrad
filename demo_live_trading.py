"""
🧪 Demo: Trading Ao Vivo na Testnet
Testa o executor com modelos reais
"""

import yaml
from src.execution.live_trader import LiveTrader

print("="*70)
print("🧪 DEMO: LIVE TRADING NA TESTNET")
print("="*70)

# Carrega config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

print(f"\n📋 Configurações:")
print(f"  Modo: {config.get('mode', 'paper')}")
print(f"  Symbol: {config['data']['symbol']}")
print(f"  Timeframe: {config['data']['timeframe']}")
print(f"  Position Size: {config['environment']['position_size']*100}%")
print(f"  Leverage: {config['environment']['leverage']}x")
print(f"  Sentiment: {'✅ Ativado' if config.get('sentiment', {}).get('enabled', False) else '❌ Desativado'}")

print(f"\n⚠️  IMPORTANTE:")
print(f"  - Pressione Ctrl+C para parar o trading")
print(f"  - Posições abertas serão fechadas automaticamente")
print(f"  - Verificação a cada 60 segundos")

input(f"\n▶️  Pressione ENTER para iniciar...\n")

# Inicializa trader
trader = LiveTrader()

# Executa por 10 iterações (~10 minutos) para demo
# Para rodar infinitamente, use max_iterations=None
trader.run(max_iterations=10, sleep_seconds=60)

print("\n" + "="*70)
print("✅ DEMO CONCLUÍDO")
print("="*70)
