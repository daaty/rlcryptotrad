"""
🧪 Teste da Conexão com Binance Testnet
Verifica se as credenciais estão funcionando
"""

import os
from binance.client import Client
from dotenv import load_dotenv

# Carrega .env
load_dotenv()

print("="*70)
print("🧪 TESTANDO BINANCE TESTNET")
print("="*70)

# Configura client com testnet
client = Client(
    api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
    api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
    testnet=True
)

print("\n📋 Credenciais:")
print(f"  API Key: {os.getenv('BINANCE_TESTNET_API_KEY')[:20]}...")
print(f"  Secret: {os.getenv('BINANCE_TESTNET_SECRET_KEY')[:20]}...")

try:
    # Testa conexão - Futures USDT Testnet
    print("\n🔗 Testando conexão com Futures USDT...")
    
    # Pega saldo de futures
    balance = client.futures_account_balance()
    
    print("\n✅ CONEXÃO ESTABELECIDA!")
    print("\n💰 Saldo da Conta Testnet (Futures USDT):")
    
    # Mostra principais saldos
    for asset in balance:
        if float(asset['balance']) > 0:
            print(f"  {asset['asset']}: {float(asset['balance']):,.2f}")
    
    # Testa buscar preço atual
    print("\n📊 Preço Atual BTC/USDT:")
    ticker = client.futures_ticker(symbol='BTCUSDT')
    print(f"  Last: ${float(ticker['lastPrice']):,.2f}")
    
    # Testa buscar algumas velas
    print("\n📈 Últimas 5 velas (15m):")
    candles = client.futures_klines(symbol='BTCUSDT', interval='15m', limit=5)
    for i, candle in enumerate(candles, 1):
        open_, high, low, close, volume = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])
        print(f"  {i}. O: ${open_:,.2f} | H: ${high:,.2f} | L: ${low:,.2f} | C: ${close:,.2f}")
    
    print("\n" + "="*70)
    print("✅ TESTNET CONFIGURADO CORRETAMENTE!")
    print("="*70)
    print("\n💡 Próximos passos:")
    print("  1. Execute: python -m src.data.data_collector")
    print("  2. Isso coletará dados REAIS da testnet")
    print("  3. Depois treine: python -m src.training.ensemble_trainer")
    print("\n🎯 Agora você pode treinar com dados em tempo real sem risco!")
    
except Exception as e:
    print(f"\n❌ ERRO ao conectar: {e}")
    print("\n🔧 Verifique:")
    print("  1. Chaves copiadas corretamente no .env")
    print("  2. Testnet ativo em: https://testnet.binancefuture.com")
    print("  3. IP não bloqueado (verifique nas configurações da API)")
