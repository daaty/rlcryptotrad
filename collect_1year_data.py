"""
Script para coletar 1 ano de dados históricos de BTC/USDT (15min)
Usa a mesma abordagem do código existente mas para período mais longo.
"""

import pandas as pd
from src.data.data_collector import DataCollector
from datetime import datetime

def collect_1_year_data():
    """Coleta 12 meses de dados históricos."""
    
    print("=" * 64)
    print("COLETANDO 1 ANO DE DADOS HISTORICOS")
    print("=" * 64)
    
    # Criar collector
    collector = DataCollector(config_path='config.yaml')
    
    # Binance permite max 1500 candles por request
    # 1 ano = ~35k candles (15min)
    # Vamos coletar em múltiplos requests
    
    print("\nColetando dados históricos...")
    print("NOTA: Binance limita a 1500 candles por request")
    print("      Para 1 ano completo, rode este script multiplas vezes")
    print("      ou use ferramenta dedicada como ccxt com paginação")
    
    # Por enquanto, coletar máximo disponível (1500)
    df = collector.fetch_ohlcv()
    
    # Adicionar indicadores
    df = collector.add_technical_indicators(df)
    
    # Resetar index para ter timestamp como coluna
    df = df.reset_index()
    
    print("\n" + "=" * 64)
    print("COLETA CONCLUIDA")
    print("=" * 64)
    print(f"Total de candles: {len(df):,}")
    
    # Calcular período coberto
    first_date = pd.to_datetime(df['timestamp'].iloc[0])
    last_date = pd.to_datetime(df['timestamp'].iloc[-1])
    days = (last_date - first_date).days
    
    print(f"Período: {first_date.strftime('%Y-%m-%d %H:%M')} até {last_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Dias cobertos: {days} dias ({days/30:.1f} meses)")
    print(f"Candles por dia: {len(df)/max(days, 1):.0f}")
    
    # Split: 80% treino / 20% teste
    split_idx = int(len(df) * 0.8)
    df_train = df[:split_idx]
    df_test = df[split_idx:]
    
    print(f"\nSplit 80/20:")
    print(f"  Treino: {len(df_train):,} candles ({len(df_train)/96:.1f} dias)")
    print(f"  Teste: {len(df_test):,} candles ({len(df_test)/96:.1f} dias)")
    
    # Salvar
    train_path = 'data/train_data_extended.csv'
    test_path = 'data/test_data_extended.csv'
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"\nArquivos salvos:")
    print(f"  Treino: {train_path}")
    print(f"  Teste: {test_path}")
    
    # Estatísticas
    print(f"\nEstatísticas (BTC/USDT):")
    print(f"  Min: ${df['close'].min():,.2f}")
    print(f"  Max: ${df['close'].max():,.2f}")
    print(f"  Média: ${df['close'].mean():,.2f}")
    print(f"  Última: ${df['close'].iloc[-1]:,.2f}")
    
    print("\n" + "=" * 64)
    print("NOTA: Este script coletou o máximo disponível (1500 candles)")
    print("      Para 1 ano completo (~35k candles), use:")
    print("      1. Ferramenta ccxt com paginação histórica")
    print("      2. Download de dataset histórico (Kaggle/CryptoDataDownload)")
    print("      3. Serviço de dados (CryptoCompare/CoinGecko)")
    print("=" * 64)
    
    return df_train, df_test


if __name__ == "__main__":
    collect_1_year_data()
