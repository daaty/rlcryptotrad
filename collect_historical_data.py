"""
Script para coletar 6 meses de dados históricos da Binance.
Faz múltiplas requisições para contornar limite de 1000 candles.
"""

import os
import pandas as pd
import talib
import yaml
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def fetch_historical_data(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    months: int = 6,
    testnet: bool = False
) -> pd.DataFrame:
    """
    Coleta dados históricos da Binance em múltiplas requisições.
    
    Args:
        symbol: Par de trading (ex: BTCUSDT)
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        months: Número de meses para coletar
        testnet: Se True, usa Binance Testnet
    
    Returns:
        DataFrame com dados OHLCV
    """
    # Inicializa cliente
    if testnet:
        client = Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True
        )
        print(f"🧪 Coletando do Binance TESTNET")
    else:
        # Sem autenticação para dados públicos
        client = Client()
        print(f"📊 Coletando do Binance (dados públicos)")
    
    # Mapeamento de intervalos
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY
    }
    binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE)
    
    # Calcula datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    print(f"\n{'='*70}")
    print(f"COLETANDO DADOS HISTÓRICOS")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    print(f"Duração: {months} meses (~{months * 30} dias)")
    print(f"{'='*70}\n")
    
    # Lista para armazenar DataFrames
    all_data = []
    
    # Calcula quantos requests são necessários
    # 15m: 96 candles/dia, 1000 candles = ~10.4 dias por request
    candles_per_day = {
        '1m': 1440,
        '5m': 288,
        '15m': 96,
        '30m': 48,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    
    candles_needed = candles_per_day.get(interval, 96) * months * 30
    requests_needed = (candles_needed // 1000) + 1
    
    print(f"📈 Candles esperados: ~{candles_needed:,}")
    print(f"🔄 Requests necessários: {requests_needed}")
    print()
    
    # Coleta dados em blocos de 1000 candles
    current_start = start_date
    request_num = 0
    
    while current_start < end_date:
        request_num += 1
        
        try:
            # Converte para timestamp em millisegundos
            start_ts = int(current_start.timestamp() * 1000)
            
            print(f"Request {request_num}/{requests_needed}: {current_start.strftime('%Y-%m-%d %H:%M')}...", end=" ")
            
            # Faz requisição
            klines = client.futures_klines(
                symbol=symbol,
                interval=binance_interval,
                startTime=start_ts,
                limit=1000
            )
            
            if not klines:
                print("❌ Sem dados")
                break
            
            # Converte para DataFrame
            df_chunk = pd.DataFrame(
                klines,
                columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore']
            )
            
            # Seleciona colunas
            df_chunk = df_chunk[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            df_chunk.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Converte tipos
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_chunk[col] = df_chunk[col].astype(float)
            
            all_data.append(df_chunk)
            
            print(f"✅ {len(df_chunk)} candles")
            
            # Atualiza data inicial para próxima requisição
            current_start = df_chunk['timestamp'].iloc[-1]
            
            # Evita rate limit
            import time
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            break
    
    # Concatena todos os DataFrames
    if not all_data:
        raise ValueError("Nenhum dado foi coletado!")
    
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicatas
    df_final = df_final.drop_duplicates(subset='timestamp', keep='first')
    df_final = df_final.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"✅ COLETA CONCLUÍDA!")
    print(f"{'='*70}")
    print(f"Total de candles: {len(df_final):,}")
    print(f"Período real: {df_final['timestamp'].iloc[0]} a {df_final['timestamp'].iloc[-1]}")
    print(f"Duração: {df_final['timestamp'].iloc[-1] - df_final['timestamp'].iloc[0]}")
    print(f"{'='*70}\n")
    
    return df_final


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos usando TA-Lib.
    """
    print("🔧 Calculando indicadores técnicos...")
    
    df = df.copy()
    
    # RSI
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    
    # SMAs
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BBL_20_2.0'] = lower
    df['BBM_20_2.0'] = middle
    df['BBU_20_2.0'] = upper
    df['BBB_20_2.0'] = (upper - lower) / middle
    df['BBP_20_2.0'] = (df['close'] - lower) / (upper - lower)
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_12_26_9'] = macd
    df['MACDs_12_26_9'] = macdsignal
    df['MACDh_12_26_9'] = macdhist
    
    # Returns
    df['open_return'] = df['open'].pct_change()
    df['high_return'] = df['high'].pct_change()
    df['low_return'] = df['low'].pct_change()
    df['close_return'] = df['close'].pct_change()
    
    # Remove NaN inicial
    df = df.dropna().reset_index(drop=True)
    
    # Normaliza indicadores para [0, 1]
    indicators_to_normalize = [
        'RSI_14', 'SMA_20', 'SMA_50', 'BBL_20_2.0', 'BBM_20_2.0', 
        'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'MACD_12_26_9', 
        'MACDs_12_26_9', 'MACDh_12_26_9'
    ]
    
    for col in indicators_to_normalize:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
    
    print(f"✅ Indicadores calculados: {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    
    return df


def main():
    """Função principal"""
    
    # Parâmetros
    symbol = "BTCUSDT"
    interval = "15m"
    months = 6
    testnet = False  # True para testnet, False para dados públicos
    
    # Coleta dados
    df = fetch_historical_data(symbol, interval, months, testnet)
    
    # Adiciona indicadores
    df = add_technical_indicators(df)
    
    # Análise dos dados
    print("\n📊 ANÁLISE DOS DADOS:")
    print(f"   Shape: {df.shape}")
    print(f"   Colunas: {list(df.columns)}")
    print(f"\n💰 Estatísticas do Close:")
    print(f"   Min: ${df['close'].min():,.2f}")
    print(f"   Max: ${df['close'].max():,.2f}")
    print(f"   Média: ${df['close'].mean():,.2f}")
    print(f"   Range: ${df['close'].max() - df['close'].min():,.2f} ({((df['close'].max() / df['close'].min()) - 1) * 100:.1f}%)")
    
    returns = df['close'].pct_change()
    print(f"\n📈 Volatilidade:")
    print(f"   Std: {returns.std() * 100:.4f}%")
    print(f"   Max drawdown: {returns.min() * 100:.2f}%")
    print(f"   Max pump: {returns.max() * 100:.2f}%")
    
    # Salva dados
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "train_data_6m.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dados salvos em: {output_path}")
    print(f"   Tamanho: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Backup do arquivo antigo
    old_file = output_dir / "train_data.csv"
    if old_file.exists():
        backup_file = output_dir / "train_data_backup.csv"
        old_file.rename(backup_file)
        print(f"   Backup do arquivo antigo: {backup_file}")
        
        # Copia novo para train_data.csv
        import shutil
        shutil.copy(output_path, old_file)
        print(f"   Novo arquivo copiado para: {old_file}")
    
    print("\n🎯 Próximo passo: Execute 'python -m src.training.ensemble_trainer' para retreinar os modelos")


if __name__ == "__main__":
    main()
