"""
Script para baixar e processar dados de mercado da Binance.
Utiliza python-binance e TA-Lib para gerar features técnicas.
"""

import os
import pandas as pd
import talib
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()


class DataCollector:
    """Coleta e processa dados de mercado para treinamento do agente."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Configuração de dados
        self.symbol = self.config['data']['symbol']
        self.timeframe = self.config['data']['timeframe']
        self.limit = self.config['data']['limit']
        self.mode = self.config.get('mode', 'paper')
        
        # Inicializa cliente baseado no modo
        if self.mode == 'testnet':
            self.client = Client(
                api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
                api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
                testnet=True
            )
            print(f"🧪 Modo TESTNET ativado (dinheiro simulado)")
        elif self.mode == 'live':
            self.client = Client(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_SECRET_KEY')
            )
            print(f"⚠️  Modo LIVE ativado (DINHEIRO REAL!)")
        else:
            # Modo paper não precisa de autenticação
            self.client = Client()
            print(f"📝 Modo PAPER ativado (simulação local)")
        
    def fetch_ohlcv(self, since: str = None) -> pd.DataFrame:
        """
        Baixa dados OHLCV da Binance Futures.
        
        Args:
            since: Data inicial no formato 'YYYY-MM-DD' (opcional)
            
        Returns:
            DataFrame com colunas [timestamp, open, high, low, close, volume]
        """
        print(f"📊 Baixando {self.limit} candles de {self.symbol} ({self.timeframe})...")
        
        # Converte símbolo (BTC/USDT -> BTCUSDT)
        symbol = self.symbol.replace('/', '')
        
        # Converte timeframe
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        interval = interval_map.get(self.timeframe, Client.KLINE_INTERVAL_15MINUTE)
        
        # Converte data se fornecida
        start_str = None
        if since:
            since_dt = datetime.strptime(since, '%Y-%m-%d')
            start_str = since_dt.strftime('%d %b, %Y')
        
        # Baixa dados
        ohlcv = self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=self.limit
        )
        
        # Converte para DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore']
        )
        
        # Seleciona apenas colunas necessárias
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Converte tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Baixados {len(df)} candles | Período: {df.index[0]} a {df.index[-1]}")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame usando TA-Lib.
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com indicadores adicionados
        """
        print("🔧 Calculando indicadores técnicos...")
        
        df_copy = df.copy()
        
        # Converte para numpy arrays
        close = df_copy['close'].values
        high = df_copy['high'].values
        low = df_copy['low'].values
        
        # Itera pelos indicadores configurados
        for indicator in self.config['indicators']:
            name = indicator['name']
            
            if name == 'rsi':
                df_copy['RSI_14'] = talib.RSI(close, timeperiod=indicator['length'])
                
            elif name == 'sma':
                df_copy[f'SMA_{indicator["length"]}'] = talib.SMA(close, timeperiod=indicator['length'])
                
            elif name == 'bbands':
                upper, middle, lower = talib.BBANDS(
                    close,
                    timeperiod=indicator['length'],
                    nbdevup=indicator['std'],
                    nbdevdn=indicator['std']
                )
                df_copy['BBL_20_2.0'] = lower
                df_copy['BBM_20_2.0'] = middle
                df_copy['BBU_20_2.0'] = upper
                df_copy['BBB_20_2.0'] = (upper - lower) / middle  # Bandwidth
                df_copy['BBP_20_2.0'] = (close - lower) / (upper - lower)  # %B
                
            elif name == 'macd':
                macd, signal, hist = talib.MACD(
                    close,
                    fastperiod=indicator['fast'],
                    slowperiod=indicator['slow'],
                    signalperiod=indicator['signal']
                )
                df_copy['MACD_12_26_9'] = macd
                df_copy['MACDs_12_26_9'] = signal
                df_copy['MACDh_12_26_9'] = hist
        
        # Remove linhas com NaN (primeiros candles)
        df_copy.dropna(inplace=True)
        
        print(f"✅ Indicadores adicionados | Features: {len(df_copy.columns)}")
        
        return df_copy
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza os dados para melhorar o treinamento do modelo RL.
        
        Usa diferentes estratégias:
        - Preços: Log returns
        - Volume: Log scale
        - Indicadores: Min-Max normalization
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame normalizado
        """
        print("📏 Normalizando dados...")
        
        df_norm = df.copy()
        
        # 1. Preços (OHLC) -> Log Returns (mas mantém originais também)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_norm[f'{col}_return'] = np.log(df_norm[col] / df_norm[col].shift(1))
        
        # MANTÉM colunas de preço originais (necessário para environment)
        # df_norm.drop(columns=price_cols, inplace=True)  # REMOVIDO
        
        # 2. Volume -> Log scale
        if 'volume' in df_norm.columns:
            df_norm['volume'] = np.log1p(df_norm['volume'])  # log1p evita log(0)
        
        # 3. Indicadores -> Min-Max normalization (0 a 1)
        indicator_cols = [col for col in df_norm.columns if col not in price_cols + ['volume']]
        
        for col in indicator_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            
            if max_val - min_val != 0:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
        
        # Remove NaN gerados pelos log returns
        df_norm.dropna(inplace=True)
        
        print(f"✅ Dados normalizados | Shape: {df_norm.shape}")
        
        return df_norm
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Divide os dados em treino, validação e teste.
        
        Args:
            df: DataFrame completo
            
        Returns:
            Tupla (train_df, val_df, test_df)
        """
        print("✂️ Dividindo dados...")
        
        n = len(df)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Salva o DataFrame processado."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        filepath = data_dir / filename
        df.to_csv(filepath)
        print(f"💾 Dados salvos em: {filepath}")
    
    def run(self, since: str = None):
        """
        Executa o pipeline completo de coleta e processamento.
        
        Args:
            since: Data inicial (opcional)
        """
        print("\n🚀 Iniciando coleta de dados...\n")
        
        # 1. Baixa dados
        df = self.fetch_ohlcv(since)
        
        # 2. Adiciona indicadores
        df = self.add_technical_indicators(df)
        
        # 3. Salva dados brutos processados
        self.save_data(df, 'market_data_raw.csv')
        
        # 4. Normaliza
        df_norm = self.normalize_data(df)
        
        # 5. Divide em treino/val/teste
        train, val, test = self.split_data(df_norm)
        
        # 6. Salva splits
        self.save_data(train, 'train_data.csv')
        self.save_data(val, 'val_data.csv')
        self.save_data(test, 'test_data.csv')
        
        print("\n✅ Pipeline de dados concluído com sucesso!")
        
        return train, val, test


if __name__ == "__main__":
    # Executa a coleta
    collector = DataCollector()
    collector.run()
