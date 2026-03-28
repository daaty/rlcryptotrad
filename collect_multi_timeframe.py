"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🕐 COLETOR MULTI-TIMEFRAME - BTC/USDT (15m, 1h, 4h)                ║
║                                                                              ║
║  Baixa dados históricos em 3 timeframes diferentes para treinar modelos     ║
║  com análise multi-temporal.                                                 ║
║                                                                              ║
║  📊 TIMEFRAMES:                                                              ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  - 15m: Tático (reações rápidas, volatilidade)                              ║
║  - 1h:  Operacional (contexto médio prazo)                                  ║
║  - 4h:  Estratégico (tendências macro)                                      ║
║                                                                              ║
║  💾 OUTPUT:                                                                  ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  - data/train_btcusdt_36m_15m_YYYYMMDD.csv                                  ║
║  - data/train_btcusdt_36m_1h_YYYYMMDD.csv                                   ║
║  - data/train_btcusdt_36m_4h_YYYYMMDD.csv                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import talib
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import time


class MultiTimeframeCollector:
    """Coletor de dados multi-timeframe para treino RL."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Inicializar CCXT
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        print("✅ CCXT Binance inicializado\n")
    
    def fetch_ohlcv_historical(
        self,
        symbol: str,
        timeframe: str,
        months: int = 36
    ) -> pd.DataFrame:
        """
        Coleta dados históricos com paginação automática.
        
        Args:
            symbol: Par (ex: 'BTC/USDT')
            timeframe: Intervalo (15m, 1h, 4h, etc)
            months: Meses para coletar (padrão: 36 = 3 anos)
            
        Returns:
            DataFrame com OHLCV completo
        """
        print(f"\n{'='*70}")
        print(f"📊 COLETANDO: {symbol} - {timeframe}")
        print(f"{'='*70}")
        
        # Calcular timestamps
        now = datetime.now()
        start_date = now - timedelta(days=months * 30)
        since = int(start_date.timestamp() * 1000)
        
        print(f"Período: {start_date.strftime('%Y-%m-%d')} a {now.strftime('%Y-%m-%d')}")
        
        # Coletar em páginas
        all_candles = []
        page = 1
        
        while True:
            try:
                print(f"  [Página {page}] Coletando...", end=' ')
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                if not ohlcv or len(ohlcv) == 0:
                    print("Sem mais dados")
                    break
                
                all_candles.extend(ohlcv)
                print(f"✅ {len(ohlcv)} candles | Total: {len(all_candles):,}")
                
                # Próxima página
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                
                # Verificar se chegou ao presente
                last_date = datetime.fromtimestamp(last_timestamp / 1000)
                if last_date >= now:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ ERRO: {e}")
                break
        
        # Converter para DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Limpar duplicatas
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Converter tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Estatísticas
        print(f"\n✅ COLETA CONCLUÍDA:")
        print(f"   Candles: {len(df):,}")
        print(f"   Período: {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")
        print(f"   Preço: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos usando TA-Lib.
        """
        print(f"\n🔧 Calculando indicadores para {timeframe}...")
        
        df = df.set_index('timestamp')
        df_copy = df.copy()
        
        close = df_copy['close'].values
        high = df_copy['high'].values
        low = df_copy['low'].values
        volume = df_copy['volume'].values
        
        # =================================================================
        # NORMALIZAÇÃO RELATIVA AO CLOSE (scale-invariant)
        # Garante que training CSV == dashboard inference:
        #   - preços absolutos (SMA, EMA, BB) → divididos por close
        #   - RSI → dividido por 100 (já é 0-100)
        #   - MACD/Signal/Hist → divididos por close (ratio adimensional)
        #   - ATR → dividido por close (volatilidade relativa)
        #   - Volume_MA20 → dividido por volume (ratio relativo)
        #   - OHLCV: mantido em bruto (mesma escala em CSV e dashboard)
        # =================================================================
        
        # Indicadores do config.yaml
        for indicator in self.config['indicators']:
            name = indicator['name']
            
            if name == 'rsi':
                df_copy['RSI_14'] = talib.RSI(close, timeperiod=indicator['length']) / 100.0
                
            elif name == 'sma':
                sma = talib.SMA(close, timeperiod=indicator['length'])
                df_copy[f'SMA_{indicator["length"]}'] = sma / (close + 1e-8)
                
            elif name == 'bbands':
                upper, middle, lower = talib.BBANDS(
                    close,
                    timeperiod=indicator['length'],
                    nbdevup=indicator['std'],
                    nbdevdn=indicator['std']
                )
                df_copy['BBL_20_2.0'] = lower  / (close + 1e-8)
                df_copy['BBM_20_2.0'] = middle / (close + 1e-8)
                df_copy['BBU_20_2.0'] = upper  / (close + 1e-8)
                df_copy['BBB_20_2.0'] = (upper - lower) / (middle + 1e-8)
                df_copy['BBP_20_2.0'] = (close - lower) / (upper - lower + 1e-8)
                
            elif name == 'macd':
                macd, signal, hist = talib.MACD(
                    close,
                    fastperiod=indicator['fast'],
                    slowperiod=indicator['slow'],
                    signalperiod=indicator['signal']
                )
                df_copy['MACD_12_26_9']  = macd   / (close + 1e-8)
                df_copy['MACDs_12_26_9'] = signal / (close + 1e-8)
                df_copy['MACDh_12_26_9'] = hist   / (close + 1e-8)
        
        # EMA normalizada por close → ratio ~1.0 (±alguns %)
        df_copy['EMA_9']  = talib.EMA(close, timeperiod=9)  / (close + 1e-8)
        df_copy['EMA_21'] = talib.EMA(close, timeperiod=21) / (close + 1e-8)
        
        # ATR normalizado por close → volatilidade relativa (~0.005-0.03)
        df_copy['ATR_14'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-8)
        
        # Volume MA: volume relativo à média (~0.1-5x, sem outliers)
        # volume/vol_ma > 1 = volume acima da média
        df_copy['Volume_MA_20'] = df_copy['volume'].values / (vol_ma + 1e-8)
        
        # Remover NaN
        df_copy.dropna(inplace=True)
        df_copy = df_copy.reset_index()
        
        print(f"   ✅ {len(df_copy.columns)} features | {len(df_copy):,} candles")
        
        return df_copy
    
    def collect_and_save_multi_timeframe(
        self,
        symbol: str = 'BTC/USDT',
        timeframes: list = ['15m', '1h', '4h'],
        months: int = 36,
        split_ratio: float = 0.8
    ) -> dict:
        """
        Pipeline completo: coleta múltiplos timeframes + indicadores + split + save.
        
        Args:
            symbol: Par de trading
            timeframes: Lista de timeframes
            months: Meses de histórico
            split_ratio: Proporção train/test
            
        Returns:
            Dicionário com paths dos arquivos salvos
        """
        print("\n" + "="*70)
        print(f"🚀 COLETA MULTI-TIMEFRAME: {symbol}")
        print("="*70)
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Período: {months} meses")
        print(f"Split: {int(split_ratio*100)}/{int((1-split_ratio)*100)}")
        print("="*70)
        
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d')
        Path('data').mkdir(exist_ok=True)
        
        for tf in timeframes:
            print(f"\n{'#'*70}")
            print(f"PROCESSANDO TIMEFRAME: {tf}")
            print(f"{'#'*70}")
            
            # 1. Coletar OHLCV
            df = self.fetch_ohlcv_historical(
                symbol=symbol,
                timeframe=tf,
                months=months
            )
            
            if df is None or len(df) == 0:
                print(f"❌ Falha ao coletar {tf}")
                continue
            
            # 2. Adicionar indicadores
            df = self.add_technical_indicators(df, tf)
            
            # 3. Split train/test
            split_idx = int(len(df) * split_ratio)
            df_train = df[:split_idx]
            df_test = df[split_idx:]
            
            print(f"\n📊 SPLIT:")
            print(f"   Treino: {len(df_train):,} candles")
            print(f"   Teste:  {len(df_test):,} candles")
            
            # 4. Criar nomes de arquivos
            symbol_clean = symbol.replace('/', '').lower()
            
            train_path = f'data/train_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'
            test_path = f'data/test_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'
            full_path = f'data/full_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'
            
            # 5. Salvar
            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)
            df.to_csv(full_path, index=False)
            
            print(f"\n💾 SALVOS:")
            print(f"   {train_path}")
            print(f"   {test_path}")
            print(f"   {full_path}")
            
            results[tf] = {
                'train': train_path,
                'test': test_path,
                'full': full_path,
                'train_candles': len(df_train),
                'test_candles': len(df_test)
            }
        
        print("\n" + "="*70)
        print("✅ COLETA MULTI-TIMEFRAME CONCLUÍDA!")
        print("="*70)
        
        for tf, paths in results.items():
            print(f"\n{tf}:")
            print(f"  Train: {paths['train_candles']:,} candles")
            print(f"  Test:  {paths['test_candles']:,} candles")
        
        return results


def main():
    """Executa coleta multi-timeframe."""
    collector = MultiTimeframeCollector()
    
    # Coletar BTC/USDT em 3 timeframes
    results = collector.collect_and_save_multi_timeframe(
        symbol='BTC/USDT',
        timeframes=['15m', '1h', '4h'],
        months=36,  # 3 anos de dados
        split_ratio=0.8
    )
    
    print("\n" + "="*70)
    print("📂 ARQUIVOS CRIADOS:")
    print("="*70)
    
    for tf, paths in results.items():
        print(f"\n{tf}:")
        for key, value in paths.items():
            if key not in ['train_candles', 'test_candles']:
                print(f"  {value}")
    
    print("\n✅ Pronto para treinar V16 com multi-timeframe!\n")


if __name__ == "__main__":
    main()
