"""
SCRIPT DE COLETA DE DADOS HISTÓRICOS PROFISSIONAL
Coleta 1-2 anos de dados de múltiplas criptos usando CCXT
Bypass do limite da Binance (1500 candles/request) com paginação
"""

import ccxt
import pandas as pd
import numpy as np
import talib
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import time

class HistoricalDataCollector:
    """Coletor profissional de dados históricos multi-symbol."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Inicializar CCXT (sem autenticação para dados públicos)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Respeita rate limits automático
            'options': {'defaultType': 'future'}  # Futures por padrão
        })
        
        print("[OK] CCXT Binance inicializado")
    
    def fetch_ohlcv_historical(
        self,
        symbol: str,
        timeframe: str = '15m',
        months: int = 24,  # AUMENTADO: 12 -> 24 meses (2 anos)
        max_candles: int = None
    ) -> pd.DataFrame:
        """
        Coleta dados históricos com paginação automática.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            timeframe: Intervalo (1m, 5m, 15m, 1h, 4h, 1d)
            months: Número de meses para coletar (padrão: 24 = 2 anos)
            max_candles: Limite máximo de candles (opcional)
            
        Returns:
            DataFrame com OHLCV completo
        """
        print(f"\n{'='*64}")
        print(f"COLETANDO DADOS HISTÓRICOS: {symbol}")
        print(f"{'='*64}")
        print(f"Timeframe: {timeframe}")
        print(f"Período: {months} meses")
        
        # Calcular timestamps
        now = datetime.now()
        start_date = now - timedelta(days=months * 30)
        
        # Converter para milliseconds
        since = int(start_date.timestamp() * 1000)
        
        print(f"Desde: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Até: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Coletar em páginas
        all_candles = []
        page = 1
        
        while True:
            try:
                print(f"\n[Página {page}] Coletando...")
                
                # Fetch OHLCV
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000  # CCXT otimiza automaticamente
                )
                
                if not ohlcv or len(ohlcv) == 0:
                    print("  Sem mais dados disponíveis")
                    break
                
                # Adicionar à coleção
                all_candles.extend(ohlcv)
                
                print(f"  Coletados: {len(ohlcv)} candles")
                print(f"  Total acumulado: {len(all_candles):,}")
                
                # Atualizar since para próxima página
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                
                # Verificar se atingiu limite
                if max_candles and len(all_candles) >= max_candles:
                    print(f"  Limite atingido: {max_candles:,}")
                    break
                
                # Verificar se já coletou até agora
                last_date = datetime.fromtimestamp(last_timestamp / 1000)
                if last_date >= now:
                    print("  Chegou ao presente")
                    break
                
                page += 1
                
                # Rate limiting (CCXT já faz, mas adicional para segurança)
                time.sleep(0.5)
                
            except ccxt.BaseError as e:
                print(f"  ERRO CCXT: {e}")
                break
            except Exception as e:
                print(f"  ERRO: {e}")
                break
        
        # Converter para DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Remover duplicatas
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Converter timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Converter tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        print(f"\n{'='*64}")
        print(f"COLETA CONCLUÍDA: {symbol}")
        print(f"{'='*64}")
        print(f"Total de candles: {len(df):,}")
        
        # Estatísticas de período
        first_date = df['timestamp'].iloc[0]
        last_date = df['timestamp'].iloc[-1]
        days = (last_date - first_date).days
        
        print(f"Período real: {first_date.strftime('%Y-%m-%d')} a {last_date.strftime('%Y-%m-%d')}")
        print(f"Dias cobertos: {days} ({days/30:.1f} meses)")
        print(f"Preço: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
        print(f"Última: ${df['close'].iloc[-1]:,.2f}")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos usando TA-Lib.
        Usa mesma configuração do config.yaml para consistência.
        """
        print("\n[CALC] Calculando indicadores técnicos...")
        
        df = df.set_index('timestamp')
        df_copy = df.copy()
        
        # Converter para numpy arrays
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
                df_copy['BBB_20_2.0'] = (upper - lower) / middle
                df_copy['BBP_20_2.0'] = (close - lower) / (upper - lower)
                
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
        
        # Remove NaN
        df_copy.dropna(inplace=True)
        df_copy = df_copy.reset_index()
        
        print(f"[OK] {len(df_copy.columns)} features calculadas")
        print(f"[OK] {len(df_copy):,} candles após limpeza")
        
        return df_copy
    
    def collect_and_save(
        self,
        symbol: str,
        months: int = 24,  # AUMENTADO: 12 -> 24 meses
        split_ratio: float = 0.8
    ) -> tuple:
        """
        Pipeline completo: coleta + indicadores + split + salvar.
        
        Args:
            symbol: Par de trading
            months: Meses de histórico (padrão: 24 = 2 anos)
            split_ratio: Proporção train/test (0.8 = 80/20)
            
        Returns:
            (df_train, df_test, paths)
        """
        # 1. Coletar OHLCV
        df = self.fetch_ohlcv_historical(
            symbol=symbol,
            timeframe=self.config['data']['timeframe'],
            months=months
        )
        
        if df is None or len(df) == 0:
            raise ValueError(f"Falha ao coletar dados de {symbol}")
        
        # 2. Adicionar indicadores
        df = self.add_technical_indicators(df)
        
        # 3. Split train/test
        split_idx = int(len(df) * split_ratio)
        df_train = df[:split_idx]
        df_test = df[split_idx:]
        
        print(f"\n[SPLIT] {int(split_ratio*100)}/{int((1-split_ratio)*100)}:")
        print(f"  Treino: {len(df_train):,} candles ({len(df_train)/96:.0f} dias)")
        print(f"  Teste: {len(df_test):,} candles ({len(df_test)/96:.0f} dias)")
        
        # 4. Criar nomes de arquivos
        symbol_clean = symbol.replace('/', '').lower()
        timestamp = datetime.now().strftime('%Y%m%d')
        
        train_path = f'data/train_{symbol_clean}_{months}m_{timestamp}.csv'
        test_path = f'data/test_{symbol_clean}_{months}m_{timestamp}.csv'
        full_path = f'data/full_{symbol_clean}_{months}m_{timestamp}.csv'
        
        # 5. Salvar
        Path('data').mkdir(exist_ok=True)
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        df.to_csv(full_path, index=False)
        
        print(f"\n[SAVE] Arquivos salvos:")
        print(f"  {train_path}")
        print(f"  {test_path}")
        print(f"  {full_path}")
        
        return df_train, df_test, {
            'train': train_path,
            'test': test_path,
            'full': full_path
        }


def collect_multi_symbol(symbols: list, months: int = 24):
    """
    Coleta dados históricos para múltiplos símbolos.
    
    Args:
        symbols: Lista de símbolos (ex: ['BTC/USDT', 'ETH/USDT'])
        months: Meses de histórico para cada (padrão: 24 = 2 anos)
    """
    collector = HistoricalDataCollector()
    
    print("\n" + "="*64)
    print(f"COLETA MULTI-SYMBOL: {len(symbols)} moedas")
    print("="*64)
    
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processando {symbol}...")
        
        try:
            df_train, df_test, paths = collector.collect_and_save(
                symbol=symbol,
                months=months
            )
            
            results[symbol] = {
                'success': True,
                'train_size': len(df_train),
                'test_size': len(df_test),
                'paths': paths
            }
            
            print(f"[OK] {symbol} concluído")
            
            # Rate limiting entre símbolos
            if i < len(symbols):
                print("\n[WAIT] Aguardando 2 segundos...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ ERRO em {symbol}: {e}")
            results[symbol] = {
                'success': False,
                'error': str(e)
            }
    
    # Resumo final
    print("\n" + "="*64)
    print("RESUMO DA COLETA")
    print("="*64)
    
    for symbol, result in results.items():
        if result['success']:
            print(f"[OK] {symbol}: {result['train_size']:,} train / {result['test_size']:,} test")
        else:
            print(f"[ERRO] {symbol}: {result['error']}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Argumentos: python collect_historical_data.py [symbol] [months]
    # Exemplo: python collect_historical_data.py BTC/USDT 12
    
    if len(sys.argv) > 1:
        # Modo single symbol
        symbol = sys.argv[1]
        months = int(sys.argv[2]) if len(sys.argv) > 2 else 24  # AUMENTADO: 12 -> 24
        
        collector = HistoricalDataCollector()
        collector.collect_and_save(symbol=symbol, months=months)
    
    else:
        # Modo multi-symbol padrão
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']  # +ADA
        months = 24  # AUMENTADO: 1 ano -> 2 anos (24 meses)
        
        print("\n[START] COLETA AUTOMATICA MULTI-SYMBOL")
        print(f"Símbolos: {', '.join(symbols)}")
        print(f"Período: {months} meses cada (~{months*30} dias)")
        print(f"Estimativa: ~{months*30*96:,} candles por símbolo (15min)")
        print("\nPressione CTRL+C para cancelar...\n")
        
        time.sleep(3)
        
        results = collect_multi_symbol(symbols, months=months)
        
        print("\n[DONE] COLETA COMPLETA!")
        print("Próximo passo: python train_multi_symbol.py")
