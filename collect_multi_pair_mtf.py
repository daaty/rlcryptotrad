"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       📊 COLETOR MULTI-PAR MULTI-TIMEFRAME - V18 DATASET                    ║
║                                                                              ║
║  Baixa dados históricos de MÚLTIPLOS PARES em 3 timeframes para treinar     ║
║  o modelo RecurrentPPO V18 com generalização multi-par.                     ║
║                                                                              ║
║  🪙 PARES:                                                                   ║
║  ── BTC/USDT  ETH/USDT  SOL/USDT  BNB/USDT                                 ║
║                                                                              ║
║  📊 TIMEFRAMES:                                                              ║
║  ── 15m (tático) · 1h (operacional) · 4h (estratégico)                      ║
║                                                                              ║
║  💾 OUTPUT por par (ex: BTC):                                                ║
║  ── data/train_btcusdt_36m_15m_YYYYMMDD.csv                                 ║
║  ── data/train_btcusdt_36m_1h_YYYYMMDD.csv                                  ║
║  ── data/train_btcusdt_36m_4h_YYYYMMDD.csv                                  ║
║  ── (idem para eth, sol, bnb)                                                ║
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
import sys


# ── Pares a coletar ──────────────────────────────────────────────────────────
SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
]

TIMEFRAMES  = ['15m', '1h', '4h']
MONTHS      = 36        # 3 anos de histórico
SPLIT_RATIO = 0.8       # 80% treino / 20% teste


class MultiPairCollector:
    """Pipeline de coleta multi-par multi-timeframe."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        print("✅ CCXT Binance Futures inicializado\n")

    # ── helpers ──────────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: str, months: int) -> pd.DataFrame | None:
        """Coleta OHLCV histórico com paginação."""
        now        = datetime.now()
        start_date = now - timedelta(days=months * 30)
        since      = int(start_date.timestamp() * 1000)

        print(f"\n{'─'*60}")
        print(f"  📥 {symbol} · {timeframe}  [{start_date:%Y-%m-%d} → {now:%Y-%m-%d}]")
        print(f"{'─'*60}")

        all_candles, page = [], 1
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=1000
                )
                if not candles:
                    break

                all_candles.extend(candles)
                last_ts   = candles[-1][0]
                last_date = datetime.fromtimestamp(last_ts / 1000)
                print(f"  [pág {page:3d}] +{len(candles)} candles → total {len(all_candles):,}", end='\r')

                if last_date >= now:
                    break

                since = last_ts + 1
                page += 1
                time.sleep(0.4)

            except Exception as exc:
                print(f"\n  ⚠️  Erro pág {page}: {exc}")
                time.sleep(2)
                break

        if not all_candles:
            print(f"\n  ❌ Sem dados para {symbol} {timeframe}")
            return None

        df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)

        print(f"\n  ✅ {len(df):,} candles  |  ${df['close'].min():,.2f} – ${df['close'].max():,.2f}")
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona os mesmos 20 indicadores técnicos do V17.7 (normalização
        relativa ao close — scale-invariant para multi-par).
        """
        df = df.set_index('timestamp').copy()

        close  = df['close'].values.astype(float)
        high   = df['high'].values.astype(float)
        low    = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # RSI (0-1)
        df['RSI_14'] = talib.RSI(close, timeperiod=14) / 100.0

        # SMAs relativas ao close (~1.0 ± fração)
        df['SMA_20'] = talib.SMA(close, timeperiod=20) / (close + 1e-8)
        df['SMA_50'] = talib.SMA(close, timeperiod=50) / (close + 1e-8)

        # Bollinger Bands relativas ao close
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BBL_20_2.0'] = lower  / (close + 1e-8)
        df['BBM_20_2.0'] = middle / (close + 1e-8)
        df['BBU_20_2.0'] = upper  / (close + 1e-8)
        df['BBB_20_2.0'] = (upper - lower) / (middle + 1e-8)
        df['BBP_20_2.0'] = (close - lower) / (upper - lower + 1e-8)

        # MACD relativo ao close
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_12_26_9']  = macd   / (close + 1e-8)
        df['MACDs_12_26_9'] = signal / (close + 1e-8)
        df['MACDh_12_26_9'] = hist   / (close + 1e-8)

        # EMAs relativas ao close
        df['EMA_9']  = talib.EMA(close, timeperiod=9)  / (close + 1e-8)
        df['EMA_21'] = talib.EMA(close, timeperiod=21) / (close + 1e-8)

        # ATR relativo ao close (volatilidade ~0.003-0.025)
        df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-8)

        # Volume relativo à média móvel
        vol_ma = talib.SMA(volume, timeperiod=20)
        df['Volume_MA_20'] = volume / (vol_ma + 1e-8)

        df.dropna(inplace=True)
        df = df.reset_index()
        return df

    # ── pipeline principal ───────────────────────────────────────────────────

    def collect_pair(
        self,
        symbol: str,
        timeframes: list,
        months: int,
        split_ratio: float,
        timestamp: str,
    ) -> dict:
        """Coleta e salva todos os timeframes de um par."""
        symbol_clean = symbol.replace('/', '').lower()   # 'BTC/USDT' → 'btcusdt'
        results = {}

        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, months)
            if df is None:
                continue

            df = self.add_indicators(df)

            split_idx = int(len(df) * split_ratio)
            df_train  = df[:split_idx].copy()
            df_test   = df[split_idx:].copy()

            train_path = f'data/train_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'
            test_path  = f'data/test_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'
            full_path  = f'data/full_{symbol_clean}_{months}m_{tf}_{timestamp}.csv'

            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path,   index=False)
            df.to_csv(full_path,        index=False)

            results[tf] = {
                'train': train_path,
                'test':  test_path,
                'full':  full_path,
                'n_train': len(df_train),
                'n_test':  len(df_test),
            }
            print(f"  💾 {tf}: train={len(df_train):,}  test={len(df_test):,}  → {train_path}")

        return results

    def run(
        self,
        symbols: list  = SYMBOLS,
        timeframes: list  = TIMEFRAMES,
        months: int    = MONTHS,
        split_ratio: float = SPLIT_RATIO,
    ):
        """Executa a coleta completa de todos os pares."""
        timestamp = datetime.now().strftime('%Y%m%d')
        Path('data').mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("🚀 COLETA MULTI-PAR MULTI-TIMEFRAME — V18 DATASET")
        print("="*70)
        print(f"  Pares:       {', '.join(symbols)}")
        print(f"  Timeframes:  {', '.join(timeframes)}")
        print(f"  Histórico:   {months} meses")
        print(f"  Split:       {int(split_ratio*100)}/{int((1-split_ratio)*100)}")
        print("="*70)

        all_results = {}

        for symbol in symbols:
            print(f"\n{'#'*70}")
            print(f"  🪙  PAR: {symbol}")
            print(f"{'#'*70}")

            pair_results = self.collect_pair(
                symbol     = symbol,
                timeframes = timeframes,
                months     = months,
                split_ratio= split_ratio,
                timestamp  = timestamp,
            )
            all_results[symbol] = pair_results

        # ── Resumo final ──────────────────────────────────────────────────────
        print("\n" + "="*70)
        print("✅ COLETA CONCLUÍDA — RESUMO:")
        print("="*70)
        for symbol, tfs in all_results.items():
            print(f"\n  {symbol}:")
            for tf, info in tfs.items():
                print(f"    {tf:>3}  train={info['n_train']:>7,}  test={info['n_test']:>6,}  →  {Path(info['train']).name}")

        print("\n🎯 Próximo passo:")
        print("   python train_recurrent_ppo_v18_multipair.py")
        print()

        return all_results


def main():
    collector = MultiPairCollector()
    collector.run()


if __name__ == "__main__":
    main()
