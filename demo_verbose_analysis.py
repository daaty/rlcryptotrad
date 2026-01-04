"""
🔍 Live Trader com Logs Detalhados
Executa trading mostrando todas as decisões dos modelos
"""

import os
import time
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
from pathlib import Path

from src.models.ensemble_model import EnsembleModel
from stable_baselines3 import PPO, TD3

load_dotenv()

# Configuração de logging detalhado (UTF-8 para Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_decisions.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix para encoding do Windows
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Cria diretório de logs
Path('logs').mkdir(exist_ok=True)


class VerboseTrader:
    """Trader com logging detalhado de decisões"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.mode = self.config.get('mode', 'testnet')
        self.symbol = self.config['data']['symbol'].replace('/', '')
        self.timeframe = self.config['data']['timeframe']
        
        # Cliente Binance
        if self.mode == 'testnet':
            self.client = Client(
                api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
                api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
                testnet=True
            )
            logger.info("Modo TESTNET ativado")
        else:
            raise ValueError("Use mode='testnet' para este demo")
        
        # Carrega modelos
        logger.info("="*70)
        logger.info("CARREGANDO MODELOS ENSEMBLE")
        logger.info("="*70)
        
        models = {}
        model_dir = Path("models/ensemble")
        
        for algo in self.config['ensemble']['algorithms']:
            model_path = model_dir / algo / f"{algo}_final.zip"
            
            if model_path.exists():
                if algo == 'ppo':
                    models[algo] = PPO.load(model_path)
                elif algo == 'td3':
                    models[algo] = TD3.load(model_path)
                
                logger.info(f"  ✅ {algo.upper()} carregado: {model_path}")
            else:
                logger.warning(f"  ⚠️ {algo.upper()} não encontrado: {model_path}")
        
        if not models:
            raise ValueError("Nenhum modelo encontrado! Treine primeiro com ensemble_trainer.py")
        
        # Cria ensemble
        self.ensemble = EnsembleModel(
            models=models,
            strategy=self.config['ensemble']['strategy'],
            weights=self.config['ensemble'].get('weights', {})
        )
        
        logger.info("="*70)
    
    def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Coleta dados de mercado"""
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
        }
        interval = interval_map.get(self.timeframe, Client.KLINE_INTERVAL_15MINUTE)
        
        klines = self.client.futures_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calcula MESMAS features que o treinamento
        import talib
        
        # Returns para cada coluna
        df['open_return'] = df['open'].pct_change()
        df['high_return'] = df['high'].pct_change()
        df['low_return'] = df['low'].pct_change()
        df['close_return'] = df['close'].pct_change()
        
        # Indicadores técnicos
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
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
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_12_26_9'] = macd
        df['MACDs_12_26_9'] = signal
        df['MACDh_12_26_9'] = hist
        
        # Normaliza (min-max)
        price_cols = ['open', 'high', 'low', 'close']
        for col in df.columns:
            if col not in price_cols:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        return df.dropna()
    
    def run(self, iterations: int = 20, sleep_seconds: int = 30):
        """
        Executa loop de análise com logs detalhados
        
        Args:
            iterations: Número de iterações
            sleep_seconds: Segundos entre verificações
        """
        logger.info("="*70)
        logger.info("INICIANDO ANALISE DE DECISOES DOS MODELOS")
        logger.info("="*70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Estrategia: {self.config['ensemble']['strategy']}")
        logger.info(f"Iteracoes: {iterations}")
        logger.info("="*70)
        
        for i in range(iterations):
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERACAO {i+1}/{iterations} - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            try:
                # 1. Coleta dados
                logger.info("Coletando dados de mercado...")
                market_data = self.get_market_data(limit=100)
                current_price = market_data['close'].iloc[-1]
                price_change = market_data['close_return'].iloc[-1] * 100
                
                logger.info(f"   Preco atual: ${current_price:,.2f} ({price_change:+.2f}%)")
                logger.info(f"   Volume: {market_data['volume'].iloc[-1]:,.0f}")
                logger.info(f"   RSI: {market_data['RSI_14'].iloc[-1]:.2f}")
                logger.info(f"   SMA(20): ${market_data['SMA_20'].iloc[-1]:,.2f}")
                
                # 2. Prepara observação (50 timesteps x 23 features)
                # IMPORTANTE: O environment usa TODAS as 20 features numéricas do CSV
                # (incluindo OHLCV + indicadores técnicos)
                
                # Pega últimos 50 timesteps de TODAS as features
                obs_data = market_data.iloc[-50:].values  # Shape: (50, 20)
                
                logger.info(f"\nEstado do mercado:")
                logger.info(f"   Features do CSV: {obs_data.shape[1]}")
                logger.info(f"   Shape base: {obs_data.shape}")
                
                # Environment adiciona 3 features de portfolio durante treinamento:
                # - balance / initial_balance (normalizado)
                # - position (0 = FLAT, 1 = LONG, -1 = SHORT)
                # - equity / initial_balance (normalizado)
                
                # Para inferência, simulamos essas 3 features com valores neutros
                balance_feat = np.ones((50, 1)) * 1.0  # 100% do saldo inicial
                position_feat = np.zeros((50, 1))  # FLAT (sem posição)
                equity_feat = np.ones((50, 1)) * 1.0  # Equity = saldo
                
                # Concatena: 20 features do CSV + 3 portfolio = 23 features totais
                obs = np.concatenate([
                    obs_data,
                    balance_feat,
                    position_feat,
                    equity_feat
                ], axis=1)
                
                logger.info(f"   Shape final da observacao: {obs.shape}")
                logger.info(f"   Returns ultimos 5 steps: {market_data['close_return'].iloc[-5:].values}")
                logger.info(f"   Volatilidade: {market_data['close_return'].std():.4f}")
                
                # 3. Obtém decisão do ensemble
                logger.info(f"\nANALISE DOS MODELOS:")
                logger.info("-"*70)
                
                action, voting_info = self.ensemble.predict(obs)
                
                # 4. Detalhes de cada modelo
                for model_name, vote in voting_info['votes'].items():
                    confidence = voting_info['confidences'][model_name]
                    weight = voting_info['weights'][model_name]
                    
                    action_names = ['FLAT', 'LONG', 'SHORT']
                    action_str = action_names[vote]
                    
                    logger.info(f"   {model_name.upper():>4}: {action_str} (confianca: {confidence:.1%}, peso: {weight:.1%})")
                
                logger.info("-"*70)
                
                # 5. Decisão final
                action_names = ['FLAT', 'LONG', 'SHORT']
                final_action_str = action_names[action]
                
                logger.info(f"\nDECISAO FINAL: {final_action_str}")
                logger.info(f"   Estrategia: {voting_info['strategy']}")
                logger.info(f"   Acordo: {voting_info['agreement']:.1%} dos modelos concordam")
                
                # 6. Análise da decisão
                logger.info(f"\nANALISE DA DECISAO:")
                
                if action == 0:  # FLAT
                    logger.info("   Modelos optaram por NAO operar")
                    logger.info("   Possiveis motivos:")
                    
                    if abs(price_change) < 0.1:
                        logger.info("      * Mercado lateral (movimento < 0.1%)")
                    
                    if market_data['close_return'].std() < 0.001:
                        logger.info("      * Baixa volatilidade")
                    
                    logger.info("      * Modelos nao identificaram oportunidade clara")
                    logger.info("      * Treinamento pode estar conservador (50k timesteps)")
                    
                elif action == 1:  # LONG
                    logger.info("   Modelos identificaram oportunidade de COMPRA")
                    logger.info("   Sinais detectados:")
                    
                    if price_change > 0:
                        logger.info(f"      * Preco subindo (+{price_change:.2f}%)")
                    
                    if current_price > market_data['SMA_20'].iloc[-1]:
                        logger.info("      * Preco acima da SMA(20) - tendencia de alta")
                    
                elif action == 2:  # SHORT
                    logger.info("   Modelos identificaram oportunidade de VENDA")
                    logger.info("   Sinais detectados:")
                    
                    if price_change < 0:
                        logger.info(f"      * Preco caindo ({price_change:.2f}%)")
                    
                    if current_price < market_data['SMA_20'].iloc[-1]:
                        logger.info("      * Preco abaixo da SMA(20) - tendencia de queda")
                
                # 7. Estatísticas dos votos
                logger.info(f"\nDISTRIBUICAO DOS VOTOS:")
                vote_counts = {}
                for vote in voting_info['votes'].values():
                    vote_counts[vote] = vote_counts.get(vote, 0) + 1
                
                for action_idx, count in sorted(vote_counts.items()):
                    pct = count / len(voting_info['votes']) * 100
                    logger.info(f"   {action_names[action_idx]}: {count} voto(s) ({pct:.0f}%)")
                
                logger.info("\n" + "="*70)
                logger.info(f"Aguardando {sleep_seconds}s ate proxima analise...")
                logger.info("="*70)
                
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("\n\nAnalise interrompida pelo usuario")
                break
                
            except Exception as e:
                logger.error(f"Erro: {e}")
                logger.exception(e)
        
        logger.info("\n" + "="*70)
        logger.info("ANALISE CONCLUIDA")
        logger.info("="*70)
        logger.info(f"\nCONCLUSOES:")
        logger.info(f"   * Se os modelos sempre votam FLAT:")
        logger.info(f"     - Foram treinados com apenas 50k timesteps")
        logger.info(f"     - Precisam de mais treinamento (200k+) para identificar padroes")
        logger.info(f"     - Ou ajustar reward function para incentivar acao")
        logger.info(f"\n   * Para retreinar com mais timesteps:")
        logger.info(f"     - Edite config.yaml: total_timesteps: 200000")
        logger.info(f"     - Execute: python -m src.training.ensemble_trainer")
        logger.info(f"     - Tempo estimado: ~30 minutos com GPU")


if __name__ == "__main__":
    trader = VerboseTrader()
    
    # Executa 20 análises com intervalo de 30s (~10 minutos total)
    trader.run(iterations=20, sleep_seconds=30)
