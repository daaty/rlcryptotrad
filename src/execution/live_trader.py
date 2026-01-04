"""
🤖 Live Trader - Trading Ao Vivo na Testnet/Live
Executa trades reais usando modelos treinados
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
from typing import Dict, Optional

from src.models.ensemble_model import EnsembleModel
from src.sentiment.news_collector import NewsCollector
from src.sentiment.llm_analyzer import LLMAnalyzer
from src.sentiment.sentiment_processor import SentimentProcessor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Executor de trading ao vivo com Ensemble RL + LLM Sentiment
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Caminho para config.yaml
        """
        # Carrega config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.mode = self.config.get('mode', 'testnet')
        self.symbol = self.config['data']['symbol'].replace('/', '')  # BTC/USDT -> BTCUSDT
        self.timeframe = self.config['data']['timeframe']
        
        # Inicializa cliente Binance
        if self.mode == 'testnet':
            self.client = Client(
                api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
                api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
                testnet=True
            )
            logger.info(f"🧪 Modo TESTNET ativado")
        elif self.mode == 'live':
            self.client = Client(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_SECRET_KEY')
            )
            logger.warning(f"⚠️  Modo LIVE ativado (DINHEIRO REAL!)")
        else:
            raise ValueError("LiveTrader requer mode='testnet' ou 'live'")
        
        # Parâmetros de trading
        self.position_size = self.config['environment']['position_size']
        self.leverage = self.config['environment']['leverage']
        self.commission = self.config['environment']['commission']
        
        # Estado atual
        self.current_position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.position_qty = 0.0
        
        # Carrega modelos ensemble
        logger.info("🤖 Carregando modelos ensemble...")
        self.ensemble = EnsembleModel(config=self.config['ensemble'])
        
        # Configura sentiment (opcional)
        self.use_sentiment = self.config.get('sentiment', {}).get('enabled', False)
        if self.use_sentiment:
            logger.info("📰 Configurando análise de sentimento...")
            self.news_collector = NewsCollector(config=self.config['sentiment']['news'])
            self.llm_analyzer = LLMAnalyzer(config=self.config['sentiment']['llm'])
            self.sentiment_processor = SentimentProcessor(config=self.config['sentiment']['processor'])
        
        logger.info("✅ LiveTrader inicializado")
    
    def get_account_balance(self) -> Dict:
        """Retorna saldo da conta"""
        balance = self.client.futures_account_balance()
        usdt = [b for b in balance if b['asset'] == 'USDT'][0]
        return {
            'available': float(usdt['availableBalance']),
            'total': float(usdt['balance']),
            'unrealized_pnl': float(usdt.get('crossUnPnl', 0))
        }
    
    def get_current_price(self) -> float:
        """Retorna preço atual do par"""
        ticker = self.client.futures_ticker(symbol=self.symbol)
        return float(ticker['lastPrice'])
    
    def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Coleta dados recentes do mercado
        
        Args:
            limit: Número de candles
            
        Returns:
            DataFrame com OHLCV + indicadores
        """
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
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
        
        # Adiciona indicadores técnicos básicos
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['volume_norm'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        return df.dropna()
    
    def get_sentiment_features(self) -> np.ndarray:
        """
        Coleta e processa sentimento de notícias
        
        Returns:
            Array com 9 features de sentimento ou zeros
        """
        if not self.use_sentiment:
            return np.zeros(9)
        
        try:
            # Coleta notícias das últimas 6h
            news = self.news_collector.collect_all(hours=6)
            
            if len(news) == 0:
                logger.warning("⚠️ Nenhuma notícia coletada, usando sentimento neutro")
                return np.zeros(9)
            
            # Analisa até 5 notícias
            analyzed = []
            for article in news[:5]:
                result = self.llm_analyzer.analyze_article(article)
                analyzed.append(result)
            
            # Processa sentimento
            self.sentiment_processor.process(analyzed)
            features = self.sentiment_processor.get_feature_vector()
            
            logger.info(f"📊 Sentimento: {len(news)} notícias | Score: {features[0]:.2f}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar sentimento: {e}")
            return np.zeros(9)
    
    def place_order(self, side: str, quantity: float):
        """
        Coloca ordem de mercado
        
        Args:
            side: 'BUY' ou 'SELL'
            quantity: Quantidade em BTC
        """
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"✅ Ordem executada: {side} {quantity} {self.symbol}")
            return order
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem: {e}")
            return None
    
    def close_position(self):
        """Fecha posição atual"""
        if self.current_position == 0:
            return
        
        side = 'SELL' if self.current_position == 1 else 'BUY'
        self.place_order(side, abs(self.position_qty))
        
        # Calcula P&L
        current_price = self.get_current_price()
        pnl = (current_price - self.entry_price) * self.position_qty
        pnl_pct = (pnl / (self.entry_price * abs(self.position_qty))) * 100
        
        logger.info(f"💰 Posição fechada | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        self.current_position = 0
        self.entry_price = 0.0
        self.position_qty = 0.0
    
    def open_position(self, action: int):
        """
        Abre nova posição
        
        Args:
            action: 0 (flat), 1 (long), 2 (short)
        """
        if action == 0:
            return
        
        # Calcula quantidade
        balance = self.get_account_balance()
        current_price = self.get_current_price()
        position_value = balance['available'] * self.position_size
        quantity = round((position_value / current_price) * self.leverage, 3)
        
        # Abre posição
        if action == 1:  # LONG
            self.place_order('BUY', quantity)
            self.current_position = 1
            self.position_qty = quantity
        elif action == 2:  # SHORT
            self.place_order('SELL', quantity)
            self.current_position = -1
            self.position_qty = -quantity
        
        self.entry_price = current_price
        logger.info(f"📈 Posição aberta: {['FLAT', 'LONG', 'SHORT'][action]} | Qty: {quantity} | Price: ${current_price:,.2f}")
    
    def run(self, max_iterations: Optional[int] = None, sleep_seconds: int = 60):
        """
        Executa loop de trading
        
        Args:
            max_iterations: Número máximo de iterações (None = infinito)
            sleep_seconds: Segundos entre verificações
        """
        logger.info("="*70)
        logger.info("🚀 INICIANDO LIVE TRADING")
        logger.info("="*70)
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 Iteração {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")
                
                # 1. Coleta dados de mercado
                logger.info("📈 Coletando dados de mercado...")
                market_data = self.get_market_data(limit=100)
                current_price = self.get_current_price()
                
                # 2. Coleta sentimento (opcional)
                sentiment_features = self.get_sentiment_features()
                
                # 3. Prepara observação para modelos
                # Usa últimos 50 candles como estado
                obs = market_data[['returns', 'sma_20', 'volume_norm']].iloc[-50:].values.flatten()
                obs = np.concatenate([obs, sentiment_features])
                
                # 4. Obtém decisão do ensemble
                action, voting_info = self.ensemble.predict(obs)
                
                logger.info(f"🤖 Decisão Ensemble: {['FLAT', 'LONG', 'SHORT'][action]}")
                logger.info(f"   Votos: {voting_info['votes']}")
                logger.info(f"   Acordo: {voting_info['agreement']:.1f}%")
                
                # 5. Executa trade
                if action != (self.current_position + 1):  # Mudança de posição
                    if self.current_position != 0:
                        self.close_position()
                    
                    self.open_position(action)
                else:
                    logger.info(f"⏸️  Mantendo posição atual: {['FLAT', 'LONG', 'SHORT'][self.current_position + 1]}")
                
                # 6. Mostra status
                balance = self.get_account_balance()
                logger.info(f"\n💰 Status da Conta:")
                logger.info(f"   Balance: ${balance['total']:,.2f}")
                logger.info(f"   Available: ${balance['available']:,.2f}")
                logger.info(f"   Unrealized P&L: ${balance['unrealized_pnl']:,.2f}")
                logger.info(f"   BTC Price: ${current_price:,.2f}")
                
                if self.current_position != 0:
                    unrealized_pnl = (current_price - self.entry_price) * self.position_qty
                    unrealized_pct = (unrealized_pnl / (self.entry_price * abs(self.position_qty))) * 100
                    logger.info(f"   Position P&L: ${unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)")
                
                # 7. Aguarda próxima iteração
                logger.info(f"\n⏳ Aguardando {sleep_seconds}s até próxima verificação...")
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrompido pelo usuário")
            
            # Fecha posição se estiver aberta
            if self.current_position != 0:
                logger.info("📤 Fechando posição aberta...")
                self.close_position()
            
            logger.info("✅ Encerrado com segurança")
        
        except Exception as e:
            logger.error(f"\n❌ Erro fatal: {e}")
            
            # Fecha posição se estiver aberta
            if self.current_position != 0:
                logger.info("📤 Fechando posição por segurança...")
                self.close_position()
            
            raise


if __name__ == "__main__":
    trader = LiveTrader()
    
    # Executa trading ao vivo
    # max_iterations=100 = 100 verificações (100 * 60s = ~1h40min)
    # sleep_seconds=60 = verifica a cada 1 minuto
    trader.run(max_iterations=None, sleep_seconds=60)
