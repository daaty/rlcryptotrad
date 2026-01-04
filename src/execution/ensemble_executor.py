"""
Ensemble Executor

Executa trading usando ensemble de modelos RL + análise de sentimento LLM.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import ccxt

from src.models.ensemble_model import EnsembleModel
from src.training.ensemble_trainer import EnsembleTrainer
from src.sentiment.news_collector import NewsCollector
from src.sentiment.llm_analyzer import LLMAnalyzer
from src.sentiment.sentiment_processor import SentimentProcessor
from src.risk.risk_manager import RiskManager
from src.data.data_collector import DataCollector

logger = logging.getLogger(__name__)


class EnsembleExecutor:
    """
    Executor completo com:
    - Ensemble de modelos RL (PPO + SAC + TD3)
    - Análise de sentimento com LLM
    - Gestão de risco avançada
    - Paper trading / Live trading
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Caminho para configuração
        """
        # Carrega configuração
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.symbol = self.config['data']['symbol']
        self.timeframe = self.config['data']['timeframe']
        self.mode = self.config['execution']['mode']
        self.use_ensemble = self.config['execution'].get('use_ensemble', True)
        
        # Inicializa componentes
        self._setup_exchange()
        self._setup_sentiment()
        self._setup_models()
        self._setup_risk()
        
        # Estado
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0
        self.balance = self.config['environment']['initial_balance']
        
        # Métricas
        self.trades = []
        self.sentiment_history = []
        
        logger.info("✅ EnsembleExecutor inicializado")
        logger.info(f"   Modo: {self.mode}")
        logger.info(f"   Sentimento: {self.sentiment_enabled}")
        logger.info(f"   Ensemble: {self.use_ensemble}")
    
    def _setup_exchange(self):
        """Configura conexão com exchange"""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        
        testnet = self.config['binance'].get('testnet', True)
        
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("📍 Usando Binance TESTNET")
        else:
            logger.warning("⚠️  MODO LIVE - Dinheiro real!")
        
        # Testa conexão
        try:
            self.exchange.fetch_balance()
            logger.info("✅ Conexão com Binance estabelecida")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
    
    def _setup_sentiment(self):
        """Configura análise de sentimento"""
        self.sentiment_enabled = self.config.get('sentiment', {}).get('enabled', False)
        
        if not self.sentiment_enabled:
            logger.info("⏭️  Sentimento desabilitado")
            self.news_collector = None
            self.llm_analyzer = None
            self.sentiment_processor = None
            return
        
        # News Collector
        news_config = self.config['sentiment']['news']
        self.news_collector = NewsCollector(news_config)
        
        # LLM Analyzer
        llm_config = self.config['sentiment']['llm']
        self.llm_analyzer = LLMAnalyzer(llm_config)
        
        # Sentiment Processor
        processor_config = self.config['sentiment']['processor']
        self.sentiment_processor = SentimentProcessor(processor_config)
        
        logger.info("✅ Sistema de sentimento configurado")
    
    def _setup_models(self):
        """Carrega modelos RL"""
        if self.use_ensemble:
            # Carrega ensemble
            trainer = EnsembleTrainer()
            models = trainer.load_ensemble(version='best')
            
            if not models:
                raise ValueError("❌ Nenhum modelo encontrado! Execute treinamento primeiro.")
            
            # Cria ensemble
            ensemble_config = self.config['ensemble']
            strategy = ensemble_config.get('strategy', 'weighted')
            weights = ensemble_config.get('weights')
            
            self.model = EnsembleModel(
                models=models,
                strategy=strategy,
                weights=weights
            )
            
            logger.info(f"✅ Ensemble carregado: {list(models.keys())}")
        
        else:
            # Carrega modelo único
            from stable_baselines3 import PPO, SAC, TD3
            
            model_path = self.config['execution'].get('model_path', 'models/ppo_trading_agent_best.zip')
            algo = self.config['training']['algorithm'].lower()
            
            if algo == 'ppo':
                self.model = PPO.load(model_path)
            elif algo == 'sac':
                self.model = SAC.load(model_path)
            elif algo == 'td3':
                self.model = TD3.load(model_path)
            
            logger.info(f"✅ Modelo {algo.upper()} carregado")
    
    def _setup_risk(self):
        """Configura gestão de risco"""
        self.risk_manager = RiskManager(self.config['risk_management'])
        logger.info("✅ Risk Manager configurado")
    
    def get_sentiment_features(self) -> Optional[np.ndarray]:
        """
        Coleta e processa sentimento de notícias
        
        Returns:
            Array com features de sentimento ou None
        """
        if not self.sentiment_enabled:
            return None
        
        try:
            # Coleta notícias recentes
            hours = self.config['sentiment']['news']['history_hours']
            news = self.news_collector.collect_all(hours=hours)
            
            if not news:
                logger.warning("⚠️ Nenhuma notícia coletada")
                return None
            
            # Analisa sentimento
            analyses = self.llm_analyzer.analyze_batch(news)
            
            # Processa em features
            features = self.sentiment_processor.get_feature_vector(analyses)
            
            # Salva histórico
            self.sentiment_history.append({
                'timestamp': datetime.now(),
                'features': features,
                'news_count': len(news),
                'sentiment_avg': np.mean([a['sentiment_score'] for a in analyses])
            })
            
            logger.info(f"📰 Sentimento: {features[0]:.3f} ({len(news)} notícias)")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar sentimento: {e}")
            return None
    
    def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Coleta dados de mercado atualizados"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Adiciona indicadores
            collector = DataCollector(self.config)
            df_with_indicators = collector.add_technical_indicators(df)
            df_normalized = collector.normalize_data(df_with_indicators)
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"❌ Erro ao coletar dados: {e}")
            return None
    
    def get_observation(self) -> Optional[np.ndarray]:
        """
        Prepara observação completa para o modelo:
        - Dados de mercado (OHLCV + indicadores)
        - Estado da carteira
        - Features de sentimento
        """
        # 1. Dados de mercado
        df = self.get_market_data(limit=60)  # Mais do que window_size
        if df is None or len(df) < 50:
            return None
        
        # 2. Features de sentimento
        sentiment_features = None
        if self.sentiment_enabled:
            sentiment_features = self.get_sentiment_features()
        
        # 3. Monta observação (similar ao TradingEnv)
        window_size = self.config['environment'].get('window_size', 50)
        historical_data = df.iloc[-window_size:].values
        
        # Estado da carteira
        initial_balance = self.config['environment']['initial_balance']
        portfolio_state = np.array([
            self.balance / initial_balance,
            self.position,
            self.balance / initial_balance  # equity simplificado
        ])
        
        portfolio_matrix = np.tile(portfolio_state, (window_size, 1))
        observation = np.concatenate([historical_data, portfolio_matrix], axis=1)
        
        # Adiciona sentimento se disponível
        if sentiment_features is not None:
            # Replica sentimento para toda janela
            sentiment_matrix = np.tile(sentiment_features, (window_size, 1))
            observation = np.concatenate([observation, sentiment_matrix], axis=1)
        
        return observation.astype(np.float32)
    
    def execute_action(self, action: int, current_price: float) -> bool:
        """
        Executa ação de trading
        
        Args:
            action: 0 (Flat), 1 (Long), 2 (Short)
            current_price: Preço atual
            
        Returns:
            True se executou com sucesso
        """
        # Mapeia ação
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # Se já está na posição correta
        if target_position == self.position:
            return True
        
        # Valida com risk manager
        risk_ok = self.risk_manager.validate_action(
            action=action,
            current_price=current_price,
            entry_price=self.entry_price,
            position=self.position
        )
        
        if not risk_ok:
            logger.warning(f"🚫 Ação {action} bloqueada pelo Risk Manager")
            return False
        
        # Executa
        try:
            # Fecha posição anterior se houver
            if self.position != 0:
                self._close_position(current_price)
            
            # Abre nova posição se não for Flat
            if target_position != 0:
                self._open_position(target_position, current_price)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao executar: {e}")
            return False
    
    def _open_position(self, direction: int, price: float):
        """Abre posição (Long ou Short)"""
        position_type = "LONG" if direction == 1 else "SHORT"
        
        # Calcula tamanho
        position_size_pct = self.config['environment']['position_size']
        leverage = self.config['environment']['leverage']
        
        quantity = (self.balance * position_size_pct * leverage) / price
        
        if self.mode == 'paper':
            # Paper trading - apenas simula
            self.position = direction
            self.entry_price = price
            
            logger.info(f"📈 PAPER {position_type}: {quantity:.6f} @ ${price:.2f}")
            
        else:
            # Live trading - executa ordem real
            side = 'buy' if direction == 1 else 'sell'
            
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                quantity
            )
            
            self.position = direction
            self.entry_price = order['average']
            
            logger.info(f"💰 LIVE {position_type}: {quantity:.6f} @ ${order['average']:.2f}")
        
        # Registra
        self.trades.append({
            'timestamp': datetime.now(),
            'type': 'open',
            'direction': position_type,
            'price': price,
            'quantity': quantity
        })
    
    def _close_position(self, price: float):
        """Fecha posição"""
        if self.position == 0:
            return
        
        # Calcula PnL
        price_change = (price - self.entry_price) / self.entry_price
        pnl_pct = price_change * self.position  # Long: positivo se subiu, Short: positivo se caiu
        
        position_size_pct = self.config['environment']['position_size']
        leverage = self.config['environment']['leverage']
        position_value = self.balance * position_size_pct * leverage
        
        pnl = pnl_pct * position_value
        self.balance += pnl
        
        logger.info(f"📊 FECHOU: PnL = ${pnl:.2f} ({pnl_pct*100:.2f}%)")
        
        # Registra
        self.trades.append({
            'timestamp': datetime.now(),
            'type': 'close',
            'price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        
        self.position = 0
        self.entry_price = 0
    
    def run(self, duration_hours: Optional[int] = None):
        """
        Loop principal de trading
        
        Args:
            duration_hours: Duração em horas (None = infinito)
        """
        logger.info("🚀 Iniciando trading...")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Timeframe: {self.timeframe}")
        logger.info(f"   Balance: ${self.balance:.2f}")
        
        start_time = datetime.now()
        check_interval = self.config['execution']['check_interval']
        
        try:
            iteration = 0
            
            while True:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteração {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 1. Coleta observação
                obs = self.get_observation()
                
                if obs is None:
                    logger.warning("⚠️ Observação inválida, pulando...")
                    time.sleep(check_interval)
                    continue
                
                # 2. Previsão do modelo
                action, info = self.model.predict(obs, deterministic=True)
                
                # Log detalhado para ensemble
                if self.use_ensemble:
                    logger.info(f"🤖 Votos: {info['votes']}")
                    logger.info(f"   Ação Final: {action} ({['Flat', 'Long', 'Short'][action]})")
                    logger.info(f"   Concordância: {info['agreement']:.1%}")
                else:
                    logger.info(f"🤖 Ação: {action} ({['Flat', 'Long', 'Short'][action]})")
                
                # 3. Pega preço atual
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                logger.info(f"💵 Preço: ${current_price:.2f}")
                
                # 4. Executa ação
                success = self.execute_action(action, current_price)
                
                if success:
                    logger.info(f"✅ Ação executada")
                else:
                    logger.warning(f"⚠️ Ação NÃO executada")
                
                # 5. Status
                logger.info(f"💰 Balance: ${self.balance:.2f}")
                logger.info(f"📊 Posição: {['Flat', 'Long', 'Short'][self.position + 1]}")
                
                if self.position != 0:
                    pnl_pct = ((current_price - self.entry_price) / self.entry_price) * self.position
                    logger.info(f"💹 PnL Aberto: {pnl_pct*100:.2f}%")
                
                # 6. Verifica duração
                if duration_hours:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        logger.info(f"\n⏰ Tempo limite atingido ({duration_hours}h)")
                        break
                
                # 7. Aguarda próximo ciclo
                logger.info(f"⏳ Aguardando {check_interval}s...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrompido pelo usuário")
        
        finally:
            # Fecha posição aberta
            if self.position != 0:
                ticker = self.exchange.fetch_ticker(self.symbol)
                self._close_position(ticker['last'])
            
            # Relatório final
            self.print_summary()
    
    def print_summary(self):
        """Imprime resumo da sessão"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 RESUMO DA SESSÃO")
        logger.info(f"{'='*60}")
        
        logger.info(f"Balance Final: ${self.balance:.2f}")
        logger.info(f"PnL Total: ${self.balance - self.config['environment']['initial_balance']:.2f}")
        logger.info(f"Total Trades: {len([t for t in self.trades if t['type'] == 'close'])}")
        
        # Estatísticas de trades
        closed_trades = [t for t in self.trades if t['type'] == 'close']
        if closed_trades:
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            losses = len([t for t in closed_trades if t['pnl'] < 0])
            
            logger.info(f"Wins: {wins} | Losses: {losses}")
            if len(closed_trades) > 0:
                logger.info(f"Win Rate: {wins/len(closed_trades)*100:.1f}%")
        
        # Salva logs
        if self.config.get('logging', {}).get('save_trades', True):
            df_trades = pd.DataFrame(self.trades)
            log_file = f"logs/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(log_file, index=False)
            logger.info(f"💾 Trades salvos: {log_file}")


if __name__ == '__main__':
    executor = EnsembleExecutor()
    
    # Roda por 24h (ou Ctrl+C para parar)
    executor.run(duration_hours=24)
