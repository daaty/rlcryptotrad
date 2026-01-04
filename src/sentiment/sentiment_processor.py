"""
Sentiment Processor

Processa análises de sentimento e transforma em features numéricas
para o agente de RL.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class SentimentProcessor:
    """
    Transforma sentimento de notícias em features para o agente RL.
    
    Gera múltiplas features:
    - Sentimento agregado recente (1h, 6h, 24h)
    - Tendência de sentimento (melhorando/piorando)
    - Volatilidade de sentimento
    - Distribuição de tópicos
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config:
                - lookback_hours: Lista de janelas temporais [1, 6, 24]
                - decay_factor: Peso temporal (mais recente = mais importante)
        """
        self.config = config
        self.lookback_hours = config.get('lookback_hours', [1, 6, 24])
        self.decay_factor = config.get('decay_factor', 0.95)
        
        # Cache de sentimentos processados
        self.sentiment_history = []
        
    def process(self, sentiment_analyses: List[Dict]) -> Dict:
        """
        Processa análises de sentimento em features numéricas
        
        Args:
            sentiment_analyses: Lista de análises do LLMAnalyzer
            
        Returns:
            {
                'sentiment_1h': float,
                'sentiment_6h': float,
                'sentiment_24h': float,
                'sentiment_trend': float,
                'sentiment_volatility': float,
                'sentiment_confidence': float,
                'bullish_ratio': float,
                'bearish_ratio': float,
                'news_volume': int
            }
        """
        if not sentiment_analyses:
            return self._empty_features()
        
        # Adiciona ao histórico
        self._update_history(sentiment_analyses)
        
        # Calcula features
        features = {}
        
        # 1. Sentimento agregado por janela temporal
        for hours in self.lookback_hours:
            key = f'sentiment_{hours}h'
            features[key] = self._calculate_windowed_sentiment(hours)
        
        # 2. Tendência (sentimento 1h vs 24h)
        features['sentiment_trend'] = self._calculate_trend()
        
        # 3. Volatilidade
        features['sentiment_volatility'] = self._calculate_volatility()
        
        # 4. Confiança média
        features['sentiment_confidence'] = np.mean([
            a['confidence'] for a in sentiment_analyses
        ])
        
        # 5. Distribuição bullish/bearish
        distribution = self._calculate_distribution(sentiment_analyses)
        features.update(distribution)
        
        # 6. Volume de notícias
        features['news_volume'] = len(sentiment_analyses)
        
        logger.info(f"✅ Features geradas: {len(features)} dimensões")
        return features
    
    def _update_history(self, new_analyses: List[Dict]):
        """Atualiza histórico de sentimentos"""
        for analysis in new_analyses:
            self.sentiment_history.append({
                'timestamp': analysis.get('published_at', datetime.now()),
                'score': analysis['sentiment_score'],
                'confidence': analysis['confidence']
            })
        
        # Remove entradas antigas (> 7 dias)
        cutoff = datetime.now() - timedelta(days=7)
        self.sentiment_history = [
            h for h in self.sentiment_history 
            if h['timestamp'] >= cutoff
        ]
    
    def _calculate_windowed_sentiment(self, hours: int) -> float:
        """
        Calcula sentimento médio nas últimas N horas com decay temporal
        
        Notícias mais recentes têm peso maior
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filtra janela temporal
        window_data = [
            h for h in self.sentiment_history 
            if h['timestamp'] >= cutoff
        ]
        
        if not window_data:
            return 0.0
        
        # Aplica decay temporal (mais recente = mais peso)
        now = datetime.now()
        weighted_scores = []
        weights = []
        
        for data in window_data:
            # Calcula idade em horas
            age_hours = (now - data['timestamp']).total_seconds() / 3600
            
            # Peso decresce exponencialmente com idade
            weight = np.exp(-age_hours / (hours * self.decay_factor))
            
            # Pondera também por confiança da análise
            final_weight = weight * data['confidence']
            
            weighted_scores.append(data['score'] * final_weight)
            weights.append(final_weight)
        
        # Média ponderada
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(weighted_scores) / total_weight
        else:
            return 0.0
    
    def _calculate_trend(self) -> float:
        """
        Calcula tendência de sentimento (melhorando ou piorando)
        
        Compara sentimento recente (1h) com período anterior (24h)
        
        Returns:
            -1.0 a 1.0 (negativo = piorando, positivo = melhorando)
        """
        recent = self._calculate_windowed_sentiment(1)
        baseline = self._calculate_windowed_sentiment(24)
        
        if baseline == 0:
            return 0.0
        
        # Mudança percentual normalizada
        change = (recent - baseline) / (abs(baseline) + 0.01)
        
        # Limita a [-1, 1]
        return np.clip(change, -1.0, 1.0)
    
    def _calculate_volatility(self) -> float:
        """
        Calcula volatilidade de sentimento (quão instável está)
        
        Alta volatilidade = incerteza de mercado
        
        Returns:
            0.0 a 1.0
        """
        if len(self.sentiment_history) < 2:
            return 0.0
        
        # Últimas 24h
        cutoff = datetime.now() - timedelta(hours=24)
        recent_scores = [
            h['score'] for h in self.sentiment_history 
            if h['timestamp'] >= cutoff
        ]
        
        if len(recent_scores) < 2:
            return 0.0
        
        # Desvio padrão normalizado
        std = np.std(recent_scores)
        
        # Normaliza para [0, 1] (assumindo scores entre -1 e 1)
        return min(std / 1.0, 1.0)
    
    def _calculate_distribution(self, analyses: List[Dict]) -> Dict:
        """
        Calcula distribuição bullish/bearish/neutral
        
        Returns:
            {
                'bullish_ratio': float,
                'bearish_ratio': float,
                'neutral_ratio': float
            }
        """
        total = len(analyses)
        if total == 0:
            return {
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'neutral_ratio': 1.0
            }
        
        bullish = sum(1 for a in analyses if a['sentiment_score'] > 0.3)
        bearish = sum(1 for a in analyses if a['sentiment_score'] < -0.3)
        neutral = total - bullish - bearish
        
        return {
            'bullish_ratio': bullish / total,
            'bearish_ratio': bearish / total,
            'neutral_ratio': neutral / total
        }
    
    def _empty_features(self) -> Dict:
        """Retorna features vazias quando não há dados"""
        features = {
            'sentiment_confidence': 0.0,
            'sentiment_trend': 0.0,
            'sentiment_volatility': 0.0,
            'bullish_ratio': 0.0,
            'bearish_ratio': 0.0,
            'neutral_ratio': 1.0,
            'news_volume': 0
        }
        
        # Adiciona janelas temporais
        for hours in self.lookback_hours:
            features[f'sentiment_{hours}h'] = 0.0
        
        return features
    
    def get_feature_vector(self, analyses: List[Dict]) -> np.ndarray:
        """
        Retorna features como vetor numpy normalizado para o agente RL
        
        Returns:
            Array 1D com features normalizadas
        """
        features = self.process(analyses)
        
        # Ordem fixa de features
        feature_names = [
            'sentiment_1h',
            'sentiment_6h', 
            'sentiment_24h',
            'sentiment_trend',
            'sentiment_volatility',
            'sentiment_confidence',
            'bullish_ratio',
            'bearish_ratio',
            'news_volume'
        ]
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            
            # Normaliza news_volume (log scale)
            if name == 'news_volume':
                value = np.log1p(value) / 10.0  # Assume max ~20000 notícias
            
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)
    
    def get_feature_description(self) -> List[str]:
        """Retorna nomes das features para debug"""
        return [
            'sentiment_1h',
            'sentiment_6h',
            'sentiment_24h',
            'sentiment_trend',
            'sentiment_volatility',
            'sentiment_confidence',
            'bullish_ratio',
            'bearish_ratio',
            'news_volume'
        ]
    
    def save_history(self, filepath: str):
        """Salva histórico de sentimentos"""
        df = pd.DataFrame(self.sentiment_history)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Histórico salvo: {filepath}")
    
    def load_history(self, filepath: str):
        """Carrega histórico de sentimentos"""
        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            self.sentiment_history = df.to_dict('records')
            logger.info(f"📂 Histórico carregado: {len(self.sentiment_history)} entradas")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar histórico: {e}")


if __name__ == '__main__':
    # Teste
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'lookback_hours': [1, 6, 24],
        'decay_factor': 0.95
    }
    
    processor = SentimentProcessor(config)
    
    # Simula análises
    test_analyses = [
        {
            'sentiment_score': 0.8,
            'confidence': 0.9,
            'published_at': datetime.now() - timedelta(minutes=30)
        },
        {
            'sentiment_score': -0.3,
            'confidence': 0.7,
            'published_at': datetime.now() - timedelta(hours=2)
        },
        {
            'sentiment_score': 0.5,
            'confidence': 0.8,
            'published_at': datetime.now() - timedelta(hours=12)
        }
    ]
    
    # Processa
    features = processor.process(test_analyses)
    
    print("\n📊 Features Geradas:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Vetor para RL
    vector = processor.get_feature_vector(test_analyses)
    print(f"\n🤖 Vetor RL: {vector}")
    print(f"Dimensões: {len(vector)}")
