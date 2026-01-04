"""
LLM Analyzer

Usa Large Language Models (GPT-4, Claude, ou modelos locais)
para analisar sentimento de notícias sobre mercado cripto.
"""

import os
import logging
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Escala de sentimento"""
    VERY_BEARISH = -1.0
    BEARISH = -0.5
    NEUTRAL = 0.0
    BULLISH = 0.5
    VERY_BULLISH = 1.0


class LLMAnalyzer:
    """
    Analisa sentimento de notícias usando LLMs.
    
    Suporta 3 backends:
    1. OpenAI (GPT-4/GPT-3.5) - Recomendado
    2. Anthropic (Claude) - Alternativa
    3. Local (FinBERT) - Gratuito mas menos preciso
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config:
                - provider: 'openai', 'anthropic', ou 'local'
                - model: Nome do modelo específico
                - temperature: Criatividade (0.0-1.0)
        """
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.model_name = config.get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.3)
        
        self.client = None
        self._setup_client()
        
    def _setup_client(self):
        """Configura cliente LLM baseado no provider"""
        
        if self.provider == 'openai':
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY não encontrada no .env")
                
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI configurado ({self.model_name})")
                
            except Exception as e:
                logger.error(f"❌ Erro ao configurar OpenAI: {e}")
                self._fallback_to_local()
                
        elif self.provider == 'anthropic':
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY não encontrada no .env")
                
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✅ Anthropic configurado ({self.model_name})")
                
            except Exception as e:
                logger.error(f"❌ Erro ao configurar Anthropic: {e}")
                self._fallback_to_local()
                
        else:  # local
            self._setup_local_model()
    
    def _fallback_to_local(self):
        """Fallback para modelo local se APIs falharem"""
        logger.warning("⚠️ Usando modelo local como fallback")
        self.provider = 'local'
        self._setup_local_model()
    
    def _setup_local_model(self):
        """Configura modelo local (FinBERT)"""
        try:
            from transformers import pipeline
            
            logger.info("📥 Carregando FinBERT...")
            self.client = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
            logger.info("✅ FinBERT carregado")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar FinBERT: {e}")
            self.client = None
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Analisa sentimento de uma notícia
        
        Args:
            article: Dicionário com 'title', 'description', 'content'
            
        Returns:
            {
                'sentiment_score': float (-1.0 a 1.0),
                'sentiment_label': str,
                'confidence': float (0.0 a 1.0),
                'reasoning': str,
                'topics': List[str]
            }
        """
        if not self.client:
            return self._neutral_sentiment()
        
        # Prepara texto para análise
        text = self._prepare_text(article)
        
        # Analisa baseado no provider
        if self.provider == 'openai':
            return self._analyze_with_openai(text, article['title'])
        elif self.provider == 'anthropic':
            return self._analyze_with_anthropic(text, article['title'])
        else:  # local
            return self._analyze_with_local(text)
    
    def _prepare_text(self, article: Dict) -> str:
        """Combina título, descrição e conteúdo"""
        parts = []
        
        if article.get('title'):
            parts.append(f"Título: {article['title']}")
        if article.get('description'):
            parts.append(f"Descrição: {article['description']}")
        if article.get('content'):
            # Limita conteúdo a 500 caracteres para economizar tokens
            content = article['content'][:500]
            parts.append(f"Conteúdo: {content}")
        
        return '\n'.join(parts)
    
    def _analyze_with_openai(self, text: str, title: str) -> Dict:
        """Análise usando OpenAI GPT"""
        try:
            prompt = f"""Você é um especialista em análise de sentimento de mercado cripto.

Analise a seguinte notícia e determine o sentimento para traders de Bitcoin/criptomoedas:

{text}

Responda APENAS no seguinte formato JSON:
{{
    "sentiment": "very_bearish|bearish|neutral|bullish|very_bullish",
    "confidence": 0.0-1.0,
    "reasoning": "breve explicação",
    "topics": ["lista", "de", "tópicos", "principais"]
}}

Considere:
- Notícias de regulação negativa = bearish
- Adoção institucional = bullish
- Hacks/exploits = bearish
- Upgrades técnicos = bullish
- Macro economia = contextual
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Você é um analista de sentimento de mercado cripto."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Parse resposta
            import json
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown se presente
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            # Converte para formato padrão
            sentiment_map = {
                'very_bearish': -1.0,
                'bearish': -0.5,
                'neutral': 0.0,
                'bullish': 0.5,
                'very_bullish': 1.0
            }
            
            return {
                'sentiment_score': sentiment_map.get(result['sentiment'], 0.0),
                'sentiment_label': result['sentiment'],
                'confidence': result.get('confidence', 0.7),
                'reasoning': result.get('reasoning', ''),
                'topics': result.get('topics', [])
            }
            
        except Exception as e:
            logger.error(f"Erro OpenAI: {e}")
            return self._neutral_sentiment()
    
    def _analyze_with_anthropic(self, text: str, title: str) -> Dict:
        """Análise usando Anthropic Claude"""
        try:
            prompt = f"""Analise o sentimento desta notícia de criptomoedas para traders:

{text}

Retorne JSON com: sentiment (very_bearish/bearish/neutral/bullish/very_bullish), confidence (0-1), reasoning, topics."""

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            import json
            result = json.loads(message.content[0].text)
            
            sentiment_map = {
                'very_bearish': -1.0,
                'bearish': -0.5,
                'neutral': 0.0,
                'bullish': 0.5,
                'very_bullish': 1.0
            }
            
            return {
                'sentiment_score': sentiment_map.get(result['sentiment'], 0.0),
                'sentiment_label': result['sentiment'],
                'confidence': result.get('confidence', 0.7),
                'reasoning': result.get('reasoning', ''),
                'topics': result.get('topics', [])
            }
            
        except Exception as e:
            logger.error(f"Erro Anthropic: {e}")
            return self._neutral_sentiment()
    
    def _analyze_with_local(self, text: str) -> Dict:
        """Análise usando FinBERT local"""
        try:
            # FinBERT aceita max 512 tokens
            text_short = text[:512]
            
            result = self.client(text_short)[0]
            
            # FinBERT retorna: positive, negative, neutral
            label_map = {
                'positive': 0.7,    # Bullish
                'negative': -0.7,   # Bearish
                'neutral': 0.0
            }
            
            return {
                'sentiment_score': label_map.get(result['label'], 0.0),
                'sentiment_label': result['label'],
                'confidence': result['score'],
                'reasoning': f"FinBERT classificou como {result['label']}",
                'topics': []
            }
            
        except Exception as e:
            logger.error(f"Erro FinBERT: {e}")
            return self._neutral_sentiment()
    
    def _neutral_sentiment(self) -> Dict:
        """Retorna sentimento neutro quando análise falha"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'reasoning': 'Análise não disponível',
            'topics': []
        }
    
    def analyze_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Analisa múltiplas notícias em lote
        
        Returns:
            Lista com resultados de análise para cada artigo
        """
        logger.info(f"🧠 Analisando {len(articles)} notícias...")
        
        results = []
        for i, article in enumerate(articles):
            if i % 10 == 0:
                logger.info(f"  Progresso: {i}/{len(articles)}")
            
            analysis = self.analyze_article(article)
            
            # Adiciona metadados do artigo
            analysis['article_title'] = article.get('title', '')
            analysis['article_url'] = article.get('url', '')
            analysis['published_at'] = article.get('published_at')
            
            results.append(analysis)
        
        logger.info(f"✅ Análise concluída")
        return results
    
    def get_aggregate_sentiment(self, analyses: List[Dict]) -> Dict:
        """
        Calcula sentimento agregado ponderado por confiança
        
        Returns:
            {
                'weighted_score': float,
                'average_score': float,
                'distribution': Dict,
                'confidence': float
            }
        """
        if not analyses:
            return {
                'weighted_score': 0.0,
                'average_score': 0.0,
                'distribution': {},
                'confidence': 0.0
            }
        
        # Score ponderado por confiança
        total_weight = sum(a['confidence'] for a in analyses)
        if total_weight > 0:
            weighted_score = sum(
                a['sentiment_score'] * a['confidence'] 
                for a in analyses
            ) / total_weight
        else:
            weighted_score = 0.0
        
        # Score médio simples
        average_score = sum(a['sentiment_score'] for a in analyses) / len(analyses)
        
        # Distribuição de sentimentos
        from collections import Counter
        labels = [a['sentiment_label'] for a in analyses]
        distribution = dict(Counter(labels))
        
        # Confiança média
        avg_confidence = sum(a['confidence'] for a in analyses) / len(analyses)
        
        return {
            'weighted_score': weighted_score,
            'average_score': average_score,
            'distribution': distribution,
            'confidence': avg_confidence,
            'total_articles': len(analyses)
        }


if __name__ == '__main__':
    # Teste
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'provider': 'local',  # Começa com local para teste
        'model': 'gpt-3.5-turbo'
    }
    
    analyzer = LLMAnalyzer(config)
    
    # Testa análise
    test_article = {
        'title': 'Bitcoin Surges to New All-Time High Above $100K',
        'description': 'Bitcoin reached a new milestone today as institutional adoption accelerates.',
        'content': 'Major financial institutions are now allocating significant portions of their portfolios to Bitcoin...'
    }
    
    result = analyzer.analyze_article(test_article)
    
    print("\n🧠 Análise de Sentimento:")
    print(f"Score: {result['sentiment_score']}")
    print(f"Label: {result['sentiment_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Reasoning: {result['reasoning']}")
