"""
Sentiment Analysis Module

Coleta notícias e analisa sentimento usando LLMs para enriquecer
as decisões do agente de trading.
"""

from .news_collector import NewsCollector
from .llm_analyzer import LLMAnalyzer
from .sentiment_processor import SentimentProcessor

__all__ = ['NewsCollector', 'LLMAnalyzer', 'SentimentProcessor']
