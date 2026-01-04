"""
News Collector

Coleta notícias de múltiplas fontes:
- NewsAPI (notícias globais)
- RSS Feeds (CoinDesk, CoinTelegraph, etc)
- Twitter/X (opcional, via scraping)
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests
import feedparser
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

logger = logging.getLogger(__name__)


class NewsCollector:
    """Coleta notícias relacionadas a criptomoedas e mercado financeiro"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Dicionário com configurações
                - newsapi_key: API key do NewsAPI
                - sources: Lista de fontes RSS
                - keywords: Palavras-chave para buscar
        """
        self.config = config
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.newsapi = None
        
        if self.newsapi_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.newsapi_key)
                logger.info("✅ NewsAPI configurado")
            except Exception as e:
                logger.warning(f"⚠️ NewsAPI não disponível: {e}")
        
        # RSS Feeds de crypto
        self.rss_feeds = config.get('rss_feeds', [
            'https://cointelegraph.com/rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://cryptonews.com/news/feed/',
            'https://decrypt.co/feed',
        ])
        
        self.keywords = config.get('keywords', [
            'bitcoin', 'BTC', 'cryptocurrency', 'crypto market'
        ])
        
    def collect_all(self, hours: int = 24) -> List[Dict]:
        """
        Coleta notícias de todas as fontes das últimas N horas
        
        Args:
            hours: Janela de tempo em horas
            
        Returns:
            Lista de dicionários com notícias:
            [
                {
                    'title': str,
                    'description': str,
                    'content': str,
                    'url': str,
                    'source': str,
                    'published_at': datetime,
                    'keywords': List[str]
                }
            ]
        """
        logger.info(f"📰 Coletando notícias das últimas {hours}h...")
        
        all_news = []
        
        # 1. NewsAPI (se disponível)
        if self.newsapi:
            try:
                news_api_results = self._collect_from_newsapi(hours)
                all_news.extend(news_api_results)
                logger.info(f"  ✅ NewsAPI: {len(news_api_results)} notícias")
            except Exception as e:
                logger.error(f"  ❌ Erro NewsAPI: {e}")
        
        # 2. RSS Feeds
        try:
            rss_results = self._collect_from_rss()
            all_news.extend(rss_results)
            logger.info(f"  ✅ RSS Feeds: {len(rss_results)} notícias")
        except Exception as e:
            logger.error(f"  ❌ Erro RSS: {e}")
        
        # Remove duplicatas (por URL)
        unique_news = self._deduplicate(all_news)
        
        # Filtra por janela de tempo
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered_news = [
            n for n in unique_news 
            if n['published_at'] >= cutoff_time
        ]
        
        logger.info(f"📊 Total: {len(filtered_news)} notícias únicas")
        return filtered_news
    
    def _collect_from_newsapi(self, hours: int) -> List[Dict]:
        """Coleta do NewsAPI"""
        from_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
        
        all_articles = []
        
        for keyword in self.keywords:
            try:
                response = self.newsapi.get_everything(
                    q=keyword,
                    from_param=from_date,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20
                )
                
                if response['status'] == 'ok':
                    articles = response.get('articles', [])
                    
                    for article in articles:
                        all_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                            'published_at': self._parse_datetime(article.get('publishedAt')),
                            'keywords': [keyword]
                        })
                        
            except Exception as e:
                logger.warning(f"Erro buscando '{keyword}': {e}")
                
        return all_articles
    
    def _collect_from_rss(self) -> List[Dict]:
        """Coleta de RSS feeds"""
        all_articles = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limita a 20 por feed
                    # Extrai texto completo se possível
                    content = ''
                    if hasattr(entry, 'content'):
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    
                    all_articles.append({
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'content': self._clean_html(content),
                        'url': entry.get('link', ''),
                        'source': feed.feed.get('title', feed_url),
                        'published_at': self._parse_datetime(entry.get('published')),
                        'keywords': self._extract_keywords(entry.get('title', ''))
                    })
                    
            except Exception as e:
                logger.warning(f"Erro RSS {feed_url}: {e}")
                
        return all_articles
    
    def _clean_html(self, html_text: str) -> str:
        """Remove tags HTML"""
        if not html_text:
            return ''
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(strip=True)
    
    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """Parse de data de múltiplos formatos"""
        if not date_str:
            return datetime.now(timezone.utc)
        
        try:
            # Tenta formato ISO
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            # Outros formatos comuns
            from dateutil import parser
            dt = parser.parse(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            return datetime.now(timezone.utc)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave relevantes"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords if found_keywords else ['general']
    
    def _deduplicate(self, news: List[Dict]) -> List[Dict]:
        """Remove notícias duplicadas por URL"""
        seen_urls = set()
        unique = []
        
        for article in news:
            url = article['url']
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(article)
        
        return unique
    
    def get_recent_summary(self, hours: int = 6) -> Dict:
        """
        Retorna um resumo agregado das últimas horas
        
        Returns:
            {
                'total_articles': int,
                'sources': List[str],
                'top_keywords': Dict[str, int],
                'latest_headlines': List[str]
            }
        """
        news = self.collect_all(hours=hours)
        
        sources = list(set(n['source'] for n in news))
        
        # Conta keywords
        keyword_counts = {}
        for article in news:
            for kw in article['keywords']:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        # Top 5 headlines mais recentes
        sorted_news = sorted(news, key=lambda x: x['published_at'], reverse=True)
        latest_headlines = [n['title'] for n in sorted_news[:5]]
        
        return {
            'total_articles': len(news),
            'sources': sources,
            'top_keywords': dict(sorted(keyword_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10]),
            'latest_headlines': latest_headlines
        }


if __name__ == '__main__':
    # Teste
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'keywords': ['bitcoin', 'BTC', 'cryptocurrency']
    }
    
    collector = NewsCollector(config)
    
    # Testa coleta
    news = collector.collect_all(hours=24)
    print(f"\n✅ Coletadas {len(news)} notícias")
    
    if news:
        print("\n📰 Exemplo:")
        print(f"Título: {news[0]['title']}")
        print(f"Fonte: {news[0]['source']}")
        print(f"Data: {news[0]['published_at']}")
    
    # Testa resumo
    summary = collector.get_recent_summary(hours=6)
    print(f"\n📊 Resumo 6h:")
    print(f"Total: {summary['total_articles']}")
    print(f"Fontes: {summary['sources']}")
    print(f"Top Keywords: {summary['top_keywords']}")
