"""
🚀 Script de Teste Rápido

Testa todos os componentes principais do sistema.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Testa se todas as dependências estão instaladas"""
    logger.info("🔍 Testando imports...")
    
    try:
        import gymnasium
        import stable_baselines3
        import ccxt
        import talib
        logger.info("  ✅ RL libs OK")
    except ImportError as e:
        logger.error(f"  ❌ Erro RL libs: {e}")
        return False
    
    try:
        import openai
        logger.info("  ✅ OpenAI OK")
    except ImportError:
        logger.warning("  ⚠️  OpenAI não instalado (opcional)")
    
    try:
        import anthropic
        logger.info("  ✅ Anthropic OK")
    except ImportError:
        logger.warning("  ⚠️  Anthropic não instalado (opcional)")
    
    try:
        from transformers import pipeline
        logger.info("  ✅ Transformers OK (FinBERT)")
    except ImportError:
        logger.warning("  ⚠️  Transformers não instalado (fallback)")
    
    return True


def test_config():
    """Testa se config.yaml existe e é válido"""
    logger.info("🔍 Testando configuração...")
    
    import yaml
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'environment', 'ensemble', 'sentiment']
        for key in required_keys:
            if key not in config:
                logger.error(f"  ❌ Faltando '{key}' no config.yaml")
                return False
        
        logger.info("  ✅ config.yaml OK")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Erro ao ler config.yaml: {e}")
        return False


def test_environment():
    """Testa se .env está configurado"""
    logger.info("🔍 Testando variáveis de ambiente...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    binance_key = os.getenv('BINANCE_API_KEY')
    if not binance_key or binance_key == 'your_api_key_here':
        logger.warning("  ⚠️  BINANCE_API_KEY não configurada")
    else:
        logger.info("  ✅ BINANCE_API_KEY OK")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.warning("  ⚠️  OPENAI_API_KEY não configurada (usará FinBERT)")
    else:
        logger.info("  ✅ OPENAI_API_KEY OK")
    
    newsapi_key = os.getenv('NEWSAPI_KEY')
    if not newsapi_key:
        logger.warning("  ⚠️  NEWSAPI_KEY não configurada (usará apenas RSS)")
    else:
        logger.info("  ✅ NEWSAPI_KEY OK")
    
    return True


def test_sentiment_basic():
    """Testa coleta de notícias básica (RSS)"""
    logger.info("🔍 Testando coleta de notícias...")
    
    try:
        from src.sentiment.news_collector import NewsCollector
        
        config = {
            'keywords': ['bitcoin'],
            'rss_feeds': ['https://cointelegraph.com/rss']
        }
        
        collector = NewsCollector(config)
        news = collector.collect_all(hours=24)
        
        logger.info(f"  ✅ Coletadas {len(news)} notícias")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Erro: {e}")
        return False


def test_environment_creation():
    """Testa criação do TradingEnv"""
    logger.info("🔍 Testando environment...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.environment.trading_env import TradingEnv
        
        # Cria dados fake
        df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'volume': np.random.randn(200),
            'rsi': np.random.randn(200),
            'sma_20': np.random.randn(200),
        })
        
        env = TradingEnv(df=df)
        obs, info = env.reset()
        
        logger.info(f"  ✅ Environment criado (obs shape: {obs.shape})")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Erro: {e}")
        return False


def test_directories():
    """Verifica se diretórios necessários existem"""
    logger.info("🔍 Verificando diretórios...")
    
    required_dirs = ['data', 'models', 'logs', 'src']
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            logger.info(f"  ✅ {dir_name}/ OK")
        else:
            logger.warning(f"  ⚠️  {dir_name}/ não existe, criando...")
            path.mkdir(parents=True, exist_ok=True)
    
    return True


def main():
    """Executa todos os testes"""
    logger.info("="*60)
    logger.info("🧪 TESTE COMPLETO DO SISTEMA")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuração", test_config),
        ("Variáveis de Ambiente", test_environment),
        ("Diretórios", test_directories),
        ("Trading Environment", test_environment_creation),
        ("Coleta de Notícias", test_sentiment_basic),
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"\n{'='*60}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ Erro em {name}: {e}")
            results.append((name, False))
    
    # Resumo
    logger.info(f"\n{'='*60}")
    logger.info("📊 RESUMO DOS TESTES")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 RESULTADO: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("✅ Sistema pronto para uso!")
        logger.info("\nPróximos passos:")
        logger.info("1. python -m src.data.data_collector  # Coleta dados")
        logger.info("2. python -m src.training.ensemble_trainer  # Treina modelos")
        logger.info("3. python -m src.execution.ensemble_executor  # Trading!")
    else:
        logger.warning("⚠️  Alguns testes falharam. Verifique os erros acima.")
        logger.info("\nDicas:")
        logger.info("- Instale dependências: pip install -r requirements.txt")
        logger.info("- Configure .env com suas API keys")
        logger.info("- Verifique config.yaml")
    
    logger.info(f"{'='*60}\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
