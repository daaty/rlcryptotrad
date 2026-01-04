"""
🚀 Teste do Sistema Ensemble Completo
- Coleta notícias em tempo real
- Analisa sentimento com GPT-4o-mini
- Usa ensemble voting (PPO + SAC + TD3)
- Simula trading em paper mode
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_news_and_sentiment():
    """Testa coleta de notícias e análise de sentimento"""
    logger.info("📰 Testando coleta de notícias + sentimento...")
    
    from src.sentiment.news_collector import NewsCollector
    from src.sentiment.llm_analyzer import LLMSentimentAnalyzer
    from src.sentiment.sentiment_processor import SentimentProcessor
    
    # 1. Coleta notícias
    collector = NewsCollector()
    news = collector.collect_all(hours=6)
    logger.info(f"  ✅ {len(news)} notícias coletadas")
    
    if len(news) == 0:
        logger.warning("  ⚠️  Nenhuma notícia encontrada. Pulando análise.")
        return None
    
    # 2. Analisa sentimento
    analyzer = LLMSentimentAnalyzer()
    sentiment_data = []
    
    for i, article in enumerate(news[:3]):  # Testa com 3 primeiras
        logger.info(f"  🔍 Analisando {i+1}/3: {article['title'][:60]}...")
        result = analyzer.analyze_article(article)
        sentiment_data.append(result)
        logger.info(f"    Sentimento: {result['sentiment_score']:.2f} | Confiança: {result['confidence']:.2f}")
    
    # 3. Processa features
    processor = SentimentProcessor()
    features = processor.get_feature_vector()
    logger.info(f"  ✅ Features de sentimento: {features.shape}")
    
    return features


def test_ensemble_models():
    """Testa carregamento dos modelos do ensemble"""
    logger.info("\n🤖 Testando modelos ensemble...")
    
    from src.models.ensemble_model import EnsembleModel
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    try:
        ensemble = EnsembleModel(config=config)
        logger.info(f"  ✅ Ensemble carregado: {len(ensemble.models)} modelos")
        logger.info(f"  Estratégia: {ensemble.strategy}")
        logger.info(f"  Modelos: {list(ensemble.models.keys())}")
        return ensemble
    except Exception as e:
        logger.error(f"  ❌ Erro ao carregar ensemble: {e}")
        return None


def test_prediction():
    """Testa predição do ensemble"""
    logger.info("\n🎯 Testando predição...")
    
    import numpy as np
    from src.models.ensemble_model import EnsembleModel
    import yaml
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    ensemble = EnsembleModel(config=config)
    
    # Cria observação dummy (50 timesteps x 24 features)
    dummy_obs = np.random.randn(50, 24)
    
    # Faz predição
    action, voting_info = ensemble.predict(dummy_obs)
    
    logger.info(f"  ✅ Ação final: {action}")
    logger.info(f"  Votos individuais:")
    for algo, vote in voting_info['individual_votes'].items():
        logger.info(f"    - {algo.upper()}: {vote}")
    logger.info(f"  Estratégia usada: {voting_info['strategy']}")


def test_full_simulation():
    """Simula 1 hora de trading com o sistema completo"""
    logger.info("\n🎮 SIMULAÇÃO COMPLETA (1 iteração)...")
    logger.info("="*60)
    
    from src.execution.ensemble_executor import EnsembleExecutor
    
    executor = EnsembleExecutor(mode='paper')
    
    logger.info("\n1️⃣ Coletando sentimento...")
    sentiment = executor.get_sentiment_features()
    logger.info(f"   Sentimento médio 24h: {sentiment[2]:.3f}")
    logger.info(f"   Tendência: {sentiment[3]:.3f}")
    logger.info(f"   Confiança: {sentiment[5]:.3f}")
    
    logger.info("\n2️⃣ Obtendo dados de mercado...")
    obs = executor.get_observation()
    logger.info(f"   Observação shape: {obs.shape}")
    logger.info(f"   Preço BTC: ${obs[-1][3]:.2f}")  # close price
    
    logger.info("\n3️⃣ Ensemble votando...")
    action, voting_info = executor.ensemble.predict(obs)
    logger.info(f"   Votos: {voting_info['individual_votes']}")
    logger.info(f"   Ação final: {action} ({['FLAT', 'LONG', 'SHORT'][action]})")
    
    logger.info("\n4️⃣ Executando ação...")
    result = executor.execute_action(action)
    logger.info(f"   Status: {result['status']}")
    logger.info(f"   Posição: {result['position']}")
    logger.info(f"   Balance: ${result['balance']:.2f}")
    
    logger.info("\n✅ Simulação completa!")


def main():
    logger.info("="*60)
    logger.info("🧪 TESTE COMPLETO DO SISTEMA ENSEMBLE + LLM")
    logger.info("="*60)
    
    # 1. Notícias + Sentimento
    try:
        sentiment_features = test_news_and_sentiment()
    except Exception as e:
        logger.error(f"❌ Erro no teste de sentimento: {e}")
        sentiment_features = None
    
    # 2. Ensemble
    try:
        ensemble = test_ensemble_models()
    except Exception as e:
        logger.error(f"❌ Erro no teste do ensemble: {e}")
        ensemble = None
    
    # 3. Predição
    try:
        test_prediction()
    except Exception as e:
        logger.error(f"❌ Erro no teste de predição: {e}")
    
    # 4. Simulação completa
    try:
        test_full_simulation()
    except Exception as e:
        logger.error(f"❌ Erro na simulação: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("✅ TESTES CONCLUÍDOS!")
    logger.info("="*60)
    logger.info("\n📋 Próximos passos:")
    logger.info("1. python -m src.execution.ensemble_executor  # Paper trading contínuo")
    logger.info("2. streamlit run dashboard.py  # Visualizar em tempo real")
    logger.info("3. Ajustar config.yaml e retreinar se necessário")


if __name__ == "__main__":
    main()
