"""
Ensemble Model

Combina previsões de múltiplos modelos RL (PPO, SAC, TD3)
usando diferentes estratégias de votação.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Estratégias de combinação de previsões"""
    MAJORITY = 'majority'  # Votação majoritária simples
    WEIGHTED = 'weighted'  # Ponderado por performance histórica
    CONFIDENCE = 'confidence'  # Ponderado por confiança do modelo
    BEST = 'best'  # Usa apenas o melhor modelo
    AVERAGE = 'average'  # Média das ações


class EnsembleModel:
    """
    Combina previsões de múltiplos modelos RL para decisão final.
    
    Suporta diferentes estratégias:
    - Majority: Ação mais votada
    - Weighted: Ponderado por performance (Sharpe ratio)
    - Confidence: Ponderado por confiança (valor de ação)
    - Best: Usa apenas melhor modelo
    - Average: Média das probabilidades de ação
    """
    
    def __init__(
        self,
        models: Dict,
        strategy: str = 'weighted',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            models: {
                'ppo': model,
                'sac': model,
                'td3': model
            }
            strategy: Estratégia de votação
            weights: Pesos personalizados por modelo (para strategy='weighted')
        """
        self.models = models
        self.strategy = VotingStrategy(strategy)
        
        # Pesos para votação ponderada
        if weights:
            self.weights = weights
        else:
            # Pesos padrão (iguais)
            self.weights = {name: 1.0 for name in models.keys()}
        
        # Normaliza pesos
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Estatísticas de performance
        self.model_stats = {name: {
            'predictions': 0,
            'correct': 0,
            'total_reward': 0.0
        } for name in models.keys()}
        
        logger.info(f"✅ Ensemble criado: {list(models.keys())}")
        logger.info(f"   Estratégia: {strategy}")
        logger.info(f"   Pesos: {self.weights}")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Dict]:
        """
        Faz previsão usando ensemble de modelos
        
        Args:
            observation: Estado atual do environment
            deterministic: Se True, ações são determinísticas
            
        Returns:
            (action, info) onde:
                action: Ação escolhida (0=Flat, 1=Long, 2=Short)
                info: {
                    'votes': {modelo: ação},
                    'confidences': {modelo: confiança},
                    'final_action': ação escolhida,
                    'agreement': concordância entre modelos
                }
        """
        # Coleta previsões de todos os modelos
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                action, _states = model.predict(observation, deterministic=deterministic)
                
                # Extrai ação (pode ser array ou escalar)
                if isinstance(action, np.ndarray):
                    action = float(action[0])
                else:
                    action = float(action)
                
                # Converte action contínuo [-1, 1] para discreto [0, 1, 2]
                if action < -0.33:
                    discrete_action = 2  # Short
                elif action > 0.33:
                    discrete_action = 1  # Long
                else:
                    discrete_action = 0  # Flat
                
                predictions[name] = discrete_action
                
                # Calcula confiança (valor da política)
                # Para modelos discretos, usa probabilidade da ação
                # Para contínuos, usa magnitude da ação
                confidence = self._calculate_confidence(model, observation, discrete_action)
                confidences[name] = confidence
                
            except Exception as e:
                logger.warning(f"⚠️ Erro em {name}: {e}")
                # Fallback para ação neutra
                predictions[name] = 0
                confidences[name] = 0.0
        
        # Salva confidences para uso no desempate
        self._last_confidences = confidences
        
        # Combina previsões baseado na estratégia
        final_action = self._combine_predictions(predictions, confidences)
        
        # Calcula concordância (% de modelos que concordam)
        agreement = self._calculate_agreement(predictions, final_action)
        
        # Info para debugging
        info = {
            'votes': predictions,
            'confidences': confidences,
            'weights': self.weights,
            'final_action': final_action,
            'agreement': agreement,
            'strategy': self.strategy.value
        }
        
        # Atualiza estatísticas
        for name in predictions.keys():
            self.model_stats[name]['predictions'] += 1
        
        return final_action, info
    
    def _combine_predictions(
        self,
        predictions: Dict[str, int],
        confidences: Dict[str, float]
    ) -> int:
        """Combina previsões baseado na estratégia"""
        
        if self.strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(predictions)
        
        elif self.strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(predictions)
        
        elif self.strategy == VotingStrategy.CONFIDENCE:
            return self._confidence_vote(predictions, confidences)
        
        elif self.strategy == VotingStrategy.BEST:
            return self._best_model_vote(predictions)
        
        elif self.strategy == VotingStrategy.AVERAGE:
            return self._average_vote(predictions, confidences)
        
        else:
            # Fallback
            return self._majority_vote(predictions)
    
    def _majority_vote(self, predictions: Dict[str, int]) -> int:
        """Votação majoritária simples"""
        from collections import Counter
        
        votes = list(predictions.values())
        vote_counts = Counter(votes)
        
        # Retorna ação mais votada
        most_common = vote_counts.most_common(1)[0]
        return most_common[0]
    
    def _weighted_vote(self, predictions: Dict[str, int]) -> int:
        """Votação ponderada por pesos pré-definidos com desempate por confiança"""
        # Soma pesos para cada ação
        action_weights = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for model_name, action in predictions.items():
            weight = self.weights.get(model_name, 0.0)
            action_weights[action] += weight
        
        # Verifica se há empate (diferença < 0.01)
        sorted_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)
        if abs(sorted_actions[0][1] - sorted_actions[1][1]) < 0.01:
            # EMPATE! Usa confiança como desempate
            logger.info("   ⚖️ Empate detectado! Usando confiança para desempatar...")
            return self._confidence_vote(predictions, self._last_confidences)
        
        # Retorna ação com maior peso
        return max(action_weights.items(), key=lambda x: x[1])[0]
    
    def _confidence_vote(
        self,
        predictions: Dict[str, int],
        confidences: Dict[str, float]
    ) -> int:
        """Votação ponderada por confiança do modelo"""
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for model_name, action in predictions.items():
            confidence = confidences.get(model_name, 0.0)
            action_scores[action] += confidence
        
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def _best_model_vote(self, predictions: Dict[str, int]) -> int:
        """Usa apenas o modelo com melhor performance histórica"""
        # Encontra modelo com maior taxa de acerto
        best_model = max(
            self.model_stats.items(),
            key=lambda x: x[1]['correct'] / max(x[1]['predictions'], 1)
        )[0]
        
        return predictions[best_model]
    
    def _average_vote(
        self,
        predictions: Dict[str, int],
        confidences: Dict[str, float]
    ) -> int:
        """Média ponderada de ações"""
        # Média ponderada por confiança
        total_weight = sum(confidences.values())
        
        if total_weight == 0:
            return self._majority_vote(predictions)
        
        weighted_sum = sum(
            predictions[name] * confidences[name]
            for name in predictions.keys()
        )
        
        avg_action = weighted_sum / total_weight
        
        # Arredonda para ação mais próxima
        return int(round(avg_action))
    
    def _calculate_confidence(
        self,
        model,
        observation: np.ndarray,
        action: int
    ) -> float:
        """
        Calcula confiança da previsão do modelo
        
        Para modelos estocásticos, usa probabilidade da ação.
        Para determinísticos, usa valor Q.
        """
        try:
            # Tenta obter probabilidades da política
            if hasattr(model.policy, 'get_distribution'):
                distribution = model.policy.get_distribution(observation)
                probs = distribution.distribution.probs
                
                if isinstance(probs, np.ndarray) and len(probs) > action:
                    return float(probs[action])
            
            # Fallback: usa confiança fixa baseada no tipo de modelo
            # PPO geralmente mais conservador, SAC mais confiante
            model_confidence = {
                'ppo': 0.7,
                'sac': 0.8,
                'td3': 0.75
            }
            
            model_type = type(model).__name__.lower()
            return model_confidence.get(model_type, 0.5)
            
        except:
            return 0.5  # Confiança neutra
    
    def _calculate_agreement(self, predictions: Dict[str, int], final_action: int) -> float:
        """Calcula % de concordância entre modelos"""
        if not predictions:
            return 0.0
        
        agreeing = sum(1 for action in predictions.values() if action == final_action)
        return agreeing / len(predictions)
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Atualiza pesos baseado em performance recente
        
        Args:
            performance_metrics: {
                'ppo': sharpe_ratio,
                'sac': sharpe_ratio,
                'td3': sharpe_ratio
            }
        """
        # Normaliza métricas para pesos
        total_performance = sum(max(v, 0.1) for v in performance_metrics.values())
        
        self.weights = {
            name: max(perf, 0.1) / total_performance
            for name, perf in performance_metrics.items()
        }
        
        logger.info(f"🔄 Pesos atualizados: {self.weights}")
    
    def update_stats(self, model_rewards: Dict[str, float]):
        """Atualiza estatísticas de performance dos modelos"""
        for name, reward in model_rewards.items():
            if name in self.model_stats:
                self.model_stats[name]['total_reward'] += reward
                
                # Marca como correto se reward positivo
                if reward > 0:
                    self.model_stats[name]['correct'] += 1
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        stats = {}
        
        for name, data in self.model_stats.items():
            predictions = data['predictions']
            if predictions > 0:
                accuracy = data['correct'] / predictions
                avg_reward = data['total_reward'] / predictions
            else:
                accuracy = 0.0
                avg_reward = 0.0
            
            stats[name] = {
                'predictions': predictions,
                'accuracy': accuracy,
                'avg_reward': avg_reward,
                'weight': self.weights.get(name, 0.0)
            }
        
        return stats
    
    def save_stats(self, filepath: str):
        """Salva estatísticas em arquivo"""
        import json
        
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Estatísticas salvas: {filepath}")


if __name__ == '__main__':
    # Teste com modelos mock
    logging.basicConfig(level=logging.INFO)
    
    class MockModel:
        def __init__(self, preferred_action):
            self.preferred_action = preferred_action
        
        def predict(self, obs, deterministic=True):
            return self.preferred_action, None
    
    # Cria ensemble mock
    models = {
        'ppo': MockModel(1),  # Prefere Long
        'sac': MockModel(1),  # Prefere Long
        'td3': MockModel(0),  # Prefere Flat
    }
    
    ensemble = EnsembleModel(
        models=models,
        strategy='weighted',
        weights={'ppo': 0.4, 'sac': 0.4, 'td3': 0.2}
    )
    
    # Testa previsão
    obs = np.random.randn(50, 20)
    action, info = ensemble.predict(obs)
    
    print(f"\n🎯 Previsão Ensemble:")
    print(f"Votos: {info['votes']}")
    print(f"Ação Final: {action}")
    print(f"Concordância: {info['agreement']:.2%}")
    print(f"Estratégia: {info['strategy']}")
