"""
Ensemble Training System

Treina múltiplos algoritmos RL (PPO, SAC, TD3) e combina suas previsões
para decisões mais robustas.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
import torch

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environment.trading_env import TradingEnv
from src.data.data_collector import DataCollector

logger = logging.getLogger(__name__)

# 🚀 Detecta GPU AMD (DirectML)
try:
    import torch_directml
    dml_device = torch_directml.device()
    GPU_AVAILABLE = True
    logger.info(f"🚀 GPU AMD detectada! Usando DirectML: {dml_device}")
except:
    dml_device = 'cpu'
    GPU_AVAILABLE = False
    logger.info("⚠️ GPU não disponível, usando CPU")


class EnsembleTrainer:
    """
    Treina ensemble de modelos RL:
    - PPO (Proximal Policy Optimization) - Estável
    - SAC (Soft Actor-Critic) - Agressivo, melhor para trading contínuo
    - TD3 (Twin Delayed DDPG) - Ações contínuas
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.ensemble_config = self.config.get('ensemble', {})
        self.training_config = self.config['training']
        
        # Diretórios
        self.models_dir = Path('models/ensemble')
        self.logs_dir = Path('logs/ensemble')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelos
        self.models = {}
        self.best_models = {}
        
    def train_all(
        self, 
        train_data_path: str,
        val_data_path: str,
        total_timesteps: Optional[int] = None
    ) -> Dict:
        """
        Treina todos os modelos do ensemble
        
        Args:
            train_data_path: Caminho para dados de treino
            val_data_path: Caminho para dados de validação
            total_timesteps: Total de timesteps (usa config se None)
            
        Returns:
            {
                'ppo': model,
                'sac': model,
                'td3': model
            }
        """
        logger.info("🚀 Iniciando treinamento ensemble...")
        
        if total_timesteps is None:
            total_timesteps = self.training_config['total_timesteps']
        
        algorithms = self.ensemble_config.get('algorithms', ['ppo', 'sac', 'td3'])
        
        for algo in algorithms:
            logger.info(f"\n{'='*60}")
            logger.info(f"Treinando {algo.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                model = self._train_single_algorithm(
                    algo=algo,
                    train_data_path=train_data_path,
                    val_data_path=val_data_path,
                    total_timesteps=total_timesteps
                )
                
                self.models[algo] = model
                logger.info(f"✅ {algo.upper()} treinado com sucesso!")
                
            except Exception as e:
                logger.error(f"❌ Erro ao treinar {algo.upper()}: {e}")
        
        logger.info(f"\n✅ Ensemble completo! {len(self.models)} modelos treinados")
        return self.models
    
    def _train_single_algorithm(
        self,
        algo: str,
        train_data_path: str,
        val_data_path: str,
        total_timesteps: int
    ):
        """Treina um algoritmo específico"""
        
        # Cria environments
        train_env = self._create_env(train_data_path, algo)
        val_env = self._create_env(val_data_path, algo)
        
        # Cria modelo
        model = self._create_model(algo, train_env)
        
        # Callbacks
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=str(self.models_dir / algo),
            log_path=str(self.logs_dir / algo),
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(self.models_dir / algo / 'checkpoints'),
            name_prefix=f'{algo}_model'
        )
        
        # Treina
        logger.info(f"🏋️ Treinando {algo.upper()} por {total_timesteps:,} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False  # Desabilitado para evitar problemas com tqdm
        )
        
        # Salva modelo final
        final_path = self.models_dir / algo / f'{algo}_final.zip'
        model.save(final_path)
        logger.info(f"💾 Modelo salvo: {final_path}")
        
        return model
    
    def _create_env(self, data_path: str, algo: str) -> DummyVecEnv:
        """Cria environment para treinamento"""
        
        def make_env():
            env = TradingEnv(
                data_path=data_path,
                config=self.config['environment']
            )
            # Monitor para logging
            log_dir = self.logs_dir / algo
            log_dir.mkdir(exist_ok=True)
            env = Monitor(env, str(log_dir))
            return env
        
        return DummyVecEnv([make_env])
    
    def _create_model(self, algo: str, env):
        """Cria modelo RL específico"""
        
        # Parâmetros base
        common_params = {
            'env': env,
            'verbose': 1,
            'device': dml_device if GPU_AVAILABLE else 'cpu',
            'tensorboard_log': str(self.logs_dir / algo)
        }
        
        if algo == 'ppo':
            # PPO - Mais conservador e estável
            model = PPO(
                policy='MlpPolicy',
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                **common_params
            )
            
        elif algo == 'sac':
            # SAC - Melhor para trading contínuo
            model = SAC(
                policy='MlpPolicy',
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                **common_params
            )
            
        elif algo == 'td3':
            # TD3 - Ações contínuas com menos ruído
            model = TD3(
                policy='MlpPolicy',
                learning_rate=1e-3,
                buffer_size=100000,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'episode'),
                gradient_steps=-1,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                **common_params
            )
        
        else:
            raise ValueError(f"Algoritmo desconhecido: {algo}")
        
        logger.info(f"✅ Modelo {algo.upper()} criado")
        return model
    
    def load_ensemble(self, version: str = 'best') -> Dict:
        """
        Carrega ensemble de modelos treinados
        
        Args:
            version: 'best' ou 'final'
            
        Returns:
            Dict com modelos carregados
        """
        logger.info(f"📂 Carregando ensemble ({version})...")
        
        algorithms = self.ensemble_config.get('algorithms', ['ppo', 'sac', 'td3'])
        models = {}
        
        for algo in algorithms:
            try:
                if version == 'best':
                    model_path = self.models_dir / algo / 'best_model.zip'
                else:
                    model_path = self.models_dir / algo / f'{algo}_final.zip'
                
                if not model_path.exists():
                    logger.warning(f"⚠️ Modelo {algo} não encontrado: {model_path}")
                    continue
                
                # Carrega modelo apropriado
                if algo == 'ppo':
                    model = PPO.load(model_path)
                elif algo == 'sac':
                    model = SAC.load(model_path)
                elif algo == 'td3':
                    model = TD3.load(model_path)
                
                models[algo] = model
                logger.info(f"  ✅ {algo.upper()} carregado")
                
            except Exception as e:
                logger.error(f"  ❌ Erro ao carregar {algo}: {e}")
        
        self.models = models
        logger.info(f"✅ Ensemble carregado: {len(models)} modelos")
        return models
    
    def evaluate_ensemble(self, test_data_path: str, episodes: int = 10) -> Dict:
        """
        Avalia performance de cada modelo do ensemble
        
        Returns:
            {
                'ppo': {'mean_reward': float, 'std_reward': float, ...},
                'sac': {...},
                'td3': {...}
            }
        """
        logger.info(f"📊 Avaliando ensemble em {episodes} episódios...")
        
        results = {}
        
        for algo, model in self.models.items():
            logger.info(f"\nAvaliando {algo.upper()}...")
            
            env = self._create_env(test_data_path, algo)
            
            episode_rewards = []
            episode_lengths = []
            
            for ep in range(episodes):
                obs = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0]
                    steps += 1
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                logger.info(f"  Ep {ep+1}/{episodes}: Reward={total_reward:.2f}, Steps={steps}")
            
            results[algo] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'mean_length': np.mean(episode_lengths)
            }
            
            logger.info(f"✅ {algo.upper()}: Reward={results[algo]['mean_reward']:.2f} ± {results[algo]['std_reward']:.2f}")
        
        # Identifica melhor modelo
        best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])[0]
        logger.info(f"\n🏆 Melhor modelo: {best_algo.upper()}")
        
        return results


if __name__ == '__main__':
    # Teste de treinamento
    logging.basicConfig(level=logging.INFO)
    
    trainer = EnsembleTrainer()
    
    # Treina ensemble (use timesteps menores para teste)
    models = trainer.train_all(
        train_data_path='data/train_data.csv',
        val_data_path='data/val_data.csv',
        total_timesteps=50000  # 50k timesteps com GPU (~5-10 min)
    )
    
    # Avalia
    results = trainer.evaluate_ensemble(
        test_data_path='data/test_data.csv',
        episodes=5
    )
    
    print("\n📊 Resultados:")
    for algo, metrics in results.items():
        print(f"\n{algo.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
