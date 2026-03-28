"""
Callback customizado para logar métricas no TensorBoard.
V7: Adiciona atualização de total_timesteps_trained para decaimento gradual do shaping.
"""

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Callback para logar métricas customizadas no TensorBoard.
    
    V7: Atualiza env.total_timesteps_trained para decaimento gradual do shaping.
    """
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        Chamado a cada step do ambiente.
        V7: Atualiza total_timesteps_trained no ambiente.
        """
        # V7: Atualiza contador de timesteps no ambiente para decaimento gradual
        # Acessa o ambiente real (não o wrapper)
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'total_timesteps_trained'):
                    env.total_timesteps_trained = self.num_timesteps
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Chamado ao final de cada rollout (episódio completo).
        Loga métricas customizadas do ambiente.
        """
        # Acessa ambiente real
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            
            # Pega métricas do episódio
            if hasattr(env, 'get_episode_metrics'):
                metrics = env.get_episode_metrics()
                
                # Loga cada métrica no TensorBoard
                for key, value in metrics.items():
                    self.logger.record(key, value)
                
                # V7: Loga decaimento do shaping
                if hasattr(env, 'total_timesteps_trained'):
                    shaping_decay = 1.0
                    if env.total_timesteps_trained > 500000:
                        shaping_decay = max(0.0, 1.0 - (env.total_timesteps_trained - 500000) / 1500000)
                    self.logger.record('training/shaping_decay', shaping_decay)
