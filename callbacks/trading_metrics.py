"""
Callback customizado para logar métricas do TradingEnv no TensorBoard.
Integra perfeitamente com Stable-Baselines3.
"""

from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict
import numpy as np


class TradingMetricsCallback(BaseCallback):
    """
    Callback que loga métricas customizadas do ambiente de trading no TensorBoard.
    
    Captura ao final de cada episódio:
    - Win rate
    - Total PnL
    - Liquidações
    - Sharpe Ratio
    - Profit Factor
    - Balance/Equity final
    
    Uso:
        from callbacks.trading_metrics import TradingMetricsCallback
        
        callback = TradingMetricsCallback(verbose=1)
        model.learn(total_timesteps=100000, callback=callback)
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        Chamado a cada step do ambiente.
        Detecta fim de episódio e captura métricas.
        """
        # Verifica se episódio terminou (dones é array numpy em VecEnv)
        dones = self.locals.get("dones")
        if dones is not None and np.any(dones):
            for idx, done in enumerate(dones):
                if done:
                    # Captura métricas do ambiente
                    env = self.training_env.envs[idx]
                    
                    # Chama método get_episode_metrics() do TradingEnv
                    if hasattr(env, 'get_episode_metrics'):
                        metrics = env.get_episode_metrics()
                        
                        # LOGA SEMPRE (para debug e visualização desde o início)
                        trades = int(metrics.get('episode/trades', 0))
                        liquidations = int(metrics.get('episode/liquidations', 0))
                        
                        # Loga cada métrica no TensorBoard (SEMPRE!)
                        for key, value in metrics.items():
                            self.logger.record(key, value)
                        
                        # Log adicional de progresso no terminal (só se houver atividade)
                        if self.verbose > 0 and (trades > 0 or liquidations > 0):
                            win_rate = metrics.get('episode/win_rate', 0) * 100
                            pnl = metrics.get('episode/total_pnl', 0)
                            
                            print(f"\n📊 Episode {self.num_timesteps // env.max_episode_steps}:")
                            print(f"   Win Rate: {win_rate:.1f}% | Trades: {trades} | PnL: ${pnl:.2f}")
                            if liquidations > 0:
                                print(f"   ⚠️ Liquidations: {liquidations}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Chamado ao final de cada rollout (múltiplos episódios).
        Calcula estatísticas agregadas.
        """
        # Estatísticas agregadas de múltiplos ambientes
        if len(self.training_env.envs) > 0:
            total_liquidations = 0
            total_trades = 0
            total_wins = 0
            
            for env in self.training_env.envs:
                if hasattr(env, 'liquidations'):
                    total_liquidations += env.liquidations
                if hasattr(env, 'trades'):
                    total_trades += env.trades
                if hasattr(env, 'wins'):
                    total_wins += env.wins
            
            # Loga estatísticas agregadas
            if total_trades > 0:
                self.logger.record("rollout/total_liquidations", total_liquidations)
                self.logger.record("rollout/total_trades", total_trades)
                self.logger.record("rollout/aggregate_winrate", total_wins / total_trades)


class LiquidationMonitor(BaseCallback):
    """
    Monitor específico para liquidações.
    Para o treino se liquidações excederem threshold.
    
    Uso:
        monitor = LiquidationMonitor(max_liquidations=10, check_freq=10000)
        model.learn(total_timesteps=100000, callback=monitor)
    """
    
    def __init__(self, max_liquidations: int = 50, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.max_liquidations = max_liquidations
        self.check_freq = check_freq
        self.last_check = 0
        
    def _on_step(self) -> bool:
        # Verifica a cada check_freq steps
        if self.n_calls - self.last_check >= self.check_freq:
            self.last_check = self.n_calls
            
            total_liquidations = 0
            total_trades = 0
            total_wins = 0
            total_long = 0
            total_short = 0
            
            for env in self.training_env.envs:
                if hasattr(env, 'liquidations'):
                    total_liquidations += env.liquidations
                if hasattr(env, 'trades'):
                    total_trades += env.trades
                if hasattr(env, 'wins'):
                    total_wins += env.wins
                if hasattr(env, 'long_trades'):
                    total_long += env.long_trades
                if hasattr(env, 'short_trades'):
                    total_short += env.short_trades
            
            # Loga liquidações
            self.logger.record("monitor/liquidations", total_liquidations)
            
            if self.verbose > 0:
                print(f"\n⚠️ Liquidation Check (step {self.n_calls}): {total_liquidations} total")
                
                # NOVO: Mostrar estatísticas agregadas
                if total_trades > 0:
                    win_rate = (total_wins / total_trades) * 100
                    long_pct = (total_long / total_trades) * 100 if total_trades > 0 else 0
                    short_pct = (total_short / total_trades) * 100 if total_trades > 0 else 0
                    flat_pct = 100 - long_pct - short_pct
                    
                    print(f"📊 ESTATÍSTICAS AGREGADAS:")
                    print(f"   Win Rate: {win_rate:.1f}%")
                    print(f"   Trades: {total_trades} (Long: {total_long}, Short: {total_short})")
                    print(f"   Balance: Long {long_pct:.1f}% | Short {short_pct:.1f}% | Flat {flat_pct:.1f}%")
            
            # Para treino se exceder threshold
            if total_liquidations > self.max_liquidations:
                print(f"\n🛑 TREINO INTERROMPIDO: {total_liquidations} liquidações (max: {self.max_liquidations})")
                print("   Modelo está assumindo riscos excessivos!")
                print("   Recomendação: Reduzir leverage ou ajustar reward shaping")
                return False  # Para o treino
        
        return True


class PerformanceDecayMonitor(BaseCallback):
    """
    Detecta decaimento de performance (overfitting ou collapse).
    Para treino se win rate cair consistentemente.
    
    Uso:
        monitor = PerformanceDecayMonitor(min_winrate=0.05, patience=5)
        model.learn(total_timesteps=100000, callback=monitor)
    """
    
    def __init__(self, min_winrate: float = 0.05, patience: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.min_winrate = min_winrate
        self.patience = patience
        self.bad_episodes = 0
        
    def _on_step(self) -> bool:
        # dones é array numpy em VecEnv — usar np.any() para checar
        dones = self.locals.get("dones")
        if dones is not None and np.any(dones):
            for idx, done in enumerate(dones):
                if done:
                    env = self.training_env.envs[idx]
                    
                    if hasattr(env, 'wins') and hasattr(env, 'trades'):
                        if env.trades > 0:
                            win_rate = env.wins / env.trades
                            
                            if win_rate < self.min_winrate:
                                self.bad_episodes += 1
                                
                                if self.verbose > 0:
                                    print(f"\n⚠️ Low Performance: Win rate {win_rate:.1%} < {self.min_winrate:.1%}")
                                    print(f"   Bad episodes: {self.bad_episodes}/{self.patience}")
                                
                                if self.bad_episodes >= self.patience:
                                    print(f"\n🛑 TREINO INTERROMPIDO: Performance decay detectado")
                                    print(f"   {self.patience} episódios consecutivos com winrate < {self.min_winrate:.1%}")
                                    return False
                            else:
                                self.bad_episodes = 0  # Reset contador
        
        return True


class ValueLossDivergenceMonitor(BaseCallback):
    """
    V17.2: Monitora value_loss para detectar divergência.
    Para treino se value_loss explodir (sinal de colapso).
    
    Uso:
        monitor = ValueLossDivergenceMonitor(max_value_loss=2500, patience=3)
        model.learn(total_timesteps=100000, callback=monitor)
    """
    
    def __init__(self, max_value_loss: float = 2500, patience: int = 3, verbose: int = 1):
        super().__init__(verbose)
        self.max_value_loss = max_value_loss
        self.patience = patience
        self.high_loss_count = 0
        self.last_check_step = 0
        self.check_freq = 2048  # Check após cada rollout
        
    def _on_step(self) -> bool:
        # Verifica a cada rollout
        if self.n_calls - self.last_check_step >= self.check_freq:
            self.last_check_step = self.n_calls
            
            # Captura value_loss do logger
            if hasattr(self.logger, 'name_to_value'):
                value_loss = self.logger.name_to_value.get('train/value_loss', None)
                
                if value_loss is not None and value_loss > self.max_value_loss:
                    self.high_loss_count += 1
                    
                    if self.verbose > 0:
                        print(f"\n⚠️ 🔥 Value Loss Divergence: {value_loss:.0f} > {self.max_value_loss:.0f}")
                        print(f"   High loss count: {self.high_loss_count}/{self.patience}")
                        print(f"   Critic perdendo capacidade de estimar returns!")
                    
                    if self.high_loss_count >= self.patience:
                        print(f"\n🛑 TREINO INTERROMPIDO: Value function divergiu!")
                        print(f"   Value loss > {self.max_value_loss} por {self.patience} rollouts")
                        print(f"   \u2192 Modelo não está aprendendo corretamente")
                        print(f"   \u2192 Use checkpoint anterior (antes da divergência)")
                        return False
                else:
                    self.high_loss_count = 0  # Reset se normalizar
        
        return True
