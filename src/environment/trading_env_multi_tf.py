"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🕐 TRADING ENVIRONMENT - MULTI-TIMEFRAME (V16.3)                ║
║                                                                              ║
║  Ambiente de RL que processa múltiplos timeframes simultaneamente            ║
║  para decisões mais informadas:                                              ║
║  - 15m: Tático (ação imediata)                                              ║
║  - 1h:  Operacional (contexto)                                              ║
║  - 4h:  Estratégico (tendência)                                             ║
║                                                                              ║
║  🔧 V16.3 FIXES (19/02/2026 - Análise Gemini):                              ║
║  - ✅ Look-ahead bias CORRIGIDO (usa current_step-1 // divisor)             ║
║  - ✅ Normalização básica adicionada (clip -100/+100)                        ║
║  - ✅ Bônus por cortar loss aumentado (0.03 → 0.05)                          ║
║  - ✅ Penalties rebalanceadas (lucro = loss em magnitude)                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


class TradingEnvMultiTF(gym.Env):
    """
    Ambiente de Trading com Multi-Timeframe Analysis.
    
    Observation Space:
        - Timeframe 15m: window_size candles (tático)
        - Timeframe 1h:  window_size//4 candles (operacional)
        - Timeframe 4h:  window_size//16 candles (estratégico)
        - Estado da carteira unificado
        
    Action Space:
        Contínuo [-1, 1]:
        - [-1.0, -0.1]: Short
        - (-0.1, 0.1): Flat
        - [0.1, 1.0]: Long
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data_paths: Dict[str, str],  # {'15m': path, '1h': path, '4h': path}
        window_size: int = 50,
        initial_balance: float = 10000,
        commission: float = 0.0004,
        slippage: float = 0.0005,
        leverage: float = 1.5,
        position_size: float = 0.05,
        max_episode_steps: int = 2000,
        random_start: bool = True,
        persist_balance: bool = True,
        use_sharpe_reward: bool = True,
        maintenance_margin_rate: float = 0.005,
        liquidation_threshold: float = 0.10,
        enable_indicator_shaping: bool = False  # V15: Desabilitado
    ):
        """
        Args:
            data_paths: Dicionário com paths para cada timeframe
            window_size: Janela de observação (para 15m)
            Demais parâmetros: iguais ao TradingEnv original
        """
        super().__init__()
        
        # Carregar dados de múltiplos timeframes
        self.data_paths = data_paths
        self.dfs = {}
        self.df_values = {}
        self.n_features = {}
        
        for tf, path in data_paths.items():
            df = pd.read_csv(path)
            # Remover colunas não-numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.dfs[tf] = df[numeric_cols].reset_index(drop=True)
            self.df_values[tf] = self.dfs[tf].values
            self.n_features[tf] = len(numeric_cols)
        
        # Verificar alinhamento temporal
        self._verify_timeframe_alignment()
        
        # Parâmetros
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.position_size = position_size
        self.current_position_size = position_size
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.persist_balance = persist_balance
        self.use_sharpe_reward = use_sharpe_reward
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_threshold = liquidation_threshold
        self.enable_indicator_shaping = enable_indicator_shaping
        
        # Contadores
        self.liquidations = 0
        self.episode_liquidations = 0
        self.total_timesteps_trained = 0
        self.last_24h_trades = []
        self.episode_start_trades = 0  # V16: Tracking de trades por episódio
        
        # Balance persistente
        self.persistent_balance = initial_balance
        self.persistent_equity = initial_balance
        
        # Histórico de returns
        self.returns_history = []
        
        # Action Space: Box contínuo [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: Multi-timeframe
        # 15m: window_size candles completos
        # 1h: window_size//4 candles (4x menos granular)
        # 4h: window_size//16 candles (16x menos granular)
        
        obs_15m = window_size * (self.n_features['15m'] + 3)  # +3 = balance, position, equity
        obs_1h = (window_size // 4) * self.n_features['1h']
        obs_4h = (window_size // 16) * self.n_features['4h']
        
        total_obs = obs_15m + obs_1h + obs_4h
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs,),
            dtype=np.float32
        )
        
        # Estado inicial
        self.reset()
    
    def _verify_timeframe_alignment(self):
        """
        Verifica se os timeframes estão temporalmente alinhados.
        1h deve ter ~1/4 dos candles de 15m
        4h deve ter ~1/4 dos candles de 1h (1/16 de 15m)
        """
        len_15m = len(self.dfs['15m'])
        len_1h = len(self.dfs['1h'])
        len_4h = len(self.dfs['4h'])
        
        ratio_1h = len_15m / len_1h
        ratio_4h = len_1h / len_4h
        
        print(f"\n📊 ALINHAMENTO DE TIMEFRAMES:")
        print(f"   15m: {len_15m:,} candles")
        print(f"   1h:  {len_1h:,} candles (ratio: {ratio_1h:.2f})")
        print(f"   4h:  {len_4h:,} candles (ratio: {ratio_4h:.2f})")
        
        # Verifica se as proporções estão corretas (~4:1)
        if not (3.5 < ratio_1h < 4.5):
            print(f"   ⚠️  Warning: 15m/1h ratio está fora do esperado (3.5-4.5)")
        if not (3.5 < ratio_4h < 4.5):
            print(f"   ⚠️  Warning: 1h/4h ratio está fora do esperado (3.5-4.5)")
        
        print()
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reinicia o ambiente."""
        super().reset(seed=seed)
        
        # Balance persistente
        if self.persist_balance:
            if self.persistent_balance < self.initial_balance * 0.6:
                print(f"⚠️ Balance crítico, resetando")
                self.persistent_balance = self.initial_balance
                self.persistent_equity = self.initial_balance
            
            self.balance = self.persistent_balance
            self.equity = self.persistent_equity
        else:
            self.balance = self.initial_balance
            self.equity = self.initial_balance
        
        # Estado da conta
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        
        # Métricas
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.episode_liquidations = 0
        self.last_trade_step = 0
        self.long_trades = 0
        self.short_trades = 0
        self._prev_position = 0
        self._last_position_change_step = 0
        self.episode_start_trades = self.trades  # V16: Tracking trades por episódio
        
        self.previous_equity = self.equity
        self.returns_history = []
        
        # Random start (baseado no timeframe 15m)
        if self.random_start:
            max_start = max(
                self.window_size,
                len(self.dfs['15m']) - self.max_episode_steps - 1
            )
            self.episode_start = np.random.randint(self.window_size, max_start)
        else:
            self.episode_start = self.window_size
        
        self.current_step = self.episode_start
        self.episode_length = 0
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """
        Constrói observação multi-timeframe.
        
        Returns:
            Array 1D concatenado com:
            - 15m: window_size últimos candles + portfolio state
            - 1h:  window_size//4 últimos candles
            - 4h:  window_size//16 últimos candles
        """
        # === TIMEFRAME 15m (tático) ===
        start_15m = max(0, self.current_step - self.window_size)
        end_15m = self.current_step
        window_15m = self.df_values['15m'][start_15m:end_15m]
        
        # Adicionar portfolio state ao 15m
        portfolio_state = np.array([
            [self.balance / self.initial_balance,
             self.position,
             self.equity / self.initial_balance]
        ]).repeat(len(window_15m), axis=0)
        
        obs_15m = np.hstack([window_15m, portfolio_state])
        
        # Pad se necessário
        if len(obs_15m) < self.window_size:
            padding = np.zeros((self.window_size - len(obs_15m), obs_15m.shape[1]))
            obs_15m = np.vstack([padding, obs_15m])
        
        # === TIMEFRAME 1h (operacional) ===
        # V16.3 FIX: Prevenir look-ahead bias (usar apenas candles FECHADOS)
        # Cada candle 1h = 4 candles 15m, MAS só está fechado a cada 4 steps
        current_1h = (self.current_step - 1) // 4  # -1 garante candle fechado
        window_1h_size = self.window_size // 4
        start_1h = max(0, current_1h - window_1h_size)
        end_1h = current_1h
        
        obs_1h = self.df_values['1h'][start_1h:end_1h]
        
        # Pad se necessário
        if len(obs_1h) < window_1h_size:
            padding = np.zeros((window_1h_size - len(obs_1h), obs_1h.shape[1]))
            obs_1h = np.vstack([padding, obs_1h])
        
        # === TIMEFRAME 4h (estratégico) ===
        # V16.3 FIX: Prevenir look-ahead bias (usar apenas candles FECHADOS)
        # Cada candle 4h = 16 candles 15m, MAS só está fechado a cada 16 steps
        current_4h = (self.current_step - 1) // 16  # -1 garante candle fechado
        window_4h_size = self.window_size // 16
        start_4h = max(0, current_4h - window_4h_size)
        end_4h = current_4h
        
        obs_4h = self.df_values['4h'][start_4h:end_4h]
        
        # Pad se necessário
        if len(obs_4h) < window_4h_size:
            padding = np.zeros((window_4h_size - len(obs_4h), obs_4h.shape[1]))
            obs_4h = np.vstack([padding, obs_4h])
        
        # Concatenar tudo em 1D
        observation = np.concatenate([
            obs_15m.flatten(),
            obs_1h.flatten(),
            obs_4h.flatten()
        ]).astype(np.float32)
        
        # V16.3 FIX: Normalização BÁSICA (clip extremos para evitar explosão)
        # Isso ajuda convergência da rede neural
        observation = np.clip(observation, -100, 100)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Retorna informações adicionais."""
        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'liquidations': self.episode_liquidations,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa ação no ambiente.
        LÓGICA IDÊNTICA AO V15 - apenas observation é multi-timeframe.
        """
        # Converter action
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        if action_value < -0.1:
            discrete_action = 2  # Short
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        elif action_value > 0.1:
            discrete_action = 1  # Long
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        else:
            discrete_action = 0  # Flat
            self.current_position_size = 0
        
        # Preço atual (sempre do timeframe 15m)
        current_price = self.dfs['15m'].iloc[self.current_step]['close']
        
        # Calcular PnL
        pnl = self._calculate_pnl(current_price)
        
        # Stop-loss automático (V11/V15)
        stop_loss_triggered = False
        if self.position != 0:
            unrealized_pnl = self._calculate_pnl(current_price)
            unrealized_pct = unrealized_pnl / self.balance
            
            if unrealized_pct <= -0.07:  # -7% stop
                self._close_position(current_price)
                stop_loss_triggered = True
                discrete_action = 0
                reward = -0.09
        
        # Executar ação
        if not stop_loss_triggered:
            action_reward = self._execute_action(discrete_action, current_price)
            reward = action_reward
            
            # V16.2 FIX: Penalidade por step flat (RESTAURADA do V15)
            if discrete_action == 0 and self.position == 0:
                reward -= 0.01  # -0.01 por step fora do mercado
        else:
            action_reward = 0.0
            reward = -0.08
        
        # Atualizar equity
        self.equity = self.balance + self.position_value
        
        # === REWARD SHAPING (V16.2: REBALANCEADO) ===
        
        # V16.2: Penalidade FORTE por inatividade prolongada
        time_since_trade = self.episode_length - self.last_trade_step
        if time_since_trade > 200:  # Reduzido 300 → 200 steps
            # Penalidade 10x mais forte: 0.0001 → 0.001
            inactivity_penalty = 0.001 * (time_since_trade - 200)
            reward -= inactivity_penalty
        
        # V16.2: Penalidade MUITO FORTE por holding prolongado
        if self.position != 0:
            holding_time = self.episode_length - self.last_trade_step
            if holding_time > 300:  # Reduzido 400 → 300, começa mais cedo
                # Penalidade 10x mais forte: 0.0005 → 0.005
                reward -= 0.005 * (holding_time - 300)
        
        # V16.3: Bônus/Penalidades BALANCEADAS + incentivo forte para cortar loss
        if action_reward != 0:
            if action_reward > 0.02:
                reward += 0.05  # Lucro = bônus
            elif action_reward < -0.02:
                reward -= 0.05  # PERDA GRANDE = penalidade forte (balanceado)
            # V16.3 MELHORADO: Bônus MAIOR por cortar loss cedo
            elif -0.02 < action_reward < -0.001:
                reward += 0.05  # Corta loss pequeno = EXCELENTE! (aumentado de 0.03)
        
        # Penalidade por overtrading (V16: SUAVIZADA)
        if discrete_action != 0 or (self.position == 0 and hasattr(self, '_prev_position') and self._prev_position != 0):
            self.last_24h_trades.append(self.episode_length)
        
        self.last_24h_trades = [t for t in self.last_24h_trades if self.episode_length - t <= 96]
        
        # V16: Aumentado threshold 3 → 10 trades
        if len(self.last_24h_trades) > 10:
            overtrading_penalty = (len(self.last_24h_trades) - 10) * 0.01  # Reduzido 0.03 → 0.01
            reward -= overtrading_penalty
        
        # Penalidade por flip-flop REMOVIDA (V16.1) - estava travando o modelo
        # if hasattr(self, '_prev_position') and hasattr(self, '_last_position_change_step'):
        #     steps_since_flip = self.episode_length - self._last_position_change_step
        #     if steps_since_flip < 50:
        #         if (self._prev_position == 1 and self.position == -1) or \
        #            (self._prev_position == -1 and self.position == 1):
        #             reward -= 0.02
        
        if self.position != getattr(self, '_prev_position', 0):
            self._last_position_change_step = self.episode_length
        
        self._prev_position = self.position
        
        # Calcular return
        step_return = (self.equity - self.previous_equity) / self.previous_equity
        self.returns_history.append(step_return)
        
        # Sharpe Ratio reward (V16: Peso MUITO reduzido)
        if self.use_sharpe_reward and len(self.returns_history) > 10:  # Requer 10+ samples
            returns_array = np.array(self.returns_history[-100:])
            mean_return = returns_array.mean()
            std_return = returns_array.std() + 1e-8
            sharpe = mean_return / std_return
            
            # V16: Peso MÍNIMO 0.2 → 0.05
            sharpe_reward = np.tanh(sharpe * 10) * 0.05
            reward += sharpe_reward
            
            progress = min(self.episode_length / self.max_episode_steps, 1.0)
            bonus = 0.02 * (1 - progress * 0.5)  # Reduzido 0.03 → 0.02
            penalty = 0.005 * (1 - progress * 0.3)  # Reduzido 0.01 → 0.005
            
            if step_return > 0.01:
                reward += bonus
            elif step_return < -0.01:
                reward -= penalty
        
        # Atualizar equity anterior
        self.previous_equity = self.equity
        
        # Avançar step
        self.current_step += 1
        self.episode_length += 1
        
        # Verificar terminação
        terminated = False
        truncated = False
        
        # Liquidação
        if self.equity <= self.initial_balance * (1 - self.liquidation_threshold):
            terminated = True
            reward = -1.0
            self.liquidations += 1
            self.episode_liquidations += 1
        
        # Fim do dataset
        if self.current_step >= len(self.dfs['15m']) - 1:
            truncated = True
        
        # Max episode steps
        if self.episode_length >= self.max_episode_steps:
            truncated = True
        
        # Atualizar persistent balance
        if self.persist_balance:
            self.persistent_balance = self.balance
            self.persistent_equity = self.equity
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calcula PnL da posição atual."""
        if self.position == 0:
            return 0.0
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        # position_value JÁ inclui leverage (aplicado em _open_position)
        if self.position == 1:  # Long
            pnl = self.position_value * price_change
        else:  # Short
            pnl = -self.position_value * price_change
        
        return pnl
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Executa ação de trading.
        
        Returns:
            Reward adicional (PnL % ao fechar posição)
        """
        action_reward = 0.0
        
        # Fechar posição existente se mudou direção
        if self.position != 0 and action != 0 and action != (1 if self.position == 1 else 2):
            pnl = self._calculate_pnl(current_price)
            action_reward = self._close_position(current_price)
        
        # Executar nova ação
        if action == 0:  # Flat
            if self.position != 0:
                action_reward = self._close_position(current_price)
        
        elif action == 1:  # Long
            if self.position == 0:
                self._open_position(1, current_price)
        
        elif action == 2:  # Short
            if self.position == 0:
                self._open_position(-1, current_price)
        
        return action_reward
    
    def _open_position(self, direction: int, price: float):
        """Abre posição Long (1) ou Short (-1)."""
        self.position = direction
        self.entry_price = price
        
        # Aplicar slippage
        effective_price = price * (1 + self.slippage) if direction == 1 else price * (1 - self.slippage)
        
        # Calcular valor da posição
        position_amount = self.balance * self.current_position_size
        self.position_value = position_amount * self.leverage
        
        # Descontar comissão
        commission_cost = self.position_value * self.commission
        self.balance -= commission_cost
        
        self.last_trade_step = self.episode_length
        
        if direction == 1:
            self.long_trades += 1
        else:
            self.short_trades += 1
    
    def _close_position(self, price: float) -> float:
        """
        Fecha posição e retorna reward (PnL %).
        
        Returns:
            PnL como % do balance (para usar como reward)
        """
        if self.position == 0:
            return 0.0
        
        # Calcular PnL
        pnl = self._calculate_pnl(price)
        
        # Aplicar slippage
        effective_price = price * (1 - self.slippage) if self.position == 1 else price * (1 + self.slippage)
        
        # Comissão de saída
        commission_cost = self.position_value * self.commission
        pnl -= commission_cost
        
        # Atualizar balance
        self.balance += pnl
        self.total_pnl += pnl
        
        # Estatísticas
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        # Reward: PnL como % do balance inicial
        reward = pnl / self.initial_balance
        
        # Resetar posição
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        
        return reward
