"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         🧠 TRADING ENVIRONMENT - MULTI-TIMEFRAME LSTM (V17-LSTM)            ║
║                                                                              ║
║  Ambiente adaptado para RecurrentPPO com LSTM:                              ║
║  - Observations SEQUENCIAIS (não flatten)                                   ║
║  - Shape: (seq_len, features) para LSTM processar                           ║
║  - Mantém estrutura temporal para memória de curto prazo                    ║
║                                                                              ║
║  🆚 DIFERENÇAS VS V16.3 (MLP):                                              ║
║  - V16.3: Flatten tudo → [1450] features                                    ║
║  - V17-LSTM: Sequencial → [50, 29] (50 timesteps, 29 features/step)        ║
║                                                                              ║
║  🎯 OBJETIVO: LSTM aprende dependências temporais que MLP não consegue      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


class TradingEnvMultiTFLSTM(gym.Env):
    """
    Ambiente de Trading Multi-Timeframe para LSTM.
    
    DIFERENÇA CHAVE: Observations são SEQUENCIAIS (seq_len, features)
    ao invés de flattened (features,).
    
    Observation Space:
        - Shape: (seq_len, n_features_per_step)
        - seq_len: 50 timesteps (mesma janela do V16.3)
        - features_per_step: 29 features por timestep
          * 26 features: OHLCV + indicators do 15m
          * 1 feature: 1h aggregated info
          * 1 feature: 4h aggregated info
          * 1 feature: portfolio state (balance, position, equity agregados)
        
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
        leverage: float = 1.0,
        position_size: float = 0.05,
        max_episode_steps: int = 2000,
        random_start: bool = True,
        persist_balance: bool = False,
        use_sharpe_reward: bool = False,
        maintenance_margin_rate: float = 0.005,
        liquidation_threshold: float = 0.30,
        enable_indicator_shaping: bool = False
    ):
        """
        Args:
            Todos iguais ao TradingEnvMultiTF (V16.3)
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
        
        # Parâmetros (IGUAIS V16.3)
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
        self.episode_start_trades = 0
        
        # Balance persistente
        self.persistent_balance = initial_balance
        self.persistent_equity = initial_balance
        
        # Histórico de returns
        self.returns_history = []
        
        # Action Space: Box contínuo [-1, 1] (IGUAL V16.3)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # V17-LSTM: Observation Space SEQUENCIAL
        # Shape: (seq_len=50, features_per_step=31)
        # 31 features = 20 (15m) + 4 (1h context) + 4 (4h context) + 3 (portfolio)
        #
        # Feature indices no df_values (após remover timestamp):
        #   IDX_CLOSE=3, IDX_RSI=5, IDX_BBP=12, IDX_MACDH=15
        #
        # 1h/4h context (4 features cada):
        #   [0] RSI_14          - momentum/overbought no TF maior
        #   [1] BBP_20_2.0      - posição nas bandas (0-1)
        #   [2] MACDh_12_26_9   - direção da tendência
        #   [3] close % diff    - diferença de preço normalizada vs 15m
        #
        # Portfolio (3 features SEPARADAS - não mediadas!):
        #   [0] balance / initial_balance
        #   [1] position (-1=short, 0=flat, 1=long)
        #   [2] equity / initial_balance
        self.IDX_CLOSE = 3
        self.IDX_RSI   = 5
        self.IDX_BBP   = 12
        self.IDX_MACDH = 15
        
        features_per_step = self.n_features['15m'] + 4 + 4 + 3  # 31 total
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, features_per_step),  # SEQUENCIAL!
            dtype=np.float32
        )
        
        print(f"\n🧠 LSTM Environment Initialized (V17.7 - Real Multi-TF):")
        print(f"   Obs shape: {self.observation_space.shape} (SEQUENTIAL)")
        print(f"   15m features: {self.n_features['15m']}")
        print(f"   1h context:   4 (RSI, BBP, MACDh, close%diff)")
        print(f"   4h context:   4 (RSI, BBP, MACDh, close%diff)")
        print(f"   portfolio:    3 (balance, position, equity) SEPARADOS")
        print(f"   Total per step: {features_per_step}\n")
        
        # Estado inicial
        self.reset()
    
    def _verify_timeframe_alignment(self):
        """Verifica alinhamento temporal (IGUAL V16.3)."""
        len_15m = len(self.dfs['15m'])
        len_1h = len(self.dfs['1h'])
        len_4h = len(self.dfs['4h'])
        
        ratio_1h = len_15m / len_1h
        ratio_4h = len_1h / len_4h
        
        print(f"\n📊 ALINHAMENTO DE TIMEFRAMES:")
        print(f"   15m: {len_15m:,} candles")
        print(f"   1h:  {len_1h:,} candles (ratio: {ratio_1h:.2f})")
        print(f"   4h:  {len_4h:,} candles (ratio: {ratio_4h:.2f})")
        
        if not (3.5 < ratio_1h < 4.5):
            print(f"   ⚠️  Warning: 15m/1h ratio fora do esperado")
        if not (3.5 < ratio_4h < 4.5):
            print(f"   ⚠️  Warning: 1h/4h ratio fora do esperado")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reinicia o ambiente (LÓGICA IGUAL V16.3)."""
        super().reset(seed=seed)
        
        # Balance persistente
        if self.persist_balance:
            if self.persistent_balance < self.initial_balance * 0.6:
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
        self.long_wins = 0
        self.long_losses = 0
        self.short_wins = 0
        self.short_losses = 0
        self.current_position_side = 0  # 1=long, -1=short, 0=flat
        # Per-step trade event (reset after each _get_info call)
        self._trade_just_closed = False
        self._last_closed_pnl   = 0.0
        self._last_closed_side  = 0
        self._prev_position = 0
        self._last_position_change_step = 0
        self.episode_start_trades = self.trades
        
        self.previous_equity = self.equity
        self.returns_history = []
        
        # Random start
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
        V17.7: Observação SEQUENCIAL com multi-timeframe real.
        
        Returns:
            Array 2D (seq_len=50, features_per_step=31)
            - 20: Features do 15m (OHLCV + indicadores completos)
            - 4:  1h context (RSI, BBP, MACDh, close%diff)
            - 4:  4h context (RSI, BBP, MACDh, close%diff)
            - 3:  Portfolio SEPARADO (balance_norm, position, equity_norm)
        """
        # Janela de 50 candles do 15m
        start_15m = max(0, self.current_step - self.window_size)
        end_15m = self.current_step
        window_15m = self.df_values['15m'][start_15m:end_15m]  # (50, 20)
        
        # Pad se necessário
        if len(window_15m) < self.window_size:
            padding = np.zeros((self.window_size - len(window_15m), window_15m.shape[1]))
            window_15m = np.vstack([padding, window_15m])
        
        # === CONTEXT DE 1h e 4h (4 features cada) ===
        # Para cada timestep dos 50, extraímos do candle TF maior já FECHADO
        # Índice: (step_15m - 1) // N garante que usamos o candle anterior,
        # evitando look-ahead bias (candle atual pode não ter fechado ainda)
        
        ctx_1h = np.zeros((self.window_size, 4), dtype=np.float32)
        ctx_4h = np.zeros((self.window_size, 4), dtype=np.float32)
        
        len_1h = len(self.df_values['1h'])
        len_4h = len(self.df_values['4h'])
        
        for i in range(self.window_size):
            step_15m = start_15m + i
            
            # Close price do 15m (para % diff normalizado)
            price_15m = float(window_15m[i, self.IDX_CLOSE])
            if price_15m == 0:
                price_15m = 1.0
            
            # --- 1h context ---
            idx_1h = max(0, (step_15m - 1) // 4)
            if idx_1h < len_1h:
                row_1h = self.df_values['1h'][idx_1h]
                ctx_1h[i, 0] = row_1h[self.IDX_RSI]    # RSI_14
                ctx_1h[i, 1] = row_1h[self.IDX_BBP]    # BBP_20_2.0
                ctx_1h[i, 2] = row_1h[self.IDX_MACDH]  # MACDh_12_26_9
                ctx_1h[i, 3] = (row_1h[self.IDX_CLOSE] / price_15m - 1) * 100  # close%diff
            
            # --- 4h context ---
            idx_4h = max(0, (step_15m - 1) // 16)
            if idx_4h < len_4h:
                row_4h = self.df_values['4h'][idx_4h]
                ctx_4h[i, 0] = row_4h[self.IDX_RSI]    # RSI_14
                ctx_4h[i, 1] = row_4h[self.IDX_BBP]    # BBP_20_2.0
                ctx_4h[i, 2] = row_4h[self.IDX_MACDH]  # MACDh_12_26_9
                ctx_4h[i, 3] = (row_4h[self.IDX_CLOSE] / price_15m - 1) * 100  # close%diff
        
        # === PORTFOLIO (3 colunas SEPARADAS - não mediadas!) ===
        balance_col = np.full((self.window_size, 1), self.balance / self.initial_balance)
        position_col = np.full((self.window_size, 1), float(self.position))
        equity_col   = np.full((self.window_size, 1), self.equity / self.initial_balance)
        
        # === CONCATENAR: (50,20) + (50,4) + (50,4) + (50,1) + (50,1) + (50,1) = (50,31) ===
        observation = np.hstack([
            window_15m,   # (50, 20) - features 15m
            ctx_1h,       # (50, 4)  - RSI/BBP/MACDh/close%diff 1h
            ctx_4h,       # (50, 4)  - RSI/BBP/MACDh/close%diff 4h
            balance_col,  # (50, 1)  - balance normalizado
            position_col, # (50, 1)  - posição (-1/0/1)
            equity_col    # (50, 1)  - equity normalizado
        ]).astype(np.float32)   # (50, 31)
        
        # Clip para evitar NaN/Inf extremos
        observation = np.clip(observation, -100, 100)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Retorna informações adicionais. Inclui trade_executed para o backtest."""
        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        long_wr  = (self.long_wins  / self.long_trades  * 100) if self.long_trades  > 0 else 0
        short_wr = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0

        # Evento de trade fechado (usado pelo backtest para capturar PnL por trade)
        trade_executed = self._trade_just_closed
        last_pnl       = self._last_closed_pnl
        last_side      = self._last_closed_side
        # Reset flag após consumir
        self._trade_just_closed = False
        self._last_closed_pnl   = 0.0
        self._last_closed_side  = 0

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
            # LONG/SHORT breakdown
            'long_trades':  self.long_trades,
            'short_trades': self.short_trades,
            'long_wins':    self.long_wins,
            'long_losses':  self.long_losses,
            'long_wr':      long_wr,
            'short_wins':   self.short_wins,
            'short_losses': self.short_losses,
            'short_wr':     short_wr,
            # Evento de trade fechado neste step
            'trade_executed': trade_executed,
            'pnl':            last_pnl,
            '_side':          last_side,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa ação no ambiente.
        LÓGICA IDÊNTICA AO V16.3 - apenas observation é sequencial.
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
        
        # Preço atual (do 15m)
        current_price = self.dfs['15m'].iloc[self.current_step]['close']
        
        # Calcular PnL
        pnl = self._calculate_pnl(current_price)
        
        # Stop-loss automático
        stop_loss_triggered = False
        if self.position != 0:
            unrealized_pnl = self._calculate_pnl(current_price)
            unrealized_pct = unrealized_pnl / self.balance
            
            if unrealized_pct <= -0.07:
                self._close_position(current_price)
                stop_loss_triggered = True
                discrete_action = 0
                reward = -0.09
        
        # Executar ação
        if not stop_loss_triggered:
            action_reward = self._execute_action(discrete_action, current_price)
            reward = action_reward
            
            # V17.7: Penalidade flat
            if discrete_action == 0 and self.position == 0:
                reward -= 0.0002
        else:
            action_reward = 0.0
            reward = -0.08
        
        # Atualizar equity
        self.equity = self.balance + self.position_value
        
        # === REWARD SHAPING V17.7 ===
        
        # V17.7: Penalidade inatividade (sem posição por muito tempo)
        time_since_trade = self.episode_length - self.last_trade_step
        if time_since_trade > 200:
            reward -= 0.00002 * (time_since_trade - 200)
        
        # V17.7b FIX: Penalidade holding CONSTANTE por step (NÃO linear em duração!)
        # Linear-in-duration criava reward de até -0.98/step → critic divergia (EV→-22)
        # Agora: grace period de 50 steps, depois -0.0003/step FIXO (> flat -0.0002/step)
        # Isso elimina o exploit sem criar reward variance catastrófica
        if self.position != 0:
            holding_time = self.episode_length - self.last_trade_step
            if holding_time > 50:
                reward -= 0.0003  # constante por step, ligeiramente > flat (0.0002)
        
        # Bônus/Penalidades balanceadas
        if action_reward != 0:
            if action_reward > 0.02:
                reward += 0.05
            elif action_reward < -0.02:
                reward -= 0.05
            elif -0.02 < action_reward < -0.001:
                reward += 0.05  # Corta loss cedo
        
        # Penalidade overtrading
        if discrete_action != 0 or (self.position == 0 and hasattr(self, '_prev_position') and self._prev_position != 0):
            self.last_24h_trades.append(self.episode_length)
        
        self.last_24h_trades = [t for t in self.last_24h_trades if self.episode_length - t <= 96]
        
        if len(self.last_24h_trades) > 10:
            reward -= (len(self.last_24h_trades) - 10) * 0.01
        
        if self.position != getattr(self, '_prev_position', 0):
            self._last_position_change_step = self.episode_length
        
        self._prev_position = self.position
        
        # Return
        step_return = (self.equity - self.previous_equity) / self.previous_equity
        self.returns_history.append(step_return)
        
        # V17.2 FIX: Limitar returns_history para prevenir memory leak
        if len(self.returns_history) > 2000:
            self.returns_history = self.returns_history[-2000:]
        
        # Sharpe (peso mínimo)
        if self.use_sharpe_reward and len(self.returns_history) > 10:
            returns_array = np.array(self.returns_history[-100:])
            mean_return = returns_array.mean()
            std_return = returns_array.std() + 1e-8
            sharpe = mean_return / std_return
            reward += np.tanh(sharpe * 10) * 0.05
        
        self.previous_equity = self.equity
        
        # Avançar
        self.current_step += 1
        self.episode_length += 1
        
        # Terminação
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
        
        # Max steps
        if self.episode_length >= self.max_episode_steps:
            truncated = True
        
        # Persistent balance
        if self.persist_balance:
            self.persistent_balance = self.balance
            self.persistent_equity = self.equity
        
        # V17.6 FIX: SEM reward scaling - o problema era critic overfitting, não gradiente
        # V17.4/5 scaling causava VF divergência imediata
        # Solução real: gamma=0.95 + n_epochs=4 + vf_coef=0.1 no PPO
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    
    # === MÉTODOS DE TRADING (IGUAIS V16.3) ===
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calcula PnL da posição atual."""
        if self.position == 0:
            return 0.0
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.position == 1:
            pnl = self.position_value * price_change
        else:
            pnl = -self.position_value * price_change
        
        return pnl
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Executa ação de trading."""
        action_reward = 0.0
        
        if self.position != 0 and action != 0 and action != (1 if self.position == 1 else 2):
            pnl = self._calculate_pnl(current_price)
            action_reward = self._close_position(current_price)
        
        if action == 0:
            if self.position != 0:
                action_reward = self._close_position(current_price)
        
        elif action == 1:
            if self.position == 0:
                self._open_position(1, current_price)
        
        elif action == 2:
            if self.position == 0:
                self._open_position(-1, current_price)
        
        return action_reward
    
    def _open_position(self, direction: int, price: float):
        """Abre posição Long (1) ou Short (-1)."""
        self.position = direction
        self.entry_price = price
        self.current_position_side = direction

        effective_price = price * (1 + self.slippage) if direction == 1 else price * (1 - self.slippage)
        
        position_amount = self.balance * self.current_position_size
        self.position_value = position_amount * self.leverage
        
        commission_cost = self.position_value * self.commission
        self.balance -= commission_cost
        
        self.last_trade_step = self.episode_length
        
        if direction == 1:
            self.long_trades += 1
        else:
            self.short_trades += 1
    
    def _close_position(self, price: float) -> float:
        """Fecha posição e retorna reward (PnL %)."""
        if self.position == 0:
            return 0.0
        
        pnl = self._calculate_pnl(price)
        
        effective_price = price * (1 - self.slippage) if self.position == 1 else price * (1 + self.slippage)
        
        commission_cost = self.position_value * self.commission
        pnl -= commission_cost
        
        self.balance += pnl
        self.total_pnl += pnl
        
        self.trades += 1
        if pnl > 0:
            self.wins += 1
            if self.current_position_side == 1:
                self.long_wins += 1
            else:
                self.short_wins += 1
        else:
            self.losses += 1
            if self.current_position_side == 1:
                self.long_losses += 1
            else:
                self.short_losses += 1
        
        # Sinaliza evento de trade fechado para _get_info
        self._trade_just_closed = True
        self._last_closed_pnl   = pnl
        self._last_closed_side  = self.current_position_side
        
        reward = pnl / self.initial_balance
        
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        self.current_position_side = 0
        
        return reward
