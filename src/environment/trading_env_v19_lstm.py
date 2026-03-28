"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      🧠 TRADING ENVIRONMENT - MULTI-TIMEFRAME LSTM (V19)                    ║
║                                                                              ║
║  🔧 CORREÇÕES CRÍTICAS APLICADAS (vs V18):                                  ║
║                                                                              ║
║  ❌ BUG V18: np.clip(-100, 100) destruía BTC/ETH OHLCV                     ║
║     BTC close = $27 839 → clipeado para 100 (CONSTANTE!)                   ║
║     5/20 features eram constante 100 → modelo via ruído puro               ║
║                                                                              ║
║  ✅ FIX V19: _normalize_ohlcv() ANTES do clip                               ║
║     open/high/low → % relativo ao close (range típico: -5 a +5%)           ║
║     close         → retorno % do candle anterior (range: -3 a +3%)         ║
║     volume        → ratio vol/vol_ma20 (range típico: 0.1 a 5.0)           ║
║     Depois do normalize → scale-invariant, safe para clip(-10, 10)         ║
║                                                                              ║
║  ✅ leverage: 1.0 → 1.5 (alinha com produção dashboard)                    ║
║                                                                              ║
║  ✅ stop-loss interno: -7% → -4.67% (equivalente: -7%/leverage=1.5)        ║
║     isso corresponde ao mesmo -7% de equity que o dashboard usa            ║
║                                                                              ║
║  ✅ vf_coef: tratado no script de treino (0.1→0.5)                         ║
║                                                                              ║
║  🔧 CORREÇÕES V19.1:                                                         ║
║  ❌ BUG V19: w[:,4] = w[:,4] / w[:,19] → raw_vol × vol_ma⁻¹ (enorme)      ║
║     col 19 JÁ É ratio vol/vol_ma → dividir de novo = std≈0, clip=10 fixo  ║
║  ✅ FIX V19.1: w[:,4] = w[:,19].copy() — ratio pré-calculado no CSV        ║
║                                                                              ║
║  ❌ BUG V19: reward = pnl/initial_balance → ~0.0001 por trade              ║
║     400× menor que penalidades SL (-0.08) → policy prefere flat            ║
║  ✅ FIX V19.1: reward = pnl/position_value → ±0.005 a ±0.05 (semântico)   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


class TradingEnvV19LSTM(gym.Env):
    """
    Ambiente de Trading Multi-Timeframe para LSTM — V19.

    DIFERENÇA CHAVE vs V18:
    - _normalize_ohlcv() converte OHLCV absolutos em valores relativos
      ANTES do clip. BTC/ETH passam escala normalmente, igual SOL/BNB.
    - leverage=1.5 (antes 1.0) — alinhado com dashboard de produção.
    - stop-loss interno -4.67% (= -7% / 1.5) — mesmo risco real.

    Observation Space:
        - Shape: (seq_len=50, features_per_step=31)
        - 20 features: OHLCV normalizados + indicadores do 15m
        - 4  features: 1h context (RSI, BBP, MACDh, close%diff)
        - 4  features: 4h context (RSI, BBP, MACDh, close%diff)
        - 3  features: portfolio state (balance_norm, position, equity_norm)

    Action Space:
        Contínuo [-1, 1]:
        - [-1.0, -0.1]: Short
        - (-0.1, 0.1): Flat
        - [0.1, 1.0]: Long
    """

    metadata = {'render_modes': ['human']}

    # ── Colunas do CSV (após drop de não-numéricas) ───────────────────────────
    # Ordem típica gerada por collect_multi_pair_mtf:
    #   0=open, 1=high, 2=low, 3=close, 4=volume,
    #   5=RSI_14, 6=SMA_20, 7=SMA_50, 8=EMA_9, 9=EMA_21,
    #   10=BBL_20_2.0, 11=BBU_20_2.0, 12=BBP_20_2.0, 13=BBB_20_2.0,
    #   14=MACD_12_26_9, 15=MACDh_12_26_9, 16=MACDs_12_26_9,
    #   17=ATR_14, 18=STOCHk_14_3_3, 19=Volume_MA_20
    IDX_OPEN  = 0
    IDX_HIGH  = 1
    IDX_LOW   = 2
    IDX_CLOSE = 3
    IDX_VOL   = 4
    IDX_VOL_MA = 19
    IDX_RSI   = 5
    IDX_BBP   = 12
    IDX_MACDH = 15

    # ── Stop-loss threshold ──────────────────────────────────────────────────
    # Dashboard usa -7% de equity com leverage 1.5.
    # No env, position_value usa leverage → unrealized_pct já inclui alavancagem.
    # Para obter mesmo risco: -7% de equity / leverage = -4.67% de unrealized.
    STOP_LOSS_THRESHOLD = -0.04  # V19.3: -4% do balance (era -7% → reduzia MaxDD de 25-36% → <15%)

    def __init__(
        self,
        data_paths: Dict[str, str],    # {'15m': path, '1h': path, '4h': path}
        window_size: int = 50,
        initial_balance: float = 10000,
        commission: float = 0.0004,
        slippage: float = 0.0005,
        leverage: float = 1.5,         # V19: 1.5 (era 1.0 no V18)
        position_size: float = 0.05,
        max_episode_steps: int = 2000,
        random_start: bool = True,
        persist_balance: bool = False,
        use_sharpe_reward: bool = False,
        maintenance_margin_rate: float = 0.005,
        liquidation_threshold: float = 0.30,
        enable_indicator_shaping: bool = False,
        trade_cooldown: int = 4,        # V19.3: mínimo steps entre trades (4×15m = 1h)
    ):
        super().__init__()

        # Carregar dados de múltiplos timeframes
        self.data_paths = data_paths
        self.dfs = {}
        self.df_values = {}
        self.n_features = {}

        for tf, path in data_paths.items():
            df = pd.read_csv(path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.dfs[tf] = df[numeric_cols].reset_index(drop=True)
            self.df_values[tf] = self.dfs[tf].values
            self.n_features[tf] = len(numeric_cols)

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
        self.trade_cooldown = trade_cooldown  # V19.3

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

        # Action Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: SEQUENCIAL (seq_len, features_per_step)
        # 20 (15m) + 4 (1h) + 4 (4h) + 3 (portfolio) = 31
        features_per_step = self.n_features['15m'] + 4 + 4 + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, features_per_step),
            dtype=np.float32
        )

        print(f"\n🧠 LSTM Environment Initialized (V19.3 — REDUCED OVERTRADING):")
        print(f"   Obs shape: {self.observation_space.shape} (SEQUENTIAL)")
        print(f"   15m features: {self.n_features['15m']}")
        print(f"   1h context:   4 (RSI, BBP, MACDh, close%diff)")
        print(f"   4h context:   4 (RSI, BBP, MACDh, close%diff)")
        print(f"   portfolio:    3 (balance_norm, position, equity_norm)")
        print(f"   Total per step: {features_per_step}")
        print(f"   ✅ _normalize_ohlcv() ativa → sem clip de preços absolutos")
        print(f"   ✅ leverage = {self.leverage}  (alinhado com produção)")
        print(f"   ✅ stop-loss = {self.STOP_LOSS_THRESHOLD*100:.1f}% do balance  (V19.3: era -7%)")
        print(f"   ✅ trade_cooldown = {self.trade_cooldown} steps = {self.trade_cooldown*15}min entre trades  (V19.3)")
        print(f"   ✅ overtrading limit = 6 trades/24h, penalidade 3× maior  (V19.3)")
        print(f"   ✅ small-loss bonus removido do reward shaping  (V19.3 fix)\n")

        self.reset()

    # ── Alinhamento Temporal ──────────────────────────────────────────────────
    def _verify_timeframe_alignment(self):
        """Verifica alinhamento temporal dos timeframes."""
        len_15m = len(self.dfs['15m'])
        len_1h  = len(self.dfs['1h'])
        len_4h  = len(self.dfs['4h'])

        ratio_1h = len_15m / len_1h
        ratio_4h = len_1h  / len_4h

        print(f"\n📊 ALINHAMENTO DE TIMEFRAMES:")
        print(f"   15m: {len_15m:,} candles")
        print(f"   1h:  {len_1h:,} candles (ratio: {ratio_1h:.2f})")
        print(f"   4h:  {len_4h:,} candles (ratio: {ratio_4h:.2f})")

        if not (3.5 < ratio_1h < 4.5):
            print(f"   ⚠️  Warning: 15m/1h ratio fora do esperado")
        if not (3.5 < ratio_4h < 4.5):
            print(f"   ⚠️  Warning: 1h/4h ratio fora do esperado")

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reinicia o ambiente."""
        super().reset(seed=seed)

        if self.persist_balance:
            if self.persistent_balance < self.initial_balance * 0.6:
                self.persistent_balance = self.initial_balance
                self.persistent_equity  = self.initial_balance
            self.balance = self.persistent_balance
            self.equity  = self.persistent_equity
        else:
            self.balance = self.initial_balance
            self.equity  = self.initial_balance

        # Estado da conta
        self.position       = 0
        self.entry_price    = 0
        self.position_value = 0

        # Métricas
        self.trades          = 0
        self.wins            = 0
        self.losses          = 0
        self.total_pnl       = 0
        self.episode_liquidations = 0
        self.last_trade_step = 0
        self.steps_since_last_trade = self.trade_cooldown  # V19.3: pronto para operar no início
        self.long_trades     = 0
        self.short_trades    = 0
        self.long_wins       = 0
        self.long_losses     = 0
        self.short_wins      = 0
        self.short_losses    = 0
        self.current_position_side = 0
        self._trade_just_closed    = False
        self._last_closed_pnl      = 0.0
        self._last_closed_side     = 0
        self._prev_position        = 0
        self._last_position_change_step = 0
        self.episode_start_trades  = self.trades

        self.previous_equity  = self.equity
        self.returns_history  = []

        # Random start
        if self.random_start:
            max_start = max(
                self.window_size,
                len(self.dfs['15m']) - self.max_episode_steps - 1
            )
            self.episode_start = np.random.randint(self.window_size, max_start)
        else:
            self.episode_start = self.window_size

        self.current_step  = self.episode_start
        self.episode_length = 0

        return self._get_observation(), self._get_info()

    # ── Normalização OHLCV ────────────────────────────────────────────────────
    @staticmethod
    def _normalize_ohlcv(window: np.ndarray) -> np.ndarray:
        """
        V19 FIX CRÍTICO: converte OHLCV absolutos (ex: BTC=$27 839) em
        valores relativos scale-invariant antes do clip.

        BUG V18: np.clip(-100, 100) recebia BTC close=$27 839
                 → clipeava para 100 (constante!) — o modelo via ruído.

        V19: normaliza primeiro, DEPOIS clipa com segurança (-10, 10).

        Transformações (in-place na cópia):
          open  → (open / close - 1) × 100   [% de desvio vs close]
          high  → (high / close - 1) × 100   [upper wick %]
          low   → (low  / close - 1) × 100   [lower wick %]
          close → (close / prev_close - 1) × 100  [retorno % do step]
          vol   → volume / vol_ma20           [ratio relativo]

          Colunas 5..19 (indicadores): sem mudança — já normalizados no CSV
          (RSI 0-100, BBP 0-1, MACD values menores, ATR %, etc.)

        Args:
            window: (seq_len, n_features) — cópia DA janela 15m extraída
        Returns:
            window modificado in-place (mesma shape)
        """
        w = window.copy().astype(np.float64)

        # Referência: close price de cada candle (coluna 3)
        close = w[:, 3].copy()
        close[close == 0] = 1.0  # evitar divisão por zero

        # --- open/high/low → desvio % vs close ---
        w[:, 0] = (w[:, 0] / close - 1.0) * 100   # open  vs close (tipicamente ~0)
        w[:, 1] = (w[:, 1] / close - 1.0) * 100   # high  vs close (sempre positivo)
        w[:, 2] = (w[:, 2] / close - 1.0) * 100   # low   vs close (sempre negativo)

        # --- close → retorno % em relação ao candle anterior ---
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]                    # primeiro step: retorno=0
        w[:, 3] = (close / (prev_close + 1e-10) - 1.0) * 100

        # --- volume → ratio vol / vol_ma20 ---
        # V19.1 FIX: col 19 (Volume_MA_20) JÁ É o ratio vol/vol_ma gerado no CSV
        # BUG V19: dividia raw_volume pelo ratio → raw_vol × vol_ma_abs (enorme)
        #          → clip(-10,10) = constante 10 | std≈0.0 (sinal morto)
        w[:, 4] = w[:, 19].copy()  # usar o ratio pré-computado diretamente

        # Colunas 5..19 (indicadores) já são normalizadas no CSV — manter sem alteração

        return w.astype(np.float32)

    # ── Observação ────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        """
        V19: Observação SEQUENCIAL com OHLCV normalizado + multi-timeframe real.

        Returns:
            Array 2D (seq_len=50, features_per_step=31)
            - 20: Features do 15m  (OHLCV NORMALIZADOS + indicadores)
            - 4:  1h context (RSI, BBP, MACDh, close%diff)
            - 4:  4h context (RSI, BBP, MACDh, close%diff)
            - 3:  Portfolio (balance_norm, position, equity_norm)
        """
        # Janela de 50 candles do 15m
        start_15m = max(0, self.current_step - self.window_size)
        end_15m   = self.current_step
        raw_15m   = self.df_values['15m'][start_15m:end_15m]  # (≤50, 20)

        # Pad se necessário
        if len(raw_15m) < self.window_size:
            padding = np.zeros((self.window_size - len(raw_15m), raw_15m.shape[1]))
            raw_15m = np.vstack([padding, raw_15m])

        # ✅ V19 FIX: normalizar OHLCV ANTES do clip
        window_15m = self._normalize_ohlcv(raw_15m)  # (50, 20) — scale-invariant agora

        # === CONTEXT DE 1h e 4h (4 features cada) ===
        ctx_1h = np.zeros((self.window_size, 4), dtype=np.float32)
        ctx_4h = np.zeros((self.window_size, 4), dtype=np.float32)

        len_1h = len(self.df_values['1h'])
        len_4h = len(self.df_values['4h'])

        for i in range(self.window_size):
            step_15m = start_15m + i

            # Close do 15m normalizado (retorno %, coluna 3 já foi convertida)
            # Para o close%diff dos contextos, usamos o close absoluto ORIGINAL
            price_15m = float(raw_15m[i, self.IDX_CLOSE])
            if price_15m == 0:
                price_15m = 1.0

            # --- 1h context ---
            idx_1h = max(0, (step_15m - 1) // 4)
            if idx_1h < len_1h:
                row_1h = self.df_values['1h'][idx_1h]
                ctx_1h[i, 0] = row_1h[self.IDX_RSI]    # RSI_14
                ctx_1h[i, 1] = row_1h[self.IDX_BBP]    # BBP_20_2.0
                ctx_1h[i, 2] = row_1h[self.IDX_MACDH]  # MACDh_12_26_9
                ctx_1h[i, 3] = (row_1h[self.IDX_CLOSE] / price_15m - 1) * 100

            # --- 4h context ---
            idx_4h = max(0, (step_15m - 1) // 16)
            if idx_4h < len_4h:
                row_4h = self.df_values['4h'][idx_4h]
                ctx_4h[i, 0] = row_4h[self.IDX_RSI]
                ctx_4h[i, 1] = row_4h[self.IDX_BBP]
                ctx_4h[i, 2] = row_4h[self.IDX_MACDH]
                ctx_4h[i, 3] = (row_4h[self.IDX_CLOSE] / price_15m - 1) * 100

        # === PORTFOLIO ===
        balance_col  = np.full((self.window_size, 1), self.balance / self.initial_balance)
        position_col = np.full((self.window_size, 1), float(self.position))
        equity_col   = np.full((self.window_size, 1), self.equity / self.initial_balance)

        # === CONCATENAR: (50,20)+(50,4)+(50,4)+(50,1)+(50,1)+(50,1) = (50,31) ===
        observation = np.hstack([
            window_15m,   # (50, 20) — OHLCV normalizados + indicadores
            ctx_1h,       # (50, 4)
            ctx_4h,       # (50, 4)
            balance_col,  # (50, 1)
            position_col, # (50, 1)
            equity_col    # (50, 1)
        ]).astype(np.float32)

        # Clip conservador: após normalização, valores devem estar em ~(-10, 10)
        # Clip em (-10, 10) é seguro e ainda captura spikes de volatilidade
        observation = np.clip(observation, -10, 10)

        return observation

    # ── Info ──────────────────────────────────────────────────────────────────
    def _get_info(self) -> Dict:
        """Retorna informações adicionais. Inclui trade_executed para o backtest."""
        win_rate = (self.wins / self.trades * 100)  if self.trades  > 0 else 0
        long_wr  = (self.long_wins  / self.long_trades  * 100) if self.long_trades  > 0 else 0
        short_wr = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0

        trade_executed = self._trade_just_closed
        last_pnl       = self._last_closed_pnl
        last_side      = self._last_closed_side
        self._trade_just_closed = False
        self._last_closed_pnl   = 0.0
        self._last_closed_side  = 0

        return {
            'balance':      self.balance,
            'equity':       self.equity,
            'position':     self.position,
            'trades':       self.trades,
            'wins':         self.wins,
            'losses':       self.losses,
            'win_rate':     win_rate,
            'total_pnl':    self.total_pnl,
            'liquidations': self.episode_liquidations,
            'long_trades':  self.long_trades,
            'short_trades': self.short_trades,
            'long_wins':    self.long_wins,
            'long_losses':  self.long_losses,
            'long_wr':      long_wr,
            'short_wins':   self.short_wins,
            'short_losses': self.short_losses,
            'short_wr':     short_wr,
            'trade_executed': trade_executed,
            'pnl':            last_pnl,
            '_side':          last_side,
        }

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executa ação no ambiente."""
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        if action_value < -0.1:
            discrete_action = 2   # Short
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        elif action_value > 0.1:
            discrete_action = 1   # Long
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        else:
            discrete_action = 0   # Flat
            self.current_position_size = 0

        # V19.3: Trade cooldown — forçar FLAT se ainda em cooldown após último trade
        self.steps_since_last_trade += 1
        if self.position == 0 and discrete_action != 0 and self.steps_since_last_trade < self.trade_cooldown:
            discrete_action = 0
            self.current_position_size = 0

        current_price = self.dfs['15m'].iloc[self.current_step]['close']

        pnl = self._calculate_pnl(current_price)

        # ── Stop-loss automático ──────────────────────────────────────────────
        # V19: threshold -7% do balance (position_value já inclui leverage 1.5,
        # então -7% de balance ≈ -4.67% de position notional — alinhado com prod)
        stop_loss_triggered = False
        reward = 0.0
        if self.position != 0:
            unrealized_pnl = self._calculate_pnl(current_price)
            unrealized_pct = unrealized_pnl / self.balance

            if unrealized_pct <= self.STOP_LOSS_THRESHOLD:
                # V19.1 FIX: usar reward real do _close_position (pnl/pos_val)
                # range típico: -0.3 a -0.6 dependendo do tamanho da perda
                reward = self._close_position(current_price)
                stop_loss_triggered = True
                discrete_action = 0

        # ── Ação normal ───────────────────────────────────────────────────────
        if not stop_loss_triggered:
            action_reward = self._execute_action(discrete_action, current_price)
            reward = action_reward

            if discrete_action == 0 and self.position == 0:
                reward -= 0.00005  # V19.3: reduzido de -0.0002 → -0.00005 (era forte demais, forçava overtrading)
        else:
            action_reward = 0.0
            # reward já setado acima via _close_position (penalidade natural do PnL)

        # Atualizar equity
        self.equity = self.balance + self.position_value

        # ── Reward Shaping (idêntico V17.7) ──────────────────────────────────
        time_since_trade = self.episode_length - self.last_trade_step
        if time_since_trade > 200:
            reward -= 0.00002 * (time_since_trade - 200)

        if self.position != 0:
            holding_time = self.episode_length - self.last_trade_step
            if holding_time > 50:
                reward -= 0.0003

        if action_reward != 0:
            if action_reward > 0.02:
                reward += 0.05
            elif action_reward < -0.02:
                reward -= 0.05
            # V19.3 FIX: Removido bônus por pequena perda (era +0.05 para -0.02 < r < -0.001)
            # Esse incentivo perverso premiava trades ruins e causava overtrading

        # Penalidade overtrading
        if discrete_action != 0 or (self.position == 0 and getattr(self, '_prev_position', 0) != 0):
            self.last_24h_trades.append(self.episode_length)

        self.last_24h_trades = [t for t in self.last_24h_trades
                                if self.episode_length - t <= 96]

        if len(self.last_24h_trades) > 6:  # V19.3: reduzido de 10 → 6 trades/24h
            reward -= (len(self.last_24h_trades) - 6) * 0.03  # V19.3: penalidade aumentada de 0.01 → 0.03

        if self.position != getattr(self, '_prev_position', 0):
            self._last_position_change_step = self.episode_length

        self._prev_position = self.position

        # Step return
        step_return = (self.equity - self.previous_equity) / self.previous_equity
        self.returns_history.append(step_return)

        if len(self.returns_history) > 2000:
            self.returns_history = self.returns_history[-2000:]

        if self.use_sharpe_reward and len(self.returns_history) > 10:
            returns_array = np.array(self.returns_history[-100:])
            mean_return   = returns_array.mean()
            std_return    = returns_array.std() + 1e-8
            sharpe        = mean_return / std_return
            reward       += np.tanh(sharpe * 10) * 0.05

        self.previous_equity = self.equity
        self.current_step   += 1
        self.episode_length += 1

        # ── Terminação ────────────────────────────────────────────────────────
        terminated = False
        truncated  = False

        if self.equity <= self.initial_balance * (1 - self.liquidation_threshold):
            terminated = True
            reward     = -1.0
            self.liquidations += 1
            self.episode_liquidations += 1

        if self.current_step >= len(self.dfs['15m']) - 1:
            truncated = True

        if self.episode_length >= self.max_episode_steps:
            truncated = True

        if self.persist_balance:
            self.persistent_balance = self.balance
            self.persistent_equity  = self.equity

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ── Métodos de Trading ────────────────────────────────────────────────────
    def _calculate_pnl(self, current_price: float) -> float:
        """Calcula PnL da posição atual."""
        if self.position == 0:
            return 0.0

        price_change = (current_price - self.entry_price) / self.entry_price

        if self.position == 1:
            return self.position_value * price_change
        else:
            return -self.position_value * price_change

    def _execute_action(self, action: int, current_price: float) -> float:
        """Executa ação de trading."""
        action_reward = 0.0

        if self.position != 0 and action != 0 and action != (1 if self.position == 1 else 2):
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

        position_amount = self.balance * self.current_position_size
        self.position_value = position_amount * self.leverage

        commission_cost = self.position_value * self.commission
        self.balance -= commission_cost

        self.last_trade_step = self.episode_length
        self.steps_since_last_trade = 0  # V19.3: reinicia cooldown ao abrir posição

        if direction == 1:
            self.long_trades += 1
        else:
            self.short_trades += 1

    def _close_position(self, price: float) -> float:
        """Fecha posição e retorna reward (PnL %)."""
        if self.position == 0:
            return 0.0

        pnl = self._calculate_pnl(price)

        commission_cost = self.position_value * self.commission
        pnl -= commission_cost

        self.balance   += pnl
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

        self._trade_just_closed = True
        self._last_closed_pnl   = pnl
        self._last_closed_side  = self.current_position_side

        # V19.1 FIX: normalizar reward pelo valor da posição (% return semântico)
        # range típico: ±0.005 (0.5% trade) a ±0.05 (5% swing)
        # vs. V19 bug: pnl/initial_balance → ~0.0001 (sinal 50-400× menor que SL)
        pos_val = self.position_value  # capturar ANTES de zerar
        reward = pnl / (pos_val + 1e-8)

        self.position             = 0
        self.entry_price          = 0
        self.position_value       = 0
        self.current_position_side = 0

        return reward
