"""
Ambiente de Trading para Reinforcement Learning usando Gymnasium.
Este é o coração do sistema - onde o agente aprende a operar.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Ambiente de Trading personalizado seguindo a interface do Gymnasium.
    
    Observation Space:
        - Preços OHLCV normalizados
        - Indicadores técnicos
        - Estado da carteira (saldo, posição, PnL)
        
    Action Space:
        0: Flat (Ficar de fora)
        1: Long (Comprar)
        2: Short (Vender)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame = None,
        data_path: str = None,
        config: Dict = None,
        initial_balance: float = 10000,
        commission: float = 0.0004,  # 0.04% - taxa REALISTA da Binance (taker fee)
        slippage: float = 0.0005,  # 0.05% - slippage realista em mercado líquido
        leverage: int = 3,
        position_size: float = 0.1,
        window_size: int = 50,
        max_episode_steps: int = 1500,  # NOVA: Trunca episódios para aumentar exploration
        random_start: bool = True,  # NOVA: Começa em pontos aleatórios do dataset
        sentiment_features: np.ndarray = None
    ):
        """
        Args:
            df: DataFrame com dados OHLCV e indicadores
            data_path: Caminho alternativo para carregar dados
            config: Dicionário de configuração
            initial_balance: Saldo inicial em USDT
            commission: Taxa de corretagem (0.0004 = 0.04%)
            leverage: Alavancagem máxima
            position_size: Fração do saldo por trade
            window_size: Janela de observação (candles)
            max_episode_steps: Máximo de steps por episódio (truncação para exploration)
            random_start: Se True, cada episódio começa em ponto aleatório do dataset
            sentiment_features: Features de sentimento (opcional)
        """
        super().__init__()
        
        # Carrega dados
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif data_path is not None:
            df_loaded = pd.read_csv(data_path)
            # Remove colunas não-numéricas (como timestamp)
            numeric_cols = df_loaded.select_dtypes(include=[np.number]).columns
            self.df = df_loaded[numeric_cols].reset_index(drop=True)
        else:
            raise ValueError("Forneça df ou data_path")
        
        # Carrega config
        if config:
            initial_balance = config.get('initial_balance', initial_balance)
            commission = config.get('commission', commission)
            slippage = config.get('slippage', slippage)
            leverage = config.get('leverage', leverage)
            position_size = config.get('position_size', position_size)
            window_size = config.get('window_size', window_size)
            max_episode_steps = config.get('max_episode_steps', max_episode_steps)
            random_start = config.get('random_start', random_start)
        
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage  # Slippage em fração (0.0005 = 0.05%)
        self.leverage = leverage
        self.position_size = position_size
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        
        # Features de sentimento (opcional)
        self.sentiment_features = sentiment_features
        self.n_sentiment_features = 0
        if sentiment_features is not None:
            self.n_sentiment_features = sentiment_features.shape[1] if len(sentiment_features.shape) > 1 else 1
        
        # Espaço de Ações: Box contínuo [-1, 1] para compatibilidade com SAC/TD3
        # -1 a -0.33: Short | -0.33 a 0.33: Flat | 0.33 a 1: Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Espaço de Observações: [preços, indicadores, carteira, sentimento]
        # Conta apenas colunas numéricas (exclui timestamp/strings)
        df_numeric = self.df.select_dtypes(include=[np.number])
        self.n_features = len(df_numeric.columns)
        obs_shape = (window_size, self.n_features + 3 + self.n_sentiment_features)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Estado inicial
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reinicia o ambiente para um novo episódio."""
        super().reset(seed=seed)
        
        # Estado da conta
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0
        self.position_value = 0
        
        # Métricas
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        
        # Equity anterior para reward (delta equity)
        self.previous_equity = self.initial_balance
        
        # NOVA: Random start position para diversificar exploração
        # Cada episódio vê uma parte diferente do dataset
        if self.random_start:
            # Garante espaço suficiente: [window_size, len(df) - max_episode_steps]
            max_start = max(self.window_size, len(self.df) - self.max_episode_steps - 1)
            self.episode_start = np.random.randint(self.window_size, max_start)
        else:
            self.episode_start = self.window_size
        
        # Índice temporal
        self.current_step = self.episode_start
        self.episode_length = 0  # Contador de steps do episódio atual
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa uma ação no ambiente.
        
        Args:
            action: Array [-1, 1] -> converte em 0 (Flat), 1 (Long), 2 (Short)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Converte action contínuo para discreto
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        if action_value < -0.33:
            discrete_action = 2  # Short
        elif action_value > 0.33:
            discrete_action = 1  # Long
        else:
            discrete_action = 0  # Flat
        
        # Preço atual
        current_price = self.df.loc[self.current_step, 'close']
        
        # Calcular PnL da posição anterior
        pnl = self._calculate_pnl(current_price)
        
        # Executar ação
        reward = self._execute_action(discrete_action, current_price)
        
        # Atualizar estado
        self.equity = self.balance + self.position_value
        
        # REWARD PURA: Apenas delta equity normalizado
        # O mercado ensina: Long em uptrend = positivo, Short em downtrend = positivo
        # Flat em lateral = neutro (sem penalidade artificial)
        # Custos reais (fees/slippage) já estão no balance
        reward = (self.equity - self.previous_equity) / self.initial_balance
        
        # Atualizar equity anterior
        self.previous_equity = self.equity
        
        self.current_step += 1
        self.episode_length += 1
        
        # Verificar se terminou
        # NOVA: Trunca episódio após max_episode_steps (força exploration via resets)
        terminated = self.current_step >= len(self.df) - 1
        truncated = (
            self.equity <= self.initial_balance * 0.5 or  # Stop se perder 50%
            self.episode_length >= self.max_episode_steps  # Trunca após N steps
        )
        
        # Fechar posição aberta ao terminar episódio (para métricas precisas)
        if (terminated or truncated) and self.position != 0:
            final_price = self.df.loc[self.current_step - 1, 'close']
            self._close_position(final_price)
            self.equity = self.balance  # Atualiza equity final
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Executa a ação de trading SEM reward shaping.
        
        A reward agora é PURAMENTE baseada em delta equity (calculada no step()),
        permitindo que o agente aprenda QUALQUER estratégia viável.
        
        Lógica:
        - action 0: Fecha posição se houver
        - action 1: Abre/mantém Long
        - action 2: Abre/mantém Short
        """
        # Mapeia ação para posição: 0 -> 0, 1 -> 1, 2 -> -1
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # Se a ação mudou
        if target_position != self.position:
            self.trades += 1  # Incrementa contador de trades
            
            # Fecha posição atual se existir
            if self.position != 0:
                self._close_position(current_price)
                
            # Abre nova posição se não for Flat
            if target_position != 0:
                self._open_position(target_position, current_price)
    
    def _open_position(self, position_type: int, price: float):
        """
        Abre uma posição aplicando slippage e fees realistas.
        
        Args:
            position_type: 1 (Long) ou -1 (Short)
            price: Preço de mercado base
        """
        # Aplicar slippage: Long paga mais, Short recebe menos
        if position_type == 1:  # Long
            execution_price = price * (1 + self.slippage)
        else:  # Short
            execution_price = price * (1 - self.slippage)
        
        self.position = position_type
        self.entry_price = execution_price
        
        # Tamanho da posição em USDT
        position_usdt = self.balance * self.position_size * self.leverage
        
        # Fees: cobrado sobre o valor da posição
        fee = position_usdt * self.commission
        self.balance -= fee  # Desconta fee do saldo
        
        self.position_value = position_usdt * position_type
        
    def _close_position(self, current_price: float) -> float:
        """
        Fecha a posição atual aplicando slippage e fees realistas.
        Retorna PnL realizado (já descontado balance e fees).
        """
        if self.position == 0:
            return 0
        
        # Aplicar slippage ao fechar: Long recebe menos, Short paga mais
        if self.position == 1:  # Long (vende)
            execution_price = current_price * (1 - self.slippage)
        else:  # Short (compra)
            execution_price = current_price * (1 + self.slippage)
        
        # Calcular PnL com preço de execução ajustado
        pnl = self._calculate_pnl(execution_price)
        
        # Cobrar fee ao fechar
        fee = abs(self.position_value) * self.commission
        pnl -= fee  # Desconta fee do PnL
        
        # Atualiza saldo e métricas
        self.balance += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        # Reseta posição
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        
        return pnl
    
    def _calculate_pnl(self, current_price: float) -> float:
        """
        Calcula o PnL não realizado da posição atual.
        """
        if self.position == 0 or self.entry_price == 0:
            return 0
        
        # PnL = (preço_atual - preço_entrada) * valor_posição / preço_entrada
        price_change = (current_price - self.entry_price) / self.entry_price
        
        # Long: ganha se subir | Short: ganha se cair
        pnl = price_change * abs(self.position_value) * self.position
        
        return pnl
    
    def _get_observation(self) -> np.ndarray:
        """
        Retorna a observação atual do ambiente.
        Inclui: histórico de preços/indicadores NORMALIZADOS + estado da carteira + sentimento.
        """
        # Janela de dados históricos
        start = self.current_step - self.window_size
        end = self.current_step
        
        # Exclui coluna timestamp se existir
        df_numeric = self.df.select_dtypes(include=[np.number])
        historical_data = df_numeric.iloc[start:end].values
        
        # NORMALIZAÇÃO Z-SCORE: média 0, desvio padrão 1
        # Calcula estatísticas da janela atual (evita look-ahead bias)
        mean = historical_data.mean(axis=0, keepdims=True)
        std = historical_data.std(axis=0, keepdims=True) + 1e-8  # Evita divisão por zero
        historical_data = (historical_data - mean) / std
        
        # Estado da carteira (normalizado)
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Saldo normalizado
            self.position,  # -1, 0 ou 1
            self.equity / self.initial_balance  # Equity normalizado
        ])
        
        # Replica o estado da carteira para cada timestep
        portfolio_matrix = np.tile(portfolio_state, (self.window_size, 1))
        
        # Concatena dados históricos + estado da carteira
        observation = np.concatenate([historical_data, portfolio_matrix], axis=1)
        
        # Adiciona features de sentimento se disponíveis
        if self.sentiment_features is not None and len(self.sentiment_features) > end:
            sentiment_window = self.sentiment_features[start:end]
            
            # Garante que tem a forma correta
            if len(sentiment_window.shape) == 1:
                sentiment_window = sentiment_window.reshape(-1, 1)
            
            # Replica para cada timestep se necessário
            if sentiment_window.shape[0] != self.window_size:
                # Repete primeiro valor conhecido (mais realista que zeros)
                n_missing = self.window_size - sentiment_window.shape[0]
                first_value = sentiment_window[0] if len(sentiment_window) > 0 else np.zeros(sentiment_window.shape[1])
                padding = np.tile(first_value, (n_missing, 1))
                sentiment_window = np.vstack([padding, sentiment_window])
            
            observation = np.concatenate([observation, sentiment_window], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Retorna informações adicionais sobre o estado atual."""
        win_rate = self.wins / self.trades if self.trades > 0 else 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_step': self.current_step
        }
    
    def render(self):
        """Renderiza o estado atual (modo texto)."""
        info = self._get_info()
        print(f"\n=== Step {info['current_step']} ===")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Equity: ${info['equity']:.2f}")
        print(f"Position: {['Flat', 'Long', 'Short'][self.position + 1]}")
        print(f"Trades: {info['trades']} | Wins: {info['wins']} | Losses: {info['losses']}")
        print(f"Win Rate: {info['win_rate']:.2%}")
        print(f"Total PnL: ${info['total_pnl']:.2f}")
