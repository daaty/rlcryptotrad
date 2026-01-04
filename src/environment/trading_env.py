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
        commission: float = 0.0004,
        leverage: int = 3,
        position_size: float = 0.1,
        window_size: int = 50,
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
            leverage = config.get('leverage', leverage)
            position_size = config.get('position_size', position_size)
            window_size = config.get('window_size', window_size)
        
        self.initial_balance = initial_balance
        self.commission = commission
        self.leverage = leverage
        self.position_size = position_size
        self.window_size = window_size
        
        # Features de sentimento (opcional)
        self.sentiment_features = sentiment_features
        self.n_sentiment_features = 0
        if sentiment_features is not None:
            self.n_sentiment_features = sentiment_features.shape[1] if len(sentiment_features.shape) > 1 else 1
        
        # Espaço de Ações: Box contínuo [-1, 1] para compatibilidade com SAC/TD3
        # -1 a -0.33: Short | -0.33 a 0.33: Flat | 0.33 a 1: Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Espaço de Observações: [preços, indicadores, carteira, sentimento]
        self.n_features = len(self.df.columns)
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
        
        # Índice temporal
        self.current_step = self.window_size
        
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
        self.current_step += 1
        
        # Verificar se terminou
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.equity <= self.initial_balance * 0.5  # Stop se perder 50%
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Executa a ação de trading e retorna a recompensa.
        
        Lógica:
        - action 0: Fecha posição se houver
        - action 1: Abre/mantém Long
        - action 2: Abre/mantém Short
        """
        reward = 0
        action_changed = False
        
        # Mapeia ação para posição: 0 -> 0, 1 -> 1, 2 -> -1
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # Se a ação mudou
        if target_position != self.position:
            action_changed = True
            
            # Fecha posição atual se existir
            if self.position != 0:
                pnl = self._close_position(current_price)
                reward += pnl
                
            # Abre nova posição se não for Flat
            if target_position != 0:
                self._open_position(target_position, current_price)
        
        # Penalidade por mudança de posição (simula custo de trade)
        if action_changed:
            trade_cost = self.balance * self.position_size * self.commission
            reward -= trade_cost
            
        return reward
    
    def _open_position(self, position_type: int, price: float):
        """
        Abre uma posição Long (1) ou Short (-1).
        """
        self.position = position_type
        self.entry_price = price
        
        # Calcula o valor da posição com alavancagem
        capital = self.balance * self.position_size
        self.position_value = capital * self.leverage * position_type
        
    def _close_position(self, current_price: float) -> float:
        """
        Fecha a posição atual e retorna o PnL.
        """
        if self.position == 0:
            return 0
        
        pnl = self._calculate_pnl(current_price)
        
        # Atualiza saldo e métricas
        self.balance += pnl
        self.total_pnl += pnl
        self.trades += 1
        
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
        Inclui: histórico de preços/indicadores + estado da carteira + sentimento.
        """
        # Janela de dados históricos
        start = self.current_step - self.window_size
        end = self.current_step
        
        historical_data = self.df.iloc[start:end].values
        
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
                # Preenche com zeros ou repete último valor
                padding = np.zeros((self.window_size - sentiment_window.shape[0], sentiment_window.shape[1]))
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
