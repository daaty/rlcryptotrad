"""
Módulo de Gestão de Risco (Risk Management)
Implementa regras hardcoded para proteger o capital.
"""

import numpy as np
from typing import Dict, Tuple
import yaml


class RiskManager:
    """
    Gerencia o risco das operações do agente de trading.
    
    Implementa:
    - Kelly Criterion para tamanho de posição
    - Stop Loss e Take Profit
    - Controle de Drawdown máximo
    - Limite de alavancagem
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.risk_config = config['risk_management']
        self.env_config = config['environment']
        
        # Parâmetros de risco
        self.max_leverage = self.risk_config['max_leverage']
        self.stop_loss_pct = self.risk_config['stop_loss_pct']
        self.take_profit_pct = self.risk_config['take_profit_pct']
        self.max_drawdown = self.risk_config['max_drawdown']
        self.kelly_fraction = self.risk_config['kelly_fraction']
        
        # Métricas
        self.peak_equity = self.env_config['initial_balance']
        self.initial_balance = self.env_config['initial_balance']
        
        # Circuit Breaker
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.circuit_breaker_active = False
        self.last_trades = []  # Histórico recente de trades (True=win, False=loss)
        
    def calculate_position_size(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calcula o tamanho ideal da posição usando Kelly Criterion fracionário.
        
        Kelly % = (W × P - L) / W
        Onde:
        - W = win_rate (taxa de vitória)
        - P = avg_win / avg_loss (razão de ganho/perda)
        - L = 1 - win_rate (taxa de perda)
        
        Args:
            balance: Saldo atual da conta
            win_rate: Taxa de vitórias (0 a 1)
            avg_win: Ganho médio por trade vencedor
            avg_loss: Perda média por trade perdedor
            confidence: Multiplicador de confiança (0 a 1)
            
        Returns:
            Tamanho da posição em USDT
        """
        # Previne divisão por zero
        if avg_loss == 0 or win_rate == 0:
            return balance * 0.01  # 1% default conservador
        
        # Calcula Kelly %
        loss_rate = 1 - win_rate
        profit_loss_ratio = abs(avg_win / avg_loss)
        
        kelly_pct = (win_rate * profit_loss_ratio - loss_rate) / profit_loss_ratio
        
        # Limita Kelly entre 0 e 1
        kelly_pct = np.clip(kelly_pct, 0, 1)
        
        # Aplica fração de Kelly (mais conservador)
        fractional_kelly = kelly_pct * self.kelly_fraction * confidence
        
        # Calcula tamanho da posição
        position_size = balance * fractional_kelly
        
        # Limita pelo tamanho máximo configurado
        max_position = balance * self.env_config['position_size']
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_atr_stop_loss(
        self, 
        entry_price: float, 
        atr: float, 
        position_type: int,
        multiplier: float = 2.0
    ) -> float:
        """
        Calcula Stop Loss dinâmico baseado em ATR (Average True Range).
        
        ATR mede a volatilidade do ativo. Stop Loss dinâmico se adapta:
        - Alta volatilidade → stop mais largo
        - Baixa volatilidade → stop mais apertado
        
        Args:
            entry_price: Preço de entrada
            atr: Average True Range do ativo
            position_type: 1 (Long) ou -1 (Short)
            multiplier: Multiplicador do ATR (default 2.0)
            
        Returns:
            Preço do stop loss
        """
        stop_distance = atr * multiplier
        
        if position_type == 1:  # LONG
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
            
        return stop_price
    
    def should_stop_loss(
        self, 
        entry_price: float, 
        current_price: float, 
        position_type: int,
        atr: float = None
    ) -> bool:
        """
        Verifica se deve acionar o Stop Loss.
        
        Suporta dois modos:
        1. Stop Loss fixo (% configurado)
        2. Stop Loss dinâmico (baseado em ATR se fornecido)
        
        Args:
            entry_price: Preço de entrada da posição
            current_price: Preço atual
            position_type: 1 (Long) ou -1 (Short)
            atr: Average True Range (opcional, para stop dinâmico)
            
        Returns:
            True se deve fechar a posição
        """
        if entry_price == 0:
            return False
        
        # Se ATR fornecido, usa stop dinâmico
        if atr is not None:
            stop_price = self.calculate_atr_stop_loss(entry_price, atr, position_type)
            
            if position_type == 1:  # LONG
                return current_price <= stop_price
            else:  # SHORT
                return current_price >= stop_price
        
        # Caso contrário, usa stop fixo (% configurado)
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Long: perda se preço cai | Short: perda se preço sobe
        loss_pct = -price_change_pct if position_type == 1 else price_change_pct
        
        return loss_pct >= self.stop_loss_pct
    
    def should_take_profit(
        self, 
        entry_price: float, 
        current_price: float, 
        position_type: int,
        return_level: bool = False
    ) -> tuple:
        """
        Verifica se deve acionar o Take Profit com suporte a níveis parciais.
        
        Estratégia:
        - Nível 1 (50%): +2% de lucro → fecha 50% da posição
        - Nível 2 (100%): +4% de lucro → fecha restante 50%
        
        Args:
            entry_price: Preço de entrada da posição
            current_price: Preço atual
            position_type: 1 (Long) ou -1 (Short)
            return_level: Se True, retorna (should_tp, level), senão apenas bool
            
        Returns:
            Se return_level=True: (bool, int) - (deve fechar?, nível atingido)
            Se return_level=False: bool - deve fechar posição completa?
        """
        if entry_price == 0:
            if return_level:
                return False, 0
            return False
        
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Long: lucro se preço sobe | Short: lucro se preço cai
        profit_pct = price_change_pct if position_type == 1 else -price_change_pct
        
        # Níveis de Take Profit
        tp_level_1 = self.take_profit_pct / 2  # 50% do TP target (ex: 2%)
        tp_level_2 = self.take_profit_pct      # 100% do TP target (ex: 4%)
        
        if profit_pct >= tp_level_2:
            # Lucro atingiu 100% do target
            if return_level:
                return True, 2
            return True
        elif profit_pct >= tp_level_1:
            # Lucro atingiu 50% do target
            if return_level:
                return True, 1
            return False  # Não fecha tudo ainda
        else:
            if return_level:
                return False, 0
            return False
    
    def calculate_partial_close_size(
        self,
        current_position_size: float,
        tp_level: int
    ) -> float:
        """
        Calcula quantidade a fechar em saída parcial.
        
        Args:
            current_position_size: Tamanho atual da posição
            tp_level: Nível de TP atingido (1 ou 2)
            
        Returns:
            Quantidade a fechar
        """
        if tp_level == 1:
            # Nível 1: fecha 50% da posição
            return current_position_size * 0.5
        elif tp_level == 2:
            # Nível 2: fecha todo o restante
            return current_position_size
        else:
            return 0.0
    
    def record_trade_result(self, pnl: float) -> None:
        """
        Registra resultado de um trade para circuit breaker.
        
        Args:
            pnl: Profit/Loss do trade (positivo = lucro, negativo = perda)
        """
        is_win = pnl > 0
        
        self.last_trades.append(is_win)
        
        # Mantém apenas últimos 100 trades
        if len(self.last_trades) > 100:
            self.last_trades.pop(0)
        
        # Atualiza contador de losses consecutivos
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Ativa circuit breaker se atingir limite
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.circuit_breaker_active = True
            print(f"⚠️ CIRCUIT BREAKER ATIVADO! {self.consecutive_losses} losses consecutivos")
    
    def should_allow_trade(self) -> Tuple[bool, str]:
        """
        Verifica se deve permitir novo trade (circuit breaker).
        
        Returns:
            (allow, reason): True se pode tradear, False + motivo se bloqueado
        """
        if self.circuit_breaker_active:
            return False, f"Circuit breaker ativo ({self.consecutive_losses} losses consecutivos)"
        
        return True, "OK"
    
    def reset_circuit_breaker(self) -> None:
        """
        Reseta o circuit breaker manualmente (após análise/ajustes).
        """
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        print("✅ Circuit breaker resetado")
    
    def get_trading_stats(self) -> Dict:
        """
        Retorna estatísticas de trading para análise.
        
        Returns:
            Dict com métricas: win_rate, avg_win, avg_loss, etc
        """
        if len(self.last_trades) == 0:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'consecutive_losses': self.consecutive_losses,
                'circuit_breaker_active': self.circuit_breaker_active
            }
        
        wins = sum(1 for t in self.last_trades if t)
        losses = len(self.last_trades) - wins
        win_rate = wins / len(self.last_trades) if len(self.last_trades) > 0 else 0
        
        return {
            'total_trades': len(self.last_trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.circuit_breaker_active
        }
    
    def check_drawdown(self, current_equity: float) -> Tuple[bool, float]:
        """
        Verifica se o drawdown máximo foi atingido.
        
        Args:
            current_equity: Equity atual da conta
            
        Returns:
            (should_stop, current_drawdown)
        """
        # Atualiza pico de equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calcula drawdown atual
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        should_stop = drawdown >= self.max_drawdown
        
        return should_stop, drawdown
    
    def validate_action(
        self,
        action: int,
        balance: float,
        position: int,
        entry_price: float,
        current_price: float
    ) -> Tuple[int, str]:
        """
        Valida e potencialmente modifica a ação do agente baseado em regras de risco.
        
        Args:
            action: Ação proposta pelo agente (0, 1, 2)
            balance: Saldo atual
            position: Posição atual (0, 1, -1)
            entry_price: Preço de entrada
            current_price: Preço atual
            
        Returns:
            (validated_action, reason)
        """
        # Verifica Stop Loss
        if position != 0 and self.should_stop_loss(entry_price, current_price, position):
            return 0, "STOP_LOSS_TRIGGERED"
        
        # Verifica Take Profit
        if position != 0 and self.should_take_profit(entry_price, current_price, position):
            return 0, "TAKE_PROFIT_TRIGGERED"
        
        # Verifica Drawdown
        should_stop, drawdown = self.check_drawdown(balance)
        if should_stop:
            return 0, f"MAX_DRAWDOWN_REACHED ({drawdown:.2%})"
        
        # Verifica saldo mínimo
        min_balance = self.initial_balance * 0.1  # 10% do inicial
        if balance < min_balance:
            return 0, "INSUFFICIENT_BALANCE"
        
        return action, "OK"
    
    def calculate_leverage(self, confidence: float, volatility: float) -> float:
        """
        Calcula a alavancagem dinâmica baseada em confiança e volatilidade.
        
        Args:
            confidence: Confiança do modelo (0 a 1)
            volatility: Volatilidade do ativo (desvio padrão dos retornos)
            
        Returns:
            Alavancagem ajustada
        """
        # Base: alavancagem configurada
        base_leverage = self.env_config['leverage']
        
        # Reduz alavancagem em alta volatilidade
        volatility_factor = np.clip(1 - volatility * 10, 0.2, 1.0)
        
        # Reduz alavancagem em baixa confiança
        confidence_factor = confidence
        
        # Calcula alavancagem final
        adjusted_leverage = base_leverage * volatility_factor * confidence_factor
        
        # Limita pelo máximo
        adjusted_leverage = min(adjusted_leverage, self.max_leverage)
        
        return max(1, adjusted_leverage)  # Mínimo de 1x
    
    def get_risk_metrics(self, balance: float, equity: float) -> Dict[str, float]:
        """
        Retorna métricas de risco atuais.
        
        Returns:
            Dicionário com métricas
        """
        _, drawdown = self.check_drawdown(equity)
        
        return {
            'current_drawdown': drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'equity': equity,
            'balance': balance,
            'risk_utilization': drawdown / self.max_drawdown if self.max_drawdown > 0 else 0
        }


class PositionSizer:
    """Classe auxiliar para calcular tamanhos de posição."""
    
    @staticmethod
    def calculate_quantity(
        position_size_usd: float,
        price: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calcula a quantidade de contratos baseado no tamanho em USD.
        
        Args:
            position_size_usd: Tamanho da posição em USDT
            price: Preço atual do ativo
            leverage: Alavancagem a usar
            
        Returns:
            Quantidade de contratos
        """
        return (position_size_usd * leverage) / price
    
    @staticmethod
    def calculate_stop_loss_price(
        entry_price: float,
        stop_loss_pct: float,
        position_type: int
    ) -> float:
        """
        Calcula o preço do Stop Loss.
        
        Args:
            entry_price: Preço de entrada
            stop_loss_pct: Percentual do stop (ex: 0.02 = 2%)
            position_type: 1 (Long) ou -1 (Short)
            
        Returns:
            Preço do Stop Loss
        """
        if position_type == 1:  # Long
            return entry_price * (1 - stop_loss_pct)
        else:  # Short
            return entry_price * (1 + stop_loss_pct)
    
    @staticmethod
    def calculate_take_profit_price(
        entry_price: float,
        take_profit_pct: float,
        position_type: int
    ) -> float:
        """
        Calcula o preço do Take Profit.
        
        Args:
            entry_price: Preço de entrada
            take_profit_pct: Percentual do TP (ex: 0.04 = 4%)
            position_type: 1 (Long) ou -1 (Short)
            
        Returns:
            Preço do Take Profit
        """
        if position_type == 1:  # Long
            return entry_price * (1 + take_profit_pct)
        else:  # Short
            return entry_price * (1 - take_profit_pct)
