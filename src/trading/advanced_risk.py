"""
Advanced Risk Management System
Trailing Stops, Warm-up Period, Dynamic Position Management
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """
    Gerencia trailing stops dinâmicos para cada posição
    
    Trailing stop move com o lucro mas nunca retrocede,
    garantindo proteção de ganhos parciais
    """
    
    def __init__(self, activation_pct: float = 0.03, distance_pct: float = 0.015):
        """
        Args:
            activation_pct: % de lucro para ativar trailing (ex: 3%)
            distance_pct: Distância do trailing em % (ex: 1.5%)
        """
        self.activation_pct = activation_pct
        self.distance_pct = distance_pct
        self.active_stops: Dict[str, Dict] = {}  # {symbol: {entry, highest_mark, stop_price}}
        
        logger.info(f"[TRAILING] Inicializado: ativação={activation_pct:.1%}, distância={distance_pct:.1%}")
    
    def register_position(self, symbol: str, entry_price: float, position_type: int):
        """
        Registra nova posição para tracking
        
        Args:
            symbol: Par (ex: 'BTCUSDT')
            entry_price: Preço de entrada
            position_type: 1 para LONG, -1 para SHORT
        """
        self.active_stops[symbol] = {
            'entry_price': entry_price,
            'position_type': position_type,
            'highest_mark': entry_price if position_type == 1 else entry_price,
            'lowest_mark': entry_price if position_type == -1 else entry_price,
            'stop_price': None,
            'activated': False,
            'opened_at': datetime.now()
        }
        
        logger.info(f"[TRAILING] {symbol}: Registrado @ ${entry_price:,.2f} ({'LONG' if position_type == 1 else 'SHORT'})")
    
    def update(self, symbol: str, current_price: float) -> Tuple[bool, Optional[float]]:
        """
        Atualiza trailing stop baseado no preço atual
        
        Returns:
            (should_exit, stop_price): Se deve sair e preço do stop
        """
        if symbol not in self.active_stops:
            return False, None
        
        stop_data = self.active_stops[symbol]
        entry_price = stop_data['entry_price']
        position_type = stop_data['position_type']
        
        # ===== PROTEÇÃO: Validação de preços =====
        if entry_price <= 0 or current_price <= 0:
            logger.error(f"[TRAILING] {symbol}: Preços inválidos - entry: ${entry_price}, current: ${current_price}")
            return False, None
        
        # LONG position
        if position_type == 1:
            # Atualiza highest mark
            if current_price > stop_data['highest_mark']:
                stop_data['highest_mark'] = current_price
            
            # Calcula lucro atual
            profit_pct = (current_price - entry_price) / entry_price
            
            # Ativa trailing se lucro >= activation_pct
            if not stop_data['activated'] and profit_pct >= self.activation_pct:
                stop_data['activated'] = True
                stop_data['stop_price'] = current_price * (1 - self.distance_pct)
                logger.info(f"[TRAILING] {symbol}: ATIVADO @ ${current_price:,.2f} (lucro {profit_pct:.1%})")
            
            # Se já ativado, move stop com highest mark
            if stop_data['activated']:
                new_stop = stop_data['highest_mark'] * (1 - self.distance_pct)
                
                # Stop nunca retrocede
                if new_stop > stop_data['stop_price']:
                    stop_data['stop_price'] = new_stop
                    logger.debug(f"[TRAILING] {symbol}: Stop movido para ${new_stop:,.2f}")
                
                # Verifica se deve sair
                if current_price <= stop_data['stop_price']:
                    logger.warning(f"[TRAILING] {symbol}: 🛑 STOP HIT @ ${current_price:,.2f} (stop: ${stop_data['stop_price']:,.2f})")
                    return True, stop_data['stop_price']
        
        # SHORT position
        else:
            # Atualiza lowest mark
            if current_price < stop_data['lowest_mark']:
                stop_data['lowest_mark'] = current_price
            
            # Calcula lucro atual (inverso para SHORT)
            profit_pct = (entry_price - current_price) / entry_price
            
            # Ativa trailing se lucro >= activation_pct
            if not stop_data['activated'] and profit_pct >= self.activation_pct:
                stop_data['activated'] = True
                stop_data['stop_price'] = current_price * (1 + self.distance_pct)
                logger.info(f"[TRAILING] {symbol}: ATIVADO @ ${current_price:,.2f} (lucro {profit_pct:.1%})")
            
            # Se já ativado, move stop com lowest mark
            if stop_data['activated']:
                new_stop = stop_data['lowest_mark'] * (1 + self.distance_pct)
                
                # Stop nunca retrocede (para SHORT = nunca sobe)
                if stop_data['stop_price'] is None or new_stop < stop_data['stop_price']:
                    stop_data['stop_price'] = new_stop
                    logger.debug(f"[TRAILING] {symbol}: Stop movido para ${new_stop:,.2f}")
                
                # Verifica se deve sair
                if current_price >= stop_data['stop_price']:
                    logger.warning(f"[TRAILING] {symbol}: 🛑 STOP HIT @ ${current_price:,.2f} (stop: ${stop_data['stop_price']:,.2f})")
                    return True, stop_data['stop_price']
        
        return False, stop_data.get('stop_price')
    
    def remove_position(self, symbol: str):
        """Remove posição do tracking"""
        if symbol in self.active_stops:
            del self.active_stops[symbol]
            logger.info(f"[TRAILING] {symbol}: Removido do tracking")

    def update_entry_price(self, symbol: str, new_entry: float) -> None:
        """
        Move o preço de entrada para breakeven (ou qualquer outro nível).
        Chamado após TP1 para garantir que o stop mínimo nunca caia abaixo
        do preço de entrada original.

        Args:
            symbol:    par a atualizar
            new_entry: novo preço de referência (normalmente = entry price = breakeven)
        """
        if symbol not in self.active_stops:
            return
        stop_data = self.active_stops[symbol]
        ptype = stop_data.get('position_type', 1)
        old_entry = stop_data['entry_price']
        stop_data['entry_price'] = new_entry

        # Se o trailing ainda não foi ativado, força ativação imediata no
        # breakeven — impede que stop caia abaixo do ponto de entrada.
        if not stop_data.get('activated', False):
            if ptype == 1:
                stop_data['stop_price'] = new_entry * (1 - self.distance_pct)
            else:
                stop_data['stop_price'] = new_entry * (1 + self.distance_pct)
            stop_data['activated'] = True
        else:
            # Já ativado: só eleva o stop se estiver abaixo do breakeven
            if ptype == 1:
                be_stop = new_entry * (1 - self.distance_pct)
                if stop_data.get('stop_price', 0) < be_stop:
                    stop_data['stop_price'] = be_stop
            else:
                be_stop = new_entry * (1 + self.distance_pct)
                if stop_data.get('stop_price', float('inf')) > be_stop:
                    stop_data['stop_price'] = be_stop

        logger.info(
            f"[TRAILING] {symbol}: Breakeven ativado — entry {old_entry:,.4f} → {new_entry:,.4f}, "
            f"stop={stop_data.get('stop_price', 0):,.4f}"
        )

    def get_stop_info(self, symbol: str) -> Optional[Dict]:
        """Retorna informações do stop para um símbolo"""
        return self.active_stops.get(symbol)


class WarmupManager:
    """
    Gerencia período de warm-up antes de permitir trading
    
    Aguarda coletar dados suficientes antes de operar
    """
    
    def __init__(self, required_candles: int = 50):
        """
        Args:
            required_candles: Número de candles necessários antes de operar
        """
        self.required_candles = required_candles
        self.candle_count: Dict[str, int] = {}  # {symbol: count}
        self.ready: Dict[str, bool] = {}  # {symbol: is_ready}
        
        logger.info(f"[WARMUP] Inicializado: {required_candles} candles necessários")
    
    def add_candle(self, symbol: str):
        """Incrementa contador de candles para um símbolo"""
        if symbol not in self.candle_count:
            self.candle_count[symbol] = 0
            self.ready[symbol] = False
        
        self.candle_count[symbol] += 1
        
        # Marca como ready se atingiu threshold
        if self.candle_count[symbol] >= self.required_candles and not self.ready[symbol]:
            self.ready[symbol] = True
            logger.info(f"[WARMUP] {symbol}: ✅ PRONTO ({self.candle_count[symbol]} candles)")
    
    def is_ready(self, symbol: str) -> bool:
        """Verifica se símbolo está pronto para trading"""
        return self.ready.get(symbol, False)
    
    def get_progress(self, symbol: str) -> Tuple[int, int, float]:
        """
        Retorna progresso do warm-up
        
        Returns:
            (current, required, percentage)
        """
        current = self.candle_count.get(symbol, 0)
        percentage = (current / self.required_candles) * 100
        return current, self.required_candles, percentage
    
    def reset(self, symbol: str):
        """Reseta warm-up para um símbolo"""
        self.candle_count[symbol] = 0
        self.ready[symbol] = False
        logger.info(f"[WARMUP] {symbol}: Reset")


class ScheduleManager:
    """
    Gerencia schedule de execução para diversificação temporal
    
    Cada par opera em horários específicos para evitar
    sobrecarga simultânea e melhorar diversificação
    """
    
    def __init__(self, schedule_config: Optional[Dict[str, list]] = None):
        """
        Args:
            schedule_config: {symbol: [minutos_permitidos]}
                Ex: {'BTCUSDT': [0, 15, 30, 45], 'ETHUSDT': [5, 20, 35, 50]}
        """
        self.schedule = schedule_config or {}
        
        # Se não fornecido, gera schedule automático (offset de 5 min)
        if not self.schedule:
            self._generate_auto_schedule()
        
        logger.info(f"[SCHEDULE] Inicializado com {len(self.schedule)} pares")
    
    def _generate_auto_schedule(self):
        """Gera schedule automático com offset de 5 minutos"""
        base_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        
        for idx, symbol in enumerate(base_symbols):
            offset = idx * 5  # 0, 5, 10, 15, 20
            minutes = [(offset + i * 15) % 60 for i in range(4)]  # 4 slots por hora
            self.schedule[symbol] = minutes
            logger.info(f"[SCHEDULE] {symbol}: {minutes}")
    
    def can_trade_now(self, symbol: str,
                       at_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Verifica se pode operar o símbolo no momento fornecido (ou agora).

        Args:
            symbol:  par a verificar
            at_time: timestamp de referência (default: datetime.now())
                     Use o timestamp de fechamento do candle para evitar
                     bloqueio quando a detecção ocorre alguns segundos depois
                     do fechamento real.
        Returns:
            (can_trade, reason)
        """
        if symbol not in self.schedule:
            return True, "No schedule defined"

        current_time   = at_time if at_time is not None else datetime.now()
        current_minute = current_time.minute

        allowed_minutes = self.schedule[symbol]

        if current_minute in allowed_minutes:
            return True, f"Scheduled minute: {current_minute}"
        else:
            next_minute = min([m for m in allowed_minutes if m > current_minute],
                              default=allowed_minutes[0])
            wait_time = (next_minute - current_minute) % 60
            return False, f"Wait {wait_time} minutes (next: {next_minute})"
    
    def get_next_execution(self, symbol: str) -> Optional[datetime]:
        """Retorna próximo horário de execução para um símbolo"""
        if symbol not in self.schedule:
            return None
        
        current_time = datetime.now()
        current_minute = current_time.minute
        
        allowed_minutes = self.schedule[symbol]
        
        # Próximo minuto permitido
        next_minute = min([m for m in allowed_minutes if m > current_minute],
                         default=allowed_minutes[0])
        
        # Se next_minute < current_minute, é na próxima hora
        if next_minute < current_minute:
            next_execution = current_time.replace(minute=next_minute, second=0) + timedelta(hours=1)
        else:
            next_execution = current_time.replace(minute=next_minute, second=0)
        
        return next_execution
    
    def add_symbol(self, symbol: str, minutes: list):
        """Adiciona novo símbolo ao schedule"""
        self.schedule[symbol] = minutes
        logger.info(f"[SCHEDULE] {symbol} adicionado: {minutes}")
    
    def remove_symbol(self, symbol: str):
        """Remove símbolo do schedule"""
        if symbol in self.schedule:
            del self.schedule[symbol]
            logger.info(f"[SCHEDULE] {symbol} removido")
