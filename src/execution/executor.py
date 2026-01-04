"""
Executor de Trading - Conecta o agente treinado à Binance API
Executa operações em Paper Trading ou Live.
"""

import ccxt
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Optional
import os
from dotenv import load_dotenv

from stable_baselines3 import PPO

import sys
sys.path.append(str(Path(__file__).parent.parent))

from risk.risk_manager import RiskManager, PositionSizer
from data.data_collector import DataCollector


class BinanceExecutor:
    """
    Executor que conecta o agente de RL à Binance Futures API.
    Gerencia posições, ordens e risk management.
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "paper"):
        """
        Args:
            config_path: Caminho para configuração
            mode: 'paper' ou 'live'
        """
        # Carrega configurações
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        load_dotenv()
        
        self.mode = mode
        self.symbol = self.config['data']['symbol']
        self.timeframe = self.config['data']['timeframe']
        
        # Inicializa exchange
        self._init_exchange()
        
        # Risk Manager
        self.risk_manager = RiskManager(config_path)
        self.position_sizer = PositionSizer()
        
        # Estado atual
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss_order_id = None
        self.take_profit_order_id = None
        
        # Métricas de trading
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        
        # Logging
        self._init_logging()
        
    def _init_exchange(self):
        """Inicializa a conexão com a Binance."""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if self.mode == "live":
            api_key = os.getenv('BINANCE_MAINNET_API_KEY', api_key)
            api_secret = os.getenv('BINANCE_MAINNET_API_SECRET', api_secret)
        
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Usa testnet se configurado
        if self.config['binance']['testnet'] and self.mode != "live":
            self.exchange.set_sandbox_mode(True)
            print("🧪 Usando Binance TESTNET")
        else:
            print(f"{'🔴 MODO LIVE' if self.mode == 'live' else '📄 MODO PAPER'}")
    
    def _init_logging(self):
        """Inicializa sistema de logging."""
        logs_dir = Path("logs/trading")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logs_dir / f"trading_{timestamp}.log"
        
        self._log("="*60)
        self._log(f"Trading Executor Iniciado - Modo: {self.mode.upper()}")
        self._log(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        self._log("="*60)
    
    def _log(self, message: str):
        """Registra mensagem no log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    def get_account_balance(self) -> float:
        """Obtém saldo da conta em USDT."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            return usdt_balance
        except Exception as e:
            self._log(f"❌ Erro ao obter saldo: {e}")
            return 0
    
    def get_current_price(self) -> float:
        """Obtém preço atual do símbolo."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            self._log(f"❌ Erro ao obter preço: {e}")
            return 0
    
    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """
        Coloca ordem a mercado.
        
        Args:
            side: 'buy' ou 'sell'
            quantity: Quantidade de contratos
            
        Returns:
            Informações da ordem ou None se falhar
        """
        if self.mode == "paper":
            self._log(f"📄 [PAPER] Ordem simulada: {side.upper()} {quantity} {self.symbol}")
            return {
                'id': f'paper_{int(time.time())}',
                'price': self.get_current_price(),
                'amount': quantity,
                'side': side
            }
        
        try:
            order = self.exchange.create_market_order(
                symbol=self.symbol,
                side=side,
                amount=quantity
            )
            
            self._log(f"✅ Ordem executada: {side.upper()} {quantity} @ ${order['price']:.2f}")
            return order
            
        except Exception as e:
            self._log(f"❌ Erro ao executar ordem: {e}")
            return None
    
    def place_stop_loss(self, side: str, quantity: float, stop_price: float):
        """Coloca ordem de Stop Loss."""
        if self.mode == "paper":
            self._log(f"📄 [PAPER] Stop Loss em ${stop_price:.2f}")
            return f'paper_sl_{int(time.time())}'
        
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='STOP_MARKET',
                side=side,
                amount=quantity,
                params={'stopPrice': stop_price}
            )
            
            self._log(f"🛡️ Stop Loss colocado em ${stop_price:.2f}")
            return order['id']
            
        except Exception as e:
            self._log(f"❌ Erro ao colocar Stop Loss: {e}")
            return None
    
    def place_take_profit(self, side: str, quantity: float, take_profit_price: float):
        """Coloca ordem de Take Profit."""
        if self.mode == "paper":
            self._log(f"📄 [PAPER] Take Profit em ${take_profit_price:.2f}")
            return f'paper_tp_{int(time.time())}'
        
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='TAKE_PROFIT_MARKET',
                side=side,
                amount=quantity,
                params={'stopPrice': take_profit_price}
            )
            
            self._log(f"🎯 Take Profit colocado em ${take_profit_price:.2f}")
            return order['id']
            
        except Exception as e:
            self._log(f"❌ Erro ao colocar Take Profit: {e}")
            return None
    
    def cancel_order(self, order_id: str):
        """Cancela uma ordem."""
        if self.mode == "paper" or order_id is None:
            return
        
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            self._log(f"🚫 Ordem {order_id} cancelada")
        except Exception as e:
            self._log(f"⚠️ Erro ao cancelar ordem: {e}")
    
    def execute_action(self, action: int, current_price: float, balance: float):
        """
        Executa a ação do agente.
        
        Args:
            action: 0 (Flat), 1 (Long), 2 (Short)
            current_price: Preço atual
            balance: Saldo disponível
        """
        # Mapeia ação
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # Valida com Risk Manager
        validated_action, reason = self.risk_manager.validate_action(
            action, balance, self.position, self.entry_price, current_price
        )
        
        if validated_action != action:
            self._log(f"⚠️ Ação modificada pelo Risk Manager: {reason}")
            target_position = 0 if validated_action == 0 else (1 if validated_action == 1 else -1)
        
        # Se mudou a posição
        if target_position != self.position:
            # Fecha posição atual
            if self.position != 0:
                self._close_position(current_price, balance)
            
            # Abre nova posição
            if target_position != 0:
                self._open_position(target_position, current_price, balance)
    
    def _open_position(self, position_type: int, price: float, balance: float):
        """Abre uma posição Long ou Short."""
        # Calcula tamanho da posição
        win_rate = self.wins / self.trades if self.trades > 0 else 0.5
        avg_win = 0.02  # Estimativas iniciais
        avg_loss = 0.01
        
        position_size_usd = self.risk_manager.calculate_position_size(
            balance, win_rate, avg_win, avg_loss, confidence=1.0
        )
        
        # Calcula quantidade
        leverage = self.config['environment']['leverage']
        quantity = self.position_sizer.calculate_quantity(
            position_size_usd, price, leverage
        )
        
        # Coloca ordem
        side = 'buy' if position_type == 1 else 'sell'
        order = self.place_market_order(side, quantity)
        
        if order:
            self.position = position_type
            self.entry_price = price
            self.position_size = quantity
            
            # Coloca Stop Loss e Take Profit
            stop_loss_price = self.position_sizer.calculate_stop_loss_price(
                price,
                self.risk_manager.stop_loss_pct,
                position_type
            )
            
            take_profit_price = self.position_sizer.calculate_take_profit_price(
                price,
                self.risk_manager.take_profit_pct,
                position_type
            )
            
            # Lado inverso para fechar posição
            sl_side = 'sell' if position_type == 1 else 'buy'
            
            self.stop_loss_order_id = self.place_stop_loss(sl_side, quantity, stop_loss_price)
            self.take_profit_order_id = self.place_take_profit(sl_side, quantity, take_profit_price)
            
            self._log(f"🔓 Posição {'LONG' if position_type == 1 else 'SHORT'} aberta")
            self._log(f"   Preço: ${price:.2f} | Quantidade: {quantity:.4f}")
            self._log(f"   Stop Loss: ${stop_loss_price:.2f} | Take Profit: ${take_profit_price:.2f}")
    
    def _close_position(self, price: float, balance: float):
        """Fecha a posição atual."""
        if self.position == 0:
            return
        
        # Cancela ordens pendentes
        self.cancel_order(self.stop_loss_order_id)
        self.cancel_order(self.take_profit_order_id)
        
        # Calcula PnL
        price_change = (price - self.entry_price) / self.entry_price
        pnl = price_change * self.position_size * price * self.position
        
        # Coloca ordem de fechamento
        side = 'sell' if self.position == 1 else 'buy'
        order = self.place_market_order(side, self.position_size)
        
        if order:
            # Atualiza métricas
            self.trades += 1
            self.total_pnl += pnl
            
            if pnl > 0:
                self.wins += 1
                self._log(f"✅ Posição fechada com LUCRO: ${pnl:.2f}")
            else:
                self.losses += 1
                self._log(f"❌ Posição fechada com PREJUÍZO: ${pnl:.2f}")
            
            win_rate = self.wins / self.trades
            self._log(f"📊 Trades: {self.trades} | Win Rate: {win_rate:.2%} | PnL Total: ${self.total_pnl:.2f}")
            
            # Reseta estado
            self.position = 0
            self.entry_price = 0
            self.position_size = 0
            self.stop_loss_order_id = None
            self.take_profit_order_id = None


def main():
    """Execução principal do bot de trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Executor de Trading com RL")
    parser.add_argument("--model", type=str, required=True, help="Caminho do modelo treinado")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Modo de execução")
    parser.add_argument("--interval", type=int, default=60, help="Intervalo de verificação (segundos)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 AGENTE DE TRADING COM RL - EXECUTOR")
    print("="*60)
    
    # Carrega modelo
    print(f"\n📦 Carregando modelo: {args.model}")
    model = PPO.load(args.model)
    print("✅ Modelo carregado")
    
    # Inicializa executor
    executor = BinanceExecutor(mode=args.mode)
    
    # Inicializa coletor de dados para features em tempo real
    collector = DataCollector()
    
    print(f"\n🚀 Bot iniciado! Verificando a cada {args.interval} segundos")
    print("Pressione Ctrl+C para parar.\n")
    
    try:
        while True:
            # Obtém dados atuais
            df = collector.fetch_ohlcv()
            df = collector.add_technical_indicators(df)
            df_norm = collector.normalize_data(df)
            
            # Prepara observação (últimas 50 candles)
            obs_data = df_norm.tail(50).values
            
            # Obtém saldo e estado
            balance = executor.get_account_balance()
            current_price = executor.get_current_price()
            
            # Estado da carteira
            portfolio_state = np.array([
                balance / executor.config['environment']['initial_balance'],
                executor.position,
                balance / executor.config['environment']['initial_balance']
            ])
            
            portfolio_matrix = np.tile(portfolio_state, (50, 1))
            observation = np.concatenate([obs_data, portfolio_matrix], axis=1)
            observation = observation.astype(np.float32)
            
            # Expande dimensão para batch
            observation = np.expand_dims(observation, axis=0)
            
            # Predição do agente
            action, _ = model.predict(observation, deterministic=True)
            action = int(action[0])
            
            # Executa ação
            executor.execute_action(action, current_price, balance)
            
            # Aguarda próximo ciclo
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Bot interrompido pelo usuário")
        
        # Fecha posições abertas
        if executor.position != 0:
            print("Fechando posições abertas...")
            current_price = executor.get_current_price()
            balance = executor.get_account_balance()
            executor._close_position(current_price, balance)
        
        print("\n✅ Bot encerrado com sucesso")


if __name__ == "__main__":
    main()
