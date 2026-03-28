"""
📊 Dashboard em Tempo Real - Trading Bot
Streamlit app com visualizações ao vivo
"""
from __future__ import annotations  # PEP 604 union hints on Python 3.9+

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yaml
import time
import logging
import re
from collections import deque
from datetime import datetime, timedelta
from binance.client import Client
from binance import ThreadedWebsocketManager
import os
from dotenv import load_dotenv
from pathlib import Path
from src.data.data_collector import DataCollector
from src.risk.risk_manager import RiskManager
from src.trading.advanced_risk import TrailingStopManager, WarmupManager, ScheduleManager
import talib
import threading

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET MANAGER - Reduz chamadas REST em 90%
# ═══════════════════════════════════════════════════════════════════════════

# ─── KLINE BUFFER CONSTANTS ────────────────────────────────────────────────
_KLINE_MAXLEN   = 600   # candles mantidos em memória por símbolo/intervalo
_INTERVALS_WS   = ['15m', '1h', '4h']  # TFs que o bot precisa
_KLINE_LIMIT_BOOT = {'15m': 500, '1h': 200, '4h': 100}  # candles no bootstrap

class BinanceWebSocketManager:
    """
    Gerencia conexões WebSocket persistentes com Binance Futures.
    
    Mantém um buffer OHLCV em memória por símbolo/intervalo que é:
      - Inicializado via UMA REST call de bootstrap (solicita o usuário clicar)
      - Atualizado em tempo real por kline WebSocket fechado
    Após o bootstrap, ZERO chamadas REST são feitas pela dashboard.
    """

    def __init__(self, client: Client):
        self.client   = client
        self.twm      = None
        self.lock     = threading.Lock()
        self.running  = False

        # ── User account data (balance + open positions) ──────────────────
        self.user_data: dict = {
            'balance': {'total': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0},
            'positions': [],
            'last_update': None,
        }

        # ── Kline rolling buffers ─────────────────────────────────────────
        # Structure: {symbol: {interval: deque([{open,high,low,close,volume,timestamp}, ...])}}
        self.kline_buffers: dict[str, dict[str, deque]] = {}
        self.bootstrap_done: bool = False          # True depois do 1° REST bootstrap
        self.bootstrap_symbols: list[str] = []    # símbolos que foram bootstrappados

        # ── Live ticker price (book ticker) ──────────────────────────────
        # {symbol: float}  — atualizado a cada msg do book ticker stream
        self.live_price: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        """Inicia o ThreadedWebsocketManager e subscreve user data stream."""
        if self.running:
            return
        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.client.API_KEY,
                api_secret=self.client.API_SECRET,
                testnet=True,
            )
            self.twm.start()
            self.twm.start_futures_user_socket(callback=self._handle_user_data)
            self.running = True
            logger.info("[WS] Iniciado — User Data Stream ativo")
        except Exception as exc:
            logger.error(f"[WS] Erro ao iniciar: {exc}")
            self.running = False

    def stop(self):
        """Para todos os streams WebSocket."""
        if self.twm:
            try:
                self.twm.stop()
            except Exception as exc:
                logger.warning(f"[WS] Erro ao parar: {exc}")
        self.running = False
        logger.info("[WS] Encerrado")

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: bootstrap (única REST call autorizada após início)
    # ─────────────────────────────────────────────────────────────────────

    def bootstrap_klines(self, symbols: list[str]) -> int:
        """
        Busca histórico inicial de candles via REST (chamado UMA vez pelo usuário).
        Popula self.kline_buffers com até _KLINE_MAXLEN candles por combinação.
        Retorna o total de candles carregados.
        """
        total = 0
        for sym in symbols:
            sym = sym.upper()
            self.kline_buffers.setdefault(sym, {})
            for interval in _INTERVALS_WS:
                limit = _KLINE_LIMIT_BOOT.get(interval, 200)
                try:
                    raw = self.client.futures_klines(symbol=sym, interval=interval, limit=limit)
                    buf = deque(maxlen=_KLINE_MAXLEN)
                    for k in raw:
                        buf.append({
                            'timestamp': int(k[0]),
                            'open':      float(k[1]),
                            'high':      float(k[2]),
                            'low':       float(k[3]),
                            'close':     float(k[4]),
                            'volume':    float(k[5]),
                        })
                    self.kline_buffers[sym][interval] = buf
                    total += len(buf)
                    logger.info(f"[WS-BOOT] {sym}/{interval}: {len(buf)} candles carregados")
                    time.sleep(0.25)  # respeita rate-limit mesmo no bootstrap
                except Exception as exc:
                    logger.error(f"[WS-BOOT] Erro {sym}/{interval}: {exc}")
        self.bootstrap_done = True
        self.bootstrap_symbols = [s.upper() for s in symbols]
        logger.info(f"[WS-BOOT] Bootstrap completo: {total} candles | símbolos: {self.bootstrap_symbols}")
        return total

    def bootstrap_account(self) -> bool:
        """
        Carrega balance e posições via REST (parte do bootstrap inicial).
        Popula self.user_data para que a UI mostre dados antes do 1° evento WS.
        """
        try:
            balance_raw = self.client.futures_account_balance()
            usdt = next((b for b in balance_raw if b['asset'] == 'USDT'), None)
            if usdt:
                with self.lock:
                    self.user_data['balance'] = {
                        'total':          float(usdt['balance']),
                        'available':      float(usdt['availableBalance']),
                        'unrealized_pnl': float(usdt.get('crossUnPnl', 0)),
                    }
            positions_raw = self.client.futures_position_information()
            with self.lock:
                self.user_data['positions'] = [
                    p for p in positions_raw if float(p['positionAmt']) != 0
                ]
                self.user_data['last_update'] = datetime.now()
            logger.info(f"[WS-BOOT] Account snapshot: balance=${self.user_data['balance']['total']:.2f}, "
                        f"positions={len(self.user_data['positions'])}")
            return True
        except Exception as exc:
            logger.error(f"[WS-BOOT] Erro ao carregar snapshot de conta: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: subscriptions
    # ─────────────────────────────────────────────────────────────────────

    def subscribe_klines_multi(self, symbol: str, intervals: list[str] | None = None):
        """Subscreve streams de kline para um símbolo em múltiplos intervalos."""
        if not self.twm:
            return
        if intervals is None:
            intervals = _INTERVALS_WS
        sym = symbol.upper()
        self.kline_buffers.setdefault(sym, {})
        for interval in intervals:
            self.kline_buffers[sym].setdefault(interval, deque(maxlen=_KLINE_MAXLEN))
            try:
                self.twm.start_kline_futures_socket(
                    callback=lambda msg, s=sym, i=interval: self._handle_kline(msg, s, i),
                    symbol=sym.lower(),
                    interval=interval,
                )
                logger.info(f"[WS] Subscribed kline: {sym}/{interval}")
            except Exception as exc:
                logger.warning(f"[WS] Erro ao subscrever {sym}/{interval}: {exc}")

    def subscribe_book_ticker(self, symbol: str):
        """Subscreve best bid/ask para preço live em tempo real."""
        if not self.twm:
            return
        try:
            self.twm.start_book_ticker_socket(
                callback=lambda msg: self._handle_book_ticker(msg),
                symbol=symbol.upper(),
            )
            logger.info(f"[WS] Subscribed book ticker: {symbol}")
        except Exception as exc:
            logger.warning(f"[WS] Erro ao subscrever book ticker {symbol}: {exc}")

    # Legacy compat (chamado por código existente na sidebar)
    def subscribe_kline(self, symbol: str, interval: str = '15m'):
        self.subscribe_klines_multi(symbol, [interval])

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: data getters
    # ─────────────────────────────────────────────────────────────────────

    def get_klines_df(self, symbol: str, interval: str = '15m',
                      limit: int = 200) -> 'pd.DataFrame | None':
        """
        Retorna DataFrame de candles do buffer em memória.
        Computa indicadores técnicos (RSI, MACD, BB, ATR…) no momento da leitura.
        Nunca chama REST — lê apenas do buffer WS.
        """
        sym = symbol.upper()
        buf = self.kline_buffers.get(sym, {}).get(interval)
        if not buf or len(buf) < 5:
            return None
        rows = list(buf)[-limit:]
        df = pd.DataFrame(rows)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        if len(df) < 5:
            return None

        close_arr  = df['close'].values.astype(float)
        high_arr   = df['high'].values.astype(float)
        low_arr    = df['low'].values.astype(float)
        volume_arr = df['volume'].values.astype(float)

        try:
            df['RSI_14']       = talib.RSI(close_arr, timeperiod=14) / 100.0
            df['SMA_20']       = talib.SMA(close_arr, timeperiod=20) / (close_arr + 1e-8)
            df['SMA_50']       = talib.SMA(close_arr, timeperiod=50) / (close_arr + 1e-8)
            upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20)
            df['BBL_20_2.0']   = lower  / (close_arr + 1e-8)
            df['BBM_20_2.0']   = middle / (close_arr + 1e-8)
            df['BBU_20_2.0']   = upper  / (close_arr + 1e-8)
            df['BBB_20_2.0']   = (upper - lower) / (middle + 1e-8)
            df['BBP_20_2.0']   = (close_arr - lower) / (upper - lower + 1e-8)
            macd, signal, hist = talib.MACD(close_arr)
            df['MACD_12_26_9']  = macd   / (close_arr + 1e-8)
            df['MACDs_12_26_9'] = signal / (close_arr + 1e-8)
            df['MACDh_12_26_9'] = hist   / (close_arr + 1e-8)
            df['EMA_9']        = talib.EMA(close_arr, timeperiod=9)  / (close_arr + 1e-8)
            df['EMA_21']       = talib.EMA(close_arr, timeperiod=21) / (close_arr + 1e-8)
            df['ATR_14']       = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14) / (close_arr + 1e-8)
            vol_ma             = talib.SMA(volume_arr, timeperiod=20)
            df['Volume_MA_20'] = volume_arr / (vol_ma + 1e-8)
            df['open_return']  = df['open'].pct_change()
            df['high_return']  = df['high'].pct_change()
            df['low_return']   = df['low'].pct_change()
            df['close_return'] = df['close'].pct_change()
        except Exception as exc:
            logger.warning(f"[WS-BUF] Erro ao computar indicadores {sym}/{interval}: {exc}")

        df = df.fillna(0)
        return df.reset_index(drop=True)

    def get_live_price(self, symbol: str) -> float | None:
        """Retorna último preço mid do book ticker (sub-segundo latência)."""
        sym = symbol.upper()
        # Fallback: último close da kline 15m
        price = self.live_price.get(sym)
        if price:
            return price
        buf = self.kline_buffers.get(sym, {}).get('15m')
        if buf:
            return buf[-1]['close']
        return None

    def get_balance(self) -> dict | None:
        """Balance do cache WebSocket (None se dados ainda não chegaram)."""
        with self.lock:
            if self.user_data['last_update']:
                age = (datetime.now() - self.user_data['last_update']).total_seconds()
                if age < 120:
                    return {**self.user_data['balance'], 'source': 'websocket', 'error': None}
                logger.warning(f"[WS] Balance antigo ({age:.0f}s)")
                return None
            return None

    def get_positions(self) -> dict | None:
        """Positions do cache WebSocket (None se dados ainda não chegaram)."""
        with self.lock:
            if self.user_data['last_update']:
                age = (datetime.now() - self.user_data['last_update']).total_seconds()
                if age < 120:
                    return {'positions': self.user_data['positions'],
                            'source': 'websocket', 'error': None}
            return None

    def buffer_stats(self) -> dict:
        """Retorna tamanho atual de cada buffer (para debug na UI)."""
        out = {}
        for sym, ivs in self.kline_buffers.items():
            out[sym] = {iv: len(buf) for iv, buf in ivs.items()}
        return out

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE: callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _handle_user_data(self, msg: dict):
        with self.lock:
            try:
                event_type = msg.get('e')
                if event_type == 'ACCOUNT_UPDATE':
                    data = msg.get('a', {})
                    for b in data.get('B', []):
                        if b['a'] == 'USDT':
                            self.user_data['balance'] = {
                                'total':          float(b['wb']),
                                'available':      float(b['cw']),
                                'unrealized_pnl': float(b.get('bc', 0)),
                            }
                            logger.info(f"[WS] Balance → ${self.user_data['balance']['total']:.2f}")
                    positions = []
                    for p in data.get('P', []):
                        amt = float(p['pa'])
                        if amt != 0:
                            positions.append({
                                'symbol':           p['s'],
                                'positionAmt':      str(amt),
                                'entryPrice':       p['ep'],
                                'markPrice':        p['mp'],
                                'unRealizedProfit': p['up'],
                                'leverage':         p.get('l', '1'),
                            })
                    self.user_data['positions']   = positions
                    self.user_data['last_update'] = datetime.now()
                    logger.info(f"[WS] Positions → {len(positions)} abertas")
                elif event_type == 'ORDER_TRADE_UPDATE':
                    o = msg.get('o', {})
                    logger.info(f"[WS] Order update: {o.get('s')} {o.get('S')} {o.get('X')}")
            except Exception as exc:
                logger.error(f"[WS] _handle_user_data erro: {exc}")

    def _handle_kline(self, msg: dict, symbol: str, interval: str):
        """Appends closed kline to the rolling buffer."""
        try:
            if msg.get('e') != 'kline':
                return
            k = msg['k']
            if not k.get('x'):   # x=True significa candle FECHADO
                return
            candle = {
                'timestamp': int(k['t']),
                'open':      float(k['o']),
                'high':      float(k['h']),
                'low':       float(k['l']),
                'close':     float(k['c']),
                'volume':    float(k['v']),
            }
            with self.lock:
                self.kline_buffers.setdefault(symbol, {})
                buf = self.kline_buffers[symbol].setdefault(interval, deque(maxlen=_KLINE_MAXLEN))
                buf.append(candle)
            logger.debug(f"[WS-KLINE] {symbol}/{interval} closed @ {k['c']} | buf={len(buf)}")
        except Exception as exc:
            logger.error(f"[WS] _handle_kline erro {symbol}/{interval}: {exc}")

    def _handle_book_ticker(self, msg: dict):
        try:
            sym = msg.get('s', '')
            bid = float(msg.get('b', 0))
            ask = float(msg.get('a', 0))
            if bid and ask:
                self.live_price[sym] = (bid + ask) / 2
        except Exception:
            pass


# ─── SINGLETON WS MANAGER ────────────────────────────────────────────────────
# Usando @st.cache_resource garante que o objeto (e os buffers) sobreviva
# a page-reruns do Streamlit — ZERO perda de dados entre refreshes.
@st.cache_resource
def _get_ws_manager_singleton() -> BinanceWebSocketManager:
    """Cria o BinanceWebSocketManager uma única vez por sessão de servidor."""
    cfg = load_config_raw()   # helper abaixo — carrega antes do st.cache_resource
    mode = cfg.get('mode', 'testnet')
    if mode == 'testnet':
        _client = Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True,
        )
    else:
        _client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY'),
        )
    return BinanceWebSocketManager(_client)

def load_config_raw() -> dict:
    """Carrega config.yaml sem usar st.cache (pode ser chamado antes de st init)."""
    with open('config.yaml') as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING ENGINE — thread de background, independente do ciclo do Streamlit
# ═══════════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """
    Executa inferência LSTM + gestão de posições em thread daemon.

    Arquitetura:
      • Roda independente do Streamlit (não para se o browser fechar ou
        o auto-refresh estiver desligado).
      • Detecta novo candle 15m pelo timestamp do buffer WS → ZERO polling REST.
      • Verifica TP/SL/trailing a cada tick (5 s) mesmo sem candle novo.
      • Dashboard só lê engine.state (thread-safe via lock).
    """

    TICK_INTERVAL = 5          # segundos entre verificações de TP/SL/novo candle
    MIN_BUFFER_CANDLES = 52    # buffer mínimo antes de rodar inferência

    def __init__(self):
        self.lock    = threading.Lock()
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False

        # Estado visível ao dashboard (leitura somente de fora)
        self.state: dict = {
            'running':   False,
            'symbols':   [],
            'last_tick': None,          # datetime do último candle processado
            'decisions': {},            # {sym: {action, value, price, ts, rsi}}
            'portfolio': {},            # {sym: {position, balance_norm, equity_norm}}
            'log':       deque(maxlen=400),   # ring-buffer de log
            'orders':    deque(maxlen=50),    # últimas ordens executadas
            'errors':    deque(maxlen=20),
        }
        # Último timestamp de candle 15m por símbolo (detecção de novo candle)
        self._last_candle_ts: dict[str, int] = {}
        # TP L1 parcial já executado por símbolo (evita re-disparar)
        self._tp1_done: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, symbols: list[str]):
        if self.running:
            # Atualiza lista de símbolos sem reiniciar a thread
            with self.lock:
                self.state['symbols'] = symbols
            return
        self._stop.clear()
        with self.lock:
            self.state['symbols'] = symbols
            self.state['running'] = True
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='TradingEngine'
        )
        self._thread.start()
        self._log(f"[ENGINE] ▶ Iniciado para: {symbols}")

    def stop(self):
        self._stop.set()
        self.running = False
        with self.lock:
            self.state['running'] = False
        self._log("[ENGINE] ⏹ Parado pelo usuário")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str):
        entry = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        with self.lock:
            self.state['log'].append(entry)
        logger.info(msg)

    # ── Main loop (runs in daemon thread) ─────────────────────────────────────

    def _loop(self):
        # Suprime avisos ruidosos do Streamlit em threads sem ScriptRunContext
        import logging as _lg
        for _sl_name in ('streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner_utils',
                         'streamlit.runtime.scriptrunner'):
            _sl = _lg.getLogger(_sl_name)
            _sl.addFilter(lambda r: 'ScriptRunContext' not in r.getMessage())

        # Lazy-load de recursos pesados dentro da thread
        ws_mgr    = _get_ws_manager_singleton()
        client    = get_binance_client()
        models_d  = load_models()
        risk_mgr  = load_risk_manager()
        trail_mgr = load_trailing_stop_manager()
        warmup    = load_warmup_manager()
        schedule  = load_schedule_manager()
        cfg       = load_config()

        if not models_d.get('lstm_v17'):
            self._log("[ENGINE] ❌ LSTM V17.7 não encontrado — engine abortado")
            with self.lock:
                self.state['running'] = False
                self.state['errors'].append("LSTM V17.7 não encontrado em models/")
            self.running = False
            return

        self._log("[ENGINE] ✅ LSTM V17.7 pronto — aguardando candle 15m fechado...")

        while not self._stop.is_set():
            try:
                symbols = list(self.state['symbols'])
                self._tick(ws_mgr, client, models_d, risk_mgr,
                           trail_mgr, warmup, schedule, cfg, symbols)
            except Exception as exc:
                msg = f"{datetime.now().strftime('%H:%M:%S')} ERRO tick: {exc}"
                with self.lock:
                    self.state['errors'].append(msg)
                logger.error(f"[ENGINE] {exc}", exc_info=True)
            self._stop.wait(timeout=self.TICK_INTERVAL)

        with self.lock:
            self.state['running'] = False
        self.running = False

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self, ws_mgr, client, models_d, risk_mgr,
              trail_mgr, warmup, schedule, cfg, symbols):

        # 1. TP/SL/Trailing — roda a cada tick (5 s), não espera novo candle
        ws_pos = ws_mgr.get_positions()
        positions = ws_pos['positions'] if ws_pos else []
        active_syms = {s.replace('/', '').upper() for s in symbols}
        for pos in positions:
            if pos['symbol'] in active_syms:
                self._check_tpsl(client, pos, risk_mgr, trail_mgr)

        # 2. Por símbolo: detecta novo candle 15m e roda inferência
        for sym_raw in symbols:
            sym = sym_raw.replace('/', '').upper()
            buf = ws_mgr.kline_buffers.get(sym, {}).get('15m')
            if not buf or len(buf) < self.MIN_BUFFER_CANDLES:
                continue  # buffer ainda não populado

            last_ts = buf[-1]['timestamp']
            if last_ts <= self._last_candle_ts.get(sym, 0):
                continue  # mesmo candle — sem novidade

            # Novo candle fechado!
            self._last_candle_ts[sym] = last_ts
            ts_str = datetime.fromtimestamp(last_ts / 1000).strftime('%H:%M')
            self._log(f"[ENGINE] {sym} 🕯 novo candle 15m @ {ts_str}")

            # Coleta dados do buffer WS (ZERO REST)
            df_15m = ws_mgr.get_klines_df(sym, '15m', limit=200)
            multi_tf: dict = {}
            df_1h = ws_mgr.get_klines_df(sym, '1h', limit=100)
            df_4h = ws_mgr.get_klines_df(sym, '4h', limit=60)
            if df_1h is not None:
                multi_tf['1h'] = df_1h
            if df_4h is not None:
                multi_tf['4h'] = df_4h

            if df_15m is None or len(df_15m) < 52:
                self._log(f"[ENGINE] {sym} dados 15m insuficientes no buffer")
                continue

            # Warm-up: avança usando candles já presentes no buffer WS
            # Evita aguardar 12 h por novos candles ao vivo quando já existe histórico
            cur_wu, req_wu, _ = warmup.get_progress(sym)
            if cur_wu < req_wu:
                shortcut = min(len(buf) - 1, req_wu - cur_wu)  # -1: não conta o candle atual
                for _ in range(shortcut):
                    warmup.add_candle(sym)
            warmup.add_candle(sym)  # conta o candle atual
            if not warmup.is_ready(sym):
                cur_wu, req_wu, pct_wu = warmup.get_progress(sym)
                self._log(f"[ENGINE] {sym} warm-up {cur_wu}/{req_wu} ({pct_wu:.0f}%) — aguardando")
                continue

            # Schedule — verifica contra o timestamp do candle fechado
            # (+3 min de tolerância para detecção tardia)
            candle_close_dt = datetime.fromtimestamp(last_ts / 1000)
            can_sched, reason_sched = schedule.can_trade_now(
                sym, at_time=candle_close_dt
            )
            if not can_sched:
                # Segunda tentativa com tolerância: janela de até 3 min após o fechamento
                for _grace_min in range(1, 4):
                    _grace_dt = candle_close_dt + timedelta(minutes=_grace_min)
                    can_sched, reason_sched = schedule.can_trade_now(sym, at_time=_grace_dt)
                    if can_sched:
                        break
            if not can_sched:
                self._log(f"[ENGINE] {sym} schedule: {reason_sched}")
                continue

            # Risk
            can_risk, reason_risk = risk_mgr.should_allow_trade()
            if not can_risk:
                self._log(f"[ENGINE] {sym} risk: {reason_risk}")
                continue

            # Portfolio state
            ws_bal = ws_mgr.get_balance()
            port = self.state['portfolio'].setdefault(sym, {
                'position': 0.0, 'balance_norm': 1.0, 'equity_norm': 1.0
            })

            # Observação + inferência LSTM
            obs = prepare_observation(
                market_data_15m=df_15m,
                multi_tf_data=multi_tf or None,
                balance_norm=port['balance_norm'],
                position=port['position'],
                equity_norm=port['equity_norm'],
            )
            if obs is None:
                self._log(f"[ENGINE] {sym} falha ao preparar observação")
                continue

            lstm_states   = self.state.get('lstm_states', {}).get(sym)
            ep_start      = np.ones((1,), dtype=bool) if lstm_states is None else np.zeros((1,), dtype=bool)
            action_value, final_action, new_lstm_states = lstm_predict(
                models_d['lstm_v17'], obs, lstm_states, ep_start
            )

            current_price = float(df_15m['close'].iloc[-1])
            with self.lock:
                self.state.setdefault('lstm_states', {})[sym] = new_lstm_states
                self.state['decisions'][sym] = {
                    'action': final_action,
                    'value':  round(action_value, 4),
                    'price':  current_price,
                    'ts':     datetime.now(),
                    'rsi':    round(float(df_15m['RSI_14'].iloc[-1]) * 100, 1),
                }
                port['position'] = (1.0 if final_action == 'LONG'
                                    else -1.0 if final_action == 'SHORT' else 0.0)
                self.state['portfolio'][sym] = port

            self._log(f"[ENGINE] {sym} → {final_action} (val={action_value:.3f}) @ ${current_price:,.2f}")

            # Filtro de qualidade de entrada
            can_enter, block_reason = validate_entry_quality(df_15m, final_action, current_price)
            if not can_enter:
                self._log(f"[ENGINE] {sym} entrada filtrada: {block_reason}")
                continue

            # Executa ordem (REST — apenas aqui)
            temp_cfg = cfg.copy()
            temp_cfg['data'] = cfg['data'].copy()
            temp_cfg['data']['primary_symbol'] = sym_raw
            order = execute_trade(client, final_action, current_price, temp_cfg)

            if order:
                with self.lock:
                    self.state['orders'].append({
                        'symbol': sym,
                        'side':   order['side'],
                        'qty':    order['origQty'],
                        'price':  order.get('avgPrice', 'MKT'),
                        'ts':     datetime.now().strftime('%H:%M:%S'),
                        'action': final_action,
                    })
                avg_px = float(order.get('avgPrice', current_price) or current_price)
                side_label = 'LONG' if order['side'] == 'BUY' else 'SHORT'
                trail_mgr.register_position(sym, avg_px, side_label)
                self._log(f"[ENGINE] ✅ {order['side']} {sym} id={order['orderId']}")
                with self.lock:
                    self.state['last_tick'] = datetime.now()

    # ── TP/SL/Trailing ────────────────────────────────────────────────────────

    def _check_tpsl(self, client, pos: dict, risk_mgr, trail_mgr):
        sym   = pos['symbol']
        qty   = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark  = float(pos['markPrice'])
        ptype = 1 if qty > 0 else -1
        atr   = mark * 0.02

        # Auto-registra no trailing se ainda não registrado
        if not trail_mgr.get_stop_info(sym):
            trail_mgr.register_position(sym, entry, 'LONG' if qty > 0 else 'SHORT')

        # Trailing stop
        should_exit_trail, trail_price = trail_mgr.update(sym, mark)
        if should_exit_trail:
            order = close_position_direct(client, sym, qty)
            if order:
                trail_mgr.remove_position(sym)
                pnl = (mark - entry) / entry * ptype * 100
                self._log(f"[ENGINE] 🛑 Trail stop {sym} @ ${trail_price:,.2f} P&L={pnl:+.2f}%")
            return

        # Stop Loss
        if risk_mgr.should_stop_loss(entry, mark, ptype, atr=atr):
            order = close_position_direct(client, sym, qty)
            if order:
                trail_mgr.remove_position(sym)
                pnl = (mark - entry) / entry * ptype * 100
                self._log(f"[ENGINE] 🛑 SL {sym} P&L={pnl:+.2f}% @ ${mark:,.2f}")
                self._tp1_done.discard(sym)
            return

        # Take Profit
        should_tp, tp_level = risk_mgr.should_take_profit(
            entry, mark, ptype, return_level=True
        )
        if should_tp:
            pnl = (mark - entry) / entry * ptype * 100
            if tp_level == 2:
                order = close_position_direct(client, sym, qty)
                if order:
                    trail_mgr.remove_position(sym)
                    self._tp1_done.discard(sym)
                    self._log(f"[ENGINE] 🎯 TP L2 (100%) {sym} +{pnl:.2f}%")
            elif tp_level == 1 and sym not in self._tp1_done:
                order = close_position_direct(client, sym, qty / 2)
                if order:
                    self._tp1_done.add(sym)
                    self._log(f"[ENGINE] 🎯 TP L1 (50%) {sym} +{pnl:.2f}%")


@st.cache_resource
def get_trading_engine() -> TradingEngine:
    """Singleton TradingEngine — sobrevive a qualquer rerun/F5/reload."""
    return TradingEngine()


# ── Sincroniza session_state com o singleton (sobrevive a Streamlit re-runs) ────
# _get_ws_manager_singleton() é chamado AQUI para garantir que o objeto exista
# antes de qualquer renderização de UI.
_ws_singleton = _get_ws_manager_singleton()
# Sempre mantém session_state apontando para o mesmo objeto (compat legado)
st.session_state['ws_manager'] = _ws_singleton

# Controle de conexão REST: o usuário PRECISA clicar em 'Conectar REST' para
# ativar chamadas REST. Impede ban infinito por restart automático.
# Flag é resetada a cada reinicio do servidor (intencional!).
if '_rest_connected' not in st.session_state:
    st.session_state['_rest_connected'] = False

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════
# BAN STATE PERSISTENCE — Sobrevive a page reloads (session_state é reset a cada reload!)
# ═════════════════════════════════════════════════════════════════════════
_BAN_FILE = Path("logs/.ban_state.json")
_REST_RATE_FILE = Path("logs/.last_rest_call")  # persiste entre restarts
_REST_COOLDOWN_SECS = 90  # nunca faça REST calls com menos de 90s de intervalo


def _rest_rate_ok() -> tuple[bool, float]:
    """
    Verifica cooldown entre chamadas REST (persiste entre restarts via arquivo).
    Retorna (pode_chamar: bool, segundos_para_liberar: float).
    """
    try:
        if _REST_RATE_FILE.exists():
            last_call = float(_REST_RATE_FILE.read_text().strip())
            elapsed = time.time() - last_call
            if elapsed < _REST_COOLDOWN_SECS:
                return False, _REST_COOLDOWN_SECS - elapsed
    except Exception:
        pass
    return True, 0.0


def _touch_rest_rate():
    """Registra timestamp da última chamada REST em arquivo."""
    try:
        _REST_RATE_FILE.parent.mkdir(exist_ok=True)
        _REST_RATE_FILE.write_text(str(time.time()))
    except Exception:
        pass


def _is_banned() -> tuple[bool, float]:
    """
    Verifica se o IP está banido consultando session_state e arquivo persistente.
    Retorna (banido: bool, segundos_restantes: float).
    Atualiza session_state a partir do arquivo se necessário (sobrevive a reload).
    """
    import json as _json
    # 1) Tenta restaurar do arquivo se session_state estiver vazio
    if 'ban_expires_at' not in st.session_state:
        try:
            if _BAN_FILE.exists():
                data = _json.loads(_BAN_FILE.read_text())
                expires_at = float(data.get('ban_expires_at', 0))
                if expires_at > time.time():
                    st.session_state['ban_expires_at'] = expires_at
                    st.session_state['last_ban_time'] = datetime.fromtimestamp(
                        float(data.get('banned_at', time.time()))
                    )
                else:
                    _BAN_FILE.unlink(missing_ok=True)  # ban expirado, limpa
        except Exception:
            pass

    expires_at = st.session_state.get('ban_expires_at', 0)
    remaining = expires_at - time.time()
    if remaining > 0:
        return True, remaining
    return False, 0.0


def _register_ban(error_str: str, context: str = ''):
    """
    Detecta e persiste ban a partir da mensagem de erro da Binance.
    Salva em session_state E em arquivo para sobreviver a page reloads.
    """
    import json as _json
    if 'banned' not in error_str.lower() and '-1003' not in error_str:
        return
    match = re.search(r'banned until (\d+)', error_str)
    if match:
        ban_expires_ms = int(match.group(1))
        ban_expires_at = ban_expires_ms / 1000
    else:
        ban_expires_at = time.time() + 600  # fallback: assume 10 min se não tiver timestamp
    st.session_state['ban_expires_at'] = ban_expires_at
    st.session_state['last_ban_time'] = datetime.now()
    try:
        _BAN_FILE.parent.mkdir(exist_ok=True)
        _BAN_FILE.write_text(_json.dumps({
            'ban_expires_at': ban_expires_at,
            'banned_at': time.time(),
        }))
    except Exception:
        pass
    expires_str = datetime.fromtimestamp(ban_expires_at).strftime('%H:%M:%S')
    remaining_min = (ban_expires_at - time.time()) / 60
    tag = f"[{context}] " if context else ""
    logging.getLogger(__name__).error(
        f"{tag}IP BANIDO até {expires_str} ({remaining_min:.1f} min restantes) — "
        f"ban.json salvo, próximas chamadas REST bloqueadas automaticamente"
    )


# Restaura ban do arquivo em TODA inicialização de page (evita REST call durante ban)
_is_banned()  # efeito colateral: popula session_state se ban ainda ativo


# Configurar logging
log_file = Path("logs/trading_decisions.log")
log_file.parent.mkdir(exist_ok=True)

# Handler para arquivo com UTF-8
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

# Handler para console com UTF-8 e ignore de erros
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
# Configura stream para UTF-8
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Configura logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🤖 Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Cache config
@st.cache_resource
def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_risk_manager():
    """Carrega Risk Manager"""
    return RiskManager()

@st.cache_resource
def load_trailing_stop_manager():
    """Carrega Trailing Stop Manager"""
    config = load_config()
    risk_config = config.get('risk_management', {})
    
    activation = risk_config.get('trailing_stop_activation', 0.03)
    distance = risk_config.get('trailing_stop_distance', 0.015)
    
    return TrailingStopManager(activation_pct=activation, distance_pct=distance)

@st.cache_resource
def load_warmup_manager():
    """Carrega Warmup Manager"""
    config = load_config()
    risk_config = config.get('risk_management', {})
    
    required_candles = risk_config.get('warm_up_candles', 50)
    
    return WarmupManager(required_candles=required_candles)

@st.cache_resource
def load_schedule_manager():
    """Carrega Schedule Manager"""
    # Schedule automático ou customizado
    return ScheduleManager()

@st.cache_resource
def load_models():
    """Carrega LSTM V17.7 (RecurrentPPO 600k) - modelo principal"""
    models = {'lstm_v17': None}
    # LSTM V17.7 (RecurrentPPO, 600k steps)
    try:
        logger.info("[MODELS] Carregando modelo LSTM V17.7 (600k)...")
        from sb3_contrib import RecurrentPPO
        lstm_v17_path = "models/recurrent_ppo_v17_lstm_20260221_030417_600000_steps.zip"
        if Path(lstm_v17_path).exists():
            models['lstm_v17'] = RecurrentPPO.load(lstm_v17_path)
            logger.info(f"[MODELS] ✅ LSTM V17.7 carregado: {lstm_v17_path}")
        else:
            logger.warning(f"[MODELS] ⚠️ LSTM V17.7 não encontrado: {lstm_v17_path}")
    except Exception as e:
        logger.error(f"[MODELS] ❌ Erro ao carregar LSTM V17.7: {e}")

    return models

@st.cache_resource
def get_binance_client():
    config = load_config()
    mode = config.get('mode', 'testnet')
    
    if mode == 'testnet':
        return Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True
        )
    else:
        return Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )

def calculate_atr(df, period=14):
    """Calcula Average True Range (volatilidade)"""
    try:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = talib.ATR(high, low, close, timeperiod=period)
        return atr[-1] if len(atr) > 0 else 0
    except:
        return 0

def detect_market_regime(df):
    """
    Detecta regime de mercado: BULL, BEAR, SIDEWAYS
    
    Usa:
    - SMA 20/50 crossover
    - ADX para força da tendência
    - Volatilidade (ATR)
    """
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # SMAs
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        
        # ADX (força da tendência)
        adx = talib.ADX(high, low, close, timeperiod=14)
        
        current_price = close[-1]
        current_sma20 = sma_20[-1]
        current_sma50 = sma_50[-1]
        current_adx = adx[-1]
        
        # Lógica de detecção
        if current_adx < 20:
            # ADX baixo = mercado lateral
            return 'SIDEWAYS', current_adx
        elif current_sma20 > current_sma50 and current_price > current_sma20:
            # Preço acima das médias e SMA20 > SMA50 = BULL
            return 'BULL', current_adx
        elif current_sma20 < current_sma50 and current_price < current_sma20:
            # Preço abaixo das médias e SMA20 < SMA50 = BEAR
            return 'BEAR', current_adx
        else:
            # Transição
            return 'SIDEWAYS', current_adx
            
    except Exception as e:
        logger.error(f"[REGIME] Erro ao detectar regime: {e}")
        return 'UNKNOWN', 0

def calculate_correlation(df1, df2, period=50):
    """Calcula correlação entre dois ativos"""
    try:
        returns1 = df1['close'].pct_change().tail(period)
        returns2 = df2['close'].pct_change().tail(period)
        
        correlation = returns1.corr(returns2)
        return correlation
    except:
        return 0

def calculate_position_size_dynamic(balance, base_size, volatility_atr, current_price, 
                                   win_streak=0, regime='SIDEWAYS', confidence=1.0):
    """
    Calcula tamanho de posição dinâmico baseado em múltiplos fatores
    
    Args:
        balance: Saldo disponível
        base_size: Tamanho base (ex: 0.03 = 3%)
        volatility_atr: ATR normalizado (ATR/price)
        current_price: Preço atual
        win_streak: Número de wins consecutivos (positivo) ou losses (negativo)
        regime: BULL, BEAR, SIDEWAYS
        confidence: Nível de confiança do modelo (0-1)
    
    Returns:
        quantity: Quantidade a operar
    """
    try:
        config = load_config()
        risk_config = config.get('risk_management', {})
        
        # 1. Fator de volatilidade (menor posição em alta volatilidade)
        volatility_factor = 1.0
        if volatility_atr > 0.02:  # >2% ATR
            volatility_factor = 0.7  # Reduz 30%
        elif volatility_atr > 0.015:  # >1.5% ATR
            volatility_factor = 0.85  # Reduz 15%
        
        # 2. Fator de win streak
        streak_factor = 1.0
        if win_streak > 2:  # 3+ wins consecutivos
            streak_factor = risk_config.get('max_win_streak_multiplier', 1.2)
        elif win_streak < -2:  # 3+ losses consecutivos
            streak_factor = risk_config.get('min_win_streak_multiplier', 0.8)
        
        # 3. Fator de regime de mercado
        regime_factor = 1.0
        if regime == 'SIDEWAYS':
            regime_factor = 0.8  # Reduz 20% em mercado lateral
        elif regime == 'BULL' or regime == 'BEAR':
            regime_factor = 1.1  # Aumenta 10% em tendência forte
        
        # 4. Fator de confiança do modelo
        confidence_factor = max(0.5, confidence)  # Mínimo 50%
        
        # Calcula tamanho final
        adjusted_size = base_size * volatility_factor * streak_factor * regime_factor * confidence_factor
        
        # Limita entre 1% e 5%
        adjusted_size = max(0.01, min(0.05, adjusted_size))
        
        # Calcula quantidade
        leverage = config['environment']['leverage']
        quantity = (balance * adjusted_size * leverage) / current_price
        
        logger.info(f"[POSITION_SIZE] Base: {base_size:.1%} | Vol: {volatility_factor:.2f} | "
                   f"Streak: {streak_factor:.2f} | Regime: {regime_factor:.2f} | "
                   f"Conf: {confidence_factor:.2f} → Final: {adjusted_size:.1%}")
        
        return round(quantity, 3)
        
    except Exception as e:
        logger.error(f"[POSITION_SIZE] Erro: {e}")
        return 0

def collect_market_data(_client, symbol: str = 'BTCUSDT',
                         interval: str = '15m', limit: int = 1000) -> 'pd.DataFrame | None':
    """
    Coleta dados OHLCV + indicadores técnicos.

    Prioridade:
      1. Buffer WebSocket em memória (ZERO chamadas REST)
      2. REST API — somente se o buffer estiver vazio E o usuário autorizou

    O buffer é mantido pelo BinanceWebSocketManager (deque de até 600 candles)
    e é populado pelo bootstrap manual + kline stream.
    """
    ws_mgr: BinanceWebSocketManager = st.session_state.get('ws_manager')

    # ── 1. Tenta WebSocket buffer primeiro ────────────────────────────────
    if ws_mgr is not None:
        df = ws_mgr.get_klines_df(symbol, interval, limit=max(limit, 200))
        if df is not None and len(df) >= 5:
            logger.debug(f"[DATA-WS] {symbol}/{interval}: {len(df)} candles do buffer WS")
            return df

    # ── 2. Fallback REST (apenas se explicitamente autorizado) ────────────────
    is_banned, remaining = _is_banned()
    if is_banned:
        logger.warning(f"[DATA-REST] Ban ativo: {remaining:.0f}s restantes. Use bootstrap WS.")
        return None

    _rest_ok = (st.session_state.get('_rest_connected', False)
                or st.session_state.get('bot_running', False))
    if not _rest_ok:
        logger.debug(f"[DATA-REST] REST desconectado — bloqueado para {symbol}/{interval}")
        return None

    try:
        _touch_rest_rate()
        logger.info(f"[DATA-REST] REST call {symbol}/{interval} (bootstrap WS ainda não feito)")
        klines = _client.futures_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        close_arr  = df['close'].values.astype(float)
        high_arr   = df['high'].values.astype(float)
        low_arr    = df['low'].values.astype(float)
        volume_arr = df['volume'].values.astype(float)

        df['RSI_14'] = talib.RSI(close_arr, timeperiod=14) / 100.0
        df['SMA_20'] = talib.SMA(close_arr, timeperiod=20) / (close_arr + 1e-8)
        df['SMA_50'] = talib.SMA(close_arr, timeperiod=50) / (close_arr + 1e-8)
        upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20)
        df['BBL_20_2.0'] = lower  / (close_arr + 1e-8)
        df['BBM_20_2.0'] = middle / (close_arr + 1e-8)
        df['BBU_20_2.0'] = upper  / (close_arr + 1e-8)
        df['BBB_20_2.0'] = (upper - lower) / (middle + 1e-8)
        df['BBP_20_2.0'] = (close_arr - lower) / (upper - lower + 1e-8)
        macd, signal, hist = talib.MACD(close_arr)
        df['MACD_12_26_9']  = macd   / (close_arr + 1e-8)
        df['MACDs_12_26_9'] = signal / (close_arr + 1e-8)
        df['MACDh_12_26_9'] = hist   / (close_arr + 1e-8)
        df['EMA_9']  = talib.EMA(close_arr, timeperiod=9)  / (close_arr + 1e-8)
        df['EMA_21'] = talib.EMA(close_arr, timeperiod=21) / (close_arr + 1e-8)
        df['ATR_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14) / (close_arr + 1e-8)
        vol_ma = talib.SMA(volume_arr, timeperiod=20)
        df['Volume_MA_20'] = volume_arr / (vol_ma + 1e-8)
        df['open_return']  = df['open'].pct_change()
        df['high_return']  = df['high'].pct_change()
        df['low_return']   = df['low'].pct_change()
        df['close_return'] = df['close'].pct_change()
        df = df.fillna(0)

        # Popula o buffer WS com estes dados para próximos ciclos
        if ws_mgr is not None:
            ws_mgr.kline_buffers.setdefault(symbol.upper(), {})
            from collections import deque as _deque
            buf = ws_mgr.kline_buffers[symbol.upper()].setdefault(
                interval, _deque(maxlen=_KLINE_MAXLEN))
            for _, row in df.iterrows():
                buf.append({
                    'timestamp': int(row['timestamp']) if 'timestamp' in row else 0,
                    'open':  float(row['open']),
                    'high':  float(row['high']),
                    'low':   float(row['low']),
                    'close': float(row['close']),
                    'volume':float(row['volume']),
                })
            logger.info(f"[DATA-REST] Buffer WS populado via REST: {symbol}/{interval} ({len(buf)} candles)")

        logger.info(f"[DATA-REST] {symbol} {interval}: {len(df)} candles")
        return df

    except Exception as e:
        error_str = str(e)
        _register_ban(error_str, 'DATA')
        logger.error(f"[DATA-REST] Erro {symbol} {interval}: {e}")
        return None

def collect_multi_timeframe_data(client, symbol: str = 'BTCUSDT') -> dict | None:
    """
    Coleta dados de múltiplos timeframes para análise contextual.
    Usa o buffer WS em memória — ZERO chamadas REST quando bootstrapped.

    Returns:
        dict: {'15m': df, '1h': df, '4h': df}  ou None se sem dados.
    """
    try:
        config = load_config()
        timeframes = config['data'].get('timeframes', {
            'tactical': '15m',
            'operational': '1h',
            'strategic': '4h'
        })

        data = {}
        for tf_name, tf_value in timeframes.items():
            df = collect_market_data(client, symbol=symbol, interval=tf_value, limit=200)
            if df is not None:
                data[tf_value] = df
                src = 'WS' if st.session_state.get('ws_manager') and \
                    st.session_state['ws_manager'].kline_buffers.get(
                        symbol.upper(), {}).get(tf_value) else 'REST'
                logger.info(f"[MULTI-TF] {symbol} {tf_value}: {len(df)} candles [{src}]")

        return data if data else None

    except Exception as e:
        logger.error(f"[MULTI-TF] Erro: {e}")
        return None

# Colunas do dataset de treino na ordem exata (sem timestamp)
# Deve bater com: select_dtypes(include=[np.number]) no CSV de treino
FEATURE_COLS_15M = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI_14', 'SMA_20', 'SMA_50',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
    'EMA_9', 'EMA_21', 'ATR_14', 'Volume_MA_20'
]  # 20 features - índices: close=3, RSI=5, BBP=12, MACDh=15

# Índices das features no array de observação da TF maior
IDX_CLOSE = 3
IDX_RSI   = 5
IDX_BBP   = 12
IDX_MACDH = 15

def prepare_observation(market_data_15m, multi_tf_data=None,
                        balance_norm=1.0, position=0.0, equity_norm=1.0):
    """
    Prepara observação para o modelo LSTM V17.7: shape (50, 31)
    
    Deve replicar EXATAMENTE a lógica de _get_observation() do
    TradingEnvMultiTFLSTM para evitar train/inference mismatch.
    
    Estrutura:
        (50, 20) 15m features   - mesma ordem do CSV de treino
        (50, 4)  1h context     - RSI, BBP, MACDh, close%diff
        (50, 4)  4h context     - RSI, BBP, MACDh, close%diff
        (50, 1)  balance_norm
        (50, 1)  position       (-1=short, 0=flat, 1=long)
        (50, 1)  equity_norm
        --------
        (50, 31) total
    """
    try:
        # === 15m: 20 features (50 candles mais recentes) ===
        obs_15m = market_data_15m[FEATURE_COLS_15M].iloc[-50:].values.copy()  # (50, 20)
        if len(obs_15m) < 50:
            pad = np.zeros((50 - len(obs_15m), 20))
            obs_15m = np.vstack([pad, obs_15m])
        
        logger.info(f"[OBS] 15m shape: {obs_15m.shape}")
        
        # === 1h / 4h context: 4 features cada ===
        ctx_1h = np.zeros((50, 4), dtype=np.float32)
        ctx_4h = np.zeros((50, 4), dtype=np.float32)
        
        if multi_tf_data is not None:
            df_1h = multi_tf_data.get('1h')
            df_4h = multi_tf_data.get('4h')
            
            if df_1h is not None and len(df_1h) > 0:
                arr_1h = df_1h[FEATURE_COLS_15M].values  # (N, 20)
                for i in range(50):  # i=0 mais antigo, i=49 mais recente
                    offset = 49 - i           # quantos candles 15m para trás
                    # +1 replica o -1 do env: garante que usamos o ÚLTIMO candle 1h
                    # JÁ FECHADO, não o da hora atual (que pode estar ainda aberto).
                    # Equivalente a: idx_1h = (step_15m - 1) // 4 no ambiente de treino.
                    idx_from_end_1h = offset // 4 + 1
                    row = max(0, len(arr_1h) - 1 - idx_from_end_1h)
                    price_15m = float(obs_15m[i, IDX_CLOSE])
                    if price_15m == 0:
                        price_15m = 1.0
                    ctx_1h[i, 0] = arr_1h[row, IDX_RSI]
                    ctx_1h[i, 1] = arr_1h[row, IDX_BBP]
                    ctx_1h[i, 2] = arr_1h[row, IDX_MACDH]
                    ctx_1h[i, 3] = (arr_1h[row, IDX_CLOSE] / price_15m - 1) * 100
            
            if df_4h is not None and len(df_4h) > 0:
                arr_4h = df_4h[FEATURE_COLS_15M].values  # (N, 20)
                for i in range(50):
                    offset = 49 - i
                    # +1 análogo ao -1 do env para 4h (16 candles 15m por candle 4h).
                    # Usa o último candle 4h fechado, não o da janela atual.
                    idx_from_end_4h = offset // 16 + 1
                    row = max(0, len(arr_4h) - 1 - idx_from_end_4h)
                    price_15m = float(obs_15m[i, IDX_CLOSE])
                    if price_15m == 0:
                        price_15m = 1.0
                    ctx_4h[i, 0] = arr_4h[row, IDX_RSI]
                    ctx_4h[i, 1] = arr_4h[row, IDX_BBP]
                    ctx_4h[i, 2] = arr_4h[row, IDX_MACDH]
                    ctx_4h[i, 3] = (arr_4h[row, IDX_CLOSE] / price_15m - 1) * 100
        
        # === Portfolio: 3 colunas SEPARADAS ===
        balance_col  = np.full((50, 1), balance_norm)
        position_col = np.full((50, 1), float(position))
        equity_col   = np.full((50, 1), equity_norm)
        
        # === Concatenar: (50,20)+(50,4)+(50,4)+(50,1)+(50,1)+(50,1) = (50,31) ===
        obs = np.hstack([
            obs_15m,      # 15m features
            ctx_1h,       # 1h context
            ctx_4h,       # 4h context
            balance_col,   
            position_col,  
            equity_col    
        ]).astype(np.float32)
        
        obs = np.clip(obs, -100, 100)
        
        logger.info(f"[OBS] Shape final: {obs.shape}")
        return obs
        
    except Exception as e:
        logger.error(f"[OBS] Erro ao preparar observação: {e}")
        import traceback; traceback.print_exc()
        return None

def lstm_predict(model, obs, lstm_states, episode_start):
    """
    Faz predição com LSTM V17.7 (RecurrentPPO), mantendo estado oculto.

    Args:
        model: RecurrentPPO model
        obs: np.array shape (50, 31)
        lstm_states: hidden states anteriores (None para início de episódio)
        episode_start: np.array bool (True se início)
    Returns:
        action_value: float
        final_action: str ("LONG", "SHORT", "FLAT")
        new_lstm_states: atualizado para próxima chamada
    """
    obs_batched = obs[np.newaxis]  # (1, 50, 31)
    action, new_lstm_states = model.predict(
        obs_batched,
        state=lstm_states,
        episode_start=episode_start,
        deterministic=True
    )
    # RecurrentPPO pode retornar array 0-d ou shape (1,) — np.squeeze garante scalar
    action_value = float(np.squeeze(action))
    if action_value < -0.1:
        final_action = "SHORT"
    elif action_value > 0.1:
        final_action = "LONG"
    else:
        final_action = "FLAT"
    logger.info(f"[LSTM] action={action_value:.3f} → {final_action}")
    return action_value, final_action, new_lstm_states

@st.cache_data(ttl=120)  # Cache de 120s (2min) para evitar ban - era 30s
def get_account_balance_cached(_client):
    """Retorna saldo da conta (WebSocket primeiro, REST como fallback)"""
    try:
        # 1) Tenta WebSocket primeiro (zero chamadas REST!)
        ws_mgr = st.session_state.get('ws_manager')
        if ws_mgr and ws_mgr.running:
            ws_balance = ws_mgr.get_balance()
            if ws_balance:
                logger.debug("[BALANCE] Dados via WebSocket ✅")
                return {**ws_balance, 'source': 'websocket'}
            else:
                # WebSocket ativo MAS sem dados ainda (aguardando primeiro evento)
                # NÃO FAZ FALLBACK PARA REST! Retorna zeros e aguarda
                logger.info("[BALANCE] WebSocket ativo mas aguardando primeiro evento...")
                return {
                    'total': 0, 
                    'available': 0, 
                    'unrealized_pnl': 0, 
                    'source': 'websocket_waiting', 
                    'error': None
                }
        
        # 2) Fallback: REST API (SÓ SE WEBSOCKET NÃO ESTIVER ATIVO)
        # Verifica ban usando expiração REAL da Binance (persiste entre reloads)
        is_banned, remaining = _is_banned()
        if is_banned:
            logger.warning(f"[BALANCE] Ban ativo: {remaining:.0f}s restantes")
            return {'total': 0, 'available': 0, 'unrealized_pnl': 0, 'source': 'banned', 'error': 'IP banned'}

        logger.debug("[BALANCE] WebSocket inativo, fallback para REST API")
        balance = _client.futures_account_balance()
        usdt = [b for b in balance if b['asset'] == 'USDT'][0]
        return {
            'total': float(usdt['balance']),
            'available': float(usdt['availableBalance']),
            'unrealized_pnl': float(usdt.get('crossUnPnl', 0)),
            'source': 'rest',
            'error': None
        }
    except Exception as e:
        error_str = str(e)
        _register_ban(error_str, 'BALANCE')  # persiste ban em arquivo + session_state
        return {'total': 0, 'available': 0, 'unrealized_pnl': 0, 'source': 'error', 'error': error_str}

def get_account_balance(client):
    """Retorna saldo da conta (wrapper sem cache para compatibilidade)"""
    result = get_account_balance_cached(client)
    if result.get('error'):
        # Não levanta exceção se houver erro - retorna zeros
        logger.warning(f"[BALANCE] Erro ao buscar saldo: {result['error']}")
        return {'total': 0, 'available': 0, 'unrealized_pnl': 0, 'source': 'error'}
    return result

@st.cache_data(ttl=120)  # Cache de 120s (2min) para evitar ban - era 30s
def get_open_positions_cached(_client):
    """Retorna posições abertas (WebSocket primeiro, REST como fallback)"""
    try:
        # 1) Tenta WebSocket primeiro
        ws_mgr = st.session_state.get('ws_manager')
        if ws_mgr and ws_mgr.running:
            ws_positions = ws_mgr.get_positions()
            if ws_positions:
                logger.debug("[POSITIONS] Dados via WebSocket ✅")
                return {**ws_positions, 'source': 'websocket'}
            else:
                # WebSocket ativo MAS sem dados ainda (aguardando primeiro evento)
                # NÃO FAZ FALLBACK PARA REST! Retorna vazio e aguarda
                logger.info("[POSITIONS] WebSocket ativo mas aguardando primeiro evento...")
                return {'positions': [], 'source': 'websocket_waiting', 'error': None}
        
        # 2) Fallback: REST API (SÓ SE WEBSOCKET NÃO ESTIVER ATIVO)
        is_banned, remaining = _is_banned()
        if is_banned:
            logger.warning(f"[POSITIONS] Ban ativo: {remaining:.0f}s restantes")
            return {'positions': [], 'source': 'banned', 'error': 'IP banned'}

        logger.debug("[POSITIONS] WebSocket inativo, fallback para REST API")
        positions = _client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        return {'positions': open_positions, 'source': 'rest', 'error': None}
    except Exception as e:
        error_str = str(e)
        _register_ban(error_str, 'POSITIONS')  # persiste ban em arquivo + session_state
        return {'positions': [], 'source': 'error', 'error': error_str}

def get_open_positions(client):
    """Retorna posições abertas (wrapper sem cache)"""
    result = get_open_positions_cached(client)
    if result.get('error'):
        # Não levanta exceção se houver erro - retorna lista vazia
        logger.warning(f"[POSITIONS] Erro ao buscar posições: {result['error']}")
        return []
    return result['positions']

def validate_entry_quality(market_data: pd.DataFrame, decision: str, current_price: float) -> tuple[bool, str]:
    """
    Valida se é um bom momento técnico para entrar na posição.
    Evita entradas aleatórias no meio de mercados lateralizados ou em topos/fundos.
    
    Returns:
        (bool, str): (pode_entrar, motivo_se_bloqueado)
    """
    try:
        if decision not in ['LONG', 'SHORT']:
            return True, ""  # FLAT sempre pode executar
        
        # Extrai indicadores técnicos do último candle
        last_candle = market_data.iloc[-1]
        candle_close = float(last_candle['close'])

        # get_klines_df armazena indicadores normalizados:
        #   RSI_14  = talib.RSI  / 100        → range [0, 1]
        #   EMA_21  = talib.EMA  / close      → ratio ≈ 0.999…1.001
        #   Volume_MA_20 = volume / vol_sma   → ratio
        # De-normaliza antes de comparar com thresholds em escala humana.
        rsi   = float(last_candle['RSI_14']) * 100          # → [0, 100]
        ema21 = float(last_candle['EMA_21']) * candle_close  # → preço absoluto

        volume       = float(last_candle['volume'])
        vol_ma20_col = last_candle.get('Volume_MA_20', None)
        if vol_ma20_col is not None and float(vol_ma20_col) > 0:
            # Volume_MA_20 = volume / vol_sma → vol_sma = volume / ratio
            vol_sma      = volume / float(vol_ma20_col)
            volume_ma20  = vol_sma
        else:
            volume_ma20 = market_data['volume'].rolling(20).mean().iloc[-1]

        # Dados do candle (preços brutos — não normalizados)
        candle_open  = float(last_candle['open'])
        candle_high  = float(last_candle['high'])
        candle_low   = float(last_candle['low'])
        candle_body  = abs(candle_close - candle_open)
        candle_range = candle_high - candle_low

        # ── FILTRO 1: RSI ──────────────────────────────────────────────────────
        # Evita comprar em topo (sobrecomprado) ou vender em fundo (sobrevendido)
        if decision == 'LONG' and rsi > 70:
            return False, f"RSI sobrecomprado ({rsi:.1f} > 70) - aguardando correção"
        if decision == 'SHORT' and rsi < 30:
            return False, f"RSI sobrevendido ({rsi:.1f} < 30) - aguardando retração"

        # ── FILTRO 2: Distância da EMA21 ───────────────────────────────────────
        # Evita entrar quando preço já se afastou muito da EMA (momentum exaurido)
        if ema21 > 0:
            distance_pct = abs(current_price - ema21) / ema21 * 100
            if distance_pct > 2.0:
                direction = "acima" if current_price > ema21 else "abaixo"
                return False, f"Preço {distance_pct:.2f}% {direction} da EMA21 - momentum exaurido"

        # ── FILTRO 3: Volume ────────────────────────────────────────────────────
        # Só entra se houver volume suficiente (confirma movimento)
        if volume_ma20 > 0 and volume < volume_ma20 * 0.7:
            return False, f"Volume fraco ({volume/volume_ma20*100:.0f}% da média) - falta confirmação"
        
        # ── FILTRO 4: Padrão de Candle ─────────────────────────────────────────
        # Evita candles de indecisão (corpo muito pequeno) - mercado lateral
        if candle_range > 0:
            body_ratio = candle_body / candle_range
            if body_ratio < 0.3:
                return False, f"Candle de indecisão (corpo {body_ratio*100:.0f}% do range) - mercado lateral"
        
        # ── FILTRO 5: Direção do candle alinhada com decisão ───────────────────
        # Para LONG, prefere candle bullish; para SHORT, prefere bearish
        candle_bullish = candle_close > candle_open
        if decision == 'LONG' and not candle_bullish:
            return False, "Candle bearish - aguardando confirmação bullish"
        if decision == 'SHORT' and candle_bullish:
            return False, "Candle bullish - aguardando confirmação bearish"
        
        # ✅ Passou em todos os filtros!
        return True, ""
    
    except Exception as e:
        logger.warning(f"[ENTRY_FILTER] Erro ao validar qualidade de entrada: {e}")
        return True, ""  # Em caso de erro, permite entrada (failsafe)

def execute_trade(client, decision, current_price, config):
    """Executa trade baseado na decisão do ensemble"""
    try:
        # Verifica se está banido antes de tentar executar trade
        if 'last_ban_time' in st.session_state:
            time_since_ban = (datetime.now() - st.session_state['last_ban_time']).total_seconds()
            if time_since_ban < 300:  # 5 minutos após ban
                logger.warning(f"[TRADE] Não é possível executar - IP banido ({int(300-time_since_ban)}s restantes)")
                return False
        
        symbol = config['data'].get('primary_symbol', 'BTC/USDT').replace('/', '')  # BTC/USDT -> BTCUSDT
        
        # Verifica posição atual
        positions = client.futures_position_information(symbol=symbol)
        
        # Encontra a posição do símbolo (pode haver múltiplas posições)
        current_position = 0.0
        for pos in positions:
            if pos['symbol'] == symbol:
                current_position = float(pos['positionAmt'])
                break
        
        logger.info(f"[TRADE] Posicao atual: {current_position} BTC, Decisao: {decision}")
        
        # Define side baseado na decisão
        if decision == 'LONG' and current_position <= 0:
            # Fecha SHORT (se houver) e abre LONG
            if current_position < 0:
                logger.info(f"[TRADE] Fechando posicao SHORT de {current_position}")
                client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=abs(current_position)
                )
            
            # Calcula quantidade para LONG
            balance = get_account_balance(client)
            position_size = config['environment']['position_size']
            leverage = config['environment']['leverage']
            quantity = (balance['available'] * position_size * leverage) / current_price
            quantity = round(quantity, 3)  # Arredonda para 3 casas decimais
            
            logger.info(f"[TRADE] Abrindo posicao LONG: {quantity} BTC @ ${current_price:,.2f}")
            order = client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"[TRADE] ✅ Ordem LONG executada: {order['orderId']}")
            return order
            
        elif decision == 'SHORT' and current_position >= 0:
            # Fecha LONG (se houver) e abre SHORT
            if current_position > 0:
                logger.info(f"[TRADE] Fechando posicao LONG de {current_position}")
                client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=current_position
                )
            
            # Calcula quantidade para SHORT
            balance = get_account_balance(client)
            position_size = config['environment']['position_size']
            leverage = config['environment']['leverage']
            quantity = (balance['available'] * position_size * leverage) / current_price
            quantity = round(quantity, 3)
            
            logger.info(f"[TRADE] Abrindo posicao SHORT: {quantity} BTC @ ${current_price:,.2f}")
            order = client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"[TRADE] ✅ Ordem SHORT executada: {order['orderId']}")
            return order
            
        elif decision == 'FLAT' and current_position != 0:
            # Fecha qualquer posição aberta
            side = 'SELL' if current_position > 0 else 'BUY'
            logger.info(f"[TRADE] Fechando posicao {side}: {abs(current_position)} BTC")
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=abs(current_position)
            )
            logger.info(f"[TRADE] ✅ Posicao fechada: {order['orderId']}")
            return order
        else:
            logger.info(f"[TRADE] Sem mudança de posição (atual: {current_position}, decisão: {decision})")
            return None
            
    except Exception as e:
        logger.error(f"[TRADE] Erro ao executar trade: {e}")
        return None

def close_position_direct(client, symbol: str, qty: float) -> dict | None:
    """Fecha uma posição aberta diretamente pelo símbolo e quantidade.
    Funciona para qualquer posição, inclusive posições legadas de modelos antigos.
    qty > 0 → posição LONG (precisa de SELL); qty < 0 → posição SHORT (precisa de BUY).
    """
    try:
        if qty == 0:
            return None
        
        # Mapa de precisão de quantidade por símbolo (casas decimais permitidas pela Binance)
        QUANTITY_PRECISION = {
            'BTCUSDT': 3,    # 0.001 BTC
            'ETHUSDT': 3,    # 0.001 ETH
            'BNBUSDT': 2,    # 0.01 BNB
            'SOLUSDT': 1,    # 0.1 SOL
            'ADAUSDT': 0,    # 1 ADA
            'DOTUSDT': 1,    # 0.1 DOT
            'MATICUSDT': 0,  # 1 MATIC
        }
        
        # Arredonda quantidade para a precisão correta do símbolo
        precision = QUANTITY_PRECISION.get(symbol, 3)  # padrão: 3 casas decimais
        quantity = round(abs(qty), precision)
        
        # Valida que a quantidade arredondada não é zero
        if quantity == 0:
            logger.warning(f"[CLOSE] ⚠️ Quantidade muito pequena após arredondamento: {symbol} qty={qty:.6f} → {quantity}")
            return None
        
        side = 'SELL' if qty > 0 else 'BUY'
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity,
            reduceOnly=True          # garante que só fecha, nunca abre nova posição
        )
        logger.info(f"[CLOSE] ✅ Posição fechada: {symbol} qty={qty:.6f} → arredondado: {quantity} → order: {order['orderId']}")
        return order
    except Exception as e:
        logger.error(f"[CLOSE] ❌ Erro ao fechar {symbol}: {e}")
        return None


def close_all_positions(client) -> list:
    """Fecha TODAS as posições abertas (inclui legadas de modelos antigos)."""
    open_positions = get_open_positions(client)
    results = []
    for pos in open_positions:
        symbol = pos['symbol']
        qty = float(pos['positionAmt'])
        result = close_position_direct(client, symbol, qty)
        results.append({'symbol': symbol, 'qty': qty, 'order': result})
    return results


@st.cache_data(ttl=120)  # Cache de 120s (2min) para evitar ban - era 30s
def get_recent_trades(_client, symbol=None, symbols=None, limit=10):
    """Retorna trades recentes com proteção anti-ban"""
    try:
        # Verifica ban (persiste entre page reloads via arquivo)
        is_banned, remaining = _is_banned()
        if is_banned:
            logger.warning(f"[TRADES] Ban ativo: {remaining:.0f}s restantes")
            return []
        
        if symbols:
            # Busca trades de múltiplos símbolos
            all_trades = []
            for sym in symbols:
                try:
                    trades = _client.futures_account_trades(symbol=sym, limit=limit)
                    all_trades.extend(trades)
                except:
                    pass  # ignora símbolos sem trades
            # Ordena por tempo (mais recentes primeiro)
            all_trades.sort(key=lambda x: x['time'], reverse=True)
            return all_trades[:limit] if limit else all_trades
        elif symbol:
            trades = _client.futures_account_trades(symbol=symbol, limit=limit)
            return trades
        else:
            return []
    except Exception as e:
        _register_ban(str(e), 'TRADES')  # persiste ban em arquivo + session_state
        return []

def get_klines(_client, symbol: str = 'BTCUSDT', interval: str = '15m', limit: int = 100) -> pd.DataFrame:
    """
    Retorna candles OHLCV para gráficos.
    Usa buffer WebSocket em memória — ZERO chamadas REST quando bootstrapped.
    Fallback para REST somente se buffer vazio E usuário autorizou.
    """
    _empty = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # 1. Tenta WS buffer
    ws_mgr: BinanceWebSocketManager | None = st.session_state.get('ws_manager')
    if ws_mgr is not None:
        df_ws = ws_mgr.get_klines_df(symbol, interval, limit=limit)
        if df_ws is not None and not df_ws.empty:
            # garante coluna timestamp como datetime para plotagem
            if 'timestamp' in df_ws.columns and not pd.api.types.is_datetime64_any_dtype(df_ws['timestamp']):
                df_ws = df_ws.copy()
                df_ws['timestamp'] = pd.to_datetime(df_ws['timestamp'], unit='ms', errors='coerce')
            return df_ws

    # 2. Fallback REST
    is_banned, _ = _is_banned()
    if is_banned:
        return _empty
    _rest_ok = (st.session_state.get('_rest_connected', False)
                or st.session_state.get('bot_running', False))
    if not _rest_ok:
        return _empty
    try:
        _touch_rest_rate()
        klines = _client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        _register_ban(str(e), 'KLINES')
        logger.error(f"[KLINES-REST] {e}")
        return _empty

def plot_candlestick(df, symbol='BTC/USDT'):
    """Gráfico de candlestick com volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Preço {symbol}', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#26a69a' if close >= open_ else '#ef5350' 
              for close, open_ in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def calculate_performance_metrics(trades):
    """Calcula métricas de performance avançadas"""
    if not trades or len(trades) < 2:
        return None
    
    df = pd.DataFrame(trades)
    df['realizedPnl'] = df['realizedPnl'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Métricas básicas
    total_trades = len(df)
    wins = len(df[df['realizedPnl'] > 0])
    losses = len(df[df['realizedPnl'] < 0])
    win_rate = (wins / total_trades) if total_trades > 0 else 0
    
    total_pnl = df['realizedPnl'].sum()
    avg_win = df[df['realizedPnl'] > 0]['realizedPnl'].mean() if wins > 0 else 0
    avg_loss = df[df['realizedPnl'] < 0]['realizedPnl'].mean() if losses > 0 else 0
    
    # Sharpe Ratio (anualizado, assumindo 365 dias)
    returns = df['realizedPnl']
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0
    
    # Profit Factor
    gross_profit = df[df['realizedPnl'] > 0]['realizedPnl'].sum()
    gross_loss = abs(df[df['realizedPnl'] < 0]['realizedPnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    df['cumulative_pnl'] = df['realizedPnl'].cumsum()
    df['running_max'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['running_max']
    max_drawdown = df['drawdown'].min()
    
    # Recovery Factor
    recovery_factor = (total_pnl / abs(max_drawdown)) if max_drawdown < 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor,
        'expectancy': expectancy
    }

def plot_pnl_chart(trades):
    """Gráfico de P&L acumulado"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['realizedPnl'] = df['realizedPnl'].astype(float)
    df['cumulative_pnl'] = df['realizedPnl'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['cumulative_pnl'],
        mode='lines+markers',
        name='P&L Acumulado',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title='P&L Acumulado',
        xaxis_title='Tempo',
        yaxis_title='P&L (USDT)',
        height=300,
        template='plotly_dark'
    )
    
    return fig

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

st.markdown('<div class="main-header">🤖 Trading Bot Dashboard</div>', unsafe_allow_html=True)

# ⚠️ ALERTA DE BAN - Verifica ban no arquivo persistente (sobrevive a reloads)
_startup_banned, _startup_remaining = _is_banned()
if _startup_banned:
    ban_expires_at = st.session_state.get('ban_expires_at', 0)
    ban_expires_str = datetime.fromtimestamp(ban_expires_at).strftime('%H:%M:%S')
    remaining_m = int(_startup_remaining) // 60
    remaining_s = int(_startup_remaining) % 60
    st.error(f"""
    🚫 **IP BANIDO PELA BINANCE API**
    
    ⏱️ Tempo restante: **{remaining_m}m {remaining_s}s** (expira às {ban_expires_str})
    
    **Por que o ban persiste mesmo após reiniciar o dashboard?**
    - ⏰ Ban expira AUTOMATICAMENTE no horário acima (não depende de trocar IP)
    - 🔄 Ban é temporário, não permanente!
    
    **O que está protegido:**
    - ✅ Todas as chamadas REST estão **bloqueadas automaticamente** até expirar
    - 💾 Estado do ban foi **salvo em arquivo** — sobrevive a reloads de página
    - 🌐 Se WebSocket estiver ativo, dados continuam chegando normalmente
    
    💡 **Aguarde {remaining_m}m {remaining_s}s. O dashboard é recargável sem problema.**
    """)
elif 'last_ban_time' in st.session_state:
    # Ban expirou nesta sessão: limpa session_state e arquivo
    for _k in ('last_ban_time', 'ban_expires_at'):
        st.session_state.pop(_k, None)
    try:
        _BAN_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    st.success("✅ Ban expirado! Chamadas REST liberadas.", icon="✅")

# Carrega config e cliente
config = load_config()
client = get_binance_client()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Status do modo
    mode = config.get('mode', 'testnet')
    if mode == 'testnet':
        st.success("🧪 Modo: TESTNET")
    elif mode == 'live':
        st.error("⚠️ Modo: LIVE (REAL)")
    else:
        st.info("📝 Modo: PAPER")
    
    st.divider()
    
    # 🆕 MULTI-PAR SELECTOR
    st.subheader("🎯 Seleção de Pares")
    st.error("🚫 **LIMITE: MAX 2 PARES**")
    st.caption("Binance Testnet: 50 calls/10s")
    st.caption("Bot faz 3 calls/par (15m+1h+4h)")
    st.caption("4 pares = 12 calls = BAN!")
    
    available_symbols = [s.replace('/', '') for s in config['data']['symbols']]
    primary_symbol = config['data']['primary_symbol'].replace('/', '')
    
    selected_symbols_raw = st.multiselect(
        "Pares Ativos",
        available_symbols,
        default=[primary_symbol] if primary_symbol in available_symbols else [available_symbols[0]],
        help="⚠️ MÁXIMO 2 pares para evitar ban!"
    )
    
    # 🔴 FORÇA LIMITE DE 2 PARES
    if len(selected_symbols_raw) > 2:
        st.error(f"❌ Máximo 2 pares permitidos! Você selecionou {len(selected_symbols_raw)}")
        selected_symbols = selected_symbols_raw[:2]
        st.warning(f"🔻 Usando apenas: {', '.join(selected_symbols)}")
    else:
        selected_symbols = selected_symbols_raw
    
    # Portfolio allocation strategy
    allocation_strategy = st.radio(
        "Estratégia de Alocação",
        ["Equal Weight", "Best Signal", "Correlation Filter"],
        help="Equal Weight: Divide capital igualmente\n"
             "Best Signal: Opera apenas o melhor sinal\n"
             "Correlation Filter: Evita pares correlacionados"
    )
    
    st.divider()
    
    # Configurações de trading
    st.subheader("📊 Parâmetros")
    for symbol in selected_symbols[:3]:  # Mostra até 3 pares
        st.text(f"📈 {symbol}")
    if len(selected_symbols) > 3:
        st.text(f"   +{len(selected_symbols)-3} mais...")
    
    st.text(f"Timeframe: {config['data']['timeframes']['tactical']}")
    st.text(f"Position Size: {config['environment']['position_size']*100}%/par")
    st.text(f"Leverage: {config['environment']['leverage']}x")

    st.divider()

    # ── REST API CONNECTION ─────────────────────────────────────────────────
    st.subheader("🔌 REST API")
    _sb_banned, _sb_ban_rem = _is_banned()
    _sb_rate_ok, _sb_rate_wait = _rest_rate_ok()
    _sb_rest_conn = st.session_state.get('_rest_connected', False)

    if _sb_banned:
        ban_exp = datetime.fromtimestamp(st.session_state.get('ban_expires_at', 0)).strftime('%H:%M:%S')
        st.error(f"🚫 Banido até {ban_exp} ({int(_sb_ban_rem//60)}m{int(_sb_ban_rem%60)}s)")
        st.caption("REST bloqueado automaticamente.")
    elif not _sb_rate_ok:
        st.warning(f"⏳ Cooldown ativo: {_sb_rate_wait:.0f}s restantes")
        st.caption(f"Intervalo mínimo: {_REST_COOLDOWN_SECS}s entre chamadas")
    elif _sb_rest_conn:
        st.success("✅ REST ativo nesta sessão")
        if st.button("🔒 Desconectar REST"):
            st.session_state['_rest_connected'] = False
            st.rerun()
    else:
        st.warning("⚪ REST desconectado (startup seguro)")
        st.caption("Nenhuma chamada REST automática no startup.")
        if st.button("🔌 Conectar REST API", type="primary",
                     help="Ativa chamadas REST. Use apenas quando NÃO estiver banido."):
            st.session_state['_rest_connected'] = True
            st.rerun()

    st.divider()

    # ── WEBSOCKET + BOOTSTRAP PANEL ────────────────────────────────────────
    st.subheader("🌐 WebSocket + Bootstrap")

    ws_mgr: BinanceWebSocketManager = st.session_state['ws_manager']
    ws_running = ws_mgr.running
    boot_done  = ws_mgr.bootstrap_done

    # Status visual
    if ws_running and boot_done:
        st.success("🟢 WS ativo + dados bootstrapped — ZERO REST calls")
    elif ws_running and not boot_done:
        st.warning("🟡 WS ativo mas sem dados. Clique Bootstrap ↓")
    else:
        st.error("🔴 WebSocket desconectado")

    # Botões de controle WS
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Iniciar WS", disabled=ws_running, key="btn_start_ws"):
            ws_mgr.start()
            st.success("WebSocket iniciado!")
            st.rerun()
    with c2:
        if st.button("⏹️ Parar WS", disabled=not ws_running, key="btn_stop_ws"):
            ws_mgr.stop()
            st.warning("WebSocket parado.")
            st.rerun()

    # Bootstrap: carrega histórico + account em 1 batch REST
    _sb_banned, _sb_rem = _is_banned()
    if _sb_banned:
        ban_exp = datetime.fromtimestamp(
            st.session_state.get('ban_expires_at', 0)).strftime('%H:%M:%S')
        st.error(f"🚫 IP banido até {ban_exp} ({int(_sb_rem//60)}m{int(_sb_rem%60)}s) — Bootstrap bloqueado")
    else:
        boot_label = (
            "⏳ Re-Bootstrap (atualiza candles)" if boot_done
            else "⚡ Bootstrap (carrega histórico + conta)"
        )
        n_rest = len(selected_symbols) * len(_INTERVALS_WS) + 2
        if st.button(boot_label, type="primary", key="btn_bootstrap",
                     help=f"Faz {n_rest} chamadas REST de uma única vez para popular buffers kline + "
                          "balance/positions. Depois disso, ZERO REST calls durante operação."):
            _touch_rest_rate()
            with st.spinner("Bootstrapping klines + account..."):
                try:
                    n_klines = ws_mgr.bootstrap_klines(selected_symbols)
                    acct_ok  = ws_mgr.bootstrap_account()
                    # subscreve streams de klines + book ticker para todos os TFs
                    if ws_mgr.running:
                        for _sym in selected_symbols:
                            ws_mgr.subscribe_klines_multi(_sym, _INTERVALS_WS)
                            ws_mgr.subscribe_book_ticker(_sym)
                    st.success(
                        f"✅ Bootstrap OK! {n_klines} candles | "
                        f"Account: {'OK' if acct_ok else 'FALHA'}"
                    )
                    logger.info(f"[BOOTSTRAP] {n_klines} klines + account OK")
                    st.rerun()
                except Exception as _bt_exc:
                    _register_ban(str(_bt_exc), 'BOOTSTRAP')
                    st.error(f"❌ Bootstrap erro: {_bt_exc}")

    # Buffer stats (mostra candles disponíveis por símbolo/TF)
    if boot_done:
        stats_lines = []
        for sym, ivs in ws_mgr.buffer_stats().items():
            for iv, cnt in ivs.items():
                icon = "🟩" if cnt >= 50 else "🟨"
                stats_lines.append(f"{icon} {sym}/{iv}: {cnt}")
        if stats_lines:
            st.caption("Buffers: " + " | ".join(stats_lines))

    if ws_mgr.user_data.get('last_update'):
        age = (datetime.now() - ws_mgr.user_data['last_update']).total_seconds()
        st.caption(f"👉 Account atualizado há {age:.0f}s")

    st.divider()
    
    # Risk Management Status
    risk_mgr = load_risk_manager()
    trailing_mgr = load_trailing_stop_manager()
    warmup_mgr = load_warmup_manager()
    schedule_mgr = load_schedule_manager()
    
    st.subheader("🛡️ Risk Management")
    
    # Circuit Breaker Status
    can_trade, reason = risk_mgr.should_allow_trade()
    if can_trade:
        st.success("✅ Trading Ativo")
    else:
        st.error(f"⛔ {reason}")
        if st.button("🔄 Reset Circuit Breaker"):
            risk_mgr.reset_circuit_breaker()
            st.success("Circuit breaker resetado!")
            st.rerun()
    
    # Trailing Stops Ativos
    active_trails = len(trailing_mgr.active_stops)
    if active_trails > 0:
        st.info(f"🎯 {active_trails} trailing stops ativos")
    
    # Trading Stats
    stats = risk_mgr.get_trading_stats()
    if stats['total_trades'] > 0:
        st.text(f"Trades: {stats['total_trades']}")
        st.text(f"Win Rate: {stats['win_rate']*100:.1f}%")
        st.text(f"Losses: {stats['consecutive_losses']}")
    
    st.divider()
    
    # 🆕 ADVANCED METRICS
    st.subheader("📊 Métricas Avançadas")
    show_correlation = st.checkbox("Correlation Matrix", value=False,
                                   help="⚠️ Habilitar faz N chamadas REST (1 por par) — só ative quando necessário")
    show_regime = st.checkbox("Market Regime", value=False,
                              help="⚠️ Habilitar faz N chamadas REST (1 por par) — só ative quando necessário")
    show_multi_tf = st.checkbox("Multi-Timeframe", value=False)
    
    st.divider()
    
    # Modelo ativo
    st.subheader("🤖 Modelo Ativo")
    st.info("🤖 **LSTM V17.7** (RecurrentPPO 600k)\n\nObs: (50, 31) | 15m+1h+4h multi-TF")
    st.divider()

    # Auto-refresh — seguro quando bootstrapped pois não faz REST calls
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh",
        value=False,
        help="Quando WebSocket bootstrapped, reruns são seguros (zero REST). "
             "Sem bootstrap, ainda pode causar REST calls e ban."
    )
    refresh_interval = st.slider(
        "Intervalo (s)", 15, 300, 30,
        help="🟢 Com WS bootstrapped: 15-30s é seguro.\n"
             "🔴 Sem WS: use mínimo 60s para evitar ban."
    )
    st.caption("💡 Com WebSocket bootstrapped o refresh não faz chamadas REST.")

    if st.button("🔄 Atualizar Agora"):
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN DATA LOAD — WebSocket-first, REST apenas como último recurso
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
_wsmgr: BinanceWebSocketManager = st.session_state['ws_manager']
_boot_done  = _wsmgr.bootstrap_done              # buffers klines populados
_ws_acct_ok = _wsmgr.user_data.get('last_update') is not None  # account bootstrapped
_banned_main, _ban_rem_main = _is_banned()

balance:          dict = {'total': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0, 'source': 'offline', 'error': None}
positions_result: dict = {'positions': [], 'source': 'offline', 'error': None}
data_source = "❓ Offline"

try:
    if _banned_main:
        ban_exp = datetime.fromtimestamp(st.session_state.get('ban_expires_at', 0)).strftime('%H:%M:%S')
        st.error(f"🚫 **IP BANIDO** até {ban_exp} — {int(_ban_rem_main//60)}m{int(_ban_rem_main%60)}s restantes. "
                 "REST desabilitado automaticamente.")
        data_source = "🚫 BANIDO"

    elif _ws_acct_ok:
        # WS tem dados de conta: usa sem qualquer chamada REST
        ws_bal = _wsmgr.get_balance()
        ws_pos = _wsmgr.get_positions()
        if ws_bal:
            balance = {**ws_bal, 'error': None}
        if ws_pos:
            positions_result = {**ws_pos, 'error': None}
        data_source = "🟢 WebSocket (0 REST calls)"

    elif st.session_state.get('_rest_connected', False) or st.session_state.get('bot_running', False):
        # REST autorizado pelo usuário
        _touch_rest_rate()
        balance          = get_account_balance_cached(client)
        positions_result = get_open_positions_cached(client)
        data_source = "🟡 REST API"

    else:
        # Modo seguro: sem WS bootstrapped e REST não autorizado
        st.info(
            "⚪ **Dashboard em modo seguro** — WebSocket não bootstrapped e REST desconectado.\n"
            "Clique **▶️ Iniciar WS** e depois **⚡ Bootstrap** na sidebar para carregar dados."
        )
        data_source = "⚪ Offline (seguro)"

    if balance.get('error'):
        st.error("⚠️ Erro ao obter balance — veja logs")
        balance = {'total': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0, 'error': None, 'source': 'error'}

except Exception as e:
    st.error(f"❌ Erro ao obter dados da conta: {e}")
    balance = {'total': 0.0, 'available': 0.0, 'unrealized_pnl': 0.0, 'error': None, 'source': 'error'}
    positions_result = {'positions': [], 'source': 'error', 'error': None}

positions = positions_result.get('positions', [])

# Barra de status global
_st1, _st2, _st3, _st4 = st.columns(4)
with _st1: st.caption(f"📡 Fonte: {data_source}")
with _st2: st.caption(f"💰 Balance: ${balance['total']:,.2f}")
with _st3: st.caption(f"📈 Posições: {len(positions)}")
with _st4:
    buf_info = " | ".join(
        f"{s}/{i}:{c}"
        for s, ivs in _wsmgr.buffer_stats().items()
        for i, c in ivs.items()
    ) if _boot_done else "sem buffer"
    st.caption(f"📦 {buf_info}")

# Tabs principais
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 Posições", "📈 Performance", "🔬 Análise Avançada", "🔍 Logs"])

with tab1:
    # Métricas principais (USA DADOS JÁ CARREGADOS ACIMA)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Balance Total",
            value=f"${balance['total']:,.2f}",
            delta=f"{balance['unrealized_pnl']:+.2f} USDT"
        )
    
    with col2:
        st.metric(
            label="💵 Disponível",
            value=f"${balance['available']:,.2f}"
        )
    
    with col3:
        st.metric(
            label="📊 Posições Abertas",
            value=len(positions)
        )
    
    with col4:
        # Calcula exposure total
        total_exposure = sum([abs(float(p['positionAmt']) * float(p['entryPrice'])) for p in positions])
        exposure_pct = (total_exposure / balance['total'] * 100) if balance['total'] > 0 else 0
        st.metric(
            label="📈 Exposure Total",
            value=f"{exposure_pct:.1f}%"
        )
    
    st.divider()
    
    # 🆕 SUB-ABAS POR SÍMBOLO
    st.subheader("📈 Gráficos Multi-Par")
    
    # Cria sub-abas para cada símbolo selecionado
    symbol_tabs = st.tabs([f"📊 {sym}" for sym in selected_symbols])
    
    for idx, trade_symbol in enumerate(selected_symbols):
        with symbol_tabs[idx]:
            symbol_binance = trade_symbol.replace('/', '')
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                timeframe_map = {
                    '1m': Client.KLINE_INTERVAL_1MINUTE,
                    '5m': Client.KLINE_INTERVAL_5MINUTE,
                    '15m': Client.KLINE_INTERVAL_15MINUTE,
                    '1h': Client.KLINE_INTERVAL_1HOUR,
                    '4h': Client.KLINE_INTERVAL_4HOUR
                }
                
                selected_tf = st.selectbox("Timeframe", list(timeframe_map.keys()), index=2, key=f"tf_{symbol_binance}")
                candles_limit = st.slider("Candles", 50, 500, 100, key=f"candles_{symbol_binance}")
            
            with col1:
                # 🔴 ANTI-BAN: delay entre símbolos para evitar burst de chamadas REST
                if idx > 0:
                    time.sleep(0.35)  # 350ms entre símbolos = máx ~3 calls/segundo

                # Reusa collect_market_data (mesma cache de Tab4 correlação/regime)
                # evita a duplicação: get_klines(limit=100) ≠ collect_market_data(limit=200)
                # que gerava 2 REST calls distintos para o mesmo símbolo/interval.
                df_chart_full = collect_market_data(client, symbol=symbol_binance,
                                                    interval=timeframe_map[selected_tf], limit=200)
                if df_chart_full is not None:
                    # collect_market_data retorna 'timestamp' como ms int; plot_candlestick precisa de datetime
                    if not pd.api.types.is_datetime64_any_dtype(df_chart_full['timestamp']):
                        df_chart_full = df_chart_full.copy()
                        df_chart_full['timestamp'] = pd.to_datetime(df_chart_full['timestamp'], unit='ms')
                    # Fatiado para os últimos N candles escolhidos pelo usuário
                    df = df_chart_full.tail(candles_limit).reset_index(drop=True)
                else:
                    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 🔴 DESABILITADO: Busca de trades históricos causa múltiplas REST calls
                # Em loop de 4 pares = 4 calls = contribui para ban
                # trades_history = get_recent_trades(client, symbol=symbol_binance, limit=20)
                trades_history = []  # Desabilitado temporáriamente
                
                # 🆕 Marcações de entradas/P&L desabilitadas (requer trades_history)
                # Busca posições abertas deste símbolo
                symbol_positions = [p for p in positions if p['symbol'] == symbol_binance]

                if df is None or df.empty:
                    st.info("⚪ Sem dados de candles. Clique **🔌 Conectar REST API** na sidebar para carregar o gráfico.")
                else:
                    # Plota gráfico SEM marcações de trades fechados (para reduzir calls)
                    fig = plot_candlestick(df, symbol=trade_symbol)
                    
                    # Adiciona marcações de posições abertas (não requer REST calls)
                    for pos in symbol_positions:
                        entry_price = float(pos['entryPrice'])
                        qty = float(pos['positionAmt'])
                        mark_price = float(pos['markPrice'])
                        pnl = float(pos['unRealizedProfit'])
                        
                        # Linha horizontal no preço de entrada
                        color = 'green' if qty > 0 else 'red'
                        fig.add_hline(
                            y=entry_price, 
                            line_dash="dash", 
                            line_color=color,
                            annotation_text=f"{'LONG' if qty > 0 else 'SHORT'} @ ${entry_price:,.2f} | P&L: ${pnl:,.2f}",
                            annotation_position="right"
                        )
                        
                        # Marca ponto de entrada (usa timestamp estimado)
                        if len(df) > 0 and 'timestamp' in df.columns:
                            fig.add_scatter(
                                x=[df['timestamp'].iloc[-1]],  # Última vela como referência
                                y=[entry_price],
                                mode='markers',
                                marker=dict(size=15, color=color, symbol='triangle-up' if qty > 0 else 'triangle-down'),
                                name=f"Entry {'LONG' if qty > 0 else 'SHORT'}",
                                showlegend=True
                            )
                    
                    st.plotly_chart(fig, width='stretch', key=f"chart_{symbol_binance}")

with tab2:
    st.subheader("💼 Posições Abertas")
    
    # 🆕 Carrega trailing manager
    trailing_mgr = load_trailing_stop_manager()
    
    if len(positions) == 0:
        st.info("📭 Nenhuma posição aberta no momento")
    else:
        # ─── 🚨 BOTÃO EMERGENCY CLOSE ALL ───────────────────────
        total_pnl_all = sum(float(p['unRealizedProfit']) for p in positions)
        pnl_color = "positive" if total_pnl_all >= 0 else "negative"
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(
                f'<span style="font-size:1.1rem">⚠️ {len(positions)} posição(ões) abertas | '
                f'P&L total não realizado: <span class="{pnl_color}"><b>${total_pnl_all:+,.2f}</b></span></span>',
                unsafe_allow_html=True
            )
        with c2:
            if st.button("🚨 Fechar TODAS", type="primary", key="close_all_btn",
                         help="Fecha imediatamente TODAS as posições via Ordem MARKET reduceOnly"):
                with st.spinner("Fechando todas as posições..."):
                    results = close_all_positions(client)
                success_count = sum(1 for r in results if r['order'])
                fail_count = len(results) - success_count
                if success_count:
                    st.success(f"✅ {success_count} posição(ões) fechada(s)")
                if fail_count:
                    st.error(f"❌ {fail_count} posição(ões) falharam — verifique os Logs")
                st.rerun()
        with c3:
            st.empty()

        st.divider()

        # Carrega Risk Manager
        risk_mgr = load_risk_manager()

        # ── Auto-registro de posições no trailing stop manager ────────────────
        # Garante que posições abertas antes do bot iniciar apareçam no trailing
        trailing_mgr = load_trailing_stop_manager()
        active_symbols = {p['symbol'] for p in positions}
        for pos_reg in positions:
            sym_reg = pos_reg['symbol']
            if not trailing_mgr.get_stop_info(sym_reg):                 # não registrada ainda
                entry_reg = float(pos_reg['entryPrice'])
                qty_reg   = float(pos_reg['positionAmt'])
                side_reg  = 'LONG' if qty_reg > 0 else 'SHORT'
                trailing_mgr.register_position(sym_reg, entry_reg, side_reg)
                logger.info(f"[TRAILING] Auto-registrado: {sym_reg} {side_reg} @ ${entry_reg:,.2f}")

        # ── Limpa flags tp1_partial de posições que já não existem ────────────
        stale_flags = [k for k in st.session_state
                       if k.startswith("tp1_partial_") and k[len("tp1_partial_"):] not in active_symbols]
        for flag in stale_flags:
            del st.session_state[flag]

        for pos in positions:
            symbol = pos['symbol']
            qty = float(pos['positionAmt'])
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            pnl_pct = (unrealized_pnl / (entry_price * abs(qty))) * 100 if qty != 0 else 0
            
            side = "LONG 🟢" if qty > 0 else "SHORT 🔴"
            position_type = 1 if qty > 0 else -1
            
            # Calcula Stop Loss e Take Profit
            # Para demo, usa ATR fictício (idealmente viria dos dados reais)
            atr_estimate = mark_price * 0.02  # ~2% do preço como ATR estimado
            
            stop_price = risk_mgr.calculate_atr_stop_loss(entry_price, atr_estimate, position_type)
            should_stop = risk_mgr.should_stop_loss(entry_price, mark_price, position_type, atr=atr_estimate)
            
            should_tp, tp_level = risk_mgr.should_take_profit(entry_price, mark_price, position_type, return_level=True)
            
            # 🆕 Verifica trailing stop
            trailing_info = trailing_mgr.get_stop_info(symbol)

            # ── AUTO TP/SL ENFORCEMENT ────────────────────────────────────────────
            # SL: fecha posição inteira imediatamente
            if should_stop:
                st.error(f"🛑 **STOP LOSS ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}% | Mark: ${mark_price:,.2f} ≤ SL: ${stop_price:,.2f}")
                st.warning("⏳ Executando fechamento automático por Stop Loss...")
                order = close_position_direct(client, symbol, qty)
                if order:
                    st.success(f"✅ Stop Loss executado: {symbol} fechado!")
                    logger.warning(f"[AUTO-SL] {symbol} fechado por stop loss @ ${mark_price:,.2f} (entry ${entry_price:,.2f}, pnl {pnl_pct:+.2f}%)")
                    # Remove trailing stop se ativo
                    trailing_mgr.remove_position(symbol)
                    st.rerun()
                else:
                    st.error(f"❌ FALHA AO EXECUTAR STOP LOSS em {symbol} — feche MANUALMENTE!")

            # TP L2 (+4%): fecha 100%
            elif should_tp and tp_level == 2:
                st.success(f"🎯 **TAKE PROFIT L2 (100%) ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}%")
                st.info("⏳ Executando fechamento automático por TP L2...")
                order = close_position_direct(client, symbol, qty)
                if order:
                    st.success(f"✅ Take Profit L2 executado: {symbol} fechado (100%)!")
                    logger.info(f"[AUTO-TP2] {symbol} fechado por TP L2 @ ${mark_price:,.2f} (entry ${entry_price:,.2f}, pnl {pnl_pct:+.2f}%)")
                    trailing_mgr.remove_position(symbol)
                    # Limpa flag de TP1 parcial se existia
                    tp1_flag = f"tp1_partial_{symbol}"
                    if tp1_flag in st.session_state:
                        del st.session_state[tp1_flag]
                    st.rerun()
                else:
                    st.error(f"❌ FALHA AO EXECUTAR TP L2 em {symbol} — feche MANUALMENTE!")

            # TP L1 (+2%): fecha 50% (partial close com proteção contra loop infinito)
            elif should_tp and tp_level == 1:
                tp1_flag = f"tp1_partial_{symbol}"
                if tp1_flag not in st.session_state:
                    st.success(f"🎯 **TAKE PROFIT L1 (50%) ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}%")
                    st.info("⏳ Executando fechamento parcial (50%) por TP L1...")
                    partial_qty = qty / 2  # 50% da posição atual
                    order = close_position_direct(client, symbol, partial_qty)
                    if order:
                        st.success(f"✅ Take Profit L1 executado: {symbol} — 50% parcialmente fechado!")
                        logger.info(f"[AUTO-TP1] {symbol} 50% fechado por TP L1 @ ${mark_price:,.2f} (entry ${entry_price:,.2f}, pnl {pnl_pct:+.2f}%)")
                        st.session_state[tp1_flag] = True  # Marca para não re-disparar na próxima render
                        st.rerun()
                    else:
                        st.error(f"❌ FALHA AO EXECUTAR TP L1 em {symbol} — feche MANUALMENTE!")
                else:
                    # TP1 já foi parcialmente fechado — aguarda TP2 ou SL
                    st.warning(f"⚠️ **TP L1 PARCIAL JÁ EXECUTADO: {symbol}** — aguardando TP L2 (+4%) ou Stop Loss")

            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{symbol}**")
                    st.text(side)
                
                with col2:
                    st.text(f"Qty: {abs(qty):.4f}")
                    st.text(f"Entry: ${entry_price:,.2f}")
                
                with col3:
                    st.text(f"Mark: ${mark_price:,.2f}")
                    leverage = pos.get('leverage', config.get('environment', {}).get('leverage', 3))
                    st.text(f"Leverage: {leverage}x")
                
                with col4:
                    # 🆕 TRAILING STOP INFO
                    if trailing_info and trailing_info.get('activated'):
                        trailing_stop = trailing_info['stop_price']
                        st.success(f"🎯 Trail: ${trailing_stop:,.0f}")
                        distance = abs(mark_price - trailing_stop) / mark_price * 100
                        st.text(f"Dist: {distance:.1f}%")
                    else:
                        # Stop Loss estático
                        stop_color = "🔴" if should_stop else "🟢"
                        st.text(f"{stop_color} SL: ${stop_price:,.0f}")
                        
                        if tp_level > 0:
                            st.text(f"✅ TP L{tp_level}")
                        else:
                            tp_target_1 = entry_price * (1.02 if qty > 0 else 0.98)
                            st.text(f"🎯 TP1: ${tp_target_1:,.0f}")
                
                with col5:
                    pnl_class = "positive" if unrealized_pnl >= 0 else "negative"
                    st.markdown(f'<p class="{pnl_class}">P&L: ${unrealized_pnl:,.2f}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="{pnl_class}">({pnl_pct:+.2f}%)</p>', unsafe_allow_html=True)

                with col6:
                    # ─── FECHAR POSIÇÃO INDIVIDUAL ───
                    if st.button("❌ Fechar", key=f"close_{symbol}",
                                 help=f"Fecha a posição {side} de {symbol} via Ordem MARKET reduceOnly"):
                        with st.spinner(f"Fechando {symbol}..."):
                            order = close_position_direct(client, symbol, qty)
                        if order:
                            st.success(f"✅ {symbol} fechado(a)!")
                        else:
                            st.error(f"❌ Falha ao fechar {symbol} — ver Logs")
                        st.rerun()
                
                # 🆕 Info adicional do trailing stop
                if trailing_info:
                    if position_type == 1:  # LONG
                        highest = trailing_info.get('highest_mark', entry_price)
                        st.text(f"📈 Max: ${highest:,.2f}")
                    else:  # SHORT
                        lowest = trailing_info.get('lowest_mark', entry_price)
                        st.text(f"📉 Min: ${lowest:,.2f}")
                    
                    if trailing_info.get('activated'):
                        st.success("🟢 Trailing ATIVO")
                    else:
                        activation_pct = config.get('risk_management', {}).get('trailing_stop_activation', 0.03) * 100
                        st.info(f"⏳ Ativa em +{activation_pct:.0f}%")
                
                st.divider()

with tab3:
    st.subheader("📊 Performance de Trades (Multi-Par)")
    
    # 🔴 DESABILITADO: Busca trades de múltiplos pares causa BAN!
    # get_recent_trades() faz loop de REST calls: 1 call por símbolo
    # 4 pares = 4 calls instantâneas = BAN!
    
    st.warning("""
    ⚠️ **Performance Multi-Par Temporariamente Desabilitada**
    
    **Motivo**: Buscar trades de 4 pares faz 4 chamadas REST instantâneas → Ban!
    
    **Alternativas**:
    1. 📊 Veja performance individual na aba **🔬 Análise Avançada**
    2. 💰 Veja P&L em tempo real na aba **💰 Posições**
    3. 📄 Veja backtest reports nos arquivos `.txt` da pasta
    
    💡 Em breve: Performance agregada via cache local (v18)
    """)
    
    # Mostra apenas par primário para evitar múltiplas calls
    primary_symbol = config['data']['primary_symbol'].replace('/', '')
    st.info(f"👁️ Exibindo apenas: **{primary_symbol}**")

    # 🔴 ANTI-BAN: Trades carregados sob demanda — NÃO executam no startup automático
    # Isso elimina 1 chamada REST (futures_account_trades) que disparava junto com as outras no boot
    if st.button("📥 Carregar Histórico de Trades", key="load_trades_btn"):
        st.session_state['_trades_loaded_tab3'] = True

    if not st.session_state.get('_trades_loaded_tab3', False):
        st.info("💡 Clique no botão acima para carregar o histórico de trades. "
                "Desabilitado no startup por padrão para evitar ban da API.")
        trades = []
    else:
        trades = get_recent_trades(client, symbol=primary_symbol, limit=100)

    if not trades and st.session_state.get('_trades_loaded_tab3', False):
        st.info("📭 Nenhum trade executado ainda")
    elif trades:
        st.divider()
        
        # Preparar DataFrame de trades
        df_trades = pd.DataFrame(trades)
        df_trades['realizedPnl'] = df_trades['realizedPnl'].astype(float)
        
        # Filtra apenas FECHAMENTOS (P&L realizado != 0)
        # Aberturas de posição sempre têm P&L = 0 e não são relevantes para análise de performance
        df_trades_closed = df_trades[df_trades['realizedPnl'] != 0].copy()
        
        if len(df_trades_closed) == 0:
            st.info("📭 Nenhum fechamento de posição ainda (apenas aberturas)")
        else:
            st.info(f"📊 Exibindo apenas **fechamentos** com P&L realizado ({len(df_trades_closed)} de {len(df_trades)} trades)")
            
            # P&L Chart (apenas fechamentos)
            pnl_chart = plot_pnl_chart(df_trades_closed.to_dict('records'))
            if pnl_chart:
                st.plotly_chart(pnl_chart, width='stretch')
            
            st.divider()
        
        # Estatísticas avançadas (usando apenas fechamentos)
        metrics = calculate_performance_metrics(df_trades_closed.to_dict('records')) if len(df_trades_closed) > 0 else None
        
        if metrics:
            st.subheader("📈 Performance Metrics")
            
            # Linha 1: Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics['total_trades'])
            
            with col2:
                win_rate_pct = metrics['win_rate'] * 100
                st.metric("Win Rate", f"{win_rate_pct:.1f}%", 
                         delta="✅" if win_rate_pct >= 50 else "⚠️")
            
            with col3:
                st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}",
                         delta=f"${metrics['total_pnl']:+,.2f}")
            
            with col4:
                sharpe = metrics['sharpe_ratio']
                sharpe_color = "✅" if sharpe > 1.5 else ("⚠️" if sharpe > 0.5 else "❌")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=sharpe_color)
            
            st.divider()
            
            # Linha 2: Métricas de risco
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Win", f"${metrics['avg_win']:,.2f}")
            
            with col2:
                st.metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")
            
            with col3:
                pf = metrics['profit_factor']
                pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
                pf_color = "✅" if pf > 1.5 else ("⚠️" if pf > 1.0 else "❌")
                st.metric("Profit Factor", pf_str, delta=pf_color)
            
            with col4:
                st.metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")
            
            st.divider()
            
            # Linha 3: Métricas avançadas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Wins", metrics['wins'], delta=f"+{metrics['wins']}")
            
            with col2:
                st.metric("Losses", metrics['losses'], delta=f"-{metrics['losses']}")
            
            with col3:
                rf = metrics['recovery_factor']
                rf_str = f"{rf:.2f}" if rf != float('inf') else "∞"
                st.metric("Recovery Factor", rf_str)
            
            with col4:
                st.metric("Expectancy", f"${metrics['expectancy']:,.2f}")
        else:
            # Estatísticas básicas (fallback para poucos trades)
            if len(df_trades_closed) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = len(df_trades_closed)
                    st.metric("Total Trades", total_trades)
                
                with col2:
                    wins = len(df_trades_closed[df_trades_closed['realizedPnl'] > 0])
                    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col3:
                    total_pnl = df_trades_closed['realizedPnl'].sum()
                    st.metric("Total P&L", f"${total_pnl:,.2f}")
                
                with col4:
                    avg_pnl = df_trades_closed['realizedPnl'].mean()
                    st.metric("Avg P&L/Trade", f"${avg_pnl:,.2f}")
        
        if len(df_trades_closed) > 0:
            st.divider()
            
            # Tabela de trades (apenas fechamentos)
            st.subheader("🗂️ Histórico de Trades (Fechamentos)")
            df_display = df_trades_closed[['time', 'symbol', 'side', 'qty', 'price', 'realizedPnl']].copy()
            df_display['time'] = pd.to_datetime(df_display['time'], unit='ms')
            df_display['realizedPnl'] = df_display['realizedPnl'].astype(float)
            
            st.dataframe(df_display, width='stretch')

with tab4:
    st.header("🔬 Análise Avançada Multi-Par")
    
    # ─── 🚨 PAINEL DE GESTÃO DE POSIÇÕES LEGADAS ──────────────────────────────
    # USA DADOS JÁ CARREGADOS (positions) ao invés de fazer nova chamada REST
    all_positions = positions  # Reutiliza dados já carregados!
    active_symbols_binance = [s.replace('/', '') for s in selected_symbols]
    legacy_positions = [p for p in all_positions if p['symbol'] not in active_symbols_binance]

    if all_positions:
        with st.expander(
            f"🚨 Gestão de Posições ({len(all_positions)} aberta(s) · "
            f"{len(legacy_positions)} legada(s) fora dos símbolos ativos)",
            expanded=bool(legacy_positions)   # abre automaticamente se houver legadas
        ):
            # Botão FECHAR TODAS no topo do painel
            ca1, ca2 = st.columns([3, 1])
            with ca2:
                if st.button("🚨 Fechar TODAS", type="primary", key="close_all_adv",
                             help="Fecha todas as posições abertas (incluindo legadas)"):
                    with st.spinner("Fechando todas as posições..."):
                        results = close_all_positions(client)
                    ok = sum(1 for r in results if r['order'])
                    st.success(f"✅ {ok}/{len(results)} posição(ões) fechada(s)")
                    st.rerun()
            with ca1:
                st.caption(
                    "⚠️ Posições **legadas** são de modelos anteriores (SAC V6/V15/TD3) e **não serão fechadas automaticamente** pelo LSTM V17.7. "
                    "Use os botões abaixo para fechá-las manualmente."
                )

            st.markdown("---")

            for pos in all_positions:
                sym = pos['symbol']
                qty = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                mark = float(pos['markPrice'])
                pnl = float(pos['unRealizedProfit'])
                side_label = "🟢 LONG" if qty > 0 else "🔴 SHORT"
                is_legacy = sym not in active_symbols_binance
                legacy_badge = " 🏷️ LEGADA" if is_legacy else ""

                pc1, pc2, pc3, pc4 = st.columns([2, 2, 2, 1])
                with pc1:
                    st.markdown(f"**{sym}**{legacy_badge}")
                    st.text(side_label)
                with pc2:
                    st.text(f"Qty:   {abs(qty):.4f}")
                    st.text(f"Entry: ${entry:,.2f}")
                with pc3:
                    pnl_class = "positive" if pnl >= 0 else "negative"
                    st.text(f"Mark:  ${mark:,.2f}")
                    st.markdown(f'<span class="{pnl_class}">P&L: ${pnl:+,.2f}</span>', unsafe_allow_html=True)
                with pc4:
                    if st.button("❌ Fechar", key=f"close_adv_{sym}",
                                 help=f"Fecha posição {side_label} de {sym}"):
                        with st.spinner(f"Fechando {sym}..."):
                            order = close_position_direct(client, sym, qty)
                        if order:
                            st.success(f"✅ {sym} fechado!")
                        else:
                            st.error(f"❌ Falha — ver Logs")
                        st.rerun()

        st.divider()

    # Carrega managers
    trailing_mgr = load_trailing_stop_manager()
    warmup_mgr = load_warmup_manager()
    schedule_mgr = load_schedule_manager()
    
    # 0. WARMUP STATUS
    st.subheader("⏳ Status de Warm-up")
    
    warmup_cols = st.columns(len(selected_symbols))
    
    for idx, symbol in enumerate(selected_symbols):
        with warmup_cols[idx]:
            current, required, pct = warmup_mgr.get_progress(symbol)
            is_ready = warmup_mgr.is_ready(symbol)
            
            if is_ready:
                st.success(f"✅ **{symbol}**")
                st.success("PRONTO")
            else:
                st.warning(f"⏳ **{symbol}**")
                st.progress(pct / 100)
                st.text(f"{current}/{required} candles")
    
    st.divider()
    
    # 0.5 SCHEDULE STATUS
    st.subheader("📅 Schedule de Execução")
    
    schedule_cols = st.columns(len(selected_symbols))
    
    for idx, symbol in enumerate(selected_symbols):
        with schedule_cols[idx]:
            can_trade, reason = schedule_mgr.can_trade_now(symbol)
            next_exec = schedule_mgr.get_next_execution(symbol)
            
            st.markdown(f"**{symbol}**")
            
            if can_trade:
                st.success("🟢 PODE OPERAR")
            else:
                st.warning(f"🔴 {reason}")
            
            if next_exec:
                st.text(f"Próximo: {next_exec.strftime('%H:%M')}")
            
            # Mostra minutos permitidos
            if symbol in schedule_mgr.schedule:
                minutes = schedule_mgr.schedule[symbol]
                st.text(f"Slots: {minutes}")
    
    st.divider()
    
    # 1. CORRELATION MATRIX
    if show_correlation and len(selected_symbols) > 1:
        st.subheader("📊 Matriz de Correlação")
        
        with st.spinner("Calculando correlações..."):
            # Coleta dados de todos os pares
            correlation_data = {}
            for symbol in selected_symbols:
                df = collect_market_data(client, symbol=symbol, interval='15m', limit=200)
                if df is not None:
                    correlation_data[symbol] = df
            
            # Calcula matriz de correlação
            if len(correlation_data) >= 2:
                correlation_matrix = pd.DataFrame(index=selected_symbols, columns=selected_symbols)
                
                for sym1 in selected_symbols:
                    for sym2 in selected_symbols:
                        if sym1 == sym2:
                            correlation_matrix.loc[sym1, sym2] = 1.0
                        elif sym1 in correlation_data and sym2 in correlation_data:
                            corr = calculate_correlation(correlation_data[sym1], correlation_data[sym2])
                            correlation_matrix.loc[sym1, sym2] = corr
                        else:
                            correlation_matrix.loc[sym1, sym2] = 0.0
                
                # Heatmap de correlação
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values.astype(float),
                    x=selected_symbols,
                    y=selected_symbols,
                    colorscale='RdYlGn_r',
                    zmid=0,
                    text=correlation_matrix.values.astype(float).round(2),
                    texttemplate='%{text}',
                    textfont={"size": 14},
                    colorbar=dict(title="Correlação")
                ))
                
                fig.update_layout(
                    title="Correlação entre Pares (50 períodos)",
                    xaxis_title="Par",
                    yaxis_title="Par",
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Alertas de correlação
                risk_config = config.get('risk_management', {})
                corr_threshold = risk_config.get('correlation_threshold', 0.70)
                
                high_correlations = []
                for i, sym1 in enumerate(selected_symbols):
                    for j, sym2 in enumerate(selected_symbols):
                        if i < j:  # Apenas metade superior da matriz
                            corr_value = float(correlation_matrix.loc[sym1, sym2])
                            if abs(corr_value) > corr_threshold:
                                high_correlations.append((sym1, sym2, corr_value))
                
                if high_correlations:
                    st.warning(f"⚠️ **Correlações Alta (>{corr_threshold:.0%}):**")
                    for sym1, sym2, corr in high_correlations:
                        st.text(f"   • {sym1} ↔ {sym2}: {corr:+.2f}")
                    st.info("💡 Evite abrir posições simultâneas em pares altamente correlacionados!")
                else:
                    st.success(f"✅ Nenhuma correlação alta detectada (threshold: {corr_threshold:.0%})")
        
        st.divider()
    
    # 2. MARKET REGIME DETECTION
    if show_regime:
        st.subheader("🌡️ Regime de Mercado por Par")
        
        regime_cols = st.columns(len(selected_symbols))
        
        for idx, symbol in enumerate(selected_symbols):
            with regime_cols[idx]:
                df = collect_market_data(client, symbol=symbol, interval='15m', limit=200)
                if df is not None:
                    regime, adx_strength = detect_market_regime(df)
                    
                    # Emoji por regime
                    regime_emoji = {
                        'BULL': '🐂',
                        'BEAR': '🐻',
                        'SIDEWAYS': '➡️',
                        'UNKNOWN': '❓'
                    }
                    
                    # Cor por regime
                    if regime == 'BULL':
                        st.success(f"{regime_emoji[regime]} **{symbol}**")
                        st.success(f"**{regime}**")
                    elif regime == 'BEAR':
                        st.error(f"{regime_emoji[regime]} **{symbol}**")
                        st.error(f"**{regime}**")
                    else:
                        st.info(f"{regime_emoji[regime]} **{symbol}**")
                        st.info(f"**{regime}**")
                    
                    st.text(f"ADX: {adx_strength:.1f}")
                    
                    # Volatilidade (ATR)
                    atr = calculate_atr(df)
                    atr_pct = (atr / df['close'].iloc[-1]) * 100
                    st.text(f"ATR: {atr_pct:.2f}%")
        
        st.divider()
    
    # 3. MULTI-TIMEFRAME ANALYSIS
    if show_multi_tf and len(selected_symbols) > 0:
        st.subheader("⏱️ Análise Multi-Timeframe")
        
        selected_pair = st.selectbox("Selecione o Par", selected_symbols)
        
        multi_tf_data = collect_multi_timeframe_data(client, symbol=selected_pair)
        
        if multi_tf_data:
            tf_cols = st.columns(len(multi_tf_data))
            
            for idx, (tf, df) in enumerate(multi_tf_data.items()):
                with tf_cols[idx]:
                    st.markdown(f"### {tf}")
                    
                    # Regime
                    regime, adx = detect_market_regime(df)
                    regime_color = {
                        'BULL': '🟢',
                        'BEAR': '🔴',
                        'SIDEWAYS': '🟡'
                    }.get(regime, '⚪')
                    
                    st.text(f"{regime_color} {regime}")
                    
                    # RSI
                    rsi = df['RSI_14'].iloc[-1] * 100
                    rsi_emoji = "🔥" if rsi > 70 else ("❄️" if rsi < 30 else "➡️")
                    st.text(f"{rsi_emoji} RSI: {rsi:.1f}")
                    
                    # Preço vs SMAs
                    current_price = df['close'].iloc[-1]
                    sma_20 = df['close'].rolling(20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    
                    if current_price > sma_20 > sma_50:
                        st.success("📈 Acima de SMAs")
                    elif current_price < sma_20 < sma_50:
                        st.error("📉 Abaixo de SMAs")
                    else:
                        st.info("↔️ Entre SMAs")
            
            # Consenso Multi-TF
            st.divider()
            regimes = [detect_market_regime(df)[0] for df in multi_tf_data.values()]
            
            if all(r == 'BULL' for r in regimes):
                st.success("✅ **CONSENSO BULL** em todos os timeframes!")
            elif all(r == 'BEAR' for r in regimes):
                st.error("⚠️ **CONSENSO BEAR** em todos os timeframes!")
            elif all(r == 'SIDEWAYS' for r in regimes):
                st.info("➡️ **CONSENSO SIDEWAYS** em todos os timeframes")
            else:
                st.warning(f"⚠️ **DIVERGÊNCIA**: {', '.join(regimes)}")
        
        st.divider()
    
    # 4. POSITION SIZING SIMULATOR
    st.subheader("🎯 Simulador de Position Sizing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sim_balance = st.number_input("Balance (USDT)", value=10000, step=1000)
        sim_win_streak = st.slider("Win Streak", -5, 5, 0, 
                                   help="Positivo = wins, Negativo = losses")
    
    with col2:
        sim_price = st.number_input("Preço Atual", value=50000, step=1000)
        sim_atr_pct = st.slider("Volatilidade ATR (%)", 0.5, 5.0, 1.5, step=0.1)
    
    # Calcula para diferentes regimes
    st.markdown("**Position Size por Regime:**")
    
    regime_results = {}
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        qty = calculate_position_size_dynamic(
            balance=sim_balance,
            base_size=config['environment']['position_size'],
            volatility_atr=sim_atr_pct/100,
            current_price=sim_price,
            win_streak=sim_win_streak,
            regime=regime,
            confidence=1.0
        )
        
        notional = qty * sim_price
        pct_of_balance = (notional / sim_balance) * 100
        regime_results[regime] = {
            'qty': qty,
            'notional': notional,
            'pct': pct_of_balance
        }
    
    result_cols = st.columns(3)
    
    for idx, (regime, data) in enumerate(regime_results.items()):
        with result_cols[idx]:
            emoji = {'BULL': '🐂', 'BEAR': '🐻', 'SIDEWAYS': '➡️'}[regime]
            st.markdown(f"### {emoji} {regime}")
            st.metric("Quantidade", f"{data['qty']:.4f}")
            st.metric("Notional", f"${data['notional']:,.2f}")
            st.metric("% Balance", f"{data['pct']:.1f}%")

with tab5:
    st.subheader("🤖 Trading Engine — Background Thread")

    engine = get_trading_engine()

    # ── Controles ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 2])

    with ctrl_c1:
        if st.button("▶ Iniciar Engine", type="primary",
                     disabled=engine.running or not st.session_state.get('_rest_connected') and not st.session_state['ws_manager'].bootstrap_done):
            engine.start(selected_symbols)
            st.success("Engine iniciada!")

    with ctrl_c2:
        if st.button("⏹ Parar Engine", type="secondary", disabled=not engine.running):
            engine.stop()
            st.info("Engine parada.")

    with ctrl_c3:
        _eng_status = "🟢 RODANDO" if engine.running else "🔴 PARADO"
        _last_tick  = engine.state.get('last_tick')
        _tick_str   = _last_tick.strftime('%H:%M:%S') if _last_tick else '—'
        st.info(f"Status: {_eng_status}  |  Último candle: {_tick_str}")

    st.divider()

    # ── Símbolos monitorados e estado do buffer ──────────────────────────────
    _eng_symbols = engine.state.get('symbols', [])
    if _eng_symbols:
        st.caption(f"Pares ativos: {', '.join(_eng_symbols)}")
    else:
        st.caption("Nenhum par ativo — clique ▶ Iniciar Engine.")

    # ── Decisões LSTM em tempo real ──────────────────────────────────────────
    _decisions = engine.state.get('decisions', {})
    if _decisions:
        st.subheader("🎯 Decisões LSTM V17.7")
        dec_rows = []
        for _sym, _dec in sorted(_decisions.items()):
            dec_rows.append({
                'Par':      _sym,
                'Ação':     _dec.get('action', '—'),
                'Confiança':f"{_dec.get('value', 0):.3f}",
                'Preço':    f"${_dec.get('price', 0):,.2f}",
                'RSI':      f"{_dec.get('rsi', 0):.1f}",
                'Horário':  _dec['ts'].strftime('%H:%M:%S') if _dec.get('ts') else '—',
            })
        st.dataframe(dec_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Sem decisões ainda — aguardando primeiro candle 15m fechado.")

    st.divider()

    # ── Ordens recentes ──────────────────────────────────────────────────────
    _orders = list(engine.state.get('orders', []))
    if _orders:
        st.subheader("📋 Ordens Recentes")
        st.dataframe(list(reversed(_orders)), use_container_width=True, hide_index=True)

    # ── Log da engine ────────────────────────────────────────────────────────
    st.subheader("📜 Log da Engine")
    _log_lines = list(engine.state.get('log', []))
    if _log_lines:
        st.text_area("Log", "\n".join(reversed(_log_lines)), height=300,
                     key="engine_log_area")
    else:
        st.caption("Log vazio.")

    # ── Erros ────────────────────────────────────────────────────────────────
    _errors = list(engine.state.get('errors', []))
    if _errors:
        st.subheader("⛔ Erros")
        for _err in reversed(_errors):
            st.error(_err)

    st.divider()

    # ── Info do sistema ──────────────────────────────────────────────────────
    st.subheader("ℹ️ Sistema")
    symbols_list = ', '.join(config['data']['symbols'])
    system_info = (
        f"Modo: {mode.upper()}\n"
        f"Symbols: {symbols_list}\n"
        f"Timeframe: {config['data']['timeframes']['tactical']}\n"
        f"Alavancagem: {config['environment']['leverage']}x\n"
        f"Tamanho de Posição: {config['environment']['position_size']*100:.0f}%\n"
        f"Última Atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.code(system_info, language="text")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    current_time = datetime.now().strftime('%H:%M:%S')
    st.text(f"⏰ Atualizado: {current_time}")

with col2:
    st.text(f"🌐 Modo: {mode.upper()}")

with col3:
    st.text(f"📊 Multi-Par Trading @ {config['data']['timeframes']['tactical']}")

# Auto-refresh — seguro quando WS bootstrapped (zero REST calls no rerun)
if auto_refresh:
    _wsmgr_ar: BinanceWebSocketManager = st.session_state['ws_manager']
    _boot_ar   = _wsmgr_ar.bootstrap_done
    _banned_ar, _ban_ar_rem = _is_banned()

    if _banned_ar and not _boot_ar:
        # Banido E sem buffer WS: para completamente para evitar REST
        st.info(f"⏸️ Auto-refresh pausado — IP banido ({int(_ban_ar_rem//60)}m{int(_ban_ar_rem%60)}s). "
                "Bootstrap o WS para continuar monitorando sem REST.")
    else:
        # Com WS bootstrapped: seguro fazer rerun (dados vêm do buffer, não de REST)
        logger.info(f"[AUTO-REFRESH] Aguardando {refresh_interval}s... (WS-bootstrapped={_boot_ar})")
        time.sleep(refresh_interval)
        logger.info("[AUTO-REFRESH] Recarregando dashboard...")
        st.rerun()
