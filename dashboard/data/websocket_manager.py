"""
WebSocket Manager — conexões persistentes com Binance Futures.
Mantém buffers OHLCV em memória, atualizados via kline stream.
Zero chamadas REST durante operação (após bootstrap inicial).

Cache em disco (kline_cache/):
  Na primeira carga busca REST e salva. Nas próximas cargas lê do cache
  e só busca o delta de candles faltantes — bootstrap em segundos.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager

from dashboard.core.config import KLINE_MAXLEN, INTERVALS_WS, KLINE_LIMIT_BOOT
from dashboard.core.logging_setup import get_logger
from dashboard.data.indicators import compute_indicators

logger = get_logger()

_CACHE_DIR = Path('kline_cache')


class BinanceWebSocketManager:
    """
    Gerencia conexões WebSocket persistentes com Binance Futures.

    Mantém um buffer OHLCV em memória por símbolo/intervalo que é:
      - Inicializado via UMA REST call de bootstrap (solicita o usuário clicar)
      - Atualizado em tempo real por kline WebSocket fechado

    Após o bootstrap, ZERO chamadas REST são feitas pela dashboard.
    Reconexão automática em caso de queda do WebSocket.
    """

    # Tempo sem mensagem após o qual o watchdog reconecta (segundos)
    _WATCHDOG_TIMEOUT = 90
    # Intervalo de verificação do watchdog
    _WATCHDOG_INTERVAL = 30
    # Intervalo do refresh REST de conta (balance + positions)
    _ACCOUNT_REFRESH_INTERVAL = 60   # segundos — peso API: ~10/min vs limite 2400
    # Intervalo de keepalive do listen key do User Data Stream
    _LISTEN_KEY_KEEPALIVE_INTERVAL = 25 * 60  # 25 minutos (limite Binance: 60 min)

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
        self.kline_buffers: dict[str, dict[str, deque]] = {}
        self.bootstrap_done: bool = False
        self.bootstrap_symbols: list[str] = []

        # ── Live ticker price ─────────────────────────────────────────────
        self.live_price: dict[str, float] = {}

        # ── Reconnect state ───────────────────────────────────────────────
        self._last_kline_tick: float = 0.0    # epoch seconds of last kline msg
        self._reconnecting: bool = False
        self._last_reconnect_time: float = 0.0  # debounce: impede storm de reconexão
        self._watchdog_thread: threading.Thread | None = None
        # ── IP ban tracking ──────────────────────────────────────────────
        # Populated when Binance returns -1003; REST skipped until then.
        self._banned_until: float = 0.0  # epoch seconds
        # ── Background account refresh (REST periódico) ───────────────────
        self._account_refresh_thread: threading.Thread | None = None
        self._listen_key_thread: threading.Thread | None = None
        self._listen_key: str | None = None          # guardado para keepalive manual

        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.client.API_KEY,
                api_secret=self.client.API_SECRET,
                testnet=True,
            )
            self.twm.start()
            self.twm.start_futures_user_socket(callback=self._handle_user_data)
            self.running = True
            self._last_kline_tick = time.time()
            logger.info("[WS] Iniciado — User Data Stream ativo")
            self._start_watchdog()
            self._start_account_refresh_thread()
            self._start_listen_key_thread()
        except Exception as exc:
            logger.error(f"[WS] Erro ao iniciar: {exc}")
            self.running = False

    def stop(self) -> None:
        """Para todos os streams WebSocket."""
        self.running = False
        if self.twm:
            try:
                self.twm.stop()
            except Exception as exc:
                logger.warning(f"[WS] Erro ao parar: {exc}")
            self.twm = None
        logger.info("[WS] Encerrado")

    def start(self) -> None:
        """Inicia (ou reinicia) o WebSocket. No-op se já estiver ativo."""
        if self.running and self.twm:
            return
        # Garante que running=True antes de reconnect (pode ter sido parado via stop())
        self.running = True
        self._reconnect()
        # Reinicia threads de background que podem ter encerrado por running=False
        self._start_watchdog()
        self._start_account_refresh_thread()
        self._start_listen_key_thread()

    def _start_watchdog(self) -> None:
        """Dispara thread de watchdog que reconecta se não chegar mensagem por _WATCHDOG_TIMEOUT s."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="WS-Watchdog"
        )
        self._watchdog_thread.start()
        logger.debug("[WS] Watchdog iniciado")

    # ─────────────────────────────────────────────────────────────────────
    # BACKGROUND: account refresh + listen key keepalive
    # ─────────────────────────────────────────────────────────────────────

    def _start_account_refresh_thread(self) -> None:
        """Inicia thread de refresh periódico REST para balance + positions.

        Custo: ~10 peso/min vs limite 2400 — completamente seguro.
        Garante que saldo e posições ficam atualizados mesmo se ACCOUNT_UPDATE
        não chegar (listen key expirada, sem ordens executadas, etc).
        """
        if self._account_refresh_thread and self._account_refresh_thread.is_alive():
            return
        self._account_refresh_thread = threading.Thread(
            target=self._account_refresh_loop,
            daemon=True,
            name="AccountRefresh",
        )
        self._account_refresh_thread.start()
        logger.info("[WS] Account refresh thread iniciado (intervalo=60s)")

    def _account_refresh_loop(self) -> None:
        """Loop periódico: aguarda 15s no boot, depois chama REST a cada 60s.
        A cada 30 iterações (~30min) também persiste buffers kline em disco."""
        time.sleep(15)  # aguarda bootstrap inicial antes da primeira chamada
        _cache_save_every = 30  # iterações (30 × 60s = 30min)
        _cache_counter = _cache_save_every - 1  # força save na primeira iteração
        while self.running:
            try:
                self._rest_refresh_account()
            except Exception as exc:
                logger.warning(f"[WS-REFRESH] Erro no loop: {exc}")
            _cache_counter += 1
            if _cache_counter >= _cache_save_every:
                _cache_counter = 0
                try:
                    self.save_kline_cache()
                except Exception as _ce:
                    logger.debug(f"[WS-CACHE] Erro no flush periódico: {_ce}")
            # Aguarda em fatias de 5s para reagir rápido ao self.running = False
            for _ in range(self._ACCOUNT_REFRESH_INTERVAL // 5):
                if not self.running:
                    return
                time.sleep(5)

    def _rest_refresh_account(self) -> None:
        """Chama REST para atualizar balance + positions. Chamado a cada 60s."""
        # ── IP-ban guard: Binance -1003 sets _banned_until; skip until then ──
        _now = time.time()
        if self._banned_until > _now:
            _remaining = int(self._banned_until - _now)
            logger.debug(f"[WS-REFRESH] IP banido — aguardando {_remaining}s")
            return

        # --- Balance ---
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
                    # Atualiza last_update aqui também — assim age_secs reflete
                    # o balance mesmo se a chamada de positions falhar.
                    self.user_data['last_update'] = datetime.now()
                logger.debug(f"[WS-REFRESH] Balance: ${float(usdt['balance']):.2f}")
        except Exception as exc:
            self._handle_ban_error(exc, '[WS-REFRESH] Balance')

        # --- Positions ---
        try:
            positions_raw = self.client.futures_position_information()
            open_pos = [p for p in positions_raw if float(p['positionAmt']) != 0]
            with self.lock:
                self.user_data['positions']   = open_pos
                self.user_data['last_update'] = datetime.now()
            logger.debug(f"[WS-REFRESH] Positions: {len(open_pos)} abertas")
        except Exception as exc:
            self._handle_ban_error(exc, '[WS-REFRESH] Positions')

    def _handle_ban_error(self, exc: Exception, label: str) -> None:
        """Trata erros REST. Para -1003 (IP ban) extrai o timestamp e registra."""
        msg = str(exc)
        logger.warning(f"{label} REST erro: {exc}")
        if '-1003' in msg or 'banned until' in msg.lower():
            # Extrai timestamp de ban da mensagem: '...banned until 1772506552742...'
            import re
            m = re.search(r'banned until (\d+)', msg)
            if m:
                ban_ts_ms = int(m.group(1))
                ban_ts_s  = ban_ts_ms / 1000.0
                self._banned_until = ban_ts_s
                # Expõe no user_data para o dashboard mostrar aviso
                with self.lock:
                    self.user_data['ban_until'] = ban_ts_s
                from datetime import datetime as _dt
                ban_str = _dt.fromtimestamp(ban_ts_s).strftime('%H:%M:%S')
                logger.warning(
                    f"[WS-REFRESH] ⛔ IP BANIDO até {ban_str} "
                    f"— REST pausado por {int(ban_ts_s - time.time())}s"
                )
            else:
                # Sem timestamp explícito: pausa 5 minutos como default
                self._banned_until = time.time() + 300
                with self.lock:
                    self.user_data['ban_until'] = self._banned_until
                logger.warning(f"[WS-REFRESH] ⛔ Rate limit sem timestamp — pausando 5min")

    def _start_listen_key_thread(self) -> None:
        """Inicia thread de keepalive do User Data Stream (escala: a cada 25 min)."""
        if self._listen_key_thread and self._listen_key_thread.is_alive():
            return
        self._listen_key_thread = threading.Thread(
            target=self._listen_key_keepalive_loop,
            daemon=True,
            name="ListenKeyKeepalive",
        )
        self._listen_key_thread.start()
        logger.info("[WS] Listen key keepalive thread iniciado (intervalo=25min)")

    def _listen_key_keepalive_loop(self) -> None:
        """Renova o listen key a cada 25 min para evitar expiração (limite: 60 min)."""
        time.sleep(self._LISTEN_KEY_KEEPALIVE_INTERVAL)
        while self.running:
            try:
                if self._listen_key:
                    self.client.futures_stream_keepalive(self._listen_key)
                    logger.debug(f"[WS] Listen key renovado: {self._listen_key[:8]}...")
                else:
                    # Busca nova listen key se não temos uma salva
                    resp = self.client.futures_stream_get_listen_key()
                    self._listen_key = resp.get('listenKey', '') if isinstance(resp, dict) else str(resp)
                    if self._listen_key:
                        self.client.futures_stream_keepalive(self._listen_key)
                        logger.debug(f"[WS] Listen key obtida e renovada")
            except Exception as exc:
                logger.debug(f"[WS] Listen key keepalive erro: {exc}")
            for _ in range(self._LISTEN_KEY_KEEPALIVE_INTERVAL // 5):
                if not self.running:
                    return
                time.sleep(5)

    def _watchdog_loop(self) -> None:
        """Loop do watchdog — verifica silêncio prolongado e reconecta."""
        while self.running:
            time.sleep(self._WATCHDOG_INTERVAL)
            if not self.running:
                break
            if not self.bootstrap_done:
                continue   # ainda não foi bootstrapped, não há expectativa de mensagens
            silence = time.time() - self._last_kline_tick
            if silence > self._WATCHDOG_TIMEOUT and not self._reconnecting:
                logger.warning(
                    f"[WS] Watchdog: {silence:.0f}s sem mensagem kline — reconectando..."
                )
                self._reconnect()

    def _reconnect(self) -> None:
        """
        Reconecta o WebSocket do zero:
        1. Para o TWM antigo
        2. Cria novo TWM
        3. Re-subscreve user data + todos os streams de kline
        Os buffers de candle em memória são preservados.
        """
        if self._reconnecting:
            return
        self._reconnecting = True
        self._last_reconnect_time = time.time()
        try:
            logger.info("[WS] Iniciando reconexão...")
            # Para o TWM antigo sem alterar self.running
            old_twm = self.twm
            self.twm = None
            if old_twm:
                try:
                    old_twm.stop()
                except Exception as exc:
                    logger.debug(f"[WS] Erro ao parar TWM antigo: {exc}")
            time.sleep(2)   # breve pausa antes de reconectar

            # Cria novo TWM
            self.twm = ThreadedWebsocketManager(
                api_key=self.client.API_KEY,
                api_secret=self.client.API_SECRET,
                testnet=True,
            )
            self.twm.start()
            self.twm.start_futures_user_socket(callback=self._handle_user_data)
            logger.info("[WS-RECONN] User Data Stream restabelecido")

            # Re-subscreve kline streams
            if self.bootstrap_symbols:
                self.subscribe_all_klines(self.bootstrap_symbols)

            self._last_kline_tick = time.time()
            logger.info(f"[WS-RECONN] ✅ Reconexão bem-sucedida ({len(self.bootstrap_symbols)} símbolos)")
        except Exception as exc:
            logger.error(f"[WS-RECONN] ❌ Falha na reconexão: {exc} — nova tentativa em 30s")
        finally:
            self._reconnecting = False

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: bootstrap (única REST call autorizada após início)
    # ─────────────────────────────────────────────────────────────────────

    # ── Cache em disco ────────────────────────────────────────────────────────

    @staticmethod
    def _cache_path(sym: str, interval: str) -> Path:
        _CACHE_DIR.mkdir(exist_ok=True)
        return _CACHE_DIR / f"{sym}_{interval}.pkl"

    @staticmethod
    def _load_cache(sym: str, interval: str) -> list[dict]:
        """Carrega buffer do disco. Retorna [] se não existir ou corrompido."""
        path = BinanceWebSocketManager._cache_path(sym, interval)
        try:
            if path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
        return []

    @staticmethod
    def _save_cache(sym: str, interval: str, candles: list[dict]) -> None:
        try:
            path = BinanceWebSocketManager._cache_path(sym, interval)
            with open(path, 'wb') as f:
                pickle.dump(candles, f)
        except Exception as exc:
            logger.debug(f"[WS-CACHE] Erro ao salvar {sym}/{interval}: {exc}")

    def bootstrap_klines(self, symbols: list[str]) -> int:
        """
        Busca histórico inicial de candles.
        1ª vez: REST completo → salva cache em disco.
        Reinicializações seguintes: carrega do cache (~ms) + busca só o delta.
        Isso evita 57 REST calls no restart → prevenção de IP ban.
        Retorna o total de candles carregados/atualizados.
        """
        total = 0
        for sym in symbols:
            sym = sym.upper()
            self.kline_buffers.setdefault(sym, {})
            for interval in INTERVALS_WS:
                limit = KLINE_LIMIT_BOOT.get(interval, 200)
                # 1. Tenta cache em disco primeiro
                cached = self._load_cache(sym, interval)
                if cached:
                    last_cached_ts = cached[-1]['timestamp']
                    # Converte intervalo para ms para calcular delta
                    ms_per_candle = {'15m': 900_000, '1h': 3_600_000, '4h': 14_400_000}
                    ms = ms_per_candle.get(interval, 900_000)
                    now_ms = int(time.time() * 1000)
                    missing = max(1, (now_ms - last_cached_ts) // ms)
                    if missing <= 3:
                        # Cache está fresco — usa direto, sem REST
                        buf: deque = deque(cached, maxlen=KLINE_MAXLEN)
                        self.kline_buffers[sym][interval] = buf
                        total += len(buf)
                        logger.info(f"[WS-BOOT] {sym}/{interval}: {len(buf)} candles (cache disco, Δ={missing})")
                        continue
                    # Cache não tão fresco — busca só delta (limite pequeno)
                    delta_limit = min(missing + 5, limit)
                    try:
                        raw = self.client.futures_klines(
                            symbol=sym, interval=interval,
                            startTime=last_cached_ts + 1, limit=delta_limit,
                        )
                        new_candles = [{
                            'timestamp': int(k[0]), 'open': float(k[1]),
                            'high': float(k[2]), 'low': float(k[3]),
                            'close': float(k[4]), 'volume': float(k[5]),
                        } for k in raw]
                        # Mescla cache + delta
                        all_candles = cached + new_candles
                        # Deduplica e mantém ordem cronológica
                        seen: set[int] = set()
                        merged: list[dict] = []
                        for c in all_candles:
                            if c['timestamp'] not in seen:
                                seen.add(c['timestamp'])
                                merged.append(c)
                        merged = sorted(merged, key=lambda x: x['timestamp'])[-limit:]
                        buf = deque(merged, maxlen=KLINE_MAXLEN)
                        self.kline_buffers[sym][interval] = buf
                        total += len(buf)
                        self._save_cache(sym, interval, list(buf))
                        logger.info(f"[WS-BOOT] {sym}/{interval}: {len(buf)} candles (cache+Δ{len(new_candles)})")
                        time.sleep(0.50)  # delta path: ~10 calls/s máx → segurança extra
                    except Exception as exc:
                        # Fallback: usa cache antigo sem delta
                        buf = deque(cached, maxlen=KLINE_MAXLEN)
                        self.kline_buffers[sym][interval] = buf
                        total += len(buf)
                        logger.warning(f"[WS-BOOT] {sym}/{interval}: usando cache antigo — {exc}")
                else:
                    # Sem cache — busca REST completo (primeira vez)
                    try:
                        raw = self.client.futures_klines(symbol=sym, interval=interval, limit=limit)
                        candles = [{
                            'timestamp': int(k[0]), 'open': float(k[1]),
                            'high': float(k[2]), 'low': float(k[3]),
                            'close': float(k[4]), 'volume': float(k[5]),
                        } for k in raw]
                        buf = deque(candles, maxlen=KLINE_MAXLEN)
                        self.kline_buffers[sym][interval] = buf
                        total += len(buf)
                        self._save_cache(sym, interval, candles)
                        logger.info(f"[WS-BOOT] {sym}/{interval}: {len(buf)} candles (REST completo)")
                        time.sleep(0.60)  # 60 candles×0.6s = ~36s para 20 símbolos × 3TF = seguro
                    except Exception as exc:
                        logger.error(f"[WS-BOOT] Erro {sym}/{interval}: {exc}")

        self.bootstrap_done = True
        self.bootstrap_symbols = [s.upper() for s in symbols]
        logger.info(f"[WS-BOOT] Bootstrap completo: {total} candles | símbolos: {self.bootstrap_symbols}")

        # ── Inicia WebSocket e subscreve streams de kline via multiplex ──────
        # Um único socket multiplexado em vez de N×M sockets individuais;
        # isso evita as 60+ conexões simultâneas que causavam timeout.
        if not self.running:
            self.start()
        self.subscribe_all_klines(self.bootstrap_symbols)

        return total

    def save_kline_cache(self) -> int:
        """
        Persiste os buffers kline em memória para o cache em disco.
        Chame periodicamente (ex: a cada hora) para manter o cache fresco.
        Retorna o número de arquivos salvos.
        """
        saved = 0
        with self.lock:
            buffers_copy = {
                sym: {iv: list(buf) for iv, buf in ivs.items()}
                for sym, ivs in self.kline_buffers.items()
            }
        for sym, ivs in buffers_copy.items():
            for interval, candles in ivs.items():
                if candles:
                    self._save_cache(sym, interval, candles)
                    saved += 1
        if saved:
            logger.info(f"[WS-CACHE] {saved} buffers salvos em disco")
        return saved

    def bootstrap_account(self) -> bool:
        """Carrega balance e posições via REST (parte do bootstrap inicial)."""
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
            logger.info(
                f"[WS-BOOT] Account snapshot: "
                f"balance=${self.user_data['balance']['total']:.2f}, "
                f"positions={len(self.user_data['positions'])}"
            )
            return True
        except Exception as exc:
            logger.error(f"[WS-BOOT] Erro ao carregar snapshot de conta: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: subscriptions
    # ─────────────────────────────────────────────────────────────────────

    def subscribe_all_klines(self, symbols: list[str]) -> None:
        """
        Subscreve streams de kline via UM único socket multiplexado.
        Apenas intervalo 15m via WS — 19 streams em vez de 57.
        1h/4h permanecem no buffer do bootstrap REST (fecham raramente e
        o cache em disco é suficientemente fresco entre sessões).
        Isso evita BinanceWebsocketQueueOverflow sem precisar de queue_limit.
        """
        if not self.twm:
            return
        streams: list[str] = []
        for sym in symbols:
            s = sym.upper()
            self.kline_buffers.setdefault(s, {})
            for interval in INTERVALS_WS:
                self.kline_buffers[s].setdefault(interval, deque(maxlen=KLINE_MAXLEN))
            # Apenas 15m via WS — mantém live_price e candle aberto atualizados
            streams.append(f"{sym.lower()}@kline_15m")
        try:
            self.twm.start_futures_multiplex_socket(
                callback=self._handle_kline_multiplex,
                streams=streams,
            )
            logger.info(f"[WS] Multiplex socket ativo: {len(streams)} streams 15m ({len(symbols)} símbolos)")
        except Exception as exc:
            logger.warning(f"[WS] Multiplex falhou, usando sockets individuais: {exc}")
            for sym in symbols:
                self.subscribe_klines_multi(sym, ['15m'])

    def subscribe_klines_multi(self, symbol: str, intervals: list[str] | None = None) -> None:
        """Subscreve streams de kline individuais para um símbolo. Prefira subscribe_all_klines."""
        if not self.twm:
            return
        if intervals is None:
            intervals = INTERVALS_WS
        sym = symbol.upper()
        self.kline_buffers.setdefault(sym, {})
        for interval in intervals:
            self.kline_buffers[sym].setdefault(interval, deque(maxlen=KLINE_MAXLEN))
            try:
                self.twm.start_kline_futures_socket(
                    callback=lambda msg, s=sym, i=interval: self._handle_kline(msg, s, i),
                    symbol=sym.lower(),
                    interval=interval,
                )
                logger.info(f"[WS] Subscribed kline: {sym}/{interval}")
            except Exception as exc:
                logger.warning(f"[WS] Erro ao subscrever {sym}/{interval}: {exc}")

    # Legacy compat
    def subscribe_kline(self, symbol: str, interval: str = '15m') -> None:
        self.subscribe_klines_multi(symbol, [interval])

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: data getters
    # ─────────────────────────────────────────────────────────────────────

    def get_klines_df(self, symbol: str, interval: str = '15m',
                      limit: int = 200) -> pd.DataFrame | None:
        """
        Retorna DataFrame de candles do buffer em memória com indicadores técnicos.
        Nunca chama REST — lê apenas do buffer WS.
        """
        sym = symbol.upper()
        buf = self.kline_buffers.get(sym, {}).get(interval)
        if not buf or len(buf) < 5:
            return None
        rows = list(buf)[-limit:]
        df = pd.DataFrame(rows)

        # Delega cálculo de indicadores ao módulo centralizado
        try:
            df = compute_indicators(df)
        except Exception as exc:
            logger.warning(f"[WS-BUF] Erro ao computar indicadores {sym}/{interval}: {exc}")
            df = df.fillna(0).reset_index(drop=True)

        if len(df) < 5:
            return None
        return df

    def get_live_price(self, symbol: str) -> float | None:
        """Retorna último preço mid do book ticker (sub-segundo latência)."""
        sym = symbol.upper()
        price = self.live_price.get(sym)
        if price:
            return price
        buf = self.kline_buffers.get(sym, {}).get('15m')
        if buf:
            return buf[-1]['close']
        return None

    def get_balance(self) -> dict | None:
        """
        Balance do cache WebSocket com unrealized_pnl recalculado em tempo real
        a partir dos mark prices do buffer kline.

        Retorna None apenas se ainda não há nenhum snapshot.
        O background refresh garante que o dado tem no máximo 60s de idade.
        """
        with self.lock:
            if not self.user_data['last_update']:
                return None
            age = int((datetime.now() - self.user_data['last_update']).total_seconds())
            bal = dict(self.user_data['balance'])

            # Recalcula unrealized_pnl usando mark price live do buffer kline
            total_upnl = 0.0
            has_live = False
            for p in self.user_data['positions']:
                sym = p.get('symbol', '')
                try:
                    entry = float(p.get('entryPrice', 0))
                    qty   = float(p.get('positionAmt', 0))
                    if entry <= 0 or qty == 0:
                        continue
                    # Prioridade: live_price > kline close > markPrice do evento WS
                    mark: float = 0.0
                    if self.live_price.get(sym):
                        mark = float(self.live_price[sym])
                    elif self.kline_buffers.get(sym, {}).get('15m'):
                        mark = float(self.kline_buffers[sym]['15m'][-1].get('close', 0))
                    else:
                        mark = float(p.get('markPrice', 0))
                    if mark:
                        total_upnl += (float(mark) - entry) * qty
                        has_live = True
                except (ValueError, TypeError, IndexError):
                    pass

            if has_live:
                bal['unrealized_pnl'] = round(total_upnl, 4)

            ban_until = self.user_data.get('ban_until', 0.0)
            ban_remaining = max(0, int(ban_until - time.time())) if ban_until else 0
            return {
                **bal,
                'source': 'websocket',
                'error': None,
                'age_secs': age,
                'ban_until': ban_until or None,
                'ban_remaining': ban_remaining,
            }

    def get_positions(self) -> dict | None:
        """
        Positions do cache WebSocket com markPrice e unRealizedProfit
        enriquecidos em tempo real a partir do buffer kline.

        Retorna None apenas se ainda não há nenhum snapshot.
        O background refresh garante que a lista de posições tem no máximo 60s.
        """
        with self.lock:
            if not self.user_data['last_update']:
                return None
            age = int((datetime.now() - self.user_data['last_update']).total_seconds())
            enriched: list[dict] = []
            for p in self.user_data['positions']:
                sym    = p.get('symbol', '')
                p_copy = dict(p)
                try:
                    entry = float(p_copy.get('entryPrice', 0))
                    qty   = float(p_copy.get('positionAmt', 0))
                    # Enriquece markPrice: live_price (sub-segundo) > kline close
                    live_p = self.live_price.get(sym)
                    buf    = self.kline_buffers.get(sym, {}).get('15m')
                    if live_p:
                        mark = live_p
                        p_copy['markPrice'] = str(round(live_p, 8))
                    elif buf:
                        mark = buf[-1]['close']
                        p_copy['markPrice'] = str(round(mark, 8))
                    else:
                        mark = float(p_copy.get('markPrice', 0))
                    # Recalcula unRealizedProfit com mark price live
                    if entry > 0 and mark > 0 and qty != 0:
                        p_copy['unRealizedProfit'] = str(round((mark - entry) * qty, 4))
                except (ValueError, TypeError, IndexError):
                    pass
                enriched.append(p_copy)
            return {
                'positions': enriched,
                'source':    'websocket',
                'error':     None,
                'age_secs':  age,
            }

    def buffer_stats(self) -> dict:
        """Retorna tamanho atual de cada buffer (para debug na UI)."""
        return {
            sym: {iv: len(buf) for iv, buf in ivs.items()}
            for sym, ivs in self.kline_buffers.items()
        }

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE: callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _handle_user_data(self, msg: dict) -> None:
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
                                'entryPrice':       p.get('ep', '0'),
                                'markPrice':        p.get('mp', '0'),   # not present in all ACCOUNT_UPDATE payloads
                                'unRealizedProfit': p.get('up', '0'),
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

    def _handle_kline_multiplex(self, msg: dict) -> None:
        """Handler para mensagens do socket multiplexado. Delega ao _handle_kline normal."""
        try:
            # Detecta mensagens de erro do TWM
            if isinstance(msg, dict) and msg.get('e') == 'error':
                logger.error(f"[WS] Erro no multiplex stream: {msg.get('m', msg)}")
                now = time.time()
                if not self._reconnecting and (now - self._last_reconnect_time) > 5:
                    self._last_reconnect_time = now
                    threading.Thread(target=self._reconnect, daemon=True, name="WS-Reconn").start()
                return
            data = msg.get('data', msg)
            if data.get('e') != 'kline':
                return
            k = data['k']
            self._handle_kline(data, k['s'], k['i'])
        except Exception as exc:
            logger.error(f"[WS] _handle_kline_multiplex erro: {exc}")

    def _handle_kline(self, msg: dict, symbol: str, interval: str) -> None:
        """
        Atualiza o buffer de candles.
        - Candle aberto (x=False): atualiza o último candle com OHLCV live
        - Candle fechado (x=True): adiciona novo candle ao buffer
        Isso garante que o gráfico reflita o preço atual dentro do candle.
        """
        try:
            if msg.get('e') != 'kline':
                return
            k = msg['k']
            candle = {
                'timestamp': int(k['t']),
                'open':      float(k['o']),
                'high':      float(k['h']),
                'low':       float(k['l']),
                'close':     float(k['c']),
                'volume':    float(k['v']),
            }
            # Atualiza live_price SEM lock (dict write é thread-safe em CPython)
            # apenas do stream 15m para evitar redundância
            if interval == '15m':
                self.live_price[symbol] = float(k['c'])
            self._last_kline_tick = time.time()  # heartbeat para watchdog

            with self.lock:
                self.kline_buffers.setdefault(symbol, {})
                buf = self.kline_buffers[symbol].setdefault(interval, deque(maxlen=KLINE_MAXLEN))
                if k.get('x'):  # candle FECHADO — adiciona ao buffer
                    buf.append(candle)
                    logger.debug(f"[WS-KLINE] {symbol}/{interval} closed @ {k['c']} | buf={len(buf)}")
                elif interval == '15m':  # candle ABERTO — só atualiza 15m (gráfico live)
                    if buf:
                        buf[-1] = candle
                    else:
                        buf.append(candle)
                # candle ABERTO de 1h/4h — descartado (reduz lock contention drasticamente)
        except Exception as exc:
            logger.error(f"[WS] _handle_kline erro {symbol}/{interval}: {exc}")

    def _handle_book_ticker(self, msg: dict) -> None:
        try:
            sym = msg.get('s', '')
            bid = float(msg.get('b', 0))
            ask = float(msg.get('a', 0))
            if bid and ask:
                self.live_price[sym] = (bid + ask) / 2
        except Exception:
            pass
