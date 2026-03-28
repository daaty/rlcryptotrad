"""
TradingEngine — thread de background que executa inferência LSTM + gestão de posições.
Independente do ciclo de render do Streamlit.
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timedelta

import numpy as np

from dashboard.core.logging_setup import get_logger
from dashboard.trading.observation import prepare_observation, lstm_predict, apply_vecnorm
from dashboard.trading.entry_filter import validate_entry_quality
from dashboard.trading.executor import execute_trade, close_position_direct
from dashboard.trading.state_persistence import save_state, load_state
from dashboard.analytics.correlation import check_correlation
from dashboard.data.trade_store import get_trade_store

logger = get_logger()


class TradingEngine:
    """
    Executa inferência LSTM + gestão de posições em thread daemon.

    Arquitetura:
      • Roda independente do Streamlit (não para se o browser fechar).
      • Detecta novo candle 15m pelo timestamp do buffer WS → ZERO polling REST.
      • Verifica TP/SL/trailing a cada tick (5 s) mesmo sem candle novo.
      • Dashboard só lê engine.state (thread-safe via lock).
    """

    TICK_INTERVAL      = 5    # segundos entre ticks
    MIN_BUFFER_CANDLES = 52   # buffer mínimo antes de rodar inferência

    def __init__(self) -> None:
        self.lock    = threading.Lock()
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False

        self.state: dict = {
            'running':              False,
            'symbols':              [],
            'last_tick':            None,
            'decisions':            {},
            'portfolio':            {},
            'log':                  deque(maxlen=400),
            'orders':               deque(maxlen=50),
            'closed_trades':        deque(maxlen=200),
            'errors':               deque(maxlen=20),
            'lstm_states':          {},
            'kill_switch_triggered': False,
            'kill_switch_reason':   '',
            'peak_equity':          0.0,
            'current_drawdown_pct': 0.0,
        }

        # ── Kill Switch por drawdown ───────────────────────────────────────────
        # Threshold padrão: 15% de drawdown desde o pico de equity na sessão.
        # Pode ser sobrescrito via config.yaml: risk_management.max_drawdown_kill_switch
        self._peak_equity: float        = 0.0
        self._kill_switch_triggered: bool = False

        self._last_candle_ts: dict[str, int] = {}
        self._tp1_done: set[str] = set()
        self._notifier = None  # preenchido em _loop após get_notifier(cfg)

        # ── Carrega histórico de trades do SQLite (sobrevive a qualquer restart) ──
        try:
            _store = get_trade_store()
            _hist_trades = _store.load_closed_trades(limit=200)
            _hist_orders = _store.load_orders(limit=50)
            if _hist_trades:
                self.state['closed_trades'] = deque(_hist_trades, maxlen=200)
                logger.info(f"[ENGINE] ↩ {len(_hist_trades)} trades carregados do banco")
            if _hist_orders:
                self.state['orders'] = deque(_hist_orders, maxlen=50)
                logger.info(f"[ENGINE] ↩ {len(_hist_orders)} ordens carregadas do banco")
        except Exception as _db_err:
            logger.warning(f"[ENGINE] Falha ao carregar histórico do banco: {_db_err}")

        # ── Restaura estado persistido (sobrevive a reinicializações) ─────────
        _saved = load_state()
        if _saved:
            self._tp1_done        = _saved['tp1_done']
            self._last_candle_ts  = _saved['last_candle_ts']
            with self.lock:
                self.state['lstm_states'] = _saved['lstm_states']
            self._saved_trail_stops = _saved['trail_stops']  # aplicado no _loop
            logger.info(f"[ENGINE] Estado restaurado: {len(self._tp1_done)} TP1, "
                        f"{len(self._last_candle_ts)} candle_ts, "
                        f"{len(_saved['lstm_states'])} lstm_states")
        else:
            self._saved_trail_stops: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, symbols: list[str]) -> None:
        if self.running:
            with self.lock:
                self.state['symbols'] = symbols
            return
        self._stop.clear()
        with self.lock:
            self.state['symbols'] = symbols
            self.state['running'] = True
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='TradingEngine')
        self._thread.start()
        self._log(f"[ENGINE] ▶ Iniciado para: {symbols}")

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        with self.lock:
            self.state['running'] = False
        self._log("[ENGINE] ⏹ Parado pelo usuário")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        entry = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        with self.lock:
            self.state['log'].append(entry)
        logger.info(msg)

    def _check_kill_switch(self, ws_bal_data: dict, cfg: dict) -> None:
        """
        Kill Switch por drawdown — para o engine automaticamente se o drawdown
        desde o pico de equity exceder o threshold configurado.

        Threshold lido de config.yaml → risk_management.max_drawdown_kill_switch
        Padrão: 0.15 (15%).

        Acionado uma única vez por sessão (não pode ser re-ativado sem restart).
        """
        threshold = float(
            cfg.get('risk_management', {}).get('max_drawdown_kill_switch', 0.15)
        )

        current_equity = float(ws_bal_data.get('total', 0.0))
        if current_equity <= 0:
            return

        # Atualizar pico
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity <= 0:
            return

        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity

        # Atualizar estado para o dashboard
        with self.lock:
            self.state['peak_equity']          = self._peak_equity
            self.state['current_drawdown_pct'] = drawdown_pct

        # Verificar threshold
        if drawdown_pct < threshold:
            return

        # ── KILL SWITCH ACIONADO ──────────────────────────────────────────
        self._kill_switch_triggered = True
        reason = (
            f"Drawdown {drawdown_pct:.1%} excedeu threshold {threshold:.1%} | "
            f"Pico=${self._peak_equity:,.2f} | "
            f"Atual=${current_equity:,.2f}"
        )
        with self.lock:
            self.state['kill_switch_triggered'] = True
            self.state['kill_switch_reason']    = reason
            self.state['errors'].append(f"🛑 KILL SWITCH: {reason}")

        self._log(f"[KILL SWITCH] 🛑 {reason}")
        logger.critical(f"[KILL SWITCH] Engine parado por drawdown: {reason}")

        # Notificar via Telegram
        if self._notifier:
            try:
                self._notifier.notify_drawdown(drawdown_pct * 100)
                self._notifier.notify_engine_error(f"🛑 KILL SWITCH ACIONADO: {reason}")
            except Exception as _ne:
                logger.warning(f"[KILL SWITCH] Telegram notify falhou: {_ne}")

        # Parar o engine
        self._stop.set()
        self.running = False
        with self.lock:
            self.state['running'] = False



    def _loop(self) -> None:
        # Suprime avisos do Streamlit em threads sem ScriptRunContext
        import logging as _lg
        for _sl_name in ('streamlit', 'streamlit.runtime',
                         'streamlit.runtime.scriptrunner_utils',
                         'streamlit.runtime.scriptrunner'):
            _sl = _lg.getLogger(_sl_name)
            _sl.addFilter(lambda r: 'ScriptRunContext' not in r.getMessage())

        # Lazy-load de recursos pesados dentro da thread
        # IMPORTANTE: usa resources_ng (NiceGUI) para compartilhar os mesmos
        # singletons do main.py — em especial o WS manager já bootstrapado.
        from dashboard.resources_ng import (
            get_ws_manager, get_binance_client, get_models,
            get_risk_manager, get_trailing_stop_manager,
            get_warmup_manager, get_schedule_manager, get_config,
        )

        ws_mgr    = get_ws_manager()
        client    = get_binance_client()
        models_d  = get_models()
        risk_mgr  = get_risk_manager()
        trail_mgr = get_trailing_stop_manager()
        warmup    = get_warmup_manager()
        schedule  = get_schedule_manager()
        cfg       = get_config()

        # ── Telegram notifier (guardado como self._notifier para acesso em _tick) ─
        from dashboard.integrations.telegram_notifier import get_notifier
        notifier = get_notifier(cfg)
        self._notifier = notifier

        if not models_d.get('lstm_v17'):
            self._log("[ENGINE] ❌ Modelo LSTM não encontrado — engine abortado")
            with self.lock:
                self.state['running'] = False
                self.state['errors'].append("Modelo LSTM não encontrado em models/")
            self.running = False
            return

        # ── 1.2 Reconciliação de posições abertas no boot ─────────────────────
        try:
            open_positions = client.futures_position_information()
            reconciled = 0
            for pos in open_positions:
                qty   = float(pos.get('positionAmt', 0))
                entry = float(pos.get('entryPrice', 0))
                sym   = pos['symbol']
                if qty == 0 or entry <= 0:
                    continue
                ptype = 1 if qty > 0 else -1
                # Restaura trailing stop se não veio do estado salvo
                if sym not in trail_mgr.active_stops:
                    if sym in self._saved_trail_stops:
                        trail_mgr.active_stops[sym] = self._saved_trail_stops[sym]
                    else:
                        trail_mgr.register_position(sym, entry, ptype)
                # Heurística: se a qty atual for < 75% de uma posição cheia,
                # assume que TP1 já foi executado e move para breakeven.
                # (não há como saber a qty original após restart sem histórico)
                self._tp1_done.add(sym)  # conservador: nunca re-executa TP1
                reconciled += 1
                self._log(f"[BOOT] Posição reconciliada: {sym} qty={qty:.4f} entry={entry:.4f}")
            if reconciled:
                self._log(f"[BOOT] {reconciled} posições reconciliadas com sucesso")
        except Exception as exc:
            self._log(f"[BOOT] Erro na reconciliação: {exc}")

        self._saved_trail_stops.clear()  # libera memória temporaria

        self._log("[ENGINE] ✅ LSTM V19 pronto — aguardando candle 15m fechado...")

        trail_mgr_ref = trail_mgr  # alias para acesso no finally
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
                if self._notifier:
                    self._notifier.notify_engine_error(str(exc))
            finally:
                # ── 1.1 Persiste estado após cada tick ───────────────────────
                try:
                    save_state(
                        tp1_done=self._tp1_done,
                        last_candle_ts=self._last_candle_ts,
                        lstm_states_map=self.state.get('lstm_states', {}),
                        trail_active_stops=trail_mgr_ref.active_stops,
                    )
                except Exception as _se:
                    logger.debug(f"[STATE] save_state error: {_se}")
            self._stop.wait(timeout=self.TICK_INTERVAL)

        with self.lock:
            self.state['running'] = False
        self.running = False

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self, ws_mgr, client, models_d, risk_mgr,
              trail_mgr, warmup, schedule, cfg, symbols) -> None:

        # 1. TP/SL/Trailing — roda a cada tick para TODAS as posições abertas
        # IMPORTANTE: o guardião de SL protege QUALQUER posição aberta na conta,
        # independente de estar na lista active_syms (evita posições órfãs sem proteção).
        ws_pos_raw = ws_mgr.get_positions()
        positions  = ws_pos_raw if isinstance(ws_pos_raw, list) else (
                     ws_pos_raw.get('positions', []) if ws_pos_raw else [])
        active_syms = {s.replace('/', '').upper() for s in symbols}
        for pos in positions:
            self._check_tpsl(client, pos, risk_mgr, trail_mgr, cfg)

        # ── Pre-fetch WS balance e positions para eliminar REST no loop ───────────
        ws_bal_data = ws_mgr.get_balance()
        ws_avail    = float((ws_bal_data or {}).get('available', 0.0)) or None

        # ── Kill Switch por drawdown ───────────────────────────────────────────
        if ws_bal_data and not self._kill_switch_triggered:
            self._check_kill_switch(ws_bal_data, cfg)
        if self._kill_switch_triggered:
            return   # engine parado pelo kill switch — não processa mais nada
        # mapa sym → positionAmt (só posições != 0)
        ws_pos_map: dict[str, float] = {}
        for p in positions:
            amt = float(p.get('positionAmt', 0))
            if amt != 0:
                ws_pos_map[p['symbol']] = amt

        # 2. Por símbolo: detecta novo candle 15m e roda inferência
        for sym_raw in symbols:
            sym = sym_raw.replace('/', '').upper()
            buf = ws_mgr.kline_buffers.get(sym, {}).get('15m')
            if not buf or len(buf) < self.MIN_BUFFER_CANDLES:
                continue

            last_ts = buf[-1]['timestamp']
            if last_ts <= self._last_candle_ts.get(sym, 0):
                continue

            self._last_candle_ts[sym] = last_ts
            ts_str = datetime.fromtimestamp(last_ts / 1000).strftime('%H:%M')
            self._log(f"[ENGINE] {sym} 🕯 novo candle 15m @ {ts_str}")

            # ── 1.4 Detector WS stale — bloqueia inferência em dados mortos ──
            # Usa o heartbeat do WS manager (última mensagem recebida), NÃO o
            # timestamp de abertura do candle (que pode ser até 14min antigo
            # durante a vida normal de um candle 15m — gera falsos positivos).
            import time as _time
            ws_age_secs = int(_time.time() - ws_mgr._last_kline_tick) if ws_mgr._last_kline_tick > 0 else 0
            if ws_age_secs > 300:  # 5 minutos sem nenhuma mensagem do stream
                self._log(
                    f"[ENGINE] ⚠️ {sym} WS STALE {ws_age_secs//60:.0f}min — "
                    f"inferência bloqueada (reconectando...)"
                )
                with self.lock:
                    self.state['decisions'].setdefault(sym, {})['ws_age_secs'] = ws_age_secs
                if self._notifier:
                    self._notifier.notify_ws_down(sym, ws_age_secs)
                continue

            df_15m  = ws_mgr.get_klines_df(sym, '15m', limit=200)
            multi_tf: dict = {}
            df_1h   = ws_mgr.get_klines_df(sym, '1h', limit=100)
            df_4h   = ws_mgr.get_klines_df(sym, '4h', limit=60)
            if df_1h is not None:
                multi_tf['1h'] = df_1h
            if df_4h is not None:
                multi_tf['4h'] = df_4h

            if df_15m is None or len(df_15m) < 52:
                self._log(f"[ENGINE] {sym} dados 15m insuficientes no buffer")
                continue

            # Warm-up
            cur_wu, req_wu, _ = warmup.get_progress(sym)
            if cur_wu < req_wu:
                shortcut = min(len(buf) - 1, req_wu - cur_wu)
                for _ in range(shortcut):
                    warmup.add_candle(sym)
            warmup.add_candle(sym)
            if not warmup.is_ready(sym):
                cur_wu, req_wu, pct_wu = warmup.get_progress(sym)
                self._log(f"[ENGINE] {sym} warm-up {cur_wu}/{req_wu} ({pct_wu:.0f}%) — aguardando")
                continue

            # Schedule
            candle_close_dt = datetime.fromtimestamp(last_ts / 1000)
            can_sched, reason_sched = schedule.can_trade_now(sym, at_time=candle_close_dt)
            if not can_sched:
                for _grace in range(1, 4):
                    _grace_dt = candle_close_dt + timedelta(minutes=_grace)
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
            # IMPORTANTE: position deriva da posição REAL no exchange (ws_pos_map),
            # nunca da última decisão do modelo. Se usarmos a decisão como context,
            # trades bloqueados (corr/filtro) fariam o LSTM ver uma posição que
            # não existe e inverter na próxima vela (ex: SHORT bloqueado → LSTM
            # vê pos=-1 e abre LONG para "fechar" a posição que nunca foi aberta).
            ws_bal = ws_mgr.get_balance()
            port   = self.state['portfolio'].setdefault(sym, {
                'position': 0.0, 'balance_norm': 1.0, 'equity_norm': 1.0,
            })
            _ws_pos_amt = ws_pos_map.get(sym, 0.0)
            port['position'] = 1.0 if _ws_pos_amt > 0 else (-1.0 if _ws_pos_amt < 0 else 0.0)

            # Observação + inferência
            # V19 (VecNormalize): normalize=True → normalize_ohlcv_v19 + aplica vecnorm stats
            # V17.7:              normalize=False → clip ±100 (sem normalização OHLCV)
            _vecnorm = models_d.get('vecnorm')
            obs = prepare_observation(
                market_data_15m=df_15m,
                multi_tf_data=multi_tf or None,
                balance_norm=port['balance_norm'],
                position=port['position'],
                equity_norm=port['equity_norm'],
                normalize=(_vecnorm is not None),
            )
            if obs is None:
                self._log(f"[ENGINE] {sym} falha ao preparar observação")
                continue
            # Aplica VecNormalize se modelo V19 — sem isso a distribuição diverge do treino
            if _vecnorm is not None:
                obs = apply_vecnorm(_vecnorm, obs)

            lstm_states  = self.state.get('lstm_states', {}).get(sym)
            ep_start     = np.ones((1,), dtype=bool) if lstm_states is None else np.zeros((1,), dtype=bool)
            action_value, final_action, new_lstm_states = lstm_predict(
                models_d['lstm_v17'], obs, lstm_states, ep_start
            )

            current_price = float(df_15m['close'].iloc[-1])
            with self.lock:
                self.state.setdefault('lstm_states', {})[sym] = new_lstm_states
                _rsi_raw = float(df_15m['RSI_14'].iloc[-1])
                # RSI está normalizado 0–1 no dataset de treino — converte para 0–100
                _rsi_display = round(_rsi_raw * 100, 1) if _rsi_raw <= 1.0 else round(_rsi_raw, 1)
                self.state['decisions'][sym] = {
                    'action':       final_action,
                    'value':        round(action_value, 4),
                    'price':        current_price,
                    'ts':           datetime.now(),
                    'rsi':          _rsi_display,
                    'ws_age_secs':  ws_age_secs,
                }
                self.state['portfolio'][sym] = port  # persiste sem sobrescrever position (já derivada do WS)

            self._log(f"[ENGINE] {sym} → {final_action} (val={action_value:.3f}) @ ${current_price:,.2f}")

            # Filtro de qualidade
            filter_mode = cfg.get('entry_filter', {}).get('mode', 'normal')
            can_enter, block_reason = validate_entry_quality(df_15m, final_action, current_price, mode=filter_mode)
            if not can_enter:
                self._log(f"[ENGINE] {sym} entrada filtrada: {block_reason}")
                continue

            # ── 2.2 Verificação de correlação com posições abertas ───────────────
            if final_action != 'FLAT':
                open_syms = [s for s, amt in ws_pos_map.items() if amt != 0 and s != sym]
                corr_threshold = cfg.get('risk_management', {}).get('correlation_threshold', 0.70)
                can_corr, corr_reason = check_correlation(
                    sym, open_syms, ws_mgr, threshold=corr_threshold
                )
                if not can_corr:
                    self._log(f"[ENGINE] {sym} bloqueado por correlação: {corr_reason}")
                    continue

            # ── 2.3 Verificação de exposição total ─────────────────────────────
            if final_action != 'FLAT':
                try:
                    ws_equity = float((ws_mgr.get_balance() or {}).get('total', 0)) or 1.0
                    total_notional = sum(
                        abs(float(p.get('positionAmt', 0))) * float(p.get('markPrice', p.get('entryPrice', 0)))
                        for p in positions if p['symbol'] in active_syms
                    )
                    exposure_pct  = total_notional / ws_equity if ws_equity > 0 else 1.0
                    max_exposure  = cfg.get('risk_management', {}).get('max_total_exposure', 0.60)
                    if exposure_pct >= max_exposure:
                        self._log(
                            f"[ENGINE] {sym} bloqueado: exposição "
                            f"{exposure_pct:.1%} >= {max_exposure:.1%}"
                        )
                        continue
                except Exception as _exp:
                    logger.debug(f"[EXP] Erro ao calcular exposição: {_exp}")

            # Executa ordem — símbolo explícito (sem hack temp_cfg)
            ws_pos_amt     = ws_pos_map.get(sym, 0.0)
            closed_trades  = list(self.state.get('closed_trades', []))
            _paper_mode    = cfg.get('mode', 'testnet') == 'paper'
            order = execute_trade(
                client, final_action, current_price, cfg,
                ws_position_amt=ws_pos_amt,
                ws_available_balance=ws_avail,
                symbol=sym,
                closed_trades=closed_trades,
                paper_mode=_paper_mode,
            )

            if order:
                _order_record = {
                    'symbol':    sym,
                    'side':      order['side'],
                    'qty':       order['origQty'],
                    'price':     order.get('avgPrice', 'MKT'),
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'action':    final_action,
                    'orderId':   str(order.get('orderId', '')),
                }
                with self.lock:
                    self.state['orders'].append(_order_record)
                # Persiste no SQLite
                try:
                    get_trade_store().save_order(_order_record)
                except Exception as _oe:
                    logger.warning(f"[ENGINE] Falha ao salvar ordem no banco: {_oe}")
                # avgPrice pode vir como string '0' em MARKET orders antes do fill assíncrono.
                # Não usar `or` direto — '0' é string truthy. Converter primeiro.
                _raw_avg = order.get('avgPrice', '0')
                avg_px   = float(_raw_avg or 0) or current_price
                # register_position espera int: 1=LONG, -1=SHORT
                trail_side = 1 if order['side'] == 'BUY' else -1
                trail_mgr.register_position(sym, avg_px, trail_side)
                self._log(f"[ENGINE] ✅ {order['side']} {sym} id={order['orderId']}")
                with self.lock:
                    self.state['last_tick'] = datetime.now()

    # ── TP/SL/Trailing ────────────────────────────────────────────────────────

    def _check_tpsl(self, client, pos: dict, risk_mgr, trail_mgr, cfg: dict) -> None:
        sym   = pos['symbol']
        qty   = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark  = float(pos['markPrice'])

        # Postura com preços ainda não preenchidos pelo WS — aguardar próximo tick
        if entry <= 0 or mark <= 0:
            return

        ptype = 1 if qty > 0 else -1
        atr   = mark * 0.02

        # Telegram notifier (carregado dentro do método para não passar por param)
        from dashboard.integrations.telegram_notifier import get_notifier
        notifier = get_notifier(cfg)

        if not trail_mgr.get_stop_info(sym):
            trail_mgr.register_position(sym, entry, ptype)  # ptype = 1 (LONG) ou -1 (SHORT)

        # ── helper: registra trade fechado ─────────────────────────────
        def _record_close(side_label: str, close_qty: float) -> None:
            pnl_usd = (mark - entry) * abs(close_qty) * ptype
            trade_record = {
                'symbol':      sym,
                'side':        side_label,
                'realizedPnl': round(pnl_usd, 4),
                'time':        int(datetime.now().timestamp() * 1000),
                'entryPrice':  entry,
                'exitPrice':   mark,
                'qty':         abs(close_qty),
            }
            with self.lock:
                self.state['closed_trades'].append(trade_record)
            # Persiste no SQLite imediatamente——sobrevive a qualquer restart
            try:
                get_trade_store().save_closed_trade(trade_record)
            except Exception as _se:
                logger.warning(f"[ENGINE] Falha ao salvar trade no banco: {_se}")

        # Trailing stop
        should_exit_trail, trail_price = trail_mgr.update(sym, mark)
        if should_exit_trail:
            order = close_position_direct(client, sym, qty, cfg)
            if order:
                trail_mgr.remove_position(sym)
                pnl = (mark - entry) / entry * ptype * 100
                _record_close('TRAIL', qty)
                notifier.notify_sl(sym, pnl)  # trail é similar a SL
                self._log(f"[ENGINE] 🛑 Trail stop {sym} @ ${trail_price:,.2f} P&L={pnl:+.2f}%")
            return

        # Stop Loss
        if risk_mgr.should_stop_loss(entry, mark, ptype, atr=atr):
            order = close_position_direct(client, sym, qty, cfg)
            if order:
                trail_mgr.remove_position(sym)
                pnl = (mark - entry) / entry * ptype * 100
                _record_close('SL', qty)
                notifier.notify_sl(sym, pnl)
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
                order = close_position_direct(client, sym, qty, cfg)
                if order:
                    trail_mgr.remove_position(sym)
                    self._tp1_done.discard(sym)
                    _record_close('TP2', qty)
                    notifier.notify_tp(sym, 2, pnl)
                    self._log(f"[ENGINE] 🎯 TP L2 (100%) {sym} +{pnl:.2f}%")
            elif tp_level == 1 and sym not in self._tp1_done:
                order = close_position_direct(client, sym, qty / 2, cfg)
                if order:
                    self._tp1_done.add(sym)
                    _record_close('TP1', qty / 2)
                    # ── 1.3 Move SL → breakeven após TP1 ──────────────────
                    trail_mgr.update_entry_price(sym, entry)
                    notifier.notify_tp(sym, 1, pnl)
                    self._log(f"[ENGINE] 🔒 Breakeven ativado {sym} @ ${entry:,.4f}")
                    self._log(f"[ENGINE] 🎯 TP L1 (50%) {sym} +{pnl:.2f}%")
