"""
LiveState — estado reativo centralizado para o dashboard NiceGUI.

O timer (2s) chama `LiveState.refresh()` que lê engine.state + ws_manager
e atualiza os campos. Os componentes NiceGUI ficam vinculados a estes campos
via binding, atualizando automaticamente sem re-render da página inteira.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any


class LiveState:
    """
    Store único para todos os valores exibidos no dashboard.
    Pode ser passado como `model` em ui.label().bind_text_from(state, 'balance')
    """

    def __init__(self):
        # ── Conta ─────────────────────────────────────────────────────────
        self.balance_str    = '$0.00'
        self.available_str  = '$0.00'
        self.pnl_str        = '+$0.00'
        self.pnl_pct_str    = '(0.00%)'
        self.pnl_positive   = True

        # ── Engine ────────────────────────────────────────────────────────
        self.engine_running  = False
        self.engine_label    = '⏹ Parado'
        self.last_tick_str   = '—'

        # ── Ban ───────────────────────────────────────────────────────────
        self.banned          = False
        self.ban_label       = ''
        self.ban_remaining_s = 0.0

        # ── Kill Switch ───────────────────────────────────────────────────
        self.kill_switch     = False
        self.ks_reason       = ''
        self.drawdown_pct    = 0.0
        self.peak_equity     = 0.0

        # ── WS ────────────────────────────────────────────────────────────
        self.ws_connected    = False
        self.ws_label        = '📡 Desconectado'
        self.ws_age_s        = 0

        # ── Posições abertas ──────────────────────────────────────────────
        self.open_positions: list[dict] = []
        self.n_positions     = 0

        # ── Decisões LSTM ─────────────────────────────────────────────────
        self.decisions: dict[str, Any] = {}

        # ── Trades/Ordens ─────────────────────────────────────────────────
        self.closed_trades: list[dict] = []
        self.orders: list[dict]        = []
        self.n_trades   = 0
        self.win_rate   = 0.0
        self.total_pnl  = 0.0
        # ── P&L breakdown ─────────────────────────────────────────────
        # unrealized: soma das posições abertas agora
        self.unrealized_pnl      = 0.0
        self.unrealized_pnl_str  = '$0.00'
        self.unrealized_pnl_pct  = 0.0
        # account: saldo_atual vs capital_inicial (inclui realizados passados)
        self.account_pnl         = 0.0
        self.account_pnl_str     = '+$0.00'
        self.account_pnl_pct_str = '(0.00%)'
        self.initial_balance     = 10_000.0
        # ── Log ───────────────────────────────────────────────────────────
        self.log_lines: list[str] = []
        self.errors: list[str]    = []

        # ── Símbolos selecionados ─────────────────────────────────────────
        self.symbols: list[str] = []

        # ── UI settings — modo visual e tema
        self.display_mode = 'Detalhado'  # 'Detalhado' ou 'Compacto'
        self.theme        = 'dark'       # 'dark' ou 'light'
        self.accent       = 'cyan'       # 'cyan', 'green', 'amber', 'purple'

        # ── Histórico P&L para gráficos (cumulativo)
        self.pnl_history_timestamps: list[str] = []
        self.pnl_history: list[float] = []

        # ── Update timestamp ──────────────────────────────────────────────
        self._last_refresh = 0.0

    # ──────────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Lê engine.state e ws_manager e atualiza todos os campos."""
        from dashboard.resources_ng import (
            get_trading_engine, get_ws_manager, is_banned, get_ban_expires_at,
        )

        now = time.time()
        self._last_refresh = now

        engine  = get_trading_engine()
        ws      = get_ws_manager()

        # ── Engine ────────────────────────────────────────────────────────
        with engine.lock:
            e_state = dict(engine.state)   # cópia rasa (thread-safe)

        self.engine_running = bool(e_state.get('running', False))
        self.engine_label   = '▶ Rodando' if self.engine_running else '⏹ Parado'
        self.symbols        = list(e_state.get('symbols', []))

        tick = e_state.get('last_tick')
        if tick:
            try:
                self.last_tick_str = datetime.fromisoformat(str(tick)).strftime('%H:%M:%S')
            except Exception:
                self.last_tick_str = str(tick)

        # ── Kill Switch ────────────────────────────────────────────────────
        self.kill_switch  = bool(e_state.get('kill_switch_triggered', False))
        self.ks_reason    = str(e_state.get('kill_switch_reason', ''))
        self.drawdown_pct = float(e_state.get('current_drawdown_pct', 0.0))
        self.peak_equity  = float(e_state.get('peak_equity', 0.0))

        # ── Decisões ──────────────────────────────────────────────────────
        self.decisions = dict(e_state.get('decisions', {}))

        # ── Trades ────────────────────────────────────────────────────────
        trades_raw = list(e_state.get('closed_trades', []))
        self.closed_trades = trades_raw[-50:]   # últimos 50 para a tabela
        self.orders        = list(e_state.get('orders', []))

        if trades_raw:
            # Suporta chaves 'pnl' (engine interna) e 'realizedPnl' (Binance)
            pnls = [
                float(t.get('pnl') or t.get('realizedPnl', 0))
                for t in trades_raw
                if t.get('pnl') is not None or t.get('realizedPnl') is not None
            ]
            self.n_trades  = len(pnls)
            self.total_pnl = sum(pnls)
            wins           = sum(1 for p in pnls if p > 0)
            self.win_rate  = wins / len(pnls) * 100 if pnls else 0.0
        else:
            self.n_trades  = 0
            self.total_pnl = 0.0
            self.win_rate  = 0.0

        # ── P&L history (timestamp + acumulado) ───────────────────────────
        try:
            cumulative = 0.0
            self.pnl_history = []
            self.pnl_history_timestamps = []
            for t in self.closed_trades[-40:]:
                pnl = float(t.get('realizedPnl', t.get('pnl', 0.0)) or 0.0)
                cumulative += pnl
                ts = t.get('time') or t.get('timestamp') or ''
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(int(ts) / 1000).strftime('%H:%M')
                self.pnl_history_timestamps.append(str(ts))
                self.pnl_history.append(cumulative)
        except Exception:
            self.pnl_history = []
            self.pnl_history_timestamps = []

        # ── Log ───────────────────────────────────────────────────────────
        self.log_lines = list(e_state.get('log', []))[-100:]
        self.errors    = list(e_state.get('errors', []))

        # ── WS / Conta ────────────────────────────────────────────────────
        try:
            from dashboard.resources_ng import get_config as _get_cfg
            _cfg     = _get_cfg()
            initial  = float(_cfg.get('environment', {}).get('initial_balance', 10_000.0))
            self.initial_balance = initial

            bal       = ws.get_balance() or {}
            total     = float(bal.get('total',         0.0))
            available = float(bal.get('available',     0.0))
            unreal    = float(bal.get('unrealized_pnl', 0.0))

            self.balance_str   = f'${total:,.2f}'
            self.available_str = f'${available:,.2f}'

            # P&L não-realizado (posições abertas agora) ← "P&L Sessão"
            self.unrealized_pnl     = unreal
            u_sign = '+' if unreal >= 0 else ''
            wallet  = total - unreal  # saldo sem considerar abertos
            u_pct   = (unreal / wallet * 100) if wallet else 0.0
            self.unrealized_pnl_str = f'{u_sign}${unreal:,.2f}'
            self.unrealized_pnl_pct = u_pct

            # P&L total da conta vs capital inicial
            acc_pnl  = total - initial
            acc_pct  = (acc_pnl / initial * 100) if initial else 0.0
            a_sign   = '+' if acc_pnl >= 0 else ''
            self.account_pnl         = acc_pnl
            self.account_pnl_str     = f'{a_sign}${acc_pnl:,.2f}'
            self.account_pnl_pct_str = f'({a_sign}{acc_pct:.2f}%)'

            # pnl_str / pnl_pct_str usados no header → mostra não-realizado
            self.pnl_positive  = unreal >= 0
            self.pnl_str       = self.unrealized_pnl_str
            self.pnl_pct_str   = f'({u_sign}{u_pct:.2f}%)'
            self.total_pnl     = unreal   # usado para colorir header
        except Exception:
            pass

        # ── Posições ──────────────────────────────────────────────────────
        try:
            raw_pos = ws.get_positions()
            pos_list = (
                raw_pos if isinstance(raw_pos, list)
                else raw_pos.get('positions', []) if raw_pos else []
            )
            self.open_positions = [
                p for p in pos_list if abs(float(p.get('positionAmt', 0))) > 0
            ]
            self.n_positions = len(self.open_positions)
        except Exception:
            self.open_positions = []
            self.n_positions    = 0

        # ── WS saúde ──────────────────────────────────────────────────────
        try:
            connected = bool(ws.connected if hasattr(ws, 'connected') else True)
            self.ws_connected = connected
            self.ws_label = '📡 Conectado' if connected else '📡 Desconectado'
        except Exception:
            self.ws_connected = False
            self.ws_label     = '📡 ?'

        # ── Ban ───────────────────────────────────────────────────────────
        banned, remaining = is_banned()
        self.banned          = banned
        self.ban_remaining_s = remaining
        if banned:
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            self.ban_label = f'⛔ IP BAN ({h}h{m:02d}m)'
        else:
            self.ban_label = ''


# ── Singleton global ──────────────────────────────────────────────────────────
_live_state: LiveState | None = None


def get_live_state() -> LiveState:
    global _live_state
    if _live_state is None:
        _live_state = LiveState()
    return _live_state
