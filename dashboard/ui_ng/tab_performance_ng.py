"""
Tab Desempenho — NiceGUI version.
Histórico de trades fechados, métricas de PnL, chart P&L, DB cross-sessão.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime

from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider, pnl_color as _pnl_color


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pnl(value) -> str:
    try:
        f = float(value)
        return f'{"+" if f >= 0 else ""}${f:,.4f}'
    except Exception:
        return '—'


def _ms_to_str(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime('%d/%m %H:%M')
    except Exception:
        return '—'


def _symbol_summary(closed_trades: list[dict]) -> list[dict]:
    rows: dict = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0, 'best': 0.0, 'worst': 0.0})
    for t in closed_trades:
        s   = t.get('symbol', '?')
        pnl = float(t.get('realizedPnl', 0))
        rows[s]['trades'] += 1
        rows[s]['wins']   += 1 if pnl > 0 else 0
        rows[s]['pnl']    += pnl
        rows[s]['best']    = max(rows[s]['best'],  pnl)
        rows[s]['worst']   = min(rows[s]['worst'], pnl)
    out = []
    for sym, r in sorted(rows.items()):
        wr = r['wins'] / r['trades'] * 100 if r['trades'] else 0
        out.append({
            'Símbolo' : sym.replace('USDT', ''),
            'Trades'  : r['trades'],
            'Win Rate': f'{wr:.0f}%',
            'P&L ($)' : round(r['pnl'],   4),
            'Melhor'  : round(r['best'],   4),
            'Pior'    : round(r['worst'],  4),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_performance_tab(state: LiveState) -> None:
    _performance_panel(state)


@ui.refreshable
def _performance_panel(state: LiveState) -> None:
    n_trades  = state.n_trades
    win_rate  = state.win_rate
    total_pnl = state.total_pnl
    pnl_color = 'text-green-400' if total_pnl >= 0 else 'text-red-400'
    pnl_sign  = '+' if total_pnl >= 0 else ''

    # ── Banner DB cross-sessão ────────────────────────────────────────────
    _render_db_stats_banner()

    section_title('📊 Desempenho da Sessão')

    # ── Métricas gerais ──────────────────────────────────────────────────────
    with ui.row().classes('gap-4 flex-wrap mb-4'):
        _card('Trades Fechados', str(n_trades), 'swap_horiz')
        _card('Win Rate',        f'{win_rate:.1f}%', 'emoji_events',
              color='text-green-400' if win_rate >= 50 else 'text-red-400')
        _card('P&L Total',       f'{pnl_sign}${total_pnl:,.4f}', 'account_balance',
              color=pnl_color)
        _card('P&L Médio/Trade',
              f'{"+" if n_trades and total_pnl/n_trades >= 0 else ""}${total_pnl/n_trades:,.4f}'
              if n_trades else '—',
              'trending_up')

    divider()

    # ── Posições abertas ──────────────────────────────────────────────────────
    _render_open_positions(state)

    # ── Sumário por símbolo ──────────────────────────────────────────────────
    closed = state.closed_trades
    if not closed:
        with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-6 text-center'):
            ui.label('📭 Sem trades fechados ainda.').classes('text-gray-400 text-lg')
        return

    section_title('📋 Sumário por Símbolo')
    sym_rows = _symbol_summary(closed)
    if sym_rows:
        ui.aggrid({
            'columnDefs': [
                {'field': 'Símbolo',  'width': 100},
                {'field': 'Trades',   'width': 80},
                {'field': 'Win Rate', 'width': 100},
                {'field': 'P&L ($)',  'width': 110,
                 'cellClassRules': {'text-green-400': 'x > 0', 'text-red-400': 'x < 0'}},
                {'field': 'Melhor',   'width': 100},
                {'field': 'Pior',     'width': 100},
            ],
            'rowData': sym_rows,
            'domLayout': 'autoHeight',
        }).classes('w-full ag-theme-alpine-dark')

    divider()

    # ── Gráfico P&L ──────────────────────────────────────────────────────────
    _render_pnl_chart(closed)

    divider()

    # ── Histórico completo ───────────────────────────────────────────────────
    section_title(f'📜 Histórico ({len(closed)} trades)')

    history = []
    for t in reversed(closed[-200:]):          # últimos 200, mais recente primeiro
        pnl = float(t.get('realizedPnl', 0))
        history.append({
            'Hora'     : _ms_to_str(t.get('time', 0)),
            'Símbolo'  : t.get('symbol', '—').replace('USDT', ''),
            'Lado'     : t.get('side', '—'),
            'Entrada'  : f"${float(t.get('entryPrice',0)):,.4f}",
            'Saída'    : f"${float(t.get('exitPrice', 0)):,.4f}",
            'P&L'      : f'{"+" if pnl>=0 else ""}${pnl:,.4f}',
            '_pnl_raw' : pnl,
        })

    ui.aggrid({
        'columnDefs': [
            {'field': 'Hora',    'width': 110},
            {'field': 'Símbolo', 'width': 90},
            {'field': 'Lado',    'width': 80},
            {'field': 'Entrada', 'width': 120},
            {'field': 'Saída',   'width': 120},
            {'field': 'P&L',     'width': 120,
             'cellClassRules': {'text-green-400': "params.data['_pnl_raw'] > 0",
                                 'text-red-400'  : "params.data['_pnl_raw'] < 0"}},
        ],
        'rowData': history,
        'domLayout': 'autoHeight',
        'pagination': True,
        'paginationPageSize': 25,
    }).classes('w-full ag-theme-alpine-dark')


# ─────────────────────────────────────────────────────────────────────────────
# DB cross-session stats banner
# ─────────────────────────────────────────────────────────────────────────────

def _render_db_stats_banner() -> None:
    """Mostra estatísticas totais do TradeStore (todas as sessões)."""
    try:
        from dashboard.data.trade_store import get_trade_store
        store    = get_trade_store()
        db_stats = store.get_stats()
        db_total = db_stats.get('total_trades', 0)
        if db_total == 0:
            return
        db_pnl  = float(db_stats.get('total_pnl', 0.0) or 0.0)
        db_wr   = (db_stats.get('wins', 0) / db_total * 100) if db_total else 0
        db_sess = db_stats.get('sessions', 0)
        first   = str(db_stats.get('first_trade', '?'))[:10]
        pnl_col = 'text-green-400' if db_pnl >= 0 else 'text-red-400'
        with ui.card().classes('w-full bg-gray-800 border border-gray-600 p-3 mb-3'):
            ui.label('🗄️ Histórico Completo (todas as sessões)').classes(
                'text-gray-300 text-sm font-semibold mb-2')
            with ui.row().classes('gap-6 flex-wrap'):
                _db_stat('📅 Trades', str(db_total))
                _db_stat('💰 PnL Acumulado', f'{"+" if db_pnl >= 0 else ""}${db_pnl:,.2f}',
                         pnl_col)
                _db_stat('🎯 Win Rate', f'{db_wr:.1f}%',
                         'text-green-400' if db_wr >= 50 else 'text-red-400')
                _db_stat('🗓️ Sessões', f'{db_sess} (desde {first})')
    except Exception:
        pass   # TradeStore indisponível — silencioso


def _db_stat(label: str, value: str, color: str = 'text-white') -> None:
    with ui.column().classes('gap-0'):
        ui.label(label).classes('text-gray-500 text-xs')
        ui.label(value).classes(f'{color} text-sm font-bold font-mono')


# ─────────────────────────────────────────────────────────────────────────────
# Posições abertas
# ─────────────────────────────────────────────────────────────────────────────

def _render_open_positions(state: LiveState) -> None:
    """Posições abertas com PnL não-realizado."""
    positions = state.open_positions
    if not positions:
        return

    total_unreal = sum(float(p.get('unRealizedProfit', 0)) for p in positions)
    unreal_color = 'text-green-400' if total_unreal >= 0 else 'text-red-400'
    sign = '+' if total_unreal >= 0 else ''

    section_title(f'🟡 Posições Abertas ({len(positions)})')
    ui.label(f'PnL Não-Realizado Total: {sign}${total_unreal:,.4f}').classes(
        f'{unreal_color} font-bold font-mono text-sm mb-2')

    rows = []
    for p in positions:
        amt  = float(p.get('positionAmt', 0))
        pnl  = float(p.get('unRealizedProfit', 0))
        ep   = float(p.get('entryPrice', 0))
        mp   = float(p.get('markPrice',  0))
        side = 'LONG' if amt > 0 else 'SHORT'
        pct  = (mp - ep) / ep * 100 * (1 if amt > 0 else -1) if ep > 0 else 0
        rows.append({
            'Símbolo': p.get('symbol', '?'),
            'Lado'   : side,
            'Qty'    : abs(amt),
            'Entrada': f'${ep:,.4f}',
            'Mark'   : f'${mp:,.4f}',
            'PnL ($)': f'{"+" if pnl>=0 else ""}{pnl:.4f}',
            'PnL (%)': f'{"+" if pct>=0 else ""}{pct:.2f}%',
        })

    ui.aggrid({
        'columnDefs': [
            {'field': 'Símbolo', 'width': 105},
            {'field': 'Lado',    'width': 80,
             'cellClassRules': {'text-green-400': "x === 'LONG'", 'text-red-400': "x === 'SHORT'"}},
            {'field': 'Qty',     'width': 95},
            {'field': 'Entrada', 'width': 115},
            {'field': 'Mark',    'width': 115},
            {'field': 'PnL ($)', 'width': 115,
             'cellClassRules': {'text-green-400': "x.startsWith('+')", 'text-red-400': "x.startsWith('-')"}},
            {'field': 'PnL (%)', 'width': 100},
        ],
        'rowData': rows,
        'domLayout': 'autoHeight',
    }).classes('w-full ag-theme-alpine-dark mb-4')
    divider()


# ─────────────────────────────────────────────────────────────────────────────
# PnL Chart
# ─────────────────────────────────────────────────────────────────────────────

def _render_pnl_chart(closed_trades: list[dict]) -> None:
    """Gráfico barras P&L por trade + linha P&L acumulado."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    if not closed_trades:
        return

    try:
        df = pd.DataFrame(closed_trades)
        if 'realizedPnl' not in df.columns:
            return

        df['realizedPnl']  = df['realizedPnl'].astype(float)
        df['cumulative']   = df['realizedPnl'].cumsum()
        df['time_label']   = df.get('time', pd.Series(range(len(df)))).apply(
            lambda ms: _ms_to_str(ms) if isinstance(ms, (int, float)) else str(ms)
        )
        bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df['realizedPnl']]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.42, 0.58],
            subplot_titles=('P&L por Trade (USDT)', 'P&L Acumulado (USDT)'),
        )
        fig.add_trace(go.Bar(
            x=df['time_label'], y=df['realizedPnl'],
            marker_color=bar_colors, name='P&L / Trade',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['time_label'], y=df['cumulative'],
            mode='lines+markers', name='Acumulado',
            line=dict(color='#7986cb', width=2),
            fill='tozeroy', fillcolor='rgba(121,134,203,0.15)',
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.25)', row=2, col=1)
        fig.update_layout(
            height=380, template='plotly_dark', showlegend=False,
            margin=dict(l=8, r=8, t=40, b=8),
            paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
        )
        fig.update_yaxes(tickprefix='$', row=1, col=1)
        fig.update_yaxes(tickprefix='$', row=2, col=1)

        section_title('📈 Gráfico P&L')
        ui.plotly(fig).classes('w-full')
    except Exception as exc:
        ui.label(f'Erro no chart P&L: {exc}').classes('text-orange-400 text-xs')


def _card(label: str, value: str, icon: str, color: str = 'text-white') -> None:
    with ui.card().classes('bg-gray-800 border border-gray-700 px-5 py-3 min-w-36'):
        with ui.row().classes('items-center gap-2 mb-1'):
            ui.icon(icon).classes('text-gray-400 text-base')
            ui.label(label).classes('text-gray-400 text-xs')
        ui.label(value).classes(f'{color} font-bold text-base font-mono')
