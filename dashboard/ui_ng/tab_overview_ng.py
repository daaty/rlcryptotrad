"""
Tab Visão Geral — NiceGUI version.
Métricas globais de conta, posições abertas, sinais e gráficos candlestick.
"""
from __future__ import annotations
from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider, pnl_color
from dashboard.ui_ng.tab_performance_ng import _render_pnl_chart


def render_overview_tab(state: LiveState) -> None:
    _overview_panel(state)
    _pnl_chart_panel(state)       # refreshado apenas quando trades mudam (tem Plotly)
    _charts_static(state)         # candlestick — renderizado uma vez, sem flicker


@ui.refreshable
def _overview_panel(state: LiveState) -> None:
    # ── Modo compacto x detalhado
    if getattr(state, 'display_mode', 'Detalhado') == 'Compacto':
        with ui.row().classes('gap-4 flex-wrap mb-4'):
            _metric('💰 Saldo Total', state.balance_str, f'Disponível: {state.available_str}')
            _metric('📈 P&L Não-Real.', state.unrealized_pnl_str, f'({state.unrealized_pnl_pct:.2f}%)')
            _metric('💼 Posições', str(state.n_positions), '')
        return

    # ── Métricas de conta ─────────────────────────────────────────────────
    with ui.row().classes('gap-4 flex-wrap mb-4'):
        _metric('💰 Saldo Total',
                state.balance_str,
                f'Disponível: {state.available_str}')

        # P&L Não-Realizado = soma das posições abertas agora
        u_sign = '+' if state.unrealized_pnl >= 0 else ''
        _metric('📊 P&L Não-Real.',
                state.unrealized_pnl_str,
                f'({u_sign}{state.unrealized_pnl_pct:.2f}%)',
                sub_label='posições abertas',
                value_css=pnl_color(state.unrealized_pnl))

        # P&L Conta = saldo atual vs capital inicial configurado
        _metric('🏦 P&L vs Inicial',
                state.account_pnl_str,
                state.account_pnl_pct_str,
                sub_label=f'capital: ${state.initial_balance:,.0f}',
                value_css=pnl_color(state.account_pnl))

        _metric('📈 Posições',  str(state.n_positions), '', sub_label='abertas')
        _metric('🏆 Trades',    str(state.n_trades),
                f'WR {state.win_rate:.1f}%' if state.n_trades else '—')
        _metric('⚙️ Engine',   state.engine_label, state.last_tick_str)

    # ── Ban alert ─────────────────────────────────────────────────────────
    if state.banned:
        with ui.card().classes('w-full bg-red-900 border border-red-500 p-3 mb-4'):
            ui.label(state.ban_label).classes('text-red-300 font-bold')

    divider()

    # ── Posições abertas ──────────────────────────────────────────────────
    section_title('💼 Posições Abertas')
    if not state.open_positions:
        ui.label('Nenhuma posição aberta.').classes('text-gray-500 text-sm')
    else:
        _positions_table(state.open_positions)

    divider()

    # ── Últimos sinais LSTM ───────────────────────────────────────────────
    section_title('🧠 Últimos Sinais')
    if not state.decisions:
        ui.label('Aguardando sinais da engine...').classes('text-gray-500 text-sm')
    else:
        with ui.row().classes('flex-wrap gap-3'):
            for sym, dec in list(state.decisions.items())[:12]:
                if not isinstance(dec, dict):
                    continue
                action = str(dec.get('action', '—')).upper()
                color = 'bg-green-800' if action == 'LONG' else (
                        'bg-red-800' if action == 'SHORT' else 'bg-gray-700')
                with ui.card().classes(f'{color} p-2 rounded text-center w-28'):
                    ui.label(sym.replace('USDT', '')).classes('text-white text-xs font-bold')
                    ui.label(action).classes('text-white text-sm font-mono')


@ui.refreshable
def _pnl_chart_panel(state: LiveState) -> None:
    """Gráfico PnL acumulado — refreshado apenas quando novos trades chegam (tem Plotly)."""
    divider()
    section_title('📉 PnL Acumulado (histórico de trades)')
    try:
        _render_pnl_chart(state.closed_trades)
    except Exception:
        ui.label('Sem trades fechados ainda.').classes('text-gray-500 text-sm')


def _charts_static(state: LiveState) -> None:
    """Seção de gráficos renderizada UMA vez — fora do @ui.refreshable."""
    divider()
    section_title('📈 Gráficos em Tempo Real')
    _render_charts_section(state)


def _metric(title: str, value: str, sub: str,
            sub_label: str = '', value_css: str = 'text-white') -> None:
    with ui.card().classes('bg-gray-800 border border-gray-700 p-3 min-w-36 rounded-lg'):
        ui.label(title).classes('text-gray-400 text-xs uppercase tracking-wide mb-1')
        ui.label(value).classes(f'{value_css} text-xl font-bold font-mono')
        if sub:
            ui.label(sub).classes('text-gray-400 text-xs mt-0.5')
        if sub_label:
            ui.label(sub_label).classes('text-gray-500 text-xs')


def _positions_table(positions: list[dict]) -> None:
    col_defs = [
        {'headerName': 'Símbolo', 'field': 'sym',     'width': 110},
        {'headerName': 'Lado',    'field': 'side',    'width': 80,
         'cellClassRules': {'text-green-400': "x === 'LONG'", 'text-red-400': "x === 'SHORT'"}},
        {'headerName': 'Qty',     'field': 'amt',     'width': 100},
        {'headerName': 'Entrada', 'field': 'entry',   'width': 120},
        {'headerName': 'Mark',    'field': 'mark',    'width': 120},
        {'headerName': 'P&L',     'field': 'pnl',     'width': 120,
         'cellClassRules': {'text-green-400': "x.startsWith('+')", 'text-red-400': "x.startsWith('-')"}},
        {'headerName': '%',       'field': 'pnl_pct', 'width': 90},
    ]
    rows = []
    for p in positions:
        amt     = float(p.get('positionAmt', 0))
        entry   = float(p.get('entryPrice', 0))
        mark    = float(p.get('markPrice', 0))
        upnl    = float(p.get('unRealizedProfit', 0))
        pnl_pct = ((mark - entry) / entry * 100 * (1 if amt > 0 else -1)) if entry else 0
        rows.append({
            'sym'    : p.get('symbol', '—').replace('USDT', ''),
            'side'   : 'LONG' if amt > 0 else 'SHORT',
            'amt'    : f'{abs(amt):.4f}',
            'entry'  : f'${entry:,.4f}',
            'mark'   : f'${mark:,.4f}',
            'pnl'    : f'{"+"if upnl>=0 else ""}${upnl:,.2f}',
            'pnl_pct': f'{"+"if pnl_pct>=0 else ""}{pnl_pct:.2f}%',
        })
    ui.aggrid({
        'columnDefs': col_defs,
        'rowData'   : rows,
        'domLayout' : 'autoHeight',
    }).classes('w-full ag-theme-alpine-dark text-sm')


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick Charts
# ─────────────────────────────────────────────────────────────────────────────

_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']
_selected_tf   = '15m'
_selected_limit = 100


def _render_charts_section(state: LiveState) -> None:
    """Gráficos candlestick — estrutura correta: um tab_panels com todos os panels."""
    from dashboard.ui_ng.symbol_selector import get_selected_symbols
    from dashboard.resources_ng import get_ws_manager

    syms = get_selected_symbols() or state.symbols
    if not syms:
        ui.label('Selecione símbolos no painel lateral.').classes('text-gray-500 text-sm')
        return

    ws = get_ws_manager()

    # Controles
    with ui.row().classes('items-center gap-4 mb-3'):
        ui.label('Timeframe:').classes('text-gray-400 text-sm')
        tf_sel = ui.toggle(_TIMEFRAMES, value='15m').props('dense').classes('text-xs')
        ui.label('Candles:').classes('text-gray-400 text-sm ml-4')
        lim_sel = ui.select(options=[50, 100, 200, 300], value=100).props('outlined dense dark').classes('w-24')

    # UMA barra de abas
    first_sym = syms[0].replace('USDT', '')
    with ui.tabs(value=first_sym).classes('w-full bg-gray-800 rounded') as sym_tabs:
        for sym in syms[:8]:
            ui.tab(sym.replace('USDT', ''), icon='show_chart')

    # UM ui.tab_panels com todos os panels DENTRO
    with ui.tab_panels(sym_tabs, value=first_sym).classes('w-full'):
        for sym in syms[:8]:
            sym_key = sym.replace('USDT', '')
            sym_upper = sym.upper().replace('/', '')
            with ui.tab_panel(sym_key):
                chart_col = ui.column().classes('w-full')

                def _draw(col=chart_col, s=sym_upper):
                    from dashboard.state_ng import get_live_state
                    col.clear()
                    with col:
                        _render_single_chart(ws, s, tf_sel, lim_sel,
                                             get_live_state().open_positions)

                _draw()

                # Redesenha ao mudar controles
                tf_sel.on_value_change(lambda _e, d=_draw: d())
                lim_sel.on_value_change(lambda _e, d=_draw: d())


def _render_single_chart(ws, sym: str, tf_sel, lim_sel, positions: list[dict]) -> None:
    """Renderiza o gráfico para um único símbolo."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    try:
        tf    = tf_sel.value if hasattr(tf_sel, 'value') else '15m'
        limit = lim_sel.value if hasattr(lim_sel, 'value') else 100
        df = ws.get_klines_df(sym.replace('/', '').upper(), tf, limit=int(limit))
        if df is None or df.empty or len(df) < 5:
            with ui.column().classes('items-center gap-3 py-8 w-full'):
                ui.label(f'⏳ Buffer vazio para {sym} [{tf}]').classes('text-gray-400 text-sm')
                ui.label('O WebSocket vai popular automaticamente. '
                         'Clique em Carregar para forçar via REST.').classes('text-gray-500 text-xs')

                def _bootstrap_sym(s=sym):
                    from dashboard.resources_ng import get_ws_manager as _gws
                    try:
                        n = _gws().bootstrap_klines([s])
                        ui.notify(f'✅ {n} candles carregados para {s}', type='positive')
                    except Exception as e:
                        ui.notify(f'Erro: {e}', type='negative')

                ui.button(f'📥 Carregar {sym}', on_click=_bootstrap_sym).props('color=primary outlined')
            return

        # Garante coluna timestamp
        if 'timestamp' not in df.columns and df.index.dtype.kind == 'M':
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        elif 'timestamp' not in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df.index)

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name=sym,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ), row=1, col=1)

        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
        if 'volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df['timestamp'], y=df['volume'],
                name='Volume', marker_color=colors, showlegend=False,
            ), row=2, col=1)

        # Marca posições abertas
        sym_pos = [p for p in positions
                   if p.get('symbol', '').upper() == sym.replace('/', '').upper()]
        for pos in sym_pos:
            ep = float(pos.get('entryPrice', 0))
            if ep > 0:
                side_color = '#26a69a' if float(pos.get('positionAmt', 0)) > 0 else '#ef5350'
                fig.add_hline(y=ep, line_dash='dash', line_color=side_color,
                              annotation_text=f"Entry {ep:,.4f}", row=1, col=1)

        fig.update_layout(
            height=460,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(l=8, r=8, t=20, b=8),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
        )
        fig.update_xaxes(showgrid=True, gridcolor='#2d2d44')
        fig.update_yaxes(showgrid=True, gridcolor='#2d2d44', tickprefix='$', row=1, col=1)

        ui.plotly(fig).classes('w-full')

    except Exception as exc:
        ui.label(f'Erro ao gerar gráfico {sym}: {exc}').classes('text-orange-400 text-xs')
