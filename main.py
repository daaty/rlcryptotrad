"""
main.py — NiceGUI Dashboard (porta 8080)
========================================
Entry-point para o dashboard migrado de Streamlit → NiceGUI.

Uso:
    python main.py
    python main.py --port 8080
    python main.py --reload   # dev mode

O dashboard antigo continua disponível em:
    streamlit run dashboard_new.py   (porta 8501)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

# Forçar backend Agg do matplotlib antes de importar nicegui.
# Evita inicialização de backends de display (Tk/Qt) que são lentos
# e causam KeyboardInterrupt em ambientes sem display.
os.environ.setdefault('MPLBACKEND', 'Agg')

# Suprimir FutureWarning do torch.load (stable_baselines3 — modelos próprios, sem risco)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', message='.*weights_only.*', category=FutureWarning)

# ── Ensure project root is in sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nicegui import app, ui

from dashboard.core.logging_setup import setup_logging
from dashboard.resources_ng import (
    get_config,
    get_binance_client,
    get_ws_manager,
    get_trading_engine,
    get_models,
)
from dashboard.state_ng import get_live_state, LiveState

# Import renderers (lazy-style references kept at top for IDE clarity)
from dashboard.ui_ng.tab_overview_ng    import render_overview_tab,    _overview_panel, _pnl_chart_panel
from dashboard.ui_ng.tab_positions_ng   import render_positions_tab,   _positions_panel
from dashboard.ui_ng.tab_performance_ng import render_performance_tab,  _performance_panel
from dashboard.ui_ng.tab_analysis_ng    import render_analysis_tab,     _analysis_panel
from dashboard.ui_ng.tab_engine_ng      import render_engine_tab,       _engine_panel
from dashboard.ui_ng.tab_challenger_ng  import render_challenger_tab,   _challenger_panel
from dashboard.ui_ng.symbol_selector    import ALL_SYMBOLS, get_selected_symbols, set_selected_symbols

logger = setup_logging()

# ── Estilo global customizado (NiceGUI) ───────────────────────────────────────
ui.add_head_html('''
<style>
    body { background: #0b1220; color: #e4e9f1; }
    .title-text { font-family: 'Inter', sans-serif; color: #5ab1f8; }
    .header-row { background: linear-gradient(90deg, #0f172a, #1b2b44); border-radius: 0.8rem; border: 1px solid #2f475e; padding: 0.8rem; }
    .metric-card { background: #131f35; border: 1px solid #2a5174; border-radius: 0.75rem; box-shadow: 0 8px 22px rgba(0,0,0,0.25); }
    .sidebar-card { background: #0f182a; border: 1px solid #2f445f; border-radius: 0.75rem; }
    .nicegui-tabs button { border-top-left-radius: 0.65rem !important; border-top-right-radius: 0.65rem !important; }
    .nicegui-tabs .active { background: #1f426e !important; color: #ffffff !important; }
</style>
''', shared=True)

# ── Startup hook ──────────────────────────────────────────────────────────────

@app.on_startup
async def _startup() -> None:
    """
    Inicializa singletons em uma thread de pool FORA do event loop NiceGUI.
    O ThreadedWebsocketManager da Binance chama loop.run_until_complete() internamente;
    isso conflita se chamado de dentro do event loop asyncio — por isso usamos run_in_executor.
    """
    def _blocking_init() -> None:
        cfg = get_config()
        get_binance_client()
        ws = get_ws_manager()        # __init__ já inicia o WebSocket + TWM
        get_trading_engine()
        get_models()            # pré-carrega LSTM
        pairs: list[str] = (
            cfg.get('trading', {}).get('symbols')
            or cfg.get('data', {}).get('symbols')
            or ['BTCUSDT', 'ETHUSDT']
        )
        syms = [p.replace('/', '').upper() for p in pairs]
        set_selected_symbols(syms)

        # Bootstrap klines (cache em disco → REST só para o delta)
        # Usa ALL_SYMBOLS para garantir que todos os pares selecionáveis
        # tenham buffers prontos antes do engine rodar inferência.
        if not ws.bootstrap_done:
            try:
                n = ws.bootstrap_klines(list(ALL_SYMBOLS))
                ws.bootstrap_account()
                logger.info('Bootstrap concluído: %d candles para %s', n, syms)
            except Exception as exc:
                logger.warning('Erro no bootstrap: %s (gráficos indisponíveis até WS popular buffer)', exc)

        logger.info('Startup completo. Pares: %s', pairs)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _blocking_init)
    except Exception as exc:
        logger.error('Erro no startup: %s', exc)


@app.on_shutdown
async def _shutdown() -> None:
    try:
        ws = get_ws_manager()
        if ws.running:
            ws.stop()
    except Exception:
        pass


# ── Header helper ─────────────────────────────────────────────────────────────

def _build_header(state: LiveState) -> None:
    """Header estático com sub-painel refreshável apenas para labels com cor dinâmica."""
    with ui.header().classes('header-row w-full mb-2'):
        with ui.row().classes('items-center justify-between w-full'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('smart_toy').classes('text-cyan-400 text-2xl')
                ui.label('Agente Trading').classes('title-text text-2xl font-bold')
                ui.label('Dashboard Profissional').classes('text-gray-300 text-sm ml-2')
        _header_metrics(state)


@ui.refreshable
def _header_metrics(state: LiveState) -> None:
    """Métricas do header — @ui.refreshable leve (sem Plotly), flicker imperceptível."""
    pnl_color = 'text-green-300' if state.total_pnl >= 0 else 'text-red-300'
    eng_color = 'text-lime-300' if state.engine_running else 'text-red-200'

    with ui.row().classes('items-center gap-4 flex-wrap'):
        with ui.card().classes('metric-card p-3'):
            ui.label('Saldo').classes('text-gray-400 text-xs uppercase').style('letter-spacing:0.07em')
            ui.label(state.balance_str).classes('text-white font-bold').style('font-size:1.05rem')
        with ui.card().classes('metric-card p-3'):
            ui.label('P&L').classes('text-gray-400 text-xs uppercase').style('letter-spacing:0.07em')
            ui.label(state.pnl_str).classes(f'{pnl_color} font-bold').style('font-size:1.05rem')
            ui.label(state.pnl_pct_str).classes('text-gray-400 text-xs mt-1')
        with ui.card().classes('metric-card p-3'):
            ui.label('Posições').classes('text-gray-400 text-xs uppercase').style('letter-spacing:0.07em')
            ui.label(str(state.n_positions)).classes('text-white font-bold').style('font-size:1.05rem')

        # ── Botão Start/Stop fixo no header ──────────────────────────────────
        async def _start_engine():
            from dashboard.resources_ng import get_trading_engine
            from dashboard.ui_ng.symbol_selector import get_selected_symbols
            eng  = get_trading_engine()
            syms = get_selected_symbols()
            if not syms:
                ui.notify('Selecione pelo menos um símbolo.', type='warning')
                return
            eng.start(syms)
            ui.notify('Engine iniciada.', type='positive')
            _header_metrics.refresh(state)

        async def _stop_engine():
            from dashboard.resources_ng import get_trading_engine
            get_trading_engine().stop()
            ui.notify('Engine parada.', type='warning')
            _header_metrics.refresh(state)

        if state.engine_running:
            ui.button('⏹ Parar Engine', on_click=_stop_engine).props(
                'color=negative').classes('text-sm self-center')
        else:
            ui.button('▶ Iniciar Engine', on_click=_start_engine).props(
                'color=positive').classes('text-sm self-center')

    with ui.row().classes('items-center gap-3 flex-wrap mt-1'):
        ui.label(state.engine_label).classes(f'{eng_color} font-bold text-sm')
        if state.banned:
            ui.label('🚫 BAN ATIVO').classes('text-red-400 font-bold text-sm')
        if state.kill_switch:
            ui.label(f'🛑 KS: {state.ks_reason[:40]}').classes('text-red-300 text-xs')

    with ui.row().classes('items-center gap-2 pt-1'):
        ui.label(f'Drawdown: {state.drawdown_pct:.2f}%').classes('text-yellow-300 text-xs')
        ui.label(f'Pico: ${state.peak_equity:,.2f}').classes('text-gray-400 text-xs')


# ── Sidebar helper ─────────────────────────────────────────────────────────────

def _build_sidebar(state: LiveState) -> None:
    with ui.left_drawer(value=True, top_corner=True, bottom_corner=True).classes(
            'bg-gray-900 border-r border-gray-700 p-4 w-56'):

        # ── Pares ─────────────────────────────────────────────────────────
        ui.label('Pares').classes('text-gray-400 text-xs font-semibold mb-2')
        selected = set(get_selected_symbols())

        async def on_toggle(sym: str, checked: bool) -> None:
            syms = set(get_selected_symbols())
            if checked:
                syms.add(sym)
            else:
                syms.discard(sym)
            set_selected_symbols(sorted(syms))

        for sym in ALL_SYMBOLS:
            chk = ui.checkbox(sym.replace('USDT', ''), value=sym in selected)
            chk.on_value_change(lambda e, s=sym: on_toggle(s, e.value))

        ui.separator().classes('my-3')

        # ── Entry Filter ─────────────────────────────────────────────────
        ui.label('🎛️ Filtro de Entrada').classes('text-gray-400 text-xs font-semibold mb-1')

        _FILTER_MODES = ['disabled', 'aggressive', 'normal', 'strict']
        _FILTER_LABELS = {
            'disabled'  : '🚫 Desativado',
            'aggressive': '⚡ Agressivo',
            'normal'    : '⚖️ Normal',
            'strict'    : '🛡️ Estrito',
        }
        _FILTER_DESC = {
            'disabled'  : 'Modelo opera livre',
            'aggressive': 'Só RSI extremo (85/15)',
            'normal'    : 'RSI 80/20 · Vol 30%',
            'strict'    : 'Todos os filtros ativos',
        }
        try:
            _cur = get_config().get('entry_filter', {}).get('mode', 'normal')
            if _cur not in _FILTER_MODES:
                _cur = 'normal'
        except Exception:
            _cur = 'normal'

        filter_sel  = ui.select(
            options={m: _FILTER_LABELS[m] for m in _FILTER_MODES},
            value=_cur,
        ).props('dense outlined dark').classes('w-full text-xs')

        _fdesc = ui.label(_FILTER_DESC.get(_cur, '')).classes('text-gray-500 text-xs mt-1')

        def _on_filter(e) -> None:
            mode = e.value
            _fdesc.set_text(_FILTER_DESC.get(mode, ''))
            try:
                import yaml as _y
                from pathlib import Path as _P
                from dashboard.resources_ng import reload_config as _rc
                _c = get_config()
                _c.setdefault('entry_filter', {})['mode'] = mode
                _p = _P('config.yaml')
                _p.write_text(_y.dump(_c, allow_unicode=True,
                                      default_flow_style=False), encoding='utf-8')
                _rc()
                ui.notify(f'Filtro: {_FILTER_LABELS[mode]}',
                          type='positive', position='top-right')
            except Exception as exc:
                ui.notify(f'Erro ao salvar filtro: {exc}', type='negative')

        filter_sel.on_value_change(_on_filter)

        ui.separator().classes('my-3')

        # ── Risk Status ──────────────────────────────────────────────────
        ui.label('🛡️ Risk Status').classes('text-gray-400 text-xs font-semibold mb-1')
        try:
            from dashboard.resources_ng import get_risk_manager, get_trailing_stop_manager
            _rm = get_risk_manager()
            _tm = get_trailing_stop_manager()
            _cfg = get_config()
            risk_cfg = _cfg.get('risk_management', {})
            env_cfg  = _cfg.get('environment', {})

            with ui.row().classes('gap-2 mb-2'):
                ui.label(f"Min Notional: ${risk_cfg.get('min_notional_usdt', 20.0):.2f}")
                ui.label(f"Kelly: {risk_cfg.get('kelly_fraction', 0.25)*100:.1f}%")
            with ui.row().classes('gap-2 mb-2'):
                ui.label(f"Max Drawdown: {risk_cfg.get('max_drawdown', 0.15)*100:.1f}%")
                ui.label(f"Exposição Total: {risk_cfg.get('max_total_exposure', 0.60)*100:.1f}%")
                ui.label(f"Leverage: {env_cfg.get('leverage', 1.5):.1f}x")

            _can, _reason = _rm.should_allow_trade()
            if _can:
                ui.label('✅ Trading Ativo').classes('text-green-400 text-xs')
            else:
                ui.label(f'⛔ {str(_reason)[:28]}').classes('text-red-400 text-xs')
                def _reset_cb(_r=_rm):
                    _r.reset_circuit_breaker()
                    ui.notify('Circuit breaker resetado!', type='positive')
                ui.button('🔄 Reset CB', on_click=_reset_cb
                          ).props('dense flat size=xs color=warning').classes('mt-1')
            if hasattr(_tm, 'active_stops') and _tm.active_stops:
                ui.label(f'🎯 {len(_tm.active_stops)} trailing ativos'
                         ).classes('text-blue-400 text-xs')
        except Exception:
            ui.label('—').classes('text-gray-500 text-xs')

        ui.separator().classes('my-3')

        # ── Status WS ────────────────────────────────────────────────────
        ui.label('Status WS').classes('text-gray-400 text-xs font-semibold mb-1')
        ui.label().bind_text_from(
            state, 'ws_connected',
            backward=lambda v: '🟢 Online' if v else '🔴 Offline'
        ).classes('text-sm')

        ui.separator().classes('my-3')
        ui.label().bind_text_from(state, 'last_tick_str').classes('text-gray-500 text-xs')

        # ── Visual settings (Executive mode) ─────────────────────────────────
        ui.label('🔧 Modo de Visualização').classes('text-gray-400 text-xs font-semibold mb-1')
        view_select = ui.select(['Detalhado', 'Compacto'], value=state.display_mode).props('dense outlined dark').classes('w-full')

        def _on_view_change(e):
            state.display_mode = e.value
            ui.notify(f'Modo de visão: {e.value}', type='positive')

        view_select.on('value_changed', _on_view_change)

        ui.label('🎨 Tema').classes('text-gray-400 text-xs font-semibold mt-3 mb-1')
        theme_select = ui.select(['dark', 'light'], value=state.theme).props('dense outlined dark').classes('w-full')

        def _on_theme_change(e):
            state.theme = e.value
            if e.value == 'dark':
                ui.dark_mode().enable()
            else:
                ui.dark_mode().disable()
            ui.notify(f'Tema alterado para: {e.value}', type='positive')

        theme_select.on('value_changed', _on_theme_change)

        ui.label('✨ Accent').classes('text-gray-400 text-xs font-semibold mt-3 mb-1')
        accent_select = ui.select(['cyan', 'green', 'amber', 'purple'], value=state.accent).props('dense outlined dark').classes('w-full')

        def _on_accent_change(e):
            state.accent = e.value
            # simples: muda cor do header e métricas via classes predefinidas
            ui.notify(f'Accent: {e.value}', type='positive')

        accent_select.on('value_changed', _on_accent_change)


# ── Main page ─────────────────────────────────────────────────────────────────

@ui.page('/')
async def dashboard() -> None:
    state = get_live_state()

    ui.dark_mode().enable()

    _build_header(state)
    _build_sidebar(state)

    # ── Global refresh: atualiza estado e todos os painéis ──────────────────
    # Rastreamento para refresh diferencial (evita re-render desnecessário)
    _prev = {'n_trades': -1, 'tick': 0}

    def _refresh_all() -> None:
        try:
            state.refresh()
        except Exception as exc:
            logger.debug('Falha ao atualizar estado: %s', exc)
            return

        _prev['tick'] += 1
        n_trades = state.n_trades

        try:
            # Header + painéis leves (sem Plotly): refresh a cada tick
            _header_metrics.refresh(state)
            _overview_panel.refresh(state)
            _positions_panel.refresh(state)

            # Performance + PnL chart: apenas quando há novos trades (contém Plotly)
            if n_trades != _prev['n_trades']:
                _performance_panel.refresh(state)
                _pnl_chart_panel.refresh(state)
                _prev['n_trades'] = n_trades

            # Engine: a cada 2 ticks (~4 s) — contém log textarea e ordens
            if _prev['tick'] % 2 == 0:
                _engine_panel.refresh(state)

            # Painéis quase estáticos: a cada 5 ticks (~10 s)
            if _prev['tick'] % 5 == 0:
                _analysis_panel.refresh(state)
                _challenger_panel.refresh(state)

        except Exception as exc:
            logger.debug('Falha ao atualizar UI: %s', exc)

    ui.timer(2.0, _refresh_all)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_labels = [
        ('overview',     '📊 Visão Geral'),
        ('positions',    '💼 Posições'),
        ('performance',  '📈 Desempenho'),
        ('analysis',     '🔬 Análise'),
        ('engine',       '⚙️ Engine'),
        ('champion',     '🏆 Champion'),
    ]

    with ui.tabs().classes(
            'bg-gray-900 border-b border-gray-700 sticky top-14 z-10') as tabs:
        for key, label in tab_labels:
            ui.tab(key, label=label).classes('text-gray-300 hover:text-white')

    with ui.tab_panels(tabs, value='overview').classes('flex-1 bg-gray-950 p-4'):

        with ui.tab_panel('overview'):
            render_overview_tab(state)

        with ui.tab_panel('positions'):
            render_positions_tab(state)

        with ui.tab_panel('performance'):
            render_performance_tab(state)

        with ui.tab_panel('analysis'):
            render_analysis_tab(state)

        with ui.tab_panel('engine'):
            render_engine_tab(state)

        with ui.tab_panel('champion'):
            render_challenger_tab(state)

# ── Chrome DevTools probe (silencia 404 no log) ──────────────────────────────

@app.get('/.well-known/appspecific/com.chrome.devtools.json')
async def _chrome_devtools() -> dict:
    return {}

# ── Health-check endpoint ─────────────────────────────────────────────────────

@app.get('/health')
async def health() -> dict:
    try:
        eng = get_trading_engine()
        state = get_live_state()
        return {
            'ok'            : True,
            'engine_running': state.engine_running,
            'ws_connected'  : state.ws_connected,
            'n_positions'   : state.n_positions,
        }
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NiceGUI Trading Dashboard')
    parser.add_argument('--port',   type=int,  default=8080, help='Porta HTTP (default: 8080)')
    parser.add_argument('--host',   type=str,  default='0.0.0.0')
    parser.add_argument('--reload', action='store_true', help='Dev mode com auto-reload')
    parser.add_argument('--no-dark', action='store_true', help='Desabilitar modo escuro')
    args = parser.parse_args()

    ui.run(
        host         = args.host,
        port         = args.port,
        title        = '🤖 Agente Trading',
        favicon      = '📊',
        dark         = not args.no_dark,
        reload       = args.reload,
        show         = False,       # não abrir browser automaticamente
        storage_secret = 'agente_trading_secret_change_in_prod',
    )
