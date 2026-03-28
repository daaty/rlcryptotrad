"""
Tab Engine — NiceGUI version.
Controle da engine: Start/Stop, kill switch, log, sinais LSTM, ordens.
"""
from __future__ import annotations
from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider, colored_pnl, pnl_color


def render_engine_tab(state: LiveState) -> None:
    """Chamado dentro de um ui.tab_panel. Usa @ui.refreshable para live update."""
    _engine_panel(state)
    # Painel estático (renderizado uma vez) logo abaixo do refreshável
    _render_risk_config_ng()


# ── Painel SL/TP/Risk Config ─────────────────────────────────────────────────

def _render_risk_config_ng() -> None:
    """Editor inline de parâmetros de risco — grava config.yaml atomicamente."""
    import os as _os
    import yaml as _yaml
    from pathlib import Path as _Path
    from dashboard.resources_ng import reload_config as _reload_config

    _CONFIG = _Path('config.yaml')

    def _load() -> dict:
        try:
            return _yaml.safe_load(_CONFIG.read_text(encoding='utf-8')) or {}
        except Exception:
            return {}

    cfg = _load()
    rm  = cfg.get('risk_management', {})
    env = cfg.get('environment', {})

    with ui.expansion('⚙️ Parâmetros de Risco / SL · TP', icon='tune').classes(
            'w-full bg-gray-800 border border-gray-700 rounded-lg mt-4 text-gray-200'):

        # ── Modo de operação ─────────────────────────────────────────────
        with ui.row().classes('items-center gap-6 mt-2 flex-wrap'):
            with ui.column().classes('gap-1'):
                ui.label('Modo de Operação').classes('text-gray-400 text-xs')
                mode_sel = ui.select(
                    options=['paper', 'testnet', 'live'],
                    value=cfg.get('mode', 'testnet'),
                ).props('outlined dense dark').classes('w-32')

            with ui.column().classes('gap-1'):
                ui.label('Position Size (%)').classes('text-gray-400 text-xs')
                pos_input = ui.number(
                    value=round(float(env.get('position_size', 0.03)) * 100, 1),
                    min=0.5, max=10.0, step=0.5, format='%.1f',
                ).props('outlined dense dark suffix="%"').classes('w-32')

        ui.separator().classes('my-2')

        # ── SL / TP ──────────────────────────────────────────────────────
        with ui.row().classes('items-center gap-6 flex-wrap'):
            with ui.column().classes('gap-1'):
                ui.label('Stop Loss (%)').classes('text-gray-400 text-xs')
                sl_input = ui.number(
                    value=round(float(rm.get('stop_loss_pct', 0.02)) * 100, 1),
                    min=0.5, max=5.0, step=0.1, format='%.1f',
                ).props('outlined dense dark suffix="%"').classes('w-32')

            with ui.column().classes('gap-1'):
                ui.label('Take Profit (%)').classes('text-gray-400 text-xs')
                tp_input = ui.number(
                    value=round(float(rm.get('take_profit_pct', 0.04)) * 100, 1),
                    min=1.0, max=10.0, step=0.5, format='%.1f',
                ).props('outlined dense dark suffix="%"').classes('w-32')

            with ui.column().classes('gap-1'):
                ui.label('Trail Activation (%)').classes('text-gray-400 text-xs')
                ta_input = ui.number(
                    value=round(float(rm.get('trailing_stop_activation', 0.03)) * 100, 2),
                    min=0.5, max=5.0, step=0.25, format='%.2f',
                ).props('outlined dense dark suffix="%"').classes('w-36')

            with ui.column().classes('gap-1'):
                ui.label('Trail Distance (%)').classes('text-gray-400 text-xs')
                td_input = ui.number(
                    value=round(float(rm.get('trailing_stop_distance', 0.015)) * 100, 2),
                    min=0.25, max=3.0, step=0.25, format='%.2f',
                ).props('outlined dense dark suffix="%"').classes('w-36')

        ui.separator().classes('my-2')

        # ── Exposição ─────────────────────────────────────────────────────
        with ui.row().classes('items-center gap-6 flex-wrap'):
            with ui.column().classes('gap-1'):
                ui.label('Exposição Total Máx (%)').classes('text-gray-400 text-xs')
                mte_input = ui.number(
                    value=round(float(rm.get('max_total_exposure', 0.60)) * 100),
                    min=10, max=100, step=5, format='%.0f',
                ).props('outlined dense dark suffix="%"').classes('w-36')

            with ui.column().classes('gap-1'):
                ui.label('Exposição Máx/Ativo (%)').classes('text-gray-400 text-xs')
                mpa_input = ui.number(
                    value=round(float(rm.get('max_exposure_per_asset', 0.25)) * 100),
                    min=5, max=50, step=5, format='%.0f',
                ).props('outlined dense dark suffix="%"').classes('w-36')

        ui.label('TP1 = SL/2 · TP2 = TP total · SL ATR = 2×ATR (quando disponível)'
                 ).classes('text-gray-500 text-xs mt-2')

        # ── Salvar ────────────────────────────────────────────────────────
        def _save() -> None:
            sl  = (sl_input.value  or 0) / 100
            tp  = (tp_input.value  or 0) / 100
            ta  = (ta_input.value  or 0) / 100
            td  = (td_input.value  or 0) / 100
            ps  = (pos_input.value or 0) / 100
            mte = (mte_input.value or 0) / 100
            mpa = (mpa_input.value or 0) / 100
            errs: list[str] = []
            if sl < 0.005:
                errs.append('SL mínimo: 0.5%')
            if tp <= sl:
                errs.append('TP deve ser maior que SL')
            if ta < sl:
                errs.append('Trail Activation deve ser >= SL')
            if mte > 1.0:
                errs.append('Exposição total não pode exceder 100%')
            if errs:
                for e in errs:
                    ui.notify(e, type='negative')
                return
            fresh = _load()
            fresh['mode'] = mode_sel.value
            fresh.setdefault('environment', {})['position_size'] = round(ps, 4)
            fresh.setdefault('risk_management', {}).update({
                'stop_loss_pct'            : round(sl, 4),
                'take_profit_pct'          : round(tp, 4),
                'trailing_stop_activation' : round(ta, 4),
                'trailing_stop_distance'   : round(td, 4),
                'max_total_exposure'       : round(mte, 4),
                'max_exposure_per_asset'   : round(mpa, 4),
            })
            tmp = str(_CONFIG) + '.tmp'
            try:
                _Path(tmp).write_text(
                    _yaml.dump(fresh, allow_unicode=True, default_flow_style=False,
                               sort_keys=False),
                    encoding='utf-8')
                _os.replace(tmp, str(_CONFIG))
                _reload_config()
                ui.notify(
                    f'✅ Salvo — SL {sl:.1%} · TP {tp:.1%} · Trail {ta:.1%}/{td:.1%}',
                    type='positive')
            except Exception as exc:
                ui.notify(f'Erro ao salvar: {exc}', type='negative')

        ui.button('💾 Salvar Configuração', on_click=_save
                  ).props('color=primary').classes('mt-3')


@ui.refreshable
def _engine_panel(state: LiveState) -> None:
    # ── Kill Switch Banner ────────────────────────────────────────────────
    if state.kill_switch:
        with ui.card().classes('w-full bg-red-900 border border-red-500 p-4 mb-4'):
            ui.label('🛑 KILL SWITCH ACIONADO').classes('text-red-300 text-xl font-bold')
            ui.label(state.ks_reason).classes('text-red-200 text-sm mt-1 font-mono')
            ui.label('Engine parada. Verifique a conta antes de reiniciar.').classes('text-red-300 text-sm mt-2')

            async def ack_kill_switch():
                from dashboard.resources_ng import get_trading_engine
                eng = get_trading_engine()
                eng._kill_switch_triggered = False
                with eng.lock:
                    eng.state['kill_switch_triggered'] = False
                    eng.state['kill_switch_reason']    = ''
                    eng.state['peak_equity']           = 0.0
                    eng.state['current_drawdown_pct']  = 0.0
                eng._peak_equity = 0.0
                ui.notify('Kill Switch reconhecido.', type='positive')
                _engine_panel.refresh(state)

            ui.button('⚡ Reconhecer Kill Switch', on_click=ack_kill_switch).classes(
                'bg-red-700 hover:bg-red-600 text-white mt-3')

    # ── Drawdown indicator ────────────────────────────────────────────────
    elif state.peak_equity > 0 and state.drawdown_pct > 0:
        dd_pct = state.drawdown_pct
        color  = 'bg-red-900' if dd_pct > 0.10 else ('bg-yellow-900' if dd_pct > 0.05 else 'bg-green-900')
        icon   = '🔴' if dd_pct > 0.10 else ('🟡' if dd_pct > 0.05 else '🟢')
        with ui.card().classes(f'w-full {color} border border-gray-600 p-2 mb-3'):
            ui.label(
                f'{icon}  Drawdown atual: {dd_pct:.1%}  |  Pico: ${state.peak_equity:,.2f}'
            ).classes('text-sm font-mono')

    # ── Controles ─────────────────────────────────────────────────────────
    with ui.row().classes('items-center gap-4 mb-4'):
        status_color = 'bg-green-700' if state.engine_running else 'bg-gray-600'
        ui.badge(state.engine_label).classes(f'{status_color} text-white text-sm px-3 py-1 rounded-full')
        if state.last_tick_str != '—':
            ui.label(f'Último tick: {state.last_tick_str}').classes('text-gray-400 text-xs')

    with ui.row().classes('gap-3 mb-6'):
        async def start_engine():
            from dashboard.resources_ng import get_trading_engine
            from dashboard.ui_ng.symbol_selector import get_selected_symbols
            eng  = get_trading_engine()
            syms = get_selected_symbols()
            if not syms:
                ui.notify('Selecione pelo menos um símbolo.', type='warning')
                return
            eng.start(syms)
            ui.notify('Engine iniciada.', type='positive')
            _engine_panel.refresh(state)

        async def stop_engine():
            from dashboard.resources_ng import get_trading_engine
            get_trading_engine().stop()
            ui.notify('Engine parada.', type='warning')
            _engine_panel.refresh(state)

        ui.button('▶ Iniciar Engine', on_click=start_engine).props(
            'color=positive').classes('text-sm')
        ui.button('⏹ Parar Engine', on_click=stop_engine).props(
            'color=negative').classes('text-sm')

    divider()

    # ── Sinais LSTM ───────────────────────────────────────────────────────
    section_title('🧠 Último Sinal LSTM')
    if not state.decisions:
        ui.label('Aguardando decisões...').classes('text-gray-500 text-sm')
    else:
        columns = [
            {'name': 'sym',    'label': 'Símbolo',   'field': 'sym',    'align': 'left'},
            {'name': 'action', 'label': 'Sinal',     'field': 'action', 'align': 'center'},
            {'name': 'conf',   'label': 'Conf',      'field': 'conf',   'align': 'right'},
            {'name': 'ts',     'label': 'Timestamp', 'field': 'ts',     'align': 'right'},
        ]
        rows = []
        for sym, dec in state.decisions.items():
            if not isinstance(dec, dict):
                continue
            action = str(dec.get('action', '—')).upper()
            rows.append({
                'sym'   : sym,
                'action': action,
                'conf'  : f"{float(dec.get('confidence', 0)):.2f}" if 'confidence' in dec else '—',
                'ts'    : str(dec.get('ts', '—'))[:8],
            })
        ui.aggrid({
            'columnDefs'   : columns,
            'rowData'      : rows,
            'domLayout'    : 'autoHeight',
            'defaultColDef': {'flex': 1, 'resizable': True, 'sortable': True, 'minWidth': 60},
        }).classes('w-full ag-theme-alpine-dark text-sm').style('max-height:250px')

    divider()

    # ── Ordens recentes ───────────────────────────────────────────────────
    section_title('📋 Ordens Recentes')
    orders = list(state.orders)[-20:]
    if not orders:
        ui.label('Nenhuma ordem registrada.').classes('text-gray-500 text-sm')
    else:
        cols = [
            {'name': 'ts',     'label': 'Hora',    'field': 'ts'},
            {'name': 'sym',    'label': 'Símbolo', 'field': 'sym'},
            {'name': 'side',   'label': 'Lado',    'field': 'side'},
            {'name': 'qty',    'label': 'Qty',     'field': 'qty'},
            {'name': 'price',  'label': 'Preço',   'field': 'price'},
            {'name': 'action', 'label': 'Ação',    'field': 'action'},
        ]
        rows = []
        for o in reversed(orders):
            rows.append({
                'ts'    : str(o.get('ts', ''))[:8],
                'sym'   : str(o.get('symbol', o.get('sym', '—'))),
                'side'  : str(o.get('side', '—')),
                'qty'   : str(o.get('qty', '—')),
                'price' : f"${float(o.get('price', 0)):,.4f}" if o.get('price') else '—',
                'action': str(o.get('action', '—')),
            })
        ui.aggrid({
            'columnDefs'   : cols,
            'rowData'      : rows,
            'domLayout'    : 'autoHeight',
            'defaultColDef': {'flex': 1, 'resizable': True, 'sortable': True, 'minWidth': 60},
        }).classes('w-full ag-theme-alpine-dark text-sm').style('max-height:300px')

    divider()

    # ── Log da engine ─────────────────────────────────────────────────────
    section_title('📝 Log')
    log_text = '\n'.join(reversed(state.log_lines[-80:]))
    ui.textarea(value=log_text).props(
        'readonly outlined dense dark'
    ).classes('w-full font-mono text-xs bg-gray-900').props(
        "style='height:300px; resize:vertical'"
    )

    # ── Erros ─────────────────────────────────────────────────────────────
    if state.errors:
        divider()
        section_title(f'⚠️ Erros ({len(state.errors)})')
        for err in list(state.errors)[-5:]:
            ui.label(str(err)).classes('text-red-400 text-xs font-mono')
