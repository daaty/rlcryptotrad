"""
Tab Análise — NiceGUI version.
Regime de mercado, Drift Detection (Z-score) e status de Warmup.
"""
from __future__ import annotations

from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider


def render_analysis_tab(state: LiveState) -> None:
    _analysis_panel(state)


@ui.refreshable
def _analysis_panel(state: LiveState) -> None:
    section_title('🔬 Análise de Mercado')

    # ── Warmup ───────────────────────────────────────────────────────────────
    _render_warmup_section()

    divider()

    # ── Schedule (janelas de operação) ───────────────────────────────────────
    _render_schedule_section()

    divider()

    # ── Regime por símbolo ───────────────────────────────────────────────────
    primary = _pick_primary_symbol(state)
    _render_regime_section(primary)

    divider()

    # ── Multi-TF ─────────────────────────────────────────────────────────────
    _render_multi_tf_section(primary, state.symbols)

    divider()

    # ── Correlação ────────────────────────────────────────────────────────────
    from dashboard.ui_ng.symbol_selector import get_selected_symbols
    syms = get_selected_symbols() or state.symbols
    if len(syms) >= 2:
        _render_correlation_section(syms)
        divider()

    # ── Simulador Position Sizing ─────────────────────────────────────────────
    _render_sizing_simulator()

    divider()

    # ── Drift Detection ──────────────────────────────────────────────────────
    _render_drift_section(primary)


def _pick_primary_symbol(state: LiveState) -> str:
    """Retorna o primeiro símbolo que tem klines disponíveis."""
    candidates = state.symbols or ['BTCUSDT']
    try:
        from dashboard.resources_ng import get_ws_manager
        ws = get_ws_manager()
        for sym in candidates:
            key = sym.replace('/', '').upper()
            df = ws.get_klines_df(key, '15m', limit=5)
            if df is not None and len(df) >= 5:
                return key
    except Exception:
        pass
    return (candidates[0].replace('/', '').upper() if candidates else 'BTCUSDT')


# ─────────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────────

def _render_warmup_section() -> None:
    try:
        from dashboard.resources_ng import get_warmup_manager, get_ws_manager
        wmgr   = get_warmup_manager()
        counts = dict(wmgr.candle_count)
        ready  = dict(wmgr.ready)
        req    = wmgr.required_candles
        # Fallback: se warmup_manager não tem counts (após bootstrap), ler kline_buffers
        if not counts:
            try:
                ws = get_ws_manager()
                for sym, tfs in ws.kline_buffers.items():
                    tf_len = len(tfs.get('15m', []))
                    if tf_len > 0:
                        counts[sym] = tf_len
                        ready[sym]  = tf_len >= req
            except Exception:
                pass
    except Exception:
        counts = {}
        ready  = {}
        req    = 50

    section_title('🔥 Warmup')

    if not counts:
        ui.label(f'⏳ Aguardando candles... (mínimo: {req} por símbolo)').classes('text-yellow-400')
        return

    all_ready = all(ready.values()) if ready else False
    if all_ready:
        ui.label(f'✅ Warmup concluído — todos os símbolos têm ≥{req} candles').classes('text-green-400')
    else:
        with ui.row().classes('flex-wrap gap-3'):
            for sym, count in counts.items():
                is_ready = ready.get(sym, False)
                color = 'bg-green-800 border-green-600' if is_ready else 'bg-yellow-900 border-yellow-600'
                icon  = '✅' if is_ready else '⏳'
                pct   = min(count / req * 100, 100)
                with ui.card().classes(f'{color} border p-3 min-w-32'):
                    ui.label(f'{icon} {sym.replace("USDT","")}').classes('text-white font-bold text-sm')
                    ui.label(f'{count}/{req} ({pct:.0f}%)').classes('text-gray-300 text-xs font-mono')


# ─────────────────────────────────────────────────────────────────────────────
# Regime
# ─────────────────────────────────────────────────────────────────────────────

def _render_regime_section(primary: str) -> None:
    section_title('📊 Regime de Mercado')

    try:
        from dashboard.resources_ng import get_ws_manager, get_config
        ws  = get_ws_manager()
        cfg = get_config()
        df  = ws.get_klines_df(primary.replace('/', '').upper(), '15m', limit=200)
        if df is None or len(df) < 50:
            ui.label('⏳ Dados insuficientes…').classes('text-gray-400')
            return
        from dashboard.analytics.regime import detect_market_regime, calculate_atr
        regime = detect_market_regime(df, cfg)
        atr    = calculate_atr(df)
        regime_colors = {
            'TREND_UP'  : ('text-green-300', '📈'),
            'TREND_DOWN': ('text-red-300',   '📉'),
            'RANGING'   : ('text-yellow-300', '↔️'),
            'VOLATILE'  : ('text-orange-300', '⚡'),
        }
        color, icon = regime_colors.get(regime, ('text-gray-300', '❓'))
        with ui.row().classes('gap-6 items-center'):
            with ui.card().classes('bg-gray-800 border border-gray-700 px-5 py-3'):
                ui.label('Regime').classes('text-gray-400 text-xs')
                ui.label(f'{icon} {regime}').classes(f'{color} font-bold text-lg')
            with ui.card().classes('bg-gray-800 border border-gray-700 px-5 py-3'):
                ui.label('ATR (15m)').classes('text-gray-400 text-xs')
                ui.label(f'${float(atr):,.4f}').classes('text-white font-mono font-bold text-lg')
    except Exception as exc:
        ui.label(f'Erro ao calcular regime: {exc}').classes('text-orange-400 text-xs')


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def _render_drift_section(primary: str) -> None:
    section_title('🔬 Drift Detection — Features V19')

    try:
        from dashboard.analytics.drift_detector import (
            load_baseline, extract_features_from_klines,
            compute_drift, overall_drift_status,
            BASELINE_PATH, FEATURE_NAMES,
        )
        from dashboard.resources_ng import get_ws_manager

        baseline = load_baseline()
        if baseline is None:
            with ui.card().classes('bg-yellow-900 border border-yellow-600 p-4'):
                ui.label(f'⚠️ Baseline não encontrado em {BASELINE_PATH}').classes('text-yellow-200')
                ui.label('Execute: python generate_feature_baseline.py').classes('text-yellow-100 text-xs font-mono')
            return

        ws     = get_ws_manager()
        sym    = primary.replace('/', '').upper()
        df_live = ws.get_klines_df(sym, '15m', limit=200)

        if df_live is None or len(df_live) < 30:
            ui.label(f'⏳ Dados live insuficientes para {sym}…').classes('text-gray-400')
            return

        live_feats = extract_features_from_klines(df_live)
        if live_feats is None:
            ui.label('Não foi possível extrair features dos klines live.').classes('text-orange-400')
            return

        results = compute_drift(live_feats, baseline)
        status  = overall_drift_status(results)

        n_alert = sum(1 for r in results if r.status == 'ALERT')
        n_warn  = sum(1 for r in results if r.status == 'WARN')

        # Banner global
        if status == 'ALERT':
            banner_cls = 'bg-red-900 border-red-500'
            banner_txt = f'🚨 DRIFT CRÍTICO — {n_alert} feature(s) com Z > 3.5'
        elif status == 'WARN':
            banner_cls = 'bg-yellow-900 border-yellow-500'
            banner_txt = f'⚠️ DRIFT MODERADO — {n_warn} feature(s) com Z > 2.0'
        else:
            banner_cls = 'bg-green-900 border-green-500'
            banner_txt = '✅ DISTRIBUIÇÃO NORMAL — regime dentro do treinamento'

        with ui.card().classes(f'{banner_cls} border p-3 w-full mb-3'):
            ui.label(banner_txt).classes('text-white font-semibold')

        # Tabela de features
        rows = []
        for r in results:
            if r.status == 'ALERT':
                status_label = '🚨 ALERT'
            elif r.status == 'WARN':
                status_label = '⚠️ WARN'
            else:
                status_label = '✅ OK'
            rows.append({
                'Status'     : status_label,
                'Feature'    : r.feature,
                'Live Mean'  : round(r.live_mean,  4),
                'Train Mean' : round(r.train_mean, 4),
                'Δ Mean'     : round(r.live_mean - r.train_mean, 4),
                'Z-Score'    : round(r.z_score, 2),
            })

        ui.aggrid({
            'columnDefs': [
                {'field': 'Status',     'width': 110},
                {'field': 'Feature',    'width': 130},
                {'field': 'Live Mean',  'width': 110},
                {'field': 'Train Mean', 'width': 110},
                {'field': 'Δ Mean',     'width': 100},
                {'field': 'Z-Score',    'width': 100,
                 'cellClassRules': {
                     'text-red-400'    : 'Math.abs(x) >= 3.5',
                     'text-yellow-400' : 'Math.abs(x) >= 2.0 && Math.abs(x) < 3.5',
                     'text-green-400'  : 'Math.abs(x) < 2.0',
                 }},
            ],
            'rowData': rows,
            'domLayout': 'autoHeight',
        }).classes('w-full ag-theme-alpine-dark')

        n_samples = baseline.get('n_samples', '?')
        gen_at    = baseline.get('generated_at', '?')
        ui.label(f'📊 Baseline: {n_samples:,} amostras · Gerado: {gen_at} · '
                 f'Live: {len(live_feats)} candles 15m de {sym}'
                 ).classes('text-gray-500 text-xs mt-2')
        ui.label('Z ≥ 2.0 = ⚠️ Warning  |  Z ≥ 3.5 = 🚨 Alert'
                 ).classes('text-gray-500 text-xs')

    except Exception as exc:
        ui.label(f'Erro no Drift Detection: {exc}').classes('text-orange-400 text-xs')


# ─────────────────────────────────────────────────────────────────────────────
# Schedule
# ─────────────────────────────────────────────────────────────────────────────

def _render_schedule_section() -> None:
    """Janelas de operação por símbolo."""
    section_title('📅 Janelas de Operação')
    try:
        from dashboard.resources_ng import get_config
        cfg   = get_config()
        sched_cfg = cfg.get('schedule', {})
        if not sched_cfg:
            ui.label('Nenhum schedule configurado no config.yaml.').classes('text-gray-500 text-sm')
            return
        # Tenta usar ScheduleManager
        try:
            from dashboard.resources import get_schedule_manager as _gsm
            sched = _gsm()
            syms  = list(sched.schedule.keys())
            if not syms:
                raise ValueError('empty')
            with ui.row().classes('flex-wrap gap-3'):
                for sym_s in syms[:8]:
                    can_trade, reason = sched.can_trade_now(sym_s)
                    color = 'bg-green-800 border-green-600' if can_trade else 'bg-red-900 border-red-700'
                    icon  = '🟢' if can_trade else '🔴'
                    with ui.card().classes(f'{color} border p-2 min-w-28'):
                        ui.label(f'{icon} {sym_s.replace("USDT","")}').classes('text-white text-xs font-bold')
                        ui.label(reason[:30]).classes('text-gray-300 text-xs')
        except Exception:
            # Fallback: mostra janelas do config
            pairs = list(sched_cfg.keys())[:8]
            with ui.row().classes('flex-wrap gap-3'):
                for p in pairs:
                    w = sched_cfg[p]
                    ui.chip(f'⏰ {p}: {w.get("start","?")}–{w.get("end","?")}').props('outline').classes('text-gray-300')
    except Exception as exc:
        ui.label(f'Erro no schedule: {exc}').classes('text-orange-400 text-xs')


# ─────────────────────────────────────────────────────────────────────────────
# Multi-TF
# ─────────────────────────────────────────────────────────────────────────────

def _render_multi_tf_section(primary: str, all_symbols: list | None = None) -> None:
    """Análise de regime + ATR em 15m, 1h e 4h para símbolos com dados."""
    section_title('📊 Análise Multi-Timeframe')
    try:
        from dashboard.resources_ng import get_ws_manager, get_config
        from dashboard.analytics.regime import detect_market_regime, calculate_atr
        ws  = get_ws_manager()
        cfg = get_config()

        # Usa símbolos que têm dados reais (max 4 para não ocupar muito espaço)
        candidates = [primary] + [s for s in (all_symbols or []) if s != primary]
        syms_with_data = []
        for s in candidates:
            k = s.replace('/', '').upper()
            df_check = ws.get_klines_df(k, '15m', limit=5)
            if df_check is not None and len(df_check) >= 5:
                syms_with_data.append(k)
            if len(syms_with_data) >= 4:
                break

        if not syms_with_data:
            ui.label('⏳ Aguardando dados kline...').classes('text-gray-400')
            return

        tf_configs = [
            ('⚡ 15m',  '15m'),
            ('⏱ 1h',   '1h'),
            ('🕓 4h',   '4h'),
        ]
        regime_styles = {
            'TREND_UP'  : ('text-green-300',  '📈'),
            'TREND_DOWN': ('text-red-300',    '📉'),
            'RANGING'   : ('text-yellow-300', '↔️'),
            'VOLATILE'  : ('text-orange-300', '⚡'),
            'TRENDING'  : ('text-green-300',  '📈'),
            'UNCERTAIN' : ('text-gray-400',   '❓'),
        }

        for sym in syms_with_data:
            short = sym.replace('USDT', '')
            ui.label(short).classes('text-gray-300 font-bold text-xs mt-3 mb-1')
            with ui.row().classes('gap-3 flex-wrap'):
                for label, tf in tf_configs:
                    try:
                        df = ws.get_klines_df(sym, tf, limit=100)
                        if df is None or len(df) < 10:
                            raise ValueError('sem dados')
                        regime = detect_market_regime(df, cfg)
                        atr    = calculate_atr(df)
                        price  = float(df['close'].iloc[-1])
                        rsi    = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df.columns else None
                        r_color, r_icon = regime_styles.get(regime, ('text-gray-300', '❓'))
                        with ui.card().classes('bg-gray-800 border border-gray-700 p-2 min-w-36'):
                            ui.label(label).classes('text-gray-400 text-xs')
                            ui.label(f'{r_icon} {regime}').classes(f'{r_color} font-bold text-xs')
                            ui.label(f'${price:,.2f}  ATR:{atr:,.2f}').classes('text-gray-300 text-xs font-mono')
                            if rsi is not None:
                                rsi_color = 'text-red-400' if rsi > 70 else ('text-green-400' if rsi < 30 else 'text-yellow-300')
                                ui.label(f'RSI {rsi:.0f}').classes(f'{rsi_color} text-xs font-mono')
                    except Exception:
                        with ui.card().classes('bg-gray-800 border border-gray-600 p-2 min-w-36'):
                            ui.label(label).classes('text-gray-400 text-xs')
                            ui.label('⏳ sem dados').classes('text-gray-600 text-xs')
    except Exception as exc:
        ui.label(f'Erro Multi-TF: {exc}').classes('text-orange-400 text-xs')


# ─────────────────────────────────────────────────────────────────────────────
# Correlação
# ─────────────────────────────────────────────────────────────────────────────

def _render_correlation_section(symbols: list[str]) -> None:
    """Matriz de correlação entre símbolos selecionados."""
    section_title('🔗 Correlação entre Símbolos')
    with ui.expansion('Mostrar matriz de correlação', icon='calculate').classes(
            'w-full bg-gray-800 border border-gray-700 rounded'):
        try:
            from dashboard.resources_ng import get_ws_manager
            from dashboard.analytics.regime import calculate_correlation
            ws   = get_ws_manager()
            syms = [s.replace('/', '').upper() for s in symbols[:4]]
            dfs  = {}
            for s in syms:
                df_ = ws.get_klines_df(s, '1h', limit=100)
                if df_ is not None and len(df_) >= 20:
                    dfs[s] = df_
            if len(dfs) < 2:
                ui.label('Dados insuficientes para correlação.').classes('text-gray-400')
                return
            sym_list = list(dfs.keys())
            rows_data = []
            for sa in sym_list:
                row = {'Símbolo': sa.replace('USDT', '')}
                for sb in sym_list:
                    corr = (1.0 if sa == sb else
                            round(calculate_correlation(dfs[sa], dfs[sb], 20), 3))
                    row[sb.replace('USDT', '')] = corr
                rows_data.append(row)
            col_defs = [{'field': 'Símbolo', 'width': 100, 'pinned': 'left'}]
            for sb in sym_list:
                sb_key = sb.replace('USDT', '')
                col_defs.append({
                    'field': sb_key, 'width': 100,
                    'cellClassRules': {
                        'text-green-400': 'x > 0.8',
                        'text-red-400'  : 'x < -0.5',
                        'text-yellow-300': 'x >= 0.5 && x <= 0.8',
                    },
                })
            ui.aggrid({
                'columnDefs': col_defs,
                'rowData'   : rows_data,
                'domLayout' : 'autoHeight',
            }).classes('w-full ag-theme-alpine-dark')
            ui.label('> 0.8 = alta correlação (verde) · < -0.5 = anti-correlação (vermelho)'
                     ).classes('text-gray-500 text-xs mt-2')
        except Exception as exc:
            ui.label(f'Erro correlação: {exc}').classes('text-orange-400 text-xs')


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizing Simulator
# ─────────────────────────────────────────────────────────────────────────────

def _render_sizing_simulator() -> None:
    """Simulador interativo de position sizing dinâmico."""
    section_title('⚖️ Simulador de Position Sizing')
    with ui.expansion('Abrir simulador', icon='calculate').classes(
            'w-full bg-gray-800 border border-gray-700 rounded'):
        try:
            from dashboard.resources_ng import get_config
            cfg     = get_config()
            rm_cfg  = cfg.get('risk_management', {})
            tr_cfg  = cfg.get('trading', {})
            base_ps = float(rm_cfg.get('position_size', 0.1))

            with ui.row().classes('gap-6 flex-wrap mt-2'):
                with ui.column().classes('gap-2'):
                    ui.label('Saldo ($)').classes('text-gray-400 text-xs')
                    bal_in = ui.number(value=1000.0, min=10, step=100, format='%.0f').props('outlined dense dark').classes('w-36')
                    ui.label('Preço ($)').classes('text-gray-400 text-xs mt-1')
                    price_in = ui.number(value=50000.0, min=0.01, step=1000, format='%.2f').props('outlined dense dark').classes('w-36')
                    ui.label('Leverage').classes('text-gray-400 text-xs mt-1')
                    lev_in = ui.number(value=float(tr_cfg.get('leverage', 3)), min=1, max=20, step=1, format='%.0f').props('outlined dense dark').classes('w-36')

                with ui.column().classes('gap-2'):
                    ui.label('ATR ($)').classes('text-gray-400 text-xs')
                    atr_in  = ui.number(value=500.0, min=1, step=10, format='%.2f').props('outlined dense dark').classes('w-36')
                    ui.label('Regime').classes('text-gray-400 text-xs mt-1')
                    reg_sel = ui.select(options=['TRENDING','RANGING','VOLATILE','UNCERTAIN'],
                                        value='TRENDING').props('outlined dense dark').classes('w-36')
                    ui.label('Confiança do Modelo').classes('text-gray-400 text-xs mt-1')
                    conf_in = ui.number(value=0.70, min=0.50, max=1.0, step=0.05, format='%.2f').props('outlined dense dark').classes('w-36')

            result_label = ui.label('').classes('text-white font-mono text-sm mt-3')

            def _calc() -> None:
                try:
                    from dashboard.analytics.performance import calculate_position_size_dynamic
                    qty = calculate_position_size_dynamic(
                        balance        = float(bal_in.value or 1000),
                        base_size      = base_ps,
                        volatility_atr = float(atr_in.value or 500),
                        current_price  = float(price_in.value or 50000),
                        leverage       = int(lev_in.value or 3),
                        win_streak     = 0,
                        regime         = reg_sel.value or 'TRENDING',
                        confidence     = float(conf_in.value or 0.7),
                        risk_config    = rm_cfg,
                    )
                    notional  = qty * float(price_in.value or 50000)
                    margin    = notional / float(lev_in.value or 3)
                    result_label.set_text(
                        f'Qty: {qty:.6f}  |  Notional: ${notional:,.2f}  |  Margem: ${margin:,.2f}'
                    )
                except Exception as ce:
                    result_label.set_text(f'Erro: {ce}')

            ui.button('⚖️ Calcular', on_click=_calc).props('color=primary').classes('mt-2')

        except Exception as exc:
            ui.label(f'Erro no simulador: {exc}').classes('text-orange-400 text-xs')
