"""
Sidebar — renderiza painel lateral e retorna configurações de sessão.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from dashboard.core.config import INTERVALS_WS, REST_COOLDOWN_SECS
from dashboard.core.ban_manager import rest_rate_ok, touch_rest_rate
from dashboard.data.websocket_manager import BinanceWebSocketManager
from dashboard.resources import (
    get_risk_manager, get_trailing_stop_manager,
    get_warmup_manager, get_schedule_manager,
    is_banned_session, register_ban_session,
)


def render_sidebar(config: dict, ws_mgr: BinanceWebSocketManager) -> dict:
    """
    Renderiza a sidebar completa.

    Returns:
        dict com:
            selected_symbols, allocation_strategy, auto_refresh,
            refresh_interval, show_correlation, show_regime, show_multi_tf
    """
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Modo
        mode = config.get('mode', 'testnet')
        {
            'testnet': lambda: st.success("🧪 Modo: TESTNET"),
            'live':    lambda: st.error("⚠️ Modo: LIVE (REAL)"),
        }.get(mode, lambda: st.info("📝 Modo: PAPER"))()

        st.divider()

        # ── Multi-par selector ────────────────────────────────────────────
        st.subheader("🎯 Seleção de Pares")
        st.success("🌐 WebSocket ativo — sem limite de pares por ban REST")
        st.caption("Dados de mercado via stream (zero REST por par).")
        st.caption("REST usado apenas para execução de ordens.")

        available_symbols = [s.replace('/', '') for s in config['data']['symbols']]
        primary_symbol    = config['data']['primary_symbol'].replace('/', '')

        selected_raw = st.multiselect(
            "Pares Ativos",
            available_symbols,
            default=[primary_symbol] if primary_symbol in available_symbols else [available_symbols[0]],
            help="Selecione quantos pares desejar — dados via WebSocket.",
        )

        # Par customizado (não listado no config)
        custom_input = st.text_input(
            "Adicionar par customizado (ex: SUIUSDT)",
            placeholder="Símbolo sem barra: SUIUSDT",
            help="Qualquer par de futuros da Binance",
        ).strip().upper()
        custom_extras = [s for s in custom_input.split(",") if s] if custom_input else []

        selected_symbols = list(dict.fromkeys(selected_raw + custom_extras)) or [available_symbols[0]]
        if custom_extras:
            st.success(f"+ Par(es) customizado(s): {', '.join(custom_extras)}")

        allocation_strategy = st.radio(
            "Estratégia de Alocação",
            ["Equal Weight", "Best Signal", "Correlation Filter"],
        )

        st.divider()

        # Parâmetros rápidos
        st.subheader("📊 Parâmetros")
        for sym in selected_symbols[:3]:
            st.text(f"📈 {sym}")
        if len(selected_symbols) > 3:
            st.text(f"   +{len(selected_symbols)-3} mais...")
        st.text(f"Timeframe: {config['data']['timeframes']['tactical']}")
        st.text(f"Position Size: {config['environment']['position_size']*100}%/par")
        st.text(f"Leverage: {config['environment']['leverage']}x")

        st.divider()

        # ── REST API ──────────────────────────────────────────────────────
        _render_rest_panel()

        st.divider()

        # ── WebSocket + Bootstrap ─────────────────────────────────────────
        _render_ws_panel(ws_mgr, selected_symbols)

        st.divider()

        # ── Risk Management ───────────────────────────────────────────────
        _render_risk_panel(config)

        st.divider()
        # ── Filtro de Entrada ─────────────────────────────────────────────
        st.subheader("🎛️ Filtro de Entrada")
        _FILTER_MODES = ["disabled", "aggressive", "normal", "strict"]
        _FILTER_LABELS = {
            "disabled":   "🚫 Desativado — modelo opera livre",
            "aggressive": "⚡ Agressivo — só RSI extremo",
            "normal":     "⚖️ Normal — RSI 80/20 · Vol 30%",
            "strict":     "🛡️ Estrito — todos os filtros",
        }
        current_mode = config.get('entry_filter', {}).get('mode', 'normal')
        if current_mode not in _FILTER_MODES:
            current_mode = 'normal'
        selected_mode = st.selectbox(
            "Modo do filtro",
            options=_FILTER_MODES,
            index=_FILTER_MODES.index(current_mode),
            format_func=lambda m: _FILTER_LABELS[m],
            help="Controla o quão rígido é o filtro técnico antes de abrir posição.",
        )
        if 'entry_filter' not in config:
            config['entry_filter'] = {}
        config['entry_filter']['mode'] = selected_mode
        st.caption(f"Modo ativo: **{selected_mode}**")

        st.divider()
        # ── Métricas avançadas ────────────────────────────────────────────
        st.subheader("📊 Métricas Avançadas")
        show_correlation = st.checkbox("Correlation Matrix", value=False)
        show_regime      = st.checkbox("Market Regime", value=False)
        show_multi_tf    = st.checkbox("Multi-Timeframe", value=False)

        st.divider()

        # Modelo ativo
        st.subheader("🤖 Modelo Ativo")
        st.info("🤖 **LSTM V17.7** (RecurrentPPO 600k)\n\nObs: (50, 31) | 15m+1h+4h multi-TF")

        st.divider()

        # Auto-refresh
        auto_refresh = st.checkbox(
            "� LIVE — Auto-atualizar", value=True,
            help="Recarrega a dashboard periodicamente usando dados do WebSocket (zero chamadas REST).",
        )
        refresh_interval = st.slider(
            "Intervalo (s)", min_value=5, max_value=120,
            value=5,
            help="5s suficiente — o WS atualiza os dados em background independentemente.",
        )
        if auto_refresh:
            st.success(f"⏱️ Atualizando a cada {refresh_interval}s (WebSocket)")
        else:
            st.warning("⏸️ Modo estático — dados só atualizam no recarregamento.")

        if st.button("🔄 Atualizar Agora"):
            st.rerun()

    return {
        'selected_symbols':    selected_symbols,
        'allocation_strategy': allocation_strategy,
        'auto_refresh':        auto_refresh,
        'refresh_interval':    refresh_interval,
        'show_correlation':    show_correlation,
        'show_regime':         show_regime,
        'show_multi_tf':       show_multi_tf,
    }


# ── Sub-painéis ───────────────────────────────────────────────────────────────

def _render_rest_panel() -> None:
    st.subheader("🔌 REST API")
    banned, ban_rem  = is_banned_session()
    rate_ok, rate_wait = rest_rate_ok()
    rest_conn        = st.session_state.get('_rest_connected', False)

    if banned:
        ban_exp = datetime.fromtimestamp(
            st.session_state.get('ban_expires_at', 0)
        ).strftime('%H:%M:%S')
        st.error(f"🚫 Banido até {ban_exp} ({int(ban_rem//60)}m{int(ban_rem%60)}s)")
        st.caption("REST bloqueado automaticamente.")
    elif not rate_ok:
        st.warning(f"⏳ Cooldown ativo: {rate_wait:.0f}s restantes")
        st.caption(f"Intervalo mínimo: {REST_COOLDOWN_SECS}s entre chamadas")
    elif rest_conn:
        st.success("✅ REST ativo nesta sessão")
        if st.button("🔒 Desconectar REST"):
            st.session_state['_rest_connected'] = False
            st.rerun()
    else:
        st.warning("⚪ REST desconectado (startup seguro)")
        st.caption("Nenhuma chamada REST automática no startup.")
        if st.button("🔌 Conectar REST API", type="primary"):
            st.session_state['_rest_connected'] = True
            st.rerun()


def _render_ws_panel(ws_mgr: BinanceWebSocketManager, selected_symbols: list[str]) -> None:
    st.subheader("🌐 WebSocket + Bootstrap")
    ws_running = ws_mgr.running
    boot_done  = ws_mgr.bootstrap_done

    if ws_running and boot_done:
        st.success("🟢 WS ativo + dados bootstrapped — ZERO REST calls")
    elif ws_running and not boot_done:
        st.warning("🟡 WS ativo mas sem dados. Clique Bootstrap ↓")
    else:
        st.error("🔴 WebSocket desconectado")

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

    banned, ban_rem = is_banned_session()
    if banned:
        ban_exp = datetime.fromtimestamp(
            st.session_state.get('ban_expires_at', 0)
        ).strftime('%H:%M:%S')
        st.error(f"🚫 IP banido até {ban_exp} ({int(ban_rem//60)}m{int(ban_rem%60)}s) — Bootstrap bloqueado")
    else:
        boot_label = (
            "⏳ Re-Bootstrap (atualiza candles)" if boot_done
            else "⚡ Bootstrap (carrega histórico + conta)"
        )
        n_rest = len(selected_symbols) * len(INTERVALS_WS) + 2
        if st.button(boot_label, type="primary", key="btn_bootstrap",
                     help=f"Faz {n_rest} chamadas REST de uma única vez."):
            touch_rest_rate()
            with st.spinner("Bootstrapping klines + account..."):
                try:
                    n_klines = ws_mgr.bootstrap_klines(selected_symbols)
                    acct_ok  = ws_mgr.bootstrap_account()
                    # subscribe_all_klines já é chamado internamente por bootstrap_klines
                    st.success(
                        f"✅ Bootstrap OK! {n_klines} candles | "
                        f"Account: {'OK' if acct_ok else 'FALHA'}"
                    )
                    st.rerun()
                except Exception as exc:
                    register_ban_session(str(exc), 'BOOTSTRAP')
                    st.error(f"❌ Bootstrap erro: {exc}")

    if boot_done:
        stats_lines = [
            f"{'🟩' if c >= 50 else '🟨'} {s}/{i}:{c}"
            for s, ivs in ws_mgr.buffer_stats().items()
            for i, c in ivs.items()
        ]
        if stats_lines:
            st.caption("Buffers: " + " | ".join(stats_lines))

    if ws_mgr.user_data.get('last_update'):
        from datetime import datetime as _dt
        age = (_dt.now() - ws_mgr.user_data['last_update']).total_seconds()
        st.caption(f"👉 Account atualizado há {age:.0f}s")


def _render_risk_panel(config: dict) -> None:
    risk_mgr    = get_risk_manager()
    trailing_mgr = get_trailing_stop_manager()

    st.subheader("🛡️ Risk Management")

    can_trade, reason = risk_mgr.should_allow_trade()
    if can_trade:
        st.success("✅ Trading Ativo")
    else:
        st.error(f"⛔ {reason}")
        if st.button("🔄 Reset Circuit Breaker"):
            risk_mgr.reset_circuit_breaker()
            st.success("Circuit breaker resetado!")
            st.rerun()

    active_trails = len(trailing_mgr.active_stops)
    if active_trails > 0:
        st.info(f"🎯 {active_trails} trailing stops ativos")

    stats = risk_mgr.get_trading_stats()
    if stats['total_trades'] > 0:
        st.text(f"Trades: {stats['total_trades']}")
        st.text(f"Win Rate: {stats['win_rate']*100:.1f}%")
        st.text(f"Losses: {stats['consecutive_losses']}")
