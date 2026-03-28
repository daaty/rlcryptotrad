"""
Tab 4 — Análise de mercado: regime, correlação, multi-TF, sizing e warmup.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.resources import get_warmup_manager, get_schedule_manager
from dashboard.data.market_data import collect_multi_timeframe_data
from dashboard.analytics.regime import (
    detect_market_regime, calculate_atr, calculate_correlation,
)
from dashboard.analytics.performance import calculate_position_size_dynamic
from dashboard.core.logging_setup import get_logger
from dashboard.analytics.drift_detector import (
    load_baseline,
    extract_features_from_klines,
    compute_drift,
    overall_drift_status,
    BASELINE_PATH,
    FEATURE_NAMES,
)

logger = get_logger()


def render_tab_analysis(
    tab,
    selected_symbols: list[str],
    client,
    config: dict,
    sidebar_state: dict,
    positions: list[dict] | None = None,
) -> None:
    with tab:
        st.subheader("🔬 Análise de Mercado")

        # ── Warmup ────────────────────────────────────────────────────────
        _render_warmup_panel(config)

        # ── Schedule ──────────────────────────────────────────────────────
        _render_schedule_panel()

        st.divider()

        if not selected_symbols:
            st.warning("Selecione ao menos um símbolo no painel lateral.")
            return

        primary = selected_symbols[0]

        # ── Regime & ATR ──────────────────────────────────────────────────
        if sidebar_state.get('show_regime', True):
            _render_regime_panel(primary, client, config)
            st.divider()

        # ── Correlação ────────────────────────────────────────────────────
        if sidebar_state.get('show_correlation', False) and len(selected_symbols) >= 2:
            _render_correlation_panel(selected_symbols, client, config)
            st.divider()

        # ── Multi-TF ──────────────────────────────────────────────────────
        if sidebar_state.get('show_multi_tf', False):
            _render_multi_tf_panel(primary, client, config)
            st.divider()

        # ── Position sizing simulator ─────────────────────────────────────
        _render_sizing_panel(config)

        # ── Legacy positions (se passadas explicitamente) ─────────────────
        if positions:
            _render_positions_legacy(positions)

        # ── Drift Detection ───────────────────────────────────────────────
        st.divider()
        _render_drift_panel(primary)


# ── Warmup ────────────────────────────────────────────────────────────────────

def _render_warmup_panel(config: dict) -> None:
    wmgr          = get_warmup_manager()
    required      = wmgr.required_candles
    candle_counts = dict(wmgr.candle_count)   # {symbol: int}
    ready_map     = dict(wmgr.ready)           # {symbol: bool}

    st.subheader("🔥 Status de Warmup")

    if not candle_counts:
        st.info(f"⏳ Aguardando candles... (mínimo: {required} por símbolo)")
        return

    all_ready = all(ready_map.values()) if ready_map else False

    if all_ready:
        st.success(f"✅ Warmup concluído — todos os símbolos têm ≥{required} candles")
    else:
        wc1, wc2 = st.columns(2)
        for i, (sym, count) in enumerate(candle_counts.items()):
            pct = min(1.0, count / required)
            col = wc1 if i % 2 == 0 else wc2
            with col:
                ready_sym = ready_map.get(sym, False)
                label = f"{sym}: {count}/{required}"
                if ready_sym:
                    st.success(f"✅ {label}")
                else:
                    st.warning(f"⏳ {label}")
                    st.progress(pct)


# ── Schedule ──────────────────────────────────────────────────────────────────

def _render_schedule_panel() -> None:
    sched = get_schedule_manager()
    st.subheader("📅 Janelas de Operação")
    symbols_sched = list(sched.schedule.keys())
    if not symbols_sched:
        st.info("Nenhum schedule configurado.")
        return
    sc1, sc2 = st.columns(2)
    for i, sym_s in enumerate(symbols_sched[:8]):
        can_trade, reason = sched.can_trade_now(sym_s)
        col = sc1 if i % 2 == 0 else sc2
        with col:
            if can_trade:
                st.success(f"🟢 {sym_s}: {reason}")
            else:
                st.warning(f"🔴 {sym_s}: {reason}")


# ── Regime ────────────────────────────────────────────────────────────────────

def _render_regime_panel(symbol: str, client, config: dict) -> None:
    st.subheader(f"🌍 Regime de Mercado — {symbol}")
    try:
        from dashboard.data.market_data import collect_market_data
        df = collect_market_data(symbol, '1h', client, limit=100)
        if df is None or df.empty:
            st.warning("Dados insuficientes para detector de regime.")
            return
        regime, conf = detect_market_regime(df)
        atr_val      = calculate_atr(df)
        current_p    = df['close'].iloc[-1]
        atr_pct      = (atr_val / current_p) * 100 if current_p else 0

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            colors = {'TRENDING': '🟢', 'RANGING': '🟡', 'VOLATILE': '🔴', 'UNCERTAIN': '⚪'}
            icon   = colors.get(regime, '⚪')
            st.metric("Regime", f"{icon} {regime}", f"Conf: {conf:.0%}")
        with rc2:
            st.metric("ATR (1h)", f"${atr_val:,.2f}", f"{atr_pct:.2f}% do preço")
        with rc3:
            st.metric("Preço Atual", f"${current_p:,.2f}")

        # Mini-guia
        regime_descriptions = {
            'TRENDING': "Mercado direcional — estratégias de tendência favorecem",
            'RANGING':  "Mercado lateral — Range trading / mean-reversion",
            'VOLATILE': "Alta volatilidade — tamanho de posição menor",
            'UNCERTAIN': "Indefinido — cautela recomendada",
        }
        st.info(f"💡 {regime_descriptions.get(regime, '')}")
    except Exception as exc:
        st.error(f"Erro ao calcular regime: {exc}")
        logger.warning(f"[TAB-ANAL] regime error: {exc}")


# ── Correlação ────────────────────────────────────────────────────────────────

def _render_correlation_panel(symbols: list[str], client, config: dict) -> None:
    st.subheader("🔗 Correlação entre Símbolos")
    try:
        from dashboard.data.market_data import collect_market_data
        dfs: dict[str, pd.DataFrame] = {}
        for s in symbols[:4]:     # no máximo 4
            df_ = collect_market_data(s, '1h', client, limit=100)
            if df_ is not None and not df_.empty:
                dfs[s] = df_

        if len(dfs) < 2:
            st.warning("Dados insuficientes para calcular correlação.")
            return

        sym_list = list(dfs.keys())
        n        = len(sym_list)
        corr_mat = pd.DataFrame(index=sym_list, columns=sym_list, dtype=float)
        for i, sa in enumerate(sym_list):
            for j, sb in enumerate(sym_list):
                corr_mat.loc[sa, sb] = calculate_correlation(dfs[sa], dfs[sb], 20) if i != j else 1.0

        corr_mat = corr_mat.astype(float).round(3)
        st.dataframe(corr_mat, width='stretch')

        # Tip
        max_corr = (
            corr_mat.where(pd.DataFrame([[i != j for j in range(n)] for i in range(n)],
                                         index=sym_list, columns=sym_list))
            .stack().abs().idxmax()
        )
        c_val = corr_mat.loc[max_corr[0], max_corr[1]]
        st.caption(f"Maior correlação: {max_corr[0]}↔{max_corr[1]} = {c_val:.3f}")
    except Exception as exc:
        st.error(f"Erro correlação: {exc}")
        logger.warning(f"[TAB-ANAL] correlation error: {exc}")


# ── Multi-TF ──────────────────────────────────────────────────────────────────

def _render_multi_tf_panel(symbol: str, client, config: dict) -> None:
    st.subheader(f"📊 Análise Multi-Timeframe — {symbol}")
    try:
        multi_tf = collect_multi_timeframe_data(symbol, client)
        if not multi_tf:
            st.warning("Dados multi-TF indisponíveis.")
            return

        for tf_label, df_tf in [
            ('⚡ 15m (Tático)',    multi_tf.get('15m')),
            ('⏱ 1h (Operacional)', multi_tf.get('1h')),
            ('🕓 4h (Estratégico)', multi_tf.get('4h')),
        ]:
            if df_tf is None or df_tf.empty:
                continue
            regime_tf, conf_tf = detect_market_regime(df_tf)
            atr_tf   = calculate_atr(df_tf)
            price_tf = df_tf['close'].iloc[-1]
            rsi_tf   = df_tf['RSI_14'].iloc[-1] if 'RSI_14' in df_tf.columns else None

            with st.expander(tf_label, expanded=(tf_label.startswith('⚡'))):
                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    st.metric("Último Preço", f"${price_tf:,.2f}")
                with tc2:
                    st.metric("Regime", regime_tf, f"{conf_tf:.0%}")
                with tc3:
                    st.metric("ATR", f"${atr_tf:,.2f}")
                with tc4:
                    if rsi_tf is not None:
                        color = "🔴" if rsi_tf > 70 else ("🟢" if rsi_tf < 30 else "🟡")
                        st.metric("RSI(14)", f"{color} {rsi_tf:.1f}")
    except Exception as exc:
        st.error(f"Erro multi-TF: {exc}")
        logger.warning(f"[TAB-ANAL] multi_tf error: {exc}")


# ── Position Sizing Simulator ─────────────────────────────────────────────────

def _render_sizing_panel(config: dict) -> None:
    st.subheader("⚖️ Simulador de Position Sizing")
    risk_cfg   = config.get('risk_management', {})
    trading_cfg= config.get('trading', {})

    with st.expander("Parâmetros do Simulador", expanded=False):
        pc1, pc2 = st.columns(2)
        with pc1:
            sim_balance   = st.number_input("Saldo ($)", value=1000.0, min_value=10.0,  step=100.0)
            sim_price     = st.number_input("Preço ($)", value=50000.0, min_value=0.1,   step=100.0)
            sim_leverage  = st.number_input("Leverage", value=float(trading_cfg.get('leverage', 3)),
                                            min_value=1.0, max_value=20.0, step=1.0)
        with pc2:
            sim_atr       = st.number_input("ATR ($)",   value=500.0,  min_value=1.0,    step=10.0)
            sim_regime    = st.selectbox("Regime",  ["TRENDING","RANGING","VOLATILE","UNCERTAIN"])
            sim_confidence= st.slider("Confiança do Modelo", 0.5, 1.0, 0.7, 0.05)

        base_size = float(risk_cfg.get('position_size', 0.1))
        win_streak = 0

        sim_qty = calculate_position_size_dynamic(
            balance        = sim_balance,
            base_size      = base_size,
            volatility_atr = sim_atr,
            current_price  = sim_price,
            leverage       = int(sim_leverage),
            win_streak     = win_streak,
            regime         = sim_regime,
            confidence     = sim_confidence,
            risk_config    = risk_cfg,
        )
        notional  = sim_qty * sim_price
        margin_req = notional / sim_leverage

        sr1, sr2, sr3 = st.columns(3)
        with sr1:
            st.metric("Quantidade", f"{sim_qty:.6f}")
        with sr2:
            st.metric("Notional", f"${notional:,.2f}")
        with sr3:
            st.metric("Margem Requerida", f"${margin_req:,.2f}")


# ── Drift Detection ───────────────────────────────────────────────────────────

def _render_drift_panel(primary_symbol: str) -> None:
    """
    Compara a distribuição atual das features V19 com o baseline de treinamento.
    Exibe alertas de Z-score para detectar regime shift que pode degradar o modelo.
    """
    from dashboard.resources import get_ws_manager

    st.subheader("🔬 Drift Detection — Features V19")

    baseline = load_baseline()
    if baseline is None:
        st.warning(
            f"Baseline não encontrado em `{BASELINE_PATH}`.  \n"
            "Execute: `python generate_feature_baseline.py`"
        )
        return

    ws_mgr = get_ws_manager()
    sym    = primary_symbol.replace('/', '').upper()   # BTCUSDT
    df_live = ws_mgr.get_klines_df(sym, '15m', limit=200)

    if df_live is None or len(df_live) < 30:
        st.info(f"⏳ Dados live insuficientes para {sym} (aguardando buffer 15m)...")
        return

    live_feats = extract_features_from_klines(df_live)
    if live_feats is None:
        st.warning("Não foi possível extrair features dos klines live.")
        return

    results = compute_drift(live_feats, baseline)
    status  = overall_drift_status(results)

    # Banner de status global
    n_alert = sum(1 for r in results if r.status == 'ALERT')
    n_warn  = sum(1 for r in results if r.status == 'WARN')

    if status == 'ALERT':
        st.error(f"🚨 DRIFT CRÍTICO — {n_alert} feature(s) com Z > 3.5. "
                 "Mercado saiu do regime de treinamento. Considere reduzir posições.")
    elif status == 'WARN':
        st.warning(f"⚠️ DRIFT MODERADO — {n_warn} feature(s) com Z > 2.0. "
                   "Mercado em transição. Monitore performance.")
    else:
        st.success("✅ DISTRIBUIÇÃO NORMAL — features dentro do regime de treinamento.")

    # Tabela de features
    rows = []
    for r in results:
        if r.status == 'ALERT':
            icon = "🚨"
        elif r.status == 'WARN':
            icon = "⚠️"
        else:
            icon = "✅"
        rows.append({
            'Status'     : f"{icon} {r.status}",
            'Feature'    : r.feature,
            'Live Mean'  : round(r.live_mean, 4),
            'Train Mean' : round(r.train_mean, 4),
            'Δ Mean'     : round(r.live_mean - r.train_mean, 4),
            'Live Std'   : round(r.live_std, 4),
            'Train Std'  : round(r.train_std, 4),
            'Z-Score'    : round(r.z_score, 2),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # Meta
    st.caption(
        f"📊 Baseline: {baseline.get('n_samples', '?'):,} amostras · "
        f"Gerado: {baseline.get('generated_at', '?')} · "
        f"Live window: {len(live_feats)} candles 15m de {sym}"
    )
    st.caption("Z ≥ 2.0 = ⚠️ Warning  |  Z ≥ 3.5 = 🚨 Alert  |  Parar engine se ≥ 2 alertas.")


# ── Posições Legacy ───────────────────────────────────────────────────────────

def _render_positions_legacy(positions: list[dict]) -> None:
    if not positions:
        return
    st.subheader("📋 Posições (Snapshot)")
    rows = []
    for p in positions:
        rows.append({
            "Símbolo":  p.get('symbol'),
            "Lado":     "LONG" if float(p.get('positionAmt', 0)) > 0 else "SHORT",
            "Qty":      float(p.get('positionAmt', 0)),
            "Entrada":  f"${float(p.get('entryPrice', 0)):,.2f}",
            "Mark":     f"${float(p.get('markPrice', 0)):,.2f}",
            "P&L":      f"${float(p.get('unRealizedProfit', 0)):,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
