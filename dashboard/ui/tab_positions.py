"""
Tab 2 — Posições abertas: visualização e fechamento manual.
TP/SL automático é responsabilidade exclusiva da TradingEngine (background thread).
Esta aba apenas EXIBE o estado das posições e oferece controle manual.
"""
from __future__ import annotations

import streamlit as st
from binance.client import Client

from dashboard.resources import (
    get_risk_manager, get_trailing_stop_manager,
    get_config,
)
from dashboard.trading.executor import close_position_direct, close_all_positions
from dashboard.core.logging_setup import get_logger

logger = get_logger()


def render_tab_positions(
    tab,
    positions: list[dict],
    client: Client,
    config: dict,
) -> None:
    with tab:
        st.subheader("💼 Posições Abertas")
        trailing_mgr = get_trailing_stop_manager()
        risk_mgr     = get_risk_manager()

        if not positions:
            st.info("📭 Nenhuma posição aberta no momento")
            return

        # ── Botão Emergency Close All ──────────────────────────────────────
        total_pnl_all = sum(float(p['unRealizedProfit']) for p in positions)
        pnl_color     = "positive" if total_pnl_all >= 0 else "negative"

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(
                f'<span style="font-size:1.1rem">⚠️ {len(positions)} posição(ões) abertas | '
                f'P&L Não Realizado: <span class="{pnl_color}"><b>${total_pnl_all:+,.2f}</b></span></span>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("🚨 Fechar TODAS", type="primary", key="close_all_btn",
                         help="Fecha imediatamente TODAS as posições via MARKET reduceOnly"):
                with st.spinner("Fechando todas as posições..."):
                    results = close_all_positions(client, positions, config)
                ok_count   = sum(1 for r in results if r['order'])
                fail_count = len(results) - ok_count
                if ok_count:
                    st.success(f"✅ {ok_count} posição(ões) fechada(s)")
                if fail_count:
                    st.error(f"❌ {fail_count} falharam — verifique os Logs")
                st.rerun()
        with c3:
            st.empty()

        st.divider()

        # ── Auto-registro no trailing stop ────────────────────────────────
        active_symbols = {p['symbol'] for p in positions}
        for pos_reg in positions:
            sym_reg = pos_reg['symbol']
            if not trailing_mgr.get_stop_info(sym_reg):
                entry_reg = float(pos_reg['entryPrice'])
                qty_reg   = float(pos_reg['positionAmt'])
                side_reg  = 'LONG' if qty_reg > 0 else 'SHORT'
                trailing_mgr.register_position(sym_reg, entry_reg, side_reg)
                logger.info(f"[TAB-POS] Auto-registrado trailing: {sym_reg} {side_reg}")

        # ── Limpa flags TP1 de posições encerradas ───────────────────────
        stale_flags = [
            k for k in st.session_state
            if k.startswith("tp1_partial_") and k[len("tp1_partial_"):] not in active_symbols
        ]
        for flag in stale_flags:
            del st.session_state[flag]

        # ── Lista de posições ─────────────────────────────────────────────
        for pos in positions:
            _render_position_row(pos, client, config, risk_mgr, trailing_mgr)


def _fmt_price(price: float) -> str:
    """Formata preço com precisão dinâmica (não trunca ativos baratos)."""
    if price <= 0:
        return "$0"
    if price < 0.01:
        return f"${price:,.6f}"
    if price < 1:
        return f"${price:,.4f}"
    if price < 100:
        return f"${price:,.2f}"
    return f"${price:,.0f}"


def _render_position_row(pos, client, config, risk_mgr, trailing_mgr) -> None:
    symbol        = pos['symbol']
    qty           = float(pos['positionAmt'])
    entry_price   = float(pos['entryPrice'])
    mark_price    = float(pos['markPrice'])
    unrealized_pnl = float(pos['unRealizedProfit'])
    pnl_pct       = (unrealized_pnl / (entry_price * abs(qty))) * 100 if qty != 0 else 0
    position_type = 1 if qty > 0 else -1
    side          = "LONG 🟢" if qty > 0 else "SHORT 🔴"

    atr_estimate = mark_price * 0.02
    stop_price   = risk_mgr.calculate_atr_stop_loss(entry_price, atr_estimate, position_type)
    trailing_info = trailing_mgr.get_stop_info(symbol)
    should_tp, tp_level = risk_mgr.should_take_profit(
        entry_price, mark_price, position_type, return_level=True
    )

    # ── Status informacional (sem execução automática — competência da Engine) ──
    if risk_mgr.should_stop_loss(entry_price, mark_price, position_type, atr=atr_estimate):
        st.error(
            f"⚠️ **SL ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}% | "
            f"Mark: ${mark_price:,.2f} | SL: ${stop_price:,.2f} — "
            "A Engine vai fechar automaticamente."
        )
    elif should_tp and tp_level == 2:
        st.success(f"🎯 **TP L2 ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}% — Engine fechando.")
    elif should_tp and tp_level == 1:
        tp1_flag = f"tp1_partial_{symbol}"
        if tp1_flag not in st.session_state:
            st.success(f"🎯 **TP L1 ATINGIDO: {symbol}** | P&L: {pnl_pct:+.2f}% — Engine fechando 50%.")
        else:
            st.warning(f"⚠️ TP L1 parcial já executado em **{symbol}** — aguardando TP L2 ou SL.")

    # ── Linha de exibição ─────────────────────────────────────────────────
    with st.container():
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 1])
        with col1:
            st.markdown(f"**{symbol}**")
            st.text(side)
        with col2:
            st.text(f"Qty: {abs(qty):.4f}")
            st.text(f"Entry: ${entry_price:,.2f}")
        with col3:
            st.text(f"Mark: ${mark_price:,.2f}")
            leverage = pos.get('leverage', config.get('environment', {}).get('leverage', 3))
            st.text(f"Leverage: {leverage}x")
        with col4:
            if trailing_info and trailing_info.get('activated'):
                trail_stop = trailing_info['stop_price']
                st.success(f"🎯 Trail: {_fmt_price(trail_stop)}")
                dist = abs(mark_price - trail_stop) / mark_price * 100
                st.text(f"Dist: {dist:.1f}%")
            else:
                stop_icon = "🔴" if risk_mgr.should_stop_loss(
                    entry_price, mark_price, position_type, atr=atr_estimate
                ) else "🟢"
                st.text(f"{stop_icon} SL: {_fmt_price(stop_price)}")
                if tp_level > 0:
                    st.text(f"✅ TP L{tp_level}")
                else:
                    tp_pct = config.get('risk_management', {}).get('take_profit_pct', 0.04) / 2
                    tp_target_1 = entry_price * (1 + tp_pct if qty > 0 else 1 - tp_pct)
                    st.text(f"🎯 TP1: {_fmt_price(tp_target_1)}")
        with col5:
            pnl_class = "positive" if unrealized_pnl >= 0 else "negative"
            st.markdown(f'<p class="{pnl_class}">P&L: ${unrealized_pnl:,.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="{pnl_class}">({pnl_pct:+.2f}%)</p>', unsafe_allow_html=True)
        with col6:
            # Fechamento MANUAL
            if st.button("❌ Fechar", key=f"close_{symbol}",
                         help=f"Fecha manualmente a posição {side} de {symbol}"):
                with st.spinner(f"Fechando {symbol}..."):
                    order = close_position_direct(client, symbol, qty, config)
                if order:
                    st.success(f"✅ {symbol} fechado!")
                else:
                    st.error(f"❌ Falha — ver Logs")
                st.rerun()

        # trailing info extra
        if trailing_info:
            if position_type == 1:
                highest = trailing_info.get('highest_mark', entry_price)
                st.text(f"📈 Max: ${highest:,.2f}")
            else:
                lowest = trailing_info.get('lowest_mark', entry_price)
                st.text(f"📉 Min: ${lowest:,.2f}")

            if trailing_info.get('activated'):
                st.success("🟢 Trailing ATIVO")
            else:
                activation_pct = config.get('risk_management', {}).get('trailing_stop_activation', 0.03) * 100
                st.info(f"⏳ Ativa em +{activation_pct:.0f}%")

        st.divider()
