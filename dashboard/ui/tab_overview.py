"""
Tab 1 — Overview: métricas de conta e gráficos multi-par.
Os gráficos são atualizados pelo fragment externo _live_dashboard (dashboard_new.py).
NÃO usar @st.fragment aqui — fragmentos aninhados causam DuplicateElementKey.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from binance.client import Client

from dashboard.data.market_data import collect_market_data
from dashboard.ui.charts import plot_candlestick


def render_tab_overview(
    tab,
    balance: dict,
    positions: list[dict],
    selected_symbols: list[str],
    client: Client,
    config: dict,
) -> None:
    with tab:
        # ── Métricas principais ───────────────────────────────────────────
        _render_metrics(balance, positions, config)

        st.divider()
        st.subheader("📈 Gráficos em Tempo Real")
        st.caption("⚡ Atualiza automaticamente via fragment externo (a cada 5s)")

        # Gráficos renderizados diretamente — sem @st.fragment aninhado
        # (o fragment externo _live_dashboard em dashboard_new.py cuida do live update)
        _render_charts_fragment(
            selected_symbols=selected_symbols,
            positions=positions,
            client=client,
            config=config,
        )


def _render_metrics(balance: dict, positions: list[dict], config: dict = None) -> None:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    total = float(balance.get('total', 0) or 0)
    avail = float(balance.get('available', 0) or 0)
    upnl  = float(balance.get('unrealized_pnl', 0) or 0)
    with col1:
        st.metric(
            "💰 Balance Total",
            f"${total:,.2f}",
            delta=f"{upnl:+.2f} USDT",
        )
    with col2:
        st.metric("💵 Disponível", f"${avail:,.2f}")
    with col3:
        st.metric("📊 Posições Abertas", len(positions))
    with col4:
        total_exposure = sum(
            abs(float(p.get('positionAmt', 0)) * float(p.get('entryPrice', 0)))
            for p in positions
        )
        exposure_pct = (total_exposure / total * 100) if total > 0 else 0
        st.metric("📈 Exposure Total", f"{exposure_pct:.1f}%")

    # ── Alerta de saldo insuficiente para trading eficiente ───────────────
    if config and avail > 0:
        rm          = config.get('risk_management', {})
        env         = config.get('environment', {})
        min_notional = float(rm.get('min_notional_usdt', 20.0))
        pos_size     = float(env.get('position_size', 0.03))
        leverage     = float(env.get('leverage', 1.5))
        commission   = float(env.get('commission', 0.0004))
        # Saldo necessário para abrir posição acima do mínimo notional
        min_balance_needed = min_notional / max(pos_size * leverage, 1e-6)
        fee_rt_pct         = commission * 2 * 100
        if avail < min_balance_needed:
            st.error(
                f"⚠️ **Saldo insuficiente para trading eficiente!**  \n"
                f"Saldo disponível: **${avail:,.2f}** — mínimo recomendado: **${min_balance_needed:,.0f} USDT**  \n"
                f"Com saldo atual, posição seria **< ${min_notional:.0f}** (após taxas de {fee_rt_pct:.2f}% round-trip o lucro seria consumido).  \n"
                f"Adicione saldo ou reduza o notional mínimo em **⚙️ Controle da Engine → Parâmetros de Risco**."
            )
        elif avail < min_balance_needed * 2:
            trade_usdt = avail * pos_size
            notional   = trade_usdt * leverage
            fee_cost   = notional * commission * 2
            st.warning(
                f"⚠️ **Saldo baixo** — posições marginais.  \n"
                f"Posição estimada: **${trade_usdt:.2f} USDT** → notional **${notional:.2f}** "
                f"| Taxa round-trip ≈ **${fee_cost:.4f}** ({fee_cost/notional*100:.2f}% da posição)"
            )

    # Add a risk summary bar to the overview
    if config:
        risk_bar_col1, risk_bar_col2, risk_bar_col3 = st.columns(3)
        with risk_bar_col1:
            st.write(f"**📉 SL**: {float(rm.get('stop_loss_pct', 0.02))*100:.1f}%")
        with risk_bar_col2:
            st.write(f"**🎯 TP**: {float(rm.get('take_profit_pct', 0.04))*100:.1f}%")
        with risk_bar_col3:
            st.write(f"**🧩 KELLY**: {float(rm.get('kelly_fraction', 0.25))*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_charts_fragment(
    selected_symbols: list[str],
    positions: list[dict],
    client,
    config: dict,
) -> None:
    """Renderiza gráficos candlestick. Atualizado pelo fragment externo _live_dashboard.
    Sem @st.fragment — fragmentos aninhados causam DuplicateElementKey.
    """
    import time
    st.caption(f"🔄 Última atualização: {time.strftime('%H:%M:%S')}")

    # ── Gráficos multi-par ‒ sub-abas por símbolo ──────────────────────
    symbol_tabs = st.tabs([f"📊 {sym}" for sym in selected_symbols])

    for idx, trade_symbol in enumerate(selected_symbols):
        with symbol_tabs[idx]:
            symbol_binance = trade_symbol.replace('/', '')

            col1, col2 = st.columns([3, 1])
            with col2:
                timeframe_map = {
                    '1m':  Client.KLINE_INTERVAL_1MINUTE,
                    '5m':  Client.KLINE_INTERVAL_5MINUTE,
                    '15m': Client.KLINE_INTERVAL_15MINUTE,
                    '1h':  Client.KLINE_INTERVAL_1HOUR,
                    '4h':  Client.KLINE_INTERVAL_4HOUR,
                }
                selected_tf   = st.selectbox("Timeframe", list(timeframe_map.keys()),
                                             index=2, key=f"tf_{symbol_binance}")
                candles_limit = st.slider("Candles", 50, 500, 100, key=f"candles_{symbol_binance}")

            with col1:
                df_chart_full = collect_market_data(
                    client, symbol=symbol_binance,
                    interval=timeframe_map[selected_tf], limit=200,
                )
                if df_chart_full is not None:
                    if not pd.api.types.is_datetime64_any_dtype(df_chart_full['timestamp']):
                        df_chart_full = df_chart_full.copy()
                        df_chart_full['timestamp'] = pd.to_datetime(
                            df_chart_full['timestamp'], unit='ms'
                        )
                    df = df_chart_full.tail(candles_limit).reset_index(drop=True)
                else:
                    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                symbol_positions = [p for p in positions if p['symbol'] == symbol_binance]

                if df.empty:
                    st.warning("⏳ Aguardando candles... Bootstrap em andamento, recarregue em instantes.")
                else:
                    fig = plot_candlestick(df, symbol=trade_symbol)

                    for pos in symbol_positions:
                        entry_price = float(pos['entryPrice'])
                        qty         = float(pos['positionAmt'])
                        pnl         = float(pos['unRealizedProfit'])
                        color       = 'green' if qty > 0 else 'red'
                        label       = 'LONG' if qty > 0 else 'SHORT'
                        fig.add_hline(
                            y=entry_price, line_dash="dash", line_color=color,
                            annotation_text=f"{label} @ ${entry_price:,.2f} | P&L: ${pnl:,.2f}",
                            annotation_position="right",
                        )
                        if len(df) > 0 and 'timestamp' in df.columns:
                            fig.add_scatter(
                                x=[df['timestamp'].iloc[-1]],
                                y=[entry_price],
                                mode='markers',
                                marker=dict(
                                    size=15, color=color,
                                    symbol='triangle-up' if qty > 0 else 'triangle-down',
                                ),
                                name=f"Entry {label}",
                                showlegend=True,
                            )

                    st.plotly_chart(fig, width='stretch', key=f"chart_{symbol_binance}")
