"""
Tab 3 — Desempenho: posições abertas, histórico fechado e métricas de PnL.
Usa engine.state (pre-carregado do SQLite) — dados persistem entre sessões.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.analytics.performance import calculate_performance_metrics
from dashboard.analytics.report_generator import generate_monthly_report
from dashboard.core.logging_setup import get_logger
from dashboard.resources import get_ws_manager
from dashboard.data.trade_store import get_trade_store

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pnl(value, dollar: bool = True) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
        prefix = "+" if f >= 0 else ""
        return f"{prefix}${f:,.4f}" if dollar else f"{prefix}{f:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _build_pnl_chart(closed_trades: list[dict]) -> go.Figure | None:
    """Gráfico combinado: barras por trade (verde/vermelho) + linha acumulada."""
    if not closed_trades:
        return None
    df = pd.DataFrame(closed_trades)
    if "realizedPnl" not in df.columns or "time" not in df.columns:
        return None

    df["realizedPnl"]    = df["realizedPnl"].astype(float)
    df["time"]           = pd.to_datetime(df["time"], unit="ms")
    df["cumulative_pnl"] = df["realizedPnl"].cumsum()
    colors               = ["#26a69a" if v >= 0 else "#ef5350" for v in df["realizedPnl"]]

    hover = (
        df["symbol"].astype(str)
        + " | " + df.get("side", pd.Series([""] * len(df))).astype(str)
        + "<br>P&L: $" + df["realizedPnl"].map(lambda x: f"{x:+.4f}")
        + "<br>Entrada: $" + df.get("entryPrice", pd.Series([0]*len(df))).map(lambda x: f"{float(x):,.4f}")
        + "<br>Saída: $"   + df.get("exitPrice",  pd.Series([0]*len(df))).map(lambda x: f"{float(x):,.4f}")
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.58],
        subplot_titles=("P&L por Trade (USDT)", "P&L Acumulado (USDT)"),
    )
    fig.add_trace(
        go.Bar(x=df["time"], y=df["realizedPnl"],
               marker_color=colors, name="P&L / Trade",
               hovertext=hover, hoverinfo="text+x"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["cumulative_pnl"],
                   mode="lines+markers", name="Acumulado",
                   line=dict(color="#7986cb", width=2),
                   fill="tozeroy", fillcolor="rgba(121,134,203,0.15)",
                   hovertemplate="$%{y:,.4f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", row=2, col=1)
    fig.update_layout(height=420, template="plotly_dark", showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=2, col=1)
    return fig


def _build_symbol_summary(closed_trades: list[dict]) -> "pd.DataFrame | None":
    if not closed_trades:
        return None
    rows: dict = defaultdict(lambda: {"Trades": 0, "Wins": 0, "PnL": 0.0, "Melhor": 0.0, "Pior": 0.0})
    for t in closed_trades:
        s   = t.get("symbol", "?")
        pnl = float(t.get("realizedPnl", 0))
        rows[s]["Trades"]  += 1
        rows[s]["Wins"]    += 1 if pnl > 0 else 0
        rows[s]["PnL"]     += pnl
        rows[s]["Melhor"]   = max(rows[s]["Melhor"], pnl)
        rows[s]["Pior"]     = min(rows[s]["Pior"],   pnl)
    data = []
    for sym, r in sorted(rows.items()):
        wr = r["Wins"] / r["Trades"] * 100 if r["Trades"] else 0
        data.append({
            "Símbolo":  sym,
            "Trades":   r["Trades"],
            "Win Rate": f"{wr:.0f}%",
            "P&L ($)":  round(r["PnL"],     4),
            "Melhor":   round(r["Melhor"],   4),
            "Pior":     round(r["Pior"],     4),
        })
    return pd.DataFrame(data)


def _ms_to_time(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%H:%M:%S")
    except Exception:
        return "—"


def _count_open_positions() -> int:
    try:
        ws  = get_ws_manager()
        pos = ws.get_positions()
        if not pos:
            return 0
        return sum(1 for p in (pos if isinstance(pos, list) else []) if float(p.get("positionAmt", 0)) != 0)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Open Positions sub-section
# ─────────────────────────────────────────────────────────────────────────────

def _render_open_positions() -> None:
    """Mostra posições abertas com PnL não-realizado do WS."""
    try:
        ws  = get_ws_manager()
        pos = ws.get_positions()
        if not pos:
            return
        active = [p for p in (pos if isinstance(pos, list) else []) if float(p.get("positionAmt", 0)) != 0]
        if not active:
            return

        st.divider()
        st.subheader(f"🟡 Posições Abertas ({len(active)})")

        total_unrealized = sum(float(p.get("unRealizedProfit", 0)) for p in active)
        color = "#26a69a" if total_unrealized >= 0 else "#ef5350"
        st.markdown(
            f"**PnL Não-Realizado Total:** "
            f"<span style='color:{color};font-size:1.1em'>"
            f"${total_unrealized:+,.4f}</span>",
            unsafe_allow_html=True,
        )

        rows = []
        for p in active:
            amt  = float(p.get("positionAmt", 0))
            pnl  = float(p.get("unRealizedProfit", 0))
            ep   = float(p.get("entryPrice", 0))
            mp   = float(p.get("markPrice", 0))
            side = "LONG" if amt > 0 else "SHORT"
            pct  = (mp - ep) / ep * 100 * (1 if amt > 0 else -1) if ep > 0 else 0
            rows.append({
                "Símbolo": p.get("symbol", "?"),
                "Lado":    side,
                "Qty":     abs(amt),
                "Entrada": f"${ep:,.4f}",
                "Mark":    f"${mp:,.4f}",
                "PnL ($)": f"{'+' if pnl >= 0 else ''}{pnl:.4f}",
                "PnL (%)": f"{'+' if pct >= 0 else ''}{pct:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    except Exception as exc:
        logger.debug(f"[TAB-PERF] _render_open_positions: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_tab_performance(tab, engine) -> None:
    with tab:
        st.subheader("📊 Desempenho Histórico")

        orders: list[dict]        = list(engine.state.get("orders", []))
        closed_trades: list[dict] = list(engine.state.get("closed_trades", []))
        is_running: bool          = engine.state.get("running", False)

        # ── Stats do banco (todas as sessões) ─────────────────────────────────
        try:
            store      = get_trade_store()
            db_stats   = store.get_stats()
            db_daily   = store.get_daily_pnl(days=30)
            db_symbols = store.get_symbol_breakdown()
            db_total   = db_stats.get('total_trades', 0)
        except Exception:
            db_stats, db_daily, db_symbols, db_total = {}, [], [], 0

        # Banner persistente: mostra totais do banco mesmo na sessão vazia
        if db_total > 0:
            _sta, _stb, _stc, _std = st.columns(4)
            _db_pnl  = db_stats.get('total_pnl', 0.0) or 0.0
            _db_wr   = (db_stats.get('wins', 0) / db_total * 100) if db_total else 0
            _db_sess = db_stats.get('sessions', 0)
            _first   = db_stats.get('first_trade', '?')[:10]
            _sta.metric("📅 Trades (histórico)",  f"{db_total}")
            _stb.metric("💰 PnL Acumulado",          f"${_db_pnl:+,.2f}",
                        delta_color="normal" if _db_pnl >= 0 else "inverse")
            _stc.metric("🎯 Win Rate Global",       f"{_db_wr:.1f}%")
            _std.metric("🗓️ Sessions",              f"{_db_sess} (desde {_first})")
            st.divider()

        # ── Estado vazio da sessão atual ───────────────────────────────────
        if not orders and not closed_trades:
            if is_running:
                st.info("⏳ Engine em execução. Aguardando o primeiro trade da sessão...")
            else:
                st.info(
                    "💭 Engine não iniciada ou nenhuma ordem nesta sessão.  \n"
                    "Inicie a engine na aba **Engine** para começar a operar."
                )
            _render_open_positions()

            # Mostra histórico do banco mesmo sem sessão ativa
            if db_total > 0 and db_symbols:
                st.subheader("📊 Histórico por Par (todas as sessões)")
                _df_sym = pd.DataFrame(db_symbols)
                _df_sym.columns = ["Par", "Trades", "PnL Total", "PnL Médio", "Wins", "Losses"]
                _df_sym["Win Rate"] = (_df_sym["Wins"] / _df_sym["Trades"] * 100).map("{:.1f}%".format)
                _df_sym["PnL Total"] = _df_sym["PnL Total"].map("${:+,.4f}".format)
                _df_sym["PnL Médio"] = _df_sym["PnL Médio"].map("${:+,.4f}".format)
                st.dataframe(_df_sym.set_index("Par"), use_container_width=True)
            return

        # ── Posições abertas ─────────────────────────────────────────────
        _render_open_positions()

        # ── Contadores de sessão ─────────────────────────────────────────
        n_entries = len(orders)
        n_closed  = len(closed_trades)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📤 Entradas (sessão)", n_entries)
        c2.metric("✅ Trades Fechados",   n_closed,
                  help="Inclui histórico carregado do banco na inicialização")
        c3.metric("🔓 Posições Abertas",  _count_open_positions())
        _sess_id = get_trade_store().current_session_id()
        c4.caption(f"🗄️ `data/trades.db`  |  sessão `{_sess_id}`")

        # ── Métricas ──────────────────────────────────────────────────────
        metrics = calculate_performance_metrics(closed_trades)

        if not metrics:
            st.divider()
            st.caption("📈 Métricas disponíveis após o primeiro trade fechado (trail/SL/TP).")
        else:
            st.divider()
            # Linha 1 — principais
            m1, m2, m3, m4 = st.columns(4)
            total_pnl = metrics["total_pnl"]
            avg_pnl   = metrics["avg_trade_pnl"]
            wr        = metrics["win_rate"] * 100
            with m1:
                st.metric("🎯 Win Rate",   f"{wr:.1f}%",
                          f"{wr-50:+.1f}pp vs 50%",
                          delta_color="normal" if wr >= 50 else "inverse")
            with m2:
                st.metric("💰 P&L Total",  f"${total_pnl:+,.4f}",
                          delta_color="normal" if total_pnl >= 0 else "inverse")
            with m3:
                st.metric("📉 P&L / Trade", f"${avg_pnl:+,.4f}",
                          delta_color="normal" if avg_pnl >= 0 else "inverse")
            with m4:
                exp = metrics["expectancy"]
                st.metric("🎲 Expectativa", f"${exp:+,.4f}",
                          delta_color="normal" if exp >= 0 else "inverse")

            # Linha 2 — avançadas
            st.divider()
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                pf = metrics["profit_factor"]
                st.metric("⚖️ Profit Factor", f"{pf:.2f}x" if not math.isinf(pf) else "∞")
            with a2:
                st.metric("📐 Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with a3:
                dd = metrics["max_drawdown"]
                st.metric("📉 Max Drawdown", f"${dd:,.4f}",
                          delta_color="inverse" if dd < 0 else "normal")
            with a4:
                st.metric("📊 W / L", f"{metrics['wins']} / {metrics['losses']}")

            # Linha 3 — médias
            st.divider()
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("🟢 Ganho Médio", f"${metrics['avg_win']:+,.4f}")
            with b2:
                st.metric("🔴 Perda Média", f"${metrics['avg_loss']:+,.4f}")
            with b3:
                rf = metrics["recovery_factor"]
                st.metric("🔁 Recovery Factor", f"{rf:.2f}x" if not math.isinf(rf) else "∞")
            with b4:
                pass

        # ── Gráfico P&L ───────────────────────────────────────────────────
        if closed_trades:
            st.divider()
            fig = _build_pnl_chart(closed_trades)
            if fig:
                st.plotly_chart(fig, use_container_width=True)  # plotly_chart usa esse param ainda

        # ── Resumo por símbolo ────────────────────────────────────────────
        sym_df = _build_symbol_summary(closed_trades)
        if sym_df is not None and not sym_df.empty:
            st.divider()
            st.subheader("📂 Resumo por Símbolo")
            styled = sym_df.style.map(
                lambda v: ("color: #26a69a" if isinstance(v, float) and v > 0
                           else "color: #ef5350" if isinstance(v, float) and v < 0 else ""),
                subset=["P&L ($)", "Melhor", "Pior"],
            ).format({"P&L ($)": "{:+.4f}", "Melhor": "{:+.4f}", "Pior": "{:+.4f}"})
            st.dataframe(styled, width='stretch', hide_index=True)

        # ── Trades fechados ───────────────────────────────────────────────
        if closed_trades:
            st.divider()
            st.subheader(f"🔒 Trades Fechados ({n_closed})")
            closed_rows = []
            for t in reversed(closed_trades):
                pnl = float(t.get("realizedPnl", 0))
                closed_rows.append({
                    "Hora":    _ms_to_time(t.get("time")),
                    "Símbolo": t.get("symbol", "?"),
                    "Motivo":  t.get("side", "—"),
                    "Entrada": f"${float(t.get('entryPrice', 0)):,.4f}",
                    "Saída":   f"${float(t.get('exitPrice',  0)):,.4f}",
                    "Qty":     t.get("qty", "—"),
                    "P&L ($)": f"{'+' if pnl >= 0 else ''}{pnl:.4f}",
                })
            st.dataframe(pd.DataFrame(closed_rows), width='stretch', hide_index=True)

        # ── Entradas da sessão ────────────────────────────────────────────
        if orders:
            st.divider()
            st.subheader(f"📤 Entradas da Sessão ({n_entries})")
            entry_rows = []
            for o in reversed(orders):
                try:
                    ts = o.get("timestamp", "")
                    ts = ts[:19].replace("T", " ") if ts else "—"
                    price_raw = o.get("price", 0)
                    try:
                        price_str = f"${float(price_raw):,.4f}"
                    except (TypeError, ValueError):
                        price_str = str(price_raw)
                    entry_rows.append({
                        "Horário": ts,
                        "Símbolo": o.get("symbol", ""),
                        "Lado":    o.get("side", ""),
                        "Ação":    o.get("action", ""),
                        "Qty":     o.get("qty", 0),
                        "Preço":   price_str,
                    })
                except Exception as exc:
                    logger.warning(f"[TAB-PERF] Erro ao parsear entrada: {exc}")
            if entry_rows:
                st.dataframe(pd.DataFrame(entry_rows), width='stretch', hide_index=True)

        # ── Histórico 30 dias (banco SQLite — todas as sessões) ───────────────
        if db_daily:
            st.divider()
            st.subheader("📅 PnL Diário — últimos 30 dias (banco)")
            _df_d = pd.DataFrame(db_daily)
            _df_d["daily_pnl"] = _df_d["daily_pnl"].astype(float)
            _colors_d = ["#26a69a" if v >= 0 else "#ef5350" for v in _df_d["daily_pnl"]]
            _fig_d = go.Figure()
            _fig_d.add_bar(x=_df_d["day"], y=_df_d["daily_pnl"],
                           marker_color=_colors_d, name="PnL diário",
                           hovertemplate="%{x}<br>$%{y:+,.4f}<extra></extra>")
            _fig_d.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            _fig_d.update_layout(height=220, template="plotly_dark", showlegend=False,
                                 margin=dict(l=10, r=10, t=10, b=10))
            _fig_d.update_yaxes(tickprefix="$")
            st.plotly_chart(_fig_d, use_container_width=True)

        # ── Breakdown global por símbolo (banco) ──────────────────────────
        if db_symbols:
            st.divider()
            st.subheader("🌐 Resultado por Par (todas as sessões)")
            _df_sym = pd.DataFrame(db_symbols)
            _df_sym.columns = ["Par", "Trades", "PnL Total", "PnL Médio", "Wins", "Losses"]
            _df_sym["Win Rate"] = (_df_sym["Wins"] / _df_sym["Trades"].replace(0, 1) * 100).map("{:.1f}%".format)
            _df_sym["PnL Total"] = _df_sym["PnL Total"].map("${:+,.4f}".format)
            _df_sym["PnL Médio"] = _df_sym["PnL Médio"].map("${:+,.4f}".format)
            st.dataframe(_df_sym.set_index("Par"), use_container_width=True)

        # ── PDF Report download ───────────────────────────────────────────────
        st.divider()
        rc1, rc2, rc3 = st.columns([2, 2, 2])
        with rc1:
            rpt_start = st.date_input("📅 De", value=None, key="rpt_start",
                                      help="Filtro de data inicial para o relatório")
        with rc2:
            rpt_end = st.date_input("📅 Até", value=None, key="rpt_end",
                                    help="Filtro de data final para o relatório")
        with rc3:
            st.write("")  # vertical alignment
            if st.button("📄 Gerar Relatório PDF", key="gen_pdf_btn"):
                if not closed_trades:
                    st.warning("Nenhum trade fechado para incluir no relatório.")
                else:
                    with st.spinner("Gerando PDF..."):
                        try:
                            pdf_bytes = generate_monthly_report(
                                closed_trades,
                                start_date=rpt_start or None,
                                end_date=rpt_end or None,
                            )
                            filename = (
                                f"performance_report"
                                f"_{datetime.now():%Y%m%d_%H%M}.pdf"
                            )
                            st.download_button(
                                label="⬇️ Baixar PDF",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf",
                                key="download_pdf_btn",
                            )
                        except Exception as exc:
                            st.error(f"Erro ao gerar PDF: {exc}")
                            logger.exception("[TAB-PERF] Erro em generate_monthly_report")
