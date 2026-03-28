"""
Tab 5 — Controle da Engine: start/stop, decisões LSTM, logs e erros.
"""
from __future__ import annotations

from collections import deque
import os
import streamlit as st
import yaml

from dashboard.core.logging_setup import get_logger

logger = get_logger()


def render_tab_engine(tab, engine, selected_symbols: list[str]) -> None:
    with tab:
        st.subheader("⚙️ Controle da Trading Engine")

        # ── Kill Switch Banner ────────────────────────────────────────────
        _render_kill_switch_banner(engine)

        # ── Controls ──────────────────────────────────────────────────────
        _render_controls(engine, selected_symbols)
        st.divider()

        # ── Risk Config Editor ────────────────────────────────────────────
        _render_risk_config(engine)
        st.divider()

        # ── LSTM Decisions ────────────────────────────────────────────────
        _render_decisions(engine)
        st.divider()

        # ── Recent Orders ─────────────────────────────────────────────────
        _render_recent_orders(engine)
        st.divider()

        # ── Log Stream ────────────────────────────────────────────────────
        _render_log_stream(engine)

        # ── Errors ────────────────────────────────────────────────────────
        _render_errors(engine)


# ── Kill Switch Banner ─────────────────────────────────────────────────────

def _render_kill_switch_banner(engine) -> None:
    """Exibe banner vermelho enorme quando o kill switch for acionado."""
    state = engine.state
    if not state.get('kill_switch_triggered', False):
        # Mostrar status do drawdown mesmo quando não acionado
        peak     = float(state.get('peak_equity', 0.0))
        dd_pct   = float(state.get('current_drawdown_pct', 0.0))
        if peak > 0 and dd_pct > 0:
            color = "🔴" if dd_pct > 0.10 else ("🟡" if dd_pct > 0.05 else "🟢")
            st.info(
                f"{color} Drawdown atual: **{dd_pct:.1%}**  |  "
                f"Pico de equity: **${peak:,.2f}**"
            )
        return

    reason = state.get('kill_switch_reason', '')
    st.error(
        "### 🛑 KILL SWITCH ACIONADO\n\n"
        f"{reason}\n\n"
        "**Engine parada automaticamente.** "
        "Verifique sua conta antes de reiniciar."
    )

    # Botão para reiniciar (com confirmação)
    if st.button("⚡ Reconhecer e Liberar Kill Switch", key="kill_switch_reset_btn",
                 type="secondary"):
        engine._kill_switch_triggered = False
        with engine.lock:
            engine.state['kill_switch_triggered'] = False
            engine.state['kill_switch_reason']    = ''
            engine.state['peak_equity']           = 0.0
            engine.state['current_drawdown_pct']  = 0.0
        engine._peak_equity = 0.0
        st.success("Kill Switch reconhecido. O engine pode ser reiniciado.")
        st.rerun()


# ── Controls ──────────────────────────────────────────────────────────────────

def _render_controls(engine, selected_symbols: list[str]) -> None:
    is_running = engine.state.get('running', False)
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        if is_running:
            st.success("🟢 Engine RODANDO")
        else:
            st.warning("🔴 Engine PARADA")

    with ec2:
        if not is_running:
            if st.button("▶️ Iniciar Engine", type="primary", key="engine_start_btn"):
                if not selected_symbols:
                    st.error("Selecione ao menos um símbolo antes de iniciar.")
                else:
                    engine.start(symbols=selected_symbols)
                    logger.info(f"[TAB-ENG] Engine iniciada: {selected_symbols}")
                    st.success(f"Engine iniciando com {len(selected_symbols)} par(es)...")
                    st.rerun()
        else:
            # Botão para atualizar símbolos sem reiniciar
            if st.button("🔄 Atualizar Pares", key="engine_update_syms_btn",
                         help="Aplica imediatamente a seleção atual de pares"):
                with engine.lock:
                    engine.state['symbols'] = selected_symbols
                st.success(f"✅ Pares atualizados: {', '.join(selected_symbols)}")
            if st.button("⏹ Parar Engine", key="engine_stop_btn"):
                engine.stop()
                logger.info("[TAB-ENG] Engine parada pelo usuário")
                st.warning("Engine parando...")
                st.rerun()

    with ec3:
        engine_syms = engine.state.get('symbols', [])
        if engine_syms:
            st.caption(f"**Pares ativos ({len(engine_syms)}):**")
            st.caption(", ".join(engine_syms))
        if is_running:
            last_tick = engine.state.get('last_tick')
            if last_tick:
                st.caption(f"Último tick: {str(last_tick)[:19]}")
            else:
                st.caption("Aguardando primeiro tick...")


# ── Risk Config Editor ───────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")


def _render_risk_config(engine) -> None:
    """Inline editor for risk management parameters — saves to config.yaml atomically."""
    with st.expander("⚙️ Parâmetros de Risco", expanded=False):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            st.error(f"Não foi possível ler config.yaml: {exc}")
            return

        rm = cfg.get("risk_management", {})
        env = cfg.get("environment", {})

        with st.form("risk_config_form"):
            st.caption("Edite os parâmetros abaixo e clique em **Salvar** para aplicar. "
                       "A engine usa os novos valores no próximo tick.")

            c1, c2, c3 = st.columns(3)
            with c1:
                mode = st.selectbox(
                    "Modo de operação",
                    options=["paper", "testnet", "live"],
                    index=["paper", "testnet", "live"].index(cfg.get("mode", "testnet")),
                    key="ri_mode",
                )
                position_size = st.slider(
                    "Position size (%)", 0.5, 10.0,
                    float(env.get("position_size", 0.03)) * 100,
                    step=0.5, key="ri_pos",
                ) / 100

            with c2:
                stop_loss_pct = st.slider(
                    "Stop Loss (%)", 0.5, 5.0,
                    float(rm.get("stop_loss_pct", 0.02)) * 100,
                    step=0.1, key="ri_sl",
                ) / 100
                take_profit_pct = st.slider(
                    "Take Profit (%)", 1.0, 10.0,
                    float(rm.get("take_profit_pct", 0.04)) * 100,
                    step=0.5, key="ri_tp",
                ) / 100

            with c3:
                trailing_activation = st.slider(
                    "Trail Activation (%)", 0.5, 5.0,
                    float(rm.get("trailing_stop_activation", 0.03)) * 100,
                    step=0.25, key="ri_ta",
                ) / 100
                trailing_distance = st.slider(
                    "Trail Distance (%)", 0.25, 3.0,
                    float(rm.get("trailing_stop_distance", 0.015)) * 100,
                    step=0.25, key="ri_td",
                ) / 100

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                max_total_exp = st.slider(
                    "Exposição Máxima Total (%)", 10, 100,
                    int(float(rm.get("max_total_exposure", 0.60)) * 100),
                    step=5, key="ri_mte",
                ) / 100
            with ec2:
                max_per_asset = st.slider(
                    "Exposição Máx. por Ativo (%)", 5, 50,
                    int(float(rm.get("max_exposure_per_asset", 0.25)) * 100),
                    step=5, key="ri_mpa",
                ) / 100
            with ec3:
                leverage = st.slider(
                    "Leverage (x)", 1, 10,
                    int(float(env.get("leverage", 1.5))),
                    step=1, key="ri_lev",
                )

            fc1, fc2 = st.columns(2)
            with fc1:
                min_notional = st.number_input(
                    "Notional mínimo por trade ($)",
                    min_value=5.0, max_value=500.0,
                    value=float(rm.get("min_notional_usdt", 20.0)),
                    step=5.0, key="ri_min_not",
                    help="Posições abaixo desse valor são bloqueadas — as taxas consumiriam o lucro.",
                )
            with fc2:
                kelly_frac = st.slider(
                    "Kelly Fraction (%)", 5, 50,
                    int(float(rm.get("kelly_fraction", 0.25)) * 100),
                    step=5, key="ri_kf",
                    help="Fração conservadora do Kelly completo. Menor = mais conservador.",
                ) / 100

            # ── Indicador visual de impacto das taxas ───────────────────────
            fee_pct = float(env.get("commission", 0.0004))
            fee_rt  = fee_pct * 2 * 100  # round trip em %
            st.info(
                f"💡 **Impacto de taxas:** Round-trip ≈ **{fee_rt:.2f}%** | "
                f"No notional mínimo ${min_notional:.0f}: taxa = "
                f"**${min_notional * fee_pct * 2:.3f} USDT**  |  "
                f"Saldo mínimo recomendado para abrir posição: "
                f"**${min_notional / max(position_size * leverage, 0.001):,.0f} USDT**"
            )

            saved = st.form_submit_button("💾 Salvar configuração", type="primary")

        if saved:
            # Validation
            errors: list[str] = []
            if stop_loss_pct < 0.005:
                errors.append("SL mínimo: 0.5%")
            if take_profit_pct <= stop_loss_pct:
                errors.append("TP deve ser maior que SL")
            if trailing_activation < stop_loss_pct:
                errors.append("Trail Activation deve ser ≥ SL")
            if max_total_exp > 1.0:
                errors.append("Exposição total não pode exceder 100%")
            if min_notional < 5.0:
                errors.append("Notional mínimo não pode ser inferior a $5")
            if leverage < 1:
                errors.append("Leverage mínimo é 1x")
            if errors:
                for e in errors:
                    st.error(e)
                return

            # Apply
            cfg["mode"] = mode
            cfg.setdefault("environment", {}).update({
                "position_size": round(position_size, 4),
                "leverage":      leverage,
            })
            cfg.setdefault("risk_management", {}).update({
                "stop_loss_pct":            round(stop_loss_pct, 4),
                "take_profit_pct":          round(take_profit_pct, 4),
                "trailing_stop_activation": round(trailing_activation, 4),
                "trailing_stop_distance":   round(trailing_distance, 4),
                "max_total_exposure":       round(max_total_exp, 4),
                "max_exposure_per_asset":   round(max_per_asset, 4),
                "min_notional_usdt":        round(min_notional, 2),
                "kelly_fraction":           round(kelly_frac, 4),
            })

            # Atomic write
            tmp = _CONFIG_PATH + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False,
                              sort_keys=False)
                os.replace(tmp, _CONFIG_PATH)
                st.success("✅ Configuração salva — aplicada no próximo tick.")
                logger.info(f"[TAB-ENG] config.yaml atualizado via UI (mode={mode}, "
                            f"sl={stop_loss_pct:.1%}, tp={take_profit_pct:.1%})")
            except Exception as exc:
                st.error(f"Erro ao salvar config.yaml: {exc}")


# ── LSTM Decisions ────────────────────────────────────────────────────────────

def _render_decisions(engine) -> None:
    st.subheader("🧠 Último Sinal LSTM por Símbolo")
    decisions: dict = engine.state.get('decisions', {})
    portfolio: dict = engine.state.get('portfolio', {})

    if not decisions:
        st.info("Nenhuma decisão ainda — engine precisa rodar pelo menos 1 ciclo.")
        return

    for sym, info in decisions.items():
        with st.expander(f"**{sym}**", expanded=True):
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1:
                action     = info.get('action', 'HOLD')
                action_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
                color = action_colors.get(action, 'gray')
                st.markdown(f"**Ação:** <span style='color:{color};font-weight:bold'>{action}</span>",
                            unsafe_allow_html=True)
            with dc2:
                conf = info.get('confidence', 0)
                st.metric("Confiança", f"{conf:.1%}")
            with dc3:
                price = info.get('price', 0)
                st.metric("Preço", f"${price:,.2f}")
            with dc4:
                regime = info.get('regime', '—')
                st.metric("Regime", regime)

            # Portfolio stats
            port = portfolio.get(sym, {})
            if port:
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.metric("Trades", port.get('trade_count', 0))
                with pc2:
                    wins    = port.get('wins', 0)
                    total_t = port.get('trade_count', 0) or 1
                    st.metric("Win Rate", f"{wins/total_t:.0%}")
                with pc3:
                    st.metric("P&L Acum.", f"${port.get('total_pnl', 0):,.2f}")

            # Raw signal
            sig_val = info.get('signal', info.get('raw_action'))
            if sig_val is not None:
                st.caption(f"Raw signal: {sig_val}")


# ── Recent Orders ─────────────────────────────────────────────────────────────

def _render_recent_orders(engine) -> None:
    st.subheader("📋 Ordens Recentes (deque 50)")
    orders_dq = engine.state.get('orders', deque())
    orders    = list(orders_dq)

    if not orders:
        st.info("Nenhuma ordem executada nesta sessão.")
        return

    rows = []
    for o in reversed(orders[-20:]):  # últimas 20
        price_raw = o.get('price', 0)
        try:
            price_str = f"${float(price_raw):,.4f}"
        except (TypeError, ValueError):
            price_str = str(price_raw)
        rows.append(
            f"[{str(o.get('timestamp',''))[:19]}] "
            f"{o.get('symbol','')} {o.get('side','')} {o.get('action','')} "
            f"qty={o.get('qty',0)} @ {price_str}"
        )
    st.text_area("Últimas 20 ordens", "\n".join(rows), height=200, disabled=True)


# ── Log Stream ────────────────────────────────────────────────────────────────

def _render_log_stream(engine) -> None:
    st.subheader("📝 Log da Engine")
    log_dq   = engine.state.get('log', deque())
    log_list = list(log_dq)

    if not log_list:
        st.info("Nenhuma mensagem de log ainda.")
        return

    # Pegar as últimas 60 linhas
    log_text = "\n".join(log_list[-60:])
    st.text_area("Log (últimas 60 linhas)", log_text, height=300, disabled=True)


# ── Errors ────────────────────────────────────────────────────────────────────

def _render_errors(engine) -> None:
    errors_dq = engine.state.get('errors', deque())
    errors    = list(errors_dq)
    if not errors:
        return

    st.divider()
    st.subheader(f"⚠️ Erros Recentes ({len(errors)})")
    for err in reversed(errors[-10:]):
        ts  = str(err.get('timestamp', ''))[:19]
        msg = err.get('message', str(err))
        st.error(f"[{ts}] {msg}")
