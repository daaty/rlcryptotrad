"""
Entry-point do Dashboard Modular — V2 (Refatorado).

Regra obrigatória Streamlit:
  st.set_page_config() DEVE ser a primeira instrução (antes de qualquer import
  de módulos que usem `st`).

Para rodar:
    streamlit run dashboard_new.py
"""
import streamlit as st
import os

# ── ① set_page_config — PRIMEIRA INSTRUÇÃO ABSOLUTA ──────────────────────────
st.set_page_config(
    page_title="🤖 Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ② Todos os outros imports DEPOIS de set_page_config ──────────────────────
from dashboard.core.logging_setup import setup_logging
from dashboard.resources import (
    get_config,
    get_binance_client,
    get_ws_manager,
    get_trading_engine,
    restore_ban_to_session,
    is_banned_session,
    init_session_defaults,
)
from dashboard.data.account_data import (
    get_account_balance_cached,
    get_open_positions_cached,
)
from dashboard.ui.sidebar import render_sidebar
from dashboard.ui.tab_overview    import render_tab_overview
from dashboard.ui.tab_positions   import render_tab_positions
from dashboard.ui.tab_performance import render_tab_performance
from dashboard.ui.tab_analysis    import render_tab_analysis
from dashboard.ui.tab_engine      import render_tab_engine
from dashboard.ui.tab_challenger  import render_tab_challenger

# ── ③ Logging e sessão ────────────────────────────────────────────────────────
logger = setup_logging()
init_session_defaults()
restore_ban_to_session()

# ── ④ Singletons (cache_resource) ────────────────────────────────────────────
config     = get_config()
client     = get_binance_client()
ws_manager = get_ws_manager()
engine     = get_trading_engine()

# ── ④-b Autenticação (opcional — ativar em config.yaml: auth.enabled: true) ───────
_auth_enabled = config.get("auth", {}).get("enabled", False)
if _auth_enabled:
    _auth_config_path = os.path.join(os.path.dirname(__file__), "auth_config.yaml")
    if not os.path.exists(_auth_config_path):
        st.error("❌ auth_config.yaml não encontrado. Crie-o ou desative auth.enabled em config.yaml.")
        st.stop()
    try:
        import streamlit_authenticator as _stauth
        _auth_cfg = config.get("auth", {})
        _authenticator = _stauth.Authenticate(
            credentials=_auth_config_path,
            cookie_name=_auth_cfg.get("cookie_name", "trading_bot_auth"),
            cookie_key=_auth_cfg.get("cookie_key", "change-this-key"),
            cookie_expiry_days=float(_auth_cfg.get("cookie_expiry_days", 7)),
        )
        _authenticator.login()
        _auth_status = st.session_state.get("authentication_status")
        if _auth_status is False:
            st.error("🔒 Usuário ou senha incorretos.")
            st.stop()
        elif _auth_status is None:
            st.info("↗️ Faça login para acessar o dashboard.")
            st.stop()
        # _auth_status is True — fall through and render dashboard
        _authenticator.logout(button_name="🚪 Sair", location="sidebar")
    except ImportError:
        st.warning("⚠️ streamlit-authenticator não instalado. Auth desativado. Instale com: pip install streamlit-authenticator")

# ── ⑤ CSS customizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #0d3f8b;
        text-align: center;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 1.2rem;
        color: #0d3f8b;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .card-box {
        background-color: #f5f7fb;
        border: 1px solid #d8e2f2;
        border-radius: 0.8rem;
        padding: 0.9rem;
        box-shadow: 0 5px 20px rgba(32, 79, 153, 0.08);
        margin-bottom: 0.65rem;
    }
    .stAlert {
        border-radius: 0.8rem !important;
    }
    .positive { color: #218838; font-weight: 700; }
    .negative { color: #c82333; font-weight: 700; }
    .neutral { color: #0c5460; font-weight: 700; }
    .stButton>button {
        border-radius: 0.65rem;
    }
    .sidebar .stTextInput>div>div>input,
    .sidebar .stSelectbox>div>div>div>select,
    .sidebar .stSlider>div>div>div>input {
        border-radius: 0.5rem;
    }
    .stTabs>div>button {
        border-top-left-radius: 0.7rem;
        border-top-right-radius: 0.7rem;
        margin-right: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── ⑥ Título ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🤖 Trading Bot Dashboard — LSTM V17.7</h1>',
            unsafe_allow_html=True)

# Paper mode banner
if config.get("mode", "testnet") == "paper":
    st.warning("📄 **PAPER MODE** — operações simuladas, nenhuma ordem real enviada à Binance.")

# ── ⑥-b Auto-bootstrap na primeira carga ─────────────────────────────────────
# Usa cache em disco (kline_cache/) — reinicializações carregam em <1s,
# fazendo REST apenas para o delta de candles faltantes (~2-3 calls vs 57).
# Flag de session_state evita double-bootstrap em re-runs do Streamlit.
if not ws_manager.bootstrap_done and not st.session_state.get('_bootstrap_started'):
    st.session_state['_bootstrap_started'] = True  # lock imediato antes do REST
    _primary = config.get('data', {}).get('primary_symbol', 'BTC/USDT').replace('/', '')
    _auto_symbols = [_primary]
    _banned_auto, _ = is_banned_session()
    if not _banned_auto:
        with st.spinner(f"⚡ Carregando {_primary} (cache/REST)..."):
            try:
                _n = ws_manager.bootstrap_klines(_auto_symbols)
                ws_manager.bootstrap_account()
                st.session_state['_rest_connected'] = True
                logger.info(f"[AUTO-BOOT] {_n} candles carregados para {_auto_symbols}")
            except Exception as _boot_err:
                from dashboard.resources import register_ban_session
                register_ban_session(str(_boot_err), 'AUTO_BOOTSTRAP')
                logger.warning(f"[AUTO-BOOT] Erro: {_boot_err}")
    else:
        st.warning("🚫 Ban ativo — bootstrap adiado. Aguarde o cooldown e recarregue a página.")

# ── ⑦ Sidebar → preferências do usuário ──────────────────────────────────────
sidebar_state    = render_sidebar(config, ws_manager)
selected_symbols = sidebar_state['selected_symbols']

# ── ⑦-b Bootstrap incremental: novos símbolos selecionados ───────────────────
# Só executa se WS já fez bootstrap inicial (garante sequência correta).
# O novo bootstrap_klines usa cache — símbolos já cacheados carregam instantâneo,
# sem REST, sem risco de ban.
_already_booted = set(ws_manager.kline_buffers.keys())
_new_symbols    = [s for s in selected_symbols if s.upper() not in _already_booted]
if _new_symbols and ws_manager.bootstrap_done:
    _boot_key = f"_boot_inc_{'_'.join(sorted(_new_symbols))}"
    if not st.session_state.get(_boot_key):
        st.session_state[_boot_key] = True  # lock por conjunto de símbolos
        _banned_inc, _ = is_banned_session()
        if not _banned_inc:
            with st.spinner(f"⚡ Bootstrapping novos pares: {', '.join(_new_symbols)}..."):
                try:
                    ws_manager.bootstrap_klines(_new_symbols)
                    logger.info(f"[BOOT-INC] Bootstrapped: {_new_symbols}")
                except Exception as _inc_err:
                    logger.warning(f"[BOOT-INC] Erro: {_inc_err}")

# ── ⑦-c Sincroniza símbolos da engine com seleção atual do sidebar ─────────────
# Faz isso a cada rerun para que mudanças no sidebar se reflitam na engine
# sem precisar parar/reiniciar. Se engine está parada, apenas prepara a lista.
if selected_symbols:
    _current_engine_syms = list(engine.state.get('symbols', []))
    if set(selected_symbols) != set(_current_engine_syms):
        with engine.lock:
            engine.state['symbols'] = selected_symbols
        logger.info(f"[SYNC] Símbolos da engine atualizados: {selected_symbols}")

# ── ⑧-a Persiste cache kline em disco (1 vez por sessão após boot) ────────────
# Garante que o restart seguinte carregue do cache e não faça 57 REST calls.
if ws_manager.bootstrap_done and not st.session_state.get('_kline_cache_saved'):
    try:
        _saved = ws_manager.save_kline_cache()
        if _saved:
            st.session_state['_kline_cache_saved'] = True
    except Exception as _ce:
        logger.debug(f"[CACHE] Erro ao salvar: {_ce}")

# ── ⑧ Fragment LIVE — status bar + todas as abas ──────────────────────────────
# O fragment re-renderiza TODO o conteúdo dinâmico a cada N segundos usando
# dados do WS singleton (in-memory). ZERO chamadas REST. ZERO st.rerun().
# Streamlit re-executa apenas o corpo do fragment → UI sempre fresca.
_auto_refresh      = sidebar_state.get('auto_refresh', True)
_page_refresh_secs = max(5, sidebar_state.get('refresh_interval', 5))

@st.fragment(run_every=_page_refresh_secs if _auto_refresh else None)
def _live_dashboard() -> None:
    import time as _time

    balance_raw   = get_account_balance_cached(client)
    positions_raw = get_open_positions_cached(client)
    balance       = balance_raw
    positions     = positions_raw.get('positions', []) if isinstance(positions_raw, dict) else []

    # ── Barra de status ───────────────────────────────────────────────────────
    _banned, _ban_expires = is_banned_session()
    # Também verifica ban detectado pelo WS manager (REST -1003)
    _ws_ban_remaining = int(balance_raw.get('ban_remaining', 0)) if isinstance(balance_raw, dict) else 0
    _ws_ban_until     = balance_raw.get('ban_until') if isinstance(balance_raw, dict) else None
    _any_ban = _banned or _ws_ban_remaining > 0

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    sb1, sb2, sb3, sb4 = st.columns(4)
    with sb1:
        if _banned:
            remaining = max(0, _ban_expires - _time.time())
            st.error(f"🚫 BAN ATIVO — {remaining:.0f}s")
        elif _ws_ban_remaining > 0:
            from datetime import datetime as _dt
            ban_str = _dt.fromtimestamp(_ws_ban_until).strftime('%H:%M:%S') if _ws_ban_until else '?'
            h, m = divmod(_ws_ban_remaining // 60, 60)
            s    = _ws_ban_remaining % 60
            t    = f"{h}h{m}m{s}s" if h else (f"{m}m{s}s" if m else f"{s}s")
            st.error(f"⛔ IP BANIDO até {ban_str} ({t})")
        else:
            st.success("✅ API: OK")
    with sb2:
        ws_symb  = list(ws_manager.kline_buffers.keys()) if ws_manager else []
        age_secs = balance_raw.get('age_secs', None) if isinstance(balance_raw, dict) else None
        if _ws_ban_remaining > 0:
            st.warning(f"⏸️ REST pausado — via WS")
        elif age_secs is not None:
            icon  = "🟢" if age_secs < 10 else ("🟡" if age_secs < 60 else "🔴")
            label = f"< 1s" if age_secs < 1 else f"{age_secs:.0f}s"
            st.success(f"{icon} Dados: {label} atrás ⚡LIVE")
        elif ws_symb:
            st.success(f"🌐 WS: {len(ws_symb)} stream(s)")
        else:
            st.warning("🌐 WS: desconectado")
    with sb3:
        engine_ok = engine.state.get('running', False)
        if engine_ok:
            st.success("⚙️ Engine: ATIVA")
        else:
            st.warning("⚙️ Engine: parada")
    with sb4:
        total_bal = float(balance.get('total', 0) if isinstance(balance, dict) else 0)
        st.metric("💰 Carteira", f"${total_bal:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    rm = config.get('risk_management', {})
    env = config.get('environment', {})
    c1.metric('🛡️ Max Drawdown', f"{float(rm.get('max_drawdown', 0.15))*100:.1f}%")
    c2.metric('⚖️ Max Exposição Total', f"{float(rm.get('max_total_exposure', 0.60))*100:.1f}%")
    c3.metric('🧩 Kelly Fraction', f"{float(rm.get('kelly_fraction', 0.25))*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Abas principais ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Visão Geral",
        "💼 Posições",
        "📊 Desempenho",
        "🔬 Análise",
        "⚙️ Engine",
        "🏆 Champion/Challenger",
    ])

    render_tab_overview(tab1, balance, positions, selected_symbols, client, config)
    render_tab_positions(tab2, positions, client, config)
    render_tab_performance(tab3, engine)
    render_tab_analysis(tab4, selected_symbols, client, config, sidebar_state, positions)
    render_tab_engine(tab5, engine, selected_symbols)
    render_tab_challenger(tab6, config)

_live_dashboard()
