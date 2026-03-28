# PLANO DE MODULARIZAÇÃO — dashboard.py
> Data de análise: 2026-02-28  
> Arquivo original: 3 228 linhas  
> Objetivo: refatorar para arquitetura profissional sem quebrar nenhuma funcionalidade existente

---

## 1. DIAGNÓSTICO — PROBLEMAS CRÍTICOS

### 1.1 Ordenação de instruções Streamlit (BUG LATENTE)
- `_get_ws_manager_singleton()` e `st.session_state['ws_manager'] = _ws_singleton` são chamados ao nível de módulo **antes** de `st.set_page_config()` (linha ~860).
- Qualquer outra chamada Streamlit antes de `set_page_config` vai causar `StreamlitAPIException` em alguns ambientes.
- `logger` é definido na linha ~866 mas as classes `BinanceWebSocketManager` e `TradingEngine` estão definidas acima e chamam `logger.*` em seus métodos — funciona por acaso (lazy), mas é uma bomba-relógio.
- **Regra a seguir**: `set_page_config()` deve ser a **primeira** chamada Streamlit; `logger` deve ser configurado **antes de qualquer classe**.

### 1.2 Duplicação massiva de código de indicadores técnicos
O bloco de cálculo (RSI, MACD, BB, ATR, SMA, EMA, Volume MA) existe em **dois lugares**:
- `BinanceWebSocketManager.get_klines_df()` (linha ~245)
- `collect_market_data()` fallback REST (linha ~1098)

São literalmente ~30 linhas copiadas duas vezes. Uma mudança num lugar não reflete no outro.

### 1.3 Lógica de TP/SL executada dentro do loop de render da UI
Em `tab2` (Posições), os blocos `if should_stop:`, `elif should_tp...` **fazem chamadas REST** (`close_position_direct`) durante a renderização do componente Streamlit. Isso é extremamente perigoso:
- Pode disparar múltiplos fechamentos se o usuário der F5 no momento errado.
- Mistura código de negócio com código de UI.
- `TradingEngine._check_tpsl()` já faz isso em background thread — há **duplicação de lógica de TP/SL** entre UI e engine.

### 1.4 Botão "❌ Fechar" duplicado
A lógica de fechar posição individual aparece em `tab2` AND `tab4` com código diferente para o mesmo endpoint (`close_position_direct`).

### 1.5 `load_dotenv()` chamado duas vezes
Linha 28 e novamente linha ~642 (após o singleton).

### 1.6 Caminho de modelo hardcoded dentro de uma função de cache
```python
lstm_v17_path = "models/recurrent_ppo_v17_lstm_20260221_030417_600000_steps.zip"
```
Deve vir de `config.yaml`.

### 1.7 `QUANTITY_PRECISION` dict embutido em função
`close_position_direct()` tem um dicionário de precisão por símbolo hardcoded. Deve ser config.

### 1.8 Tab3 (Performance) essencialmente morta
Toda a funcionalidade está comentada com um `st.warning()` de aviso. Deveria ser removida ou ter uma implementação alternativa via dados do engine (que já guarda as ordens em `engine.state['orders']`).

### 1.9 `time.sleep()` dentro de loops de render da UI
- `time.sleep(0.35)` no loop de gráficos por símbolo (aba Overview).
- `time.sleep(refresh_interval)` no bloco de auto-refresh.
Isso **bloqueia a thread principal do Streamlit**, congela a UI e pode causar timeout.
O delay de 350ms entre símbolos deveria estar no lugar do coletor de dados (WS-first já resolve isso sem sleep).

### 1.10 Não existe `if __name__ == "__main__":`
Todo o código de UI (tabs, sidebar, metricas) roda ao nível de módulo. Em Streamlit isso é normal, mas significa que não existe separação clara entre importação e execução.

### 1.11 `calculate_position_size_dynamic()` chama `load_config()` internamente
Uma função utilitária que "puxa" config via cache Streamlit. Isso acopla a lógica de negócio ao Streamlit, impossibilitando testes unitários.

### 1.12 Sem nenhum teste
Funções críticas como `prepare_observation()`, `lstm_predict()`, `validate_entry_quality()` têm zero cobertura de testes.

---

## 2. ESTRUTURA PROPOSTA DE MÓDULOS

```
dashboard/                          ← pasta nova (o arquivo raiz vira entry-point)
├── __init__.py
│
├── core/
│   ├── __init__.py
│   ├── config.py                   ← load_config(), load_config_raw(), constantes globais
│   ├── logging_setup.py            ← configuração de logging, UTF-8 handler
│   └── ban_manager.py              ← _is_banned(), _register_ban(), _rest_rate_ok(), _touch_rest_rate()
│
├── data/
│   ├── __init__.py
│   ├── indicators.py               ← _compute_indicators(df) — função única, sem duplicação
│   ├── websocket_manager.py        ← classe BinanceWebSocketManager
│   ├── market_data.py              ← collect_market_data(), collect_multi_timeframe_data(), get_klines()
│   └── account_data.py             ← get_account_balance*(), get_open_positions*(), get_recent_trades()
│
├── trading/
│   ├── __init__.py
│   ├── observation.py              ← FEATURE_COLS_15M, prepare_observation(), lstm_predict()
│   ├── executor.py                 ← execute_trade(), close_position_direct(), close_all_positions()
│   ├── engine.py                   ← classe TradingEngine (background thread)
│   └── entry_filter.py             ← validate_entry_quality()
│
├── analytics/
│   ├── __init__.py
│   ├── performance.py              ← calculate_performance_metrics(), calculate_position_size_dynamic()
│   └── regime.py                   ← detect_market_regime(), calculate_atr(), calculate_correlation()
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                  ← render_sidebar() → retorna selected_symbols, config choices
│   ├── tab_overview.py             ← render_tab_overview(tab, data)
│   ├── tab_positions.py            ← render_tab_positions(tab, data)  ← SEM lógica de TP/SL!
│   ├── tab_performance.py          ← render_tab_performance(tab, data)
│   ├── tab_analysis.py             ← render_tab_analysis(tab, data)
│   ├── tab_engine.py               ← render_tab_engine(tab, engine)
│   └── charts.py                   ← plot_candlestick(), plot_pnl_chart()
│
└── resources.py                    ← @st.cache_resource singletons (client, ws_manager, engine, models…)

dashboard.py  (entry-point mínimo, ~50 linhas)
├── st.set_page_config()  ← PRIMEIRA LINHA
├── import de módulos
└── main() com tabs e sidebar
```

---

## 3. PLANO DE IMPLEMENTAÇÃO DETALHADO (em ordem)

> Siga esta ordem estritamente para evitar quebrar a aplicação enquanto refatora.
> A cada passo, rode `streamlit run dashboard.py` para validar que não quebrou.

---

### FASE 0 — Preparação (sem mudar comportamento)

**Passo 0.1** — Criar estrutura de pastas

```
mkdir dashboard
mkdir dashboard\core
mkdir dashboard\data
mkdir dashboard\trading
mkdir dashboard\analytics
mkdir dashboard\ui
```

Criar `__init__.py` vazio em cada pasta.

**Passo 0.2** — Mover `logging_setup.py` primeiro

Extrair do dashboard.py (linhas ~850-870) para `dashboard/core/logging_setup.py`:
```python
# dashboard/core/logging_setup.py
import logging, sys
from pathlib import Path

def setup_logging() -> logging.Logger:
    log_file = Path("logs/trading_decisions.log")
    log_file.parent.mkdir(exist_ok=True)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    return logging.getLogger('trading_bot')
```

Em todos os módulos depois, faça:
```python
from dashboard.core.logging_setup import setup_logging
logger = setup_logging()
```

---

### FASE 1 — Extrair módulos puros (sem dependência Streamlit)

**Passo 1.1** — `dashboard/core/config.py`

Mover:
- `load_config_raw()`
- `load_config()` (sem `@st.cache_resource` — faremos no resources.py depois)
- Constantes: `_KLINE_MAXLEN`, `_INTERVALS_WS`, `_KLINE_LIMIT_BOOT`
- `_BAN_FILE`, `_REST_RATE_FILE`, `_REST_COOLDOWN_SECS`

**Passo 1.2** — `dashboard/core/ban_manager.py`

Mover (sem ST):
- `_is_banned()`
- `_register_ban()`
- `_rest_rate_ok()`
- `_touch_rest_rate()`

Remover dependência de `st.session_state` desta camada. O ban_manager deve funcionar
apenas com arquivo + parâmetros. Streamlit vai ler/escrever o session_state no `resources.py`.

**Passo 1.3** — `dashboard/data/indicators.py`

```python
# dashboard/data/indicators.py
import pandas as pd, numpy as np, talib

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores técnicos normalizados.
    Entrada: df com colunas [open, high, low, close, volume]
    Saída: df com colunas adicionais de indicadores
    """
    ...  # código único — extraído de get_klines_df() e collect_market_data()
```

Remover duplicação: ambos `BinanceWebSocketManager.get_klines_df()` e  
`collect_market_data()` passam a chamar `compute_indicators(df)`.

**Passo 1.4** — `dashboard/data/websocket_manager.py`

Mover toda a classe `BinanceWebSocketManager` (linhas 42-357).
Substituir `logger = logging.getLogger(__name__)` por import do logging_setup.

**Passo 1.5** — `dashboard/trading/observation.py`

Mover:
- `FEATURE_COLS_15M`
- `IDX_CLOSE`, `IDX_RSI`, `IDX_BBP`, `IDX_MACDH`
- `prepare_observation()`
- `lstm_predict()`

**Passo 1.6** — `dashboard/trading/entry_filter.py`

Mover `validate_entry_quality()`.

**Passo 1.7** — `dashboard/trading/executor.py`

Mover:
- `execute_trade()`
- `close_position_direct()`
- `close_all_positions()`

Mover `QUANTITY_PRECISION` para `config.yaml`:
```yaml
quantity_precision:
  BTCUSDT: 3
  ETHUSDT: 3
  ...
```

**Passo 1.8** — `dashboard/analytics/regime.py`

Mover:
- `detect_market_regime()`
- `calculate_atr()`
- `calculate_correlation()`

**Passo 1.9** — `dashboard/analytics/performance.py`

Mover:
- `calculate_performance_metrics()`
- `calculate_position_size_dynamic()` — remover `load_config()` interno, passar `config` como parâmetro

**Passo 1.10** — `dashboard/trading/engine.py`

Mover classe `TradingEngine` (linhas 415-629).
Importar módulos já extraídos nas etapas anteriores.

---

### FASE 2 — Extrair módulos com dependência Streamlit

**Passo 2.1** — `dashboard/resources.py`

Centralizar todos os `@st.cache_resource`:
```python
# dashboard/resources.py
import streamlit as st
from binance.client import Client
from dashboard.data.websocket_manager import BinanceWebSocketManager
from dashboard.trading.engine import TradingEngine
...

@st.cache_resource
def get_config(): ...

@st.cache_resource
def get_binance_client(): ...

@st.cache_resource
def get_ws_manager() -> BinanceWebSocketManager: ...

@st.cache_resource
def get_trading_engine() -> TradingEngine: ...

@st.cache_resource
def get_risk_manager(): ...

@st.cache_resource
def get_models(): ...
```

**Passo 2.2** — `dashboard/data/account_data.py`

Mover:
- `get_account_balance_cached()`
- `get_account_balance()`
- `get_open_positions_cached()`
- `get_open_positions()`
- `get_recent_trades()`

**Passo 2.3** — `dashboard/data/market_data.py`

Mover:
- `collect_market_data()`
- `collect_multi_timeframe_data()`
- `get_klines()`

Ambas chamam `compute_indicators()` do `indicators.py`.

---

### FASE 3 — Extrair componentes de UI

**Passo 3.1** — `dashboard/ui/charts.py`

Mover:
- `plot_candlestick()`
- `plot_pnl_chart()`

**Passo 3.2** — `dashboard/ui/sidebar.py`

Extrair todo o bloco `with st.sidebar:` em uma função:
```python
def render_sidebar(config, ws_manager) -> dict:
    """
    Renderiza a sidebar completa.
    Retorna: {
        'selected_symbols': [...],
        'allocation_strategy': '...',
        'auto_refresh': bool,
        'refresh_interval': int,
        ...
    }
    """
```

**Passo 3.3** — `dashboard/ui/tab_overview.py`

```python
def render_tab_overview(tab, balance, positions, selected_symbols, client, config):
    with tab:
        ...
```

**Passo 3.4** — `dashboard/ui/tab_positions.py`

```python
def render_tab_positions(tab, positions, client, config, trailing_mgr, risk_mgr):
    with tab:
        ...
```

**IMPORTANTE**: Remover completamente a lógica de Auto-TP/SL desta aba.
O `TradingEngine._check_tpsl()` já faz isso. A UI deve apenas EXIBIR o estado.
Manter somente o botão manual "❌ Fechar".

**Passo 3.5** — `dashboard/ui/tab_performance.py`

Reimplementar usando `engine.state['orders']` (dados do engine, zero REST):
```python
def render_tab_performance(tab, engine):
    with tab:
        orders = list(engine.state.get('orders', []))
        # exibe P&L calculado a partir das ordens do engine, não de REST calls
```

**Passo 3.6** — `dashboard/ui/tab_analysis.py`

Mover conteúdo de `tab4`.

**Passo 3.7** — `dashboard/ui/tab_engine.py`

Mover conteúdo de `tab5`.

---

### FASE 4 — Reescrever o entry-point

**Passo 4.1** — Novo `dashboard.py` (~60 linhas):

```python
"""
📊 Trading Bot Dashboard — Entry point
"""
# 1. st.set_page_config DEVE ser a primeira instrução Streamlit
import streamlit as st
st.set_page_config(
    page_title="🤖 Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Imports internos (após set_page_config)
from dashboard.core.logging_setup import setup_logging
from dashboard.core.ban_manager import restore_ban_to_session
from dashboard.resources import get_ws_manager, get_trading_engine, get_config, get_binance_client
from dashboard.ui.sidebar import render_sidebar
from dashboard.ui.tab_overview import render_tab_overview
from dashboard.ui.tab_positions import render_tab_positions
from dashboard.ui.tab_performance import render_tab_performance
from dashboard.ui.tab_analysis import render_tab_analysis
from dashboard.ui.tab_engine import render_tab_engine

logger = setup_logging()

def main():
    ws_manager = get_ws_manager()
    engine     = get_trading_engine()
    config     = get_config()
    client     = get_binance_client()

    # Restaura ban persistido em arquivo (sem fazer REST call)
    restore_ban_to_session()

    # Sidebar — retorna configurações do usuário
    sidebar_state = render_sidebar(config, ws_manager)
    selected_symbols = sidebar_state['selected_symbols']

    # Carrega dados de conta (WS-first, zero REST se bootstrapped)
    balance, positions = load_account_snapshot(ws_manager, client)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💰 Posições", "📈 Performance",
        "🔬 Análise Avançada", "⚙️ Engine"
    ])
    render_tab_overview(tab1, balance, positions, selected_symbols, client, config)
    render_tab_positions(tab2, positions, client, config)
    render_tab_performance(tab3, engine)
    render_tab_analysis(tab4, selected_symbols, client, config, sidebar_state)
    render_tab_engine(tab5, engine, selected_symbols)

    # Auto-refresh (apenas se WS bootstrapped)
    handle_auto_refresh(sidebar_state, ws_manager)

if __name__ == "__main__":
    main()
```

---

### FASE 5 — Testes e validação

**Passo 5.1** — Testes unitários para funções puras

Criar `tests/` com:
- `test_indicators.py` — valida `compute_indicators()` com df sintético
- `test_observation.py` — valida shape `(50, 31)` de `prepare_observation()`
- `test_entry_filter.py` — valida `validate_entry_quality()` com cenários LONG/SHORT/FLAT
- `test_ban_manager.py` — valida persistência de ban em arquivo

**Passo 5.2** — Validação de integração

```
streamlit run dashboard.py
```
Verificar:
- [ ] Dashboard abre sem erro
- [ ] Sidebar renderiza
- [ ] WebSocket inicia
- [ ] Bootstrap carrega candles
- [ ] Tabs renderizam sem REST call
- [ ] Engine inicia e processa candles

---

## 4. MELHORIAS ADICIONAIS (backlog pós-refatoração)

| Prioridade | Item |
|---|---|
| 🔴 Alta | Mover caminho do modelo LSTM para `config.yaml` → `models.lstm_v17.path` |
| 🔴 Alta | Mover `QUANTITY_PRECISION` para `config.yaml` → `trading.quantity_precision` |
| 🔴 Alta | Remover `time.sleep(0.35)` do loop de render — usar WS-first resolve sem sleep |
| 🟡 Média | Tab3 Performance reimplementada com `engine.state['orders']` (zero REST) |
| 🟡 Média | Configurar `config.yaml` com schema de validação (pydantic ou dataclass) |
| 🟡 Média | Adicionar `st.cache_data` com TTL para `render_sidebar` evitar re-render completo |
| 🟢 Baixa | Dark/light mode toggle na sidebar |
| 🟢 Baixa | WebSocket reconnect automático após timeout |
| 🟢 Baixa | Exportar histórico de trades para CSV diretamente da UI |

---

## 5. CHECKLIST DE EXECUÇÃO (usar para tracking)

- [ ] **FASE 0** — Pastas e logging_setup
- [ ] **FASE 1.1** — core/config.py
- [ ] **FASE 1.2** — core/ban_manager.py
- [ ] **FASE 1.3** — data/indicators.py ← remove duplicação
- [ ] **FASE 1.4** — data/websocket_manager.py
- [ ] **FASE 1.5** — trading/observation.py
- [ ] **FASE 1.6** — trading/entry_filter.py
- [ ] **FASE 1.7** — trading/executor.py ← move QUANTITY_PRECISION para config
- [ ] **FASE 1.8** — analytics/regime.py
- [ ] **FASE 1.9** — analytics/performance.py ← remove load_config() interno
- [ ] **FASE 1.10** — trading/engine.py
- [ ] **FASE 2.1** — resources.py
- [ ] **FASE 2.2** — data/account_data.py
- [ ] **FASE 2.3** — data/market_data.py
- [ ] **FASE 3.1** — ui/charts.py
- [ ] **FASE 3.2** — ui/sidebar.py
- [ ] **FASE 3.3** — ui/tab_overview.py
- [ ] **FASE 3.4** — ui/tab_positions.py ← REMOVE lógica Auto-TP/SL da UI
- [ ] **FASE 3.5** — ui/tab_performance.py ← REIMPLEMENTAR com dados do engine
- [ ] **FASE 3.6** — ui/tab_analysis.py
- [ ] **FASE 3.7** — ui/tab_engine.py
- [ ] **FASE 4.1** — Novo dashboard.py entry-point
- [ ] **FASE 5** — Testes e validação final

---

## 6. RESUMO EXECUTIVO DO QUE MAIS IMPORTA

1. **Bug de ordenação**: `set_page_config` e `logger` devem ser as **primeiras** coisas a inicializar. Corrija isso antes de qualquer outra coisa.
2. **Duplicação de indicadores**: uma única função `compute_indicators(df)` em `data/indicators.py` elimina ~60 linhas duplicadas e garante consistência.
3. **TP/SL na UI é perigoso**: remover do tab_positions, deixar 100% na TradingEngine.
4. **Tab Performance morta**: reimplementar com `engine.state['orders']` e zero REST.
5. **`time.sleep()` bloqueia UI**: remover completamente — o WebSocket resolve sem sleep.
6. **O resto é organização**: as fases 1-4 acima transformam um monólito de 3228 linhas em ~15 módulos pequenos, testáveis e legíveis.
