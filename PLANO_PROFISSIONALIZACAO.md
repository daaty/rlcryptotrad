# Plano de Profissionalização — Trading Bot Dashboard
> Gerado em 28/02/2026. Siga as fases em ordem. Marque `[x]` ao concluir cada item.

---

## Como executar este plano (instruções para agente IA)

1. Leia este arquivo inteiro antes de começar.
2. Execute uma fase por vez, na ordem indicada.
3. Sempre leia os arquivos relevantes **antes** de editar.
4. Após cada tarefa, marque `[x]` e registre o arquivo modificado.
5. Se uma tarefa depende de outra, verifique se a anterior está `[x]`.
6. Nunca apague código existente sem confirmar que o novo cobre os mesmos casos.

---

## Arquivos-chave do sistema

| Arquivo | Responsabilidade |
|---|---|
| `dashboard/trading/engine.py` | Loop principal, inferência, TP/SL |
| `dashboard/trading/executor.py` | Abertura/fechamento de ordens Binance |
| `dashboard/trading/observation.py` | Prepara observação para o modelo LSTM |
| `dashboard/trading/entry_filter.py` | Filtros de qualidade de entrada |
| `dashboard/data/websocket_manager.py` | Buffers de candles WS |
| `dashboard/resources.py` | Singletons (client, ws_mgr, engine, models) |
| `dashboard/analytics/performance.py` | Métricas de desempenho |
| `dashboard/ui/tab_performance.py` | Aba de desempenho |
| `dashboard/ui/tab_engine.py` | Aba de controle do engine |
| `dashboard/ui/tab_positions.py` | Aba de posições abertas |
| `config.yaml` | Configuração central |
| `src/risk/risk_manager.py` | RiskManager legado (CCXT) |

---

## FASE 1 — Tornar o bot seguro (prioridade máxima)
> Bugs que causam perda real de dinheiro. Resolver antes de qualquer outra fase.

### 1.1 — Persistência de estado em disco
**Risco eliminado:** Posições órfãs sem SL/TP após reinicialização.  
**Arquivo:** `dashboard/trading/engine.py`  
**Arquivo novo:** `dashboard/trading/state_persistence.py`

- [x] Criar `dashboard/trading/state_persistence.py` com funções:
  - `save_state(state_dict, path="data/engine_state.json")` — serializa `_tp1_done` (set→list), `_last_candle_ts`, `lstm_states` (ndarray→base64), trail positions
  - `load_state(path="data/engine_state.json") -> dict | None` — desserializa; retorna None se arquivo não existe ou está corrompido
  - Usar `json` + `base64` + `numpy` para serializar ndarrays
- [x] No `TradingEngine.__init__`, chamar `load_state()` e restaurar `_tp1_done`, `_last_candle_ts`, `state['lstm_states']`
- [x] No final de cada `_tick` (bloco try), chamar `save_state()` com os campos acima
- [x] Criar diretório `data/` se não existir
- [x] Testar: iniciar engine, abrir posição, reiniciar, verificar que `_tp1_done` e `_last_candle_ts` foram restaurados

### 1.2 — Reconciliação de posições no boot
**Risco eliminado:** Abrir segunda posição no mesmo ativo que já está aberto.  
**Arquivo:** `dashboard/trading/engine.py` (método `_loop`)

- [x] No início de `_loop`, após carregar recursos, chamar `client.futures_position_information()`
- [x] Para cada posição com `positionAmt != 0`:
  - Registrar no `trail_mgr` via `trail_mgr.register_position(sym, entryPrice, ptype)` se ainda não registrado
  - Adicionar sym ao `_tp1_done` se a posição for < 50% da qty de entrada (heurística: posição já teve TP1)
  - Logar posição encontrada: `[BOOT] Posição aberta encontrada: {sym} qty={qty} entry={entry}`
- [x] Logar quantidade de posições reconciliadas no final
- [x] Garantir que `trail_mgr.get_stop_info(sym)` não retorna None após reconciliação

### 1.3 — Mover SL para breakeven após TP1
**Risco eliminado:** Posição que já realizou 50% pode ainda perder tudo no SL original.  
**Arquivo:** `dashboard/trading/engine.py` (método `_check_tpsl`)

- [x] Após o bloco `elif tp_level == 1 and sym not in self._tp1_done:`, adicionar:
  ```python
  # Move SL → breakeven após TP1
  trail_mgr.update_entry_price(sym, entry)  # breakeven = preço de entrada
  self._log(f"[ENGINE] 🔒 Breakeven ativado {sym} @ ${entry:,.4f}")
  ```
- [x] Verificar se `trail_mgr` tem método `update_entry_price` — se não tiver, implementar no `TrailingStopManager`
- [x] Se `TrailingStopManager` não existir como módulo separado, procurar onde está definido antes de editar

### 1.4 — Detector de WS stale (dados mortos)
**Risco eliminado:** Inferência em candles com horas de atraso se o WS cair.  
**Arquivo:** `dashboard/trading/engine.py` (método `_tick`)

- [x] Na seção `# 2. Por símbolo: detecta novo candle 15m`, antes de processar o candle, adicionar:
  ```python
  # Verifica staleness: se último candle tem mais de 5 minutos, WS pode estar morto
  import time as _time
  last_buf_ts = buf[-1].get('timestamp', 0) / 1000  # ms → s
  if _time.time() - last_buf_ts > 300:  # 5 minutos
      self._log(f"[ENGINE] ⚠️ {sym} WS STALE — último candle há {(_time.time()-last_buf_ts)/60:.1f} min — inferência bloqueada")
      continue
  ```
- [x] Adicionar métricas de staleness ao `state['decisions'][sym]`: campo `'ws_age_secs': int(_time.time() - last_buf_ts)`
- [x] No `tab_engine.py`, mostrar indicador visual se `ws_age_secs > 120`

### 1.5 — Consolidar executors (eliminar src/)
**Risco eliminado:** Dois caminhos de execução que podem divergir.  
**Arquivos:** `src/execution/executor.py`, `src/risk/risk_manager.py`

- [x] Verificar quais scripts chamam `src/execution/executor.py` — usar `grep_search("from src.execution|from src import|import src")`
- [x] Se nenhum script do dashboard usar `src/`, adicionar `# DEPRECATED — usar dashboard/trading/executor.py` no topo de `src/execution/executor.py`
- [x] Mover lógica útil de `src/risk/risk_manager.py` (Kelly Criterion) para `dashboard/analytics/risk_calculator.py`
- [x] Não deletar `src/` — apenas marcar como deprecated e redirecionar

---

## FASE 2 — Tornar o bot lucrativo
> Melhorias de edge financeiro. Executar após todas as tarefas da Fase 1.

### 2.1 — Position sizing via Kelly real
**Arquivo:** `dashboard/trading/executor.py`  
**Arquivo:** `dashboard/analytics/risk_calculator.py` (novo ou existente)

- [x] Verificar se `RiskManager` em `src/risk/risk_manager.py` tem `calculate_position_size()` — confirmar assinatura
- [x] Criar/editar `dashboard/analytics/risk_calculator.py` com função:
  ```python
  def kelly_position_size(balance, closed_trades: list, kelly_fraction=0.25, max_pct=0.10) -> float:
      # Calcula win_rate, avg_win, avg_loss dos últimos 30 trades em closed_trades
      # Retorna fração do balance a usar (0.01 a max_pct)
  ```
- [x] Em `execute_trade`, substituir `config['environment']['position_size']` por chamada ao `kelly_position_size`
- [x] Passar `closed_trades` via parâmetro opcional `closed_trades: list | None = None`
- [x] Engine passa `list(self.state['closed_trades'])` quando chamar `execute_trade`
- [x] Logar: `[TRADE] Kelly sizing: win_rate={:.1%} → position_size={:.1%}`
- [x] Fallback: se `closed_trades` vazio ou < 10 trades, usar `config['environment']['position_size']`

### 2.2 — Verificação de correlação entre posições
**Arquivo:** `dashboard/trading/engine.py` (antes de `execute_trade`)  
**Arquivo:** `dashboard/analytics/correlation.py` (novo)

- [x] Criar `dashboard/analytics/correlation.py`:
  ```python
  def check_correlation(new_sym, open_syms, ws_mgr, threshold=0.70) -> tuple[bool, str]:
      # Busca returns das últimas 50 barras de cada par aberto
      # Calcula pearson entre new_sym e cada sym aberto
      # Retorna (can_enter, "correlação BTC/ETH: 0.91 > 0.70")
  ```
- [x] No `_tick`, antes de `execute_trade`, chamar:
  ```python
  open_syms = [s for s, p in ws_pos_map.items() if p != 0]
  can_corr, corr_reason = check_correlation(sym, open_syms, ws_mgr, threshold)
  if not can_corr:
      self._log(f"[ENGINE] {sym} bloqueado por correlação: {corr_reason}")
      continue
  ```
- [x] Usar `numpy.corrcoef` para o cálculo — sem dependências extras

### 2.3 — Verificação de exposição total
**Arquivo:** `dashboard/trading/engine.py` (método `_tick`)

- [x] Antes de `execute_trade`, calcular exposição atual:
  ```python
  ws_equity = float((ws_mgr.get_balance() or {}).get('total', 0)) or 1.0
  total_notional = sum(
      abs(float(p.get('positionAmt', 0))) * float(p.get('markPrice', 0))
      for p in positions if p['symbol'] in active_syms
  )
  exposure_pct = total_notional / ws_equity if ws_equity > 0 else 1.0
  max_exposure = cfg.get('risk_management', {}).get('max_total_exposure', 0.60)
  if exposure_pct >= max_exposure:
      self._log(f"[ENGINE] {sym} bloqueado: exposição {exposure_pct:.1%} >= {max_exposure:.1%}")
      continue
  ```

### 2.4 — Persistir LSTM states entre reinicios
> Já parcialmente coberto pela tarefa 1.1 (lstm_states no state_persistence.py)

- [x] Confirmar que `save_state` serializa `state['lstm_states']` corretamente (ndarrays → base64)
- [x] Confirmar que `load_state` restaura e que `ep_start = np.zeros((1,), dtype=bool)` é usado (não `np.ones`)
- [x] Adicionar log: `[ENGINE] {sym} lstm_state restaurado do disco` vs `[ENGINE] {sym} lstm_state novo (ep_start=True)`

### 2.5 — Corrigir RSI escala (bug de display)
**Arquivo:** `dashboard/trading/engine.py` (linha com `RSI_14 * 100`)

- [x] Buscar: `grep_search("RSI_14.*100", isRegexp=True)` em `engine.py`
- [x] Verificar escala real: se `prepare_observation` normaliza RSI para 0–1, então `*100` está correto
- [x] Se RSI já está em 0–100 (escala original), remover `* 100` no display
- [x] Verificar `dashboard/trading/observation.py` para confirmar escala do RSI normalizado

---

## FASE 3 — Operacional profissional
> Executar após Fase 1 e 2. Necessário para operar sem supervisão constante.

### 3.1 — Notificações Telegram
**Arquivo novo:** `dashboard/integrations/telegram_notifier.py`  
**Dependência:** `pip install python-telegram-bot`

- [x] Criar `dashboard/integrations/__init__.py`
- [x] Criar `dashboard/integrations/telegram_notifier.py`:
  ```python
  class TelegramNotifier:
      def __init__(self, token: str, chat_id: str): ...
      def send(self, msg: str, level='INFO'): ...  # async via threading
      def notify_trade(self, sym, side, qty, price, pnl=None): ...
      def notify_sl(self, sym, pnl_pct): ...
      def notify_drawdown(self, drawdown_pct): ...
      def notify_ws_down(self, sym): ...
  ```
- [x] Adicionar ao `config.yaml`:
  ```yaml
  notifications:
    telegram:
      enabled: false
      token: ""
      chat_id: ""
      events: ["sl", "tp", "drawdown_10pct", "ws_down", "engine_error"]
  ```
- [x] No engine, após cada `_record_close`, chamar `notifier.notify_trade(...)` se enabled
- [x] No detector WS stale (1.4), chamar `notifier.notify_ws_down(sym)` se stale > 300s
- [x] Instalar dependência apenas se `notifications.telegram.enabled: true`

### 3.2 — Logging persistente em disco
**Arquivo:** `dashboard/core/logging_setup.py`

- [x] Adicionar `RotatingFileHandler` ao logger root:
  ```python
  from logging.handlers import RotatingFileHandler
  Path("logs/trading").mkdir(parents=True, exist_ok=True)
  handler = RotatingFileHandler(
      f"logs/trading/{datetime.now().strftime('%Y-%m-%d')}.log",
      maxBytes=10_000_000, backupCount=7, encoding='utf-8'
  )
  handler.setLevel(logging.DEBUG)
  ```
- [x] Garantir que o handler não duplica se `setup_logging()` for chamado múltiplas vezes (verificar `if logger.handlers`)
- [x] Manter o `deque(maxlen=400)` em memória para o dashboard (UI não muda)
- [x] Adicionar botão "📥 Exportar logs" no `tab_engine.py` que copia último log para clipboard/download

### 3.3 — Testes unitários
**Arquivo novo:** `tests/` (diretório)

- [x] Criar `tests/__init__.py`
- [x] Criar `tests/test_executor.py`:
  - Mock `binance.Client` com `unittest.mock.MagicMock`
  - Testar: LONG abre quando `positionAmt == 0`, SHORT fecha LONG antes de abrir, FLAT fecha posição, qty == 0 retorna None
- [x] Criar `tests/test_risk_calculator.py`:
  - Testar Kelly com win_rate=0.6, avg_win=2, avg_loss=1 → resultado esperado ~0.20
  - Testar Kelly com 0 trades → retorna fallback
- [x] Criar `tests/test_entry_filter.py`:
  - Criar df mínimo com RSI/MACD/close para cada modo
  - Testar que RSI overbought bloqueia LONG
- [x] Criar `tests/test_correlation.py` (após criar o módulo em 2.2)
- [x] Criar `tests/conftest.py` com fixtures de DataFrames e configs comuns
- [x] Executar: `.\venv\Scripts\python.exe -m pytest tests/ -v`

### 3.4 — Modo paper trading isolado
**Arquivo:** `dashboard/trading/executor.py`  
**Arquivo:** `config.yaml`

- [x] Adicionar parâmetro `paper_mode` ao `execute_trade`:
  ```python
  def execute_trade(..., paper_mode: bool = False) -> dict | None:
      if paper_mode:
          # Simula fill imediato ao preço de mercado
          return {
              'orderId': f'PAPER_{int(time.time())}',
              'symbol': symbol,
              'side': 'BUY' if decision == 'LONG' else 'SELL',
              'origQty': str(quantity),
              'avgPrice': str(current_price),
              'status': 'FILLED',
          }
  ```
- [x] Engine lê `cfg.get('mode', 'testnet') == 'paper'` e passa `paper_mode=True`
- [x] Paper mode usa saldo virtual de `config['environment']['initial_balance']` armazenado em `state`
- [x] Dashboard exibe banner "📄 PAPER MODE — operações simuladas" quando ativo

---

## FASE 4 — Produto vendável
> Executar após Fases 1, 2 e 3 completas.

### 4.1 — Docker + docker-compose
**Arquivos novos:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`

- [x] Criar `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "dashboard_new.py", "--server.port=8501", "--server.headless=true"]
  ```
- [x] Criar `docker-compose.yml` com serviço `bot` e volume para `data/`, `logs/`, `models/`
- [x] Criar `.dockerignore` excluindo `venv/`, `__pycache__/`, `.env`, `*.pyc`
- [x] Verificar que `requirements.txt` está completo: `pip freeze > requirements.txt`
- [x] Testar build: `docker build -t trading-bot .`

### 4.2 — Configuração via UI (sem editar YAML)
**Arquivo:** `dashboard/ui/tab_engine.py`  
**Arquivo:** `config.yaml`

- [x] Adicionar seção "⚙️ Parâmetros de Risco" no `tab_engine.py` com `st.form`:
  - Sliders para: `stop_loss_pct`, `take_profit_pct`, `trailing_stop_activation`, `trailing_stop_distance`, `max_total_exposure`, `position_size`
  - Botão "Salvar configuração"
- [x] Ao salvar, atualizar `config.yaml` via `yaml.dump()` atomicamente (escrever em `.tmp`, depois `os.replace`)
- [x] Recarregar config no engine via `get_config.clear()` (Streamlit cache_resource clear)
- [x] Validar ranges antes de salvar: SL > 0.5%, TP > SL, exposure < 100%

### 4.3 — Relatórios PDF mensais
**Arquivo novo:** `dashboard/analytics/report_generator.py`  
**Dependência:** `pip install reportlab` ou `pip install fpdf2`

- [x] Criar `generate_monthly_report(closed_trades, start_date, end_date) -> bytes`:
  - Seções: resumo executivo, métricas chave, curva de equity, top 5 wins/losses, breakdown por símbolo
- [x] Adicionar botão "📄 Gerar Relatório PDF" em `tab_performance.py`
- [x] Usar `st.download_button` para exportar o bytes do PDF

### 4.4 — Autenticação básica
**Arquivo:** `dashboard_new.py`  
**Dependência:** `pip install streamlit-authenticator`

- [x] Instalar: `pip install streamlit-authenticator`
- [x] Criar `auth_config.yaml` com credentials (senha em bcrypt hash)
- [x] No início de `dashboard_new.py`, antes de qualquer render, verificar autenticação
- [x] Apenas mostrar dashboard se `authentication_status == True`

### 4.5 — README e documentação
**Arquivo novo:** `README.md`

- [x] Criar `README.md` com seções:
  - Visão geral do sistema (diagrama textual de arquitetura)
  - Requisitos (Python 3.11, Binance API key, CUDA opcional)
  - Instalação passo a passo (clone, venv, pip install, config.yaml)
  - Como rodar (testnet vs live)
  - Como configurar símbolos, timeframes, risk management
  - Estrutura de arquivos comentada
  - FAQ e troubleshooting comum

---

## Débito técnico crítico (resolver em paralelo com as fases)

- [x] **Remover `src/execution/executor.py`** do caminho de execução (ver 1.5)
- [x] **Corrigir multi-symbol executor** — `temp_cfg['data']['primary_symbol'] = sym_raw` é um hack; refatorar `execute_trade` para aceitar `symbol` como parâmetro explícito, não ler do config
- [x] **Verificar RSI scale** — `RSI_14 * 100` pode exibir valor errado na UI (ver 2.5)
- [x] **TP1 qty não travada** — fechar TP1 com `qty_at_entry / 2` não com `current_qty / 2` (race condition entre tick e TP1)

---

## Checklist de verificação final (antes de ir para live com dinheiro real)

- [ ] Fase 1 completa (todos os itens marcados)
- [ ] Testar: reiniciar engine com posição aberta → SL/TP ainda funcionam
- [ ] Testar: WS cair → engine bloqueia inferência, não abre trades
- [ ] Testar: TP1 acionado → SL move para breakeven
- [ ] Fase 2 completa
- [ ] Backtest com Kelly sizing vs fixed sizing → Kelly tem Sharpe superior
- [ ] Rodar 1 semana em paper mode sem crashes
- [ ] Rodar 2 semanas em testnet com capital virtual
- [ ] Verificar que drawdown não ultrapassa 15% em 1 mês testnet
- [ ] Fase 3.1 (Telegram) completa — receber notificações em tempo real
- [ ] Fase 3.2 (logging) completa — logs disponíveis para debug noturno
- [ ] Fase 3.3 (testes): `pytest tests/ -v` com 100% pass
- [ ] Code review manual: ler `engine.py`, `executor.py`, `executor.py` linha a linha procurando edge cases
- [ ] **Somente após todos os itens acima: migrar para live**
