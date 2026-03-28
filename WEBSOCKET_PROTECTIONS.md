# 🔒 Proteções Anti-Ban Implementadas

## Problema Original
Dashboard fazia **centenas de REST calls** causando ban do IP a cada 5-10 minutos.

## Solução: WebSocket + Bloqueios REST

### 1. WebSocket Auto-Ativado
- ✅ Inicia automaticamente no load do dashboard
- ✅ Zero REST calls após inicialização
- ✅ Recebe dados via **push events** em tempo real

### 2. Bloqueios de REST API quando WebSocket Ativo

#### Funções 100% Bloqueadas:
1. **`get_klines()`** - Histórico de candles
   - Retorna DataFrame vazio se WebSocket ativo
   - Log: `[KLINES] WebSocket ativo - bloqueando REST call`

2. **`get_recent_trades()`** - Histórico de trades
   - Retorna lista vazia se WebSocket ativo
   - Log: `[TRADES] WebSocket ativo - bloqueando REST call`

3. **`collect_market_data()`** - Dados de mercado para modelo
   - Retorna `None` se WebSocket ativo
   - Log: `[DATA] WebSocket ativo - bloqueando REST call`

4. **`get_account_balance_cached()`** - Saldo da conta
   - Usa cache do WebSocket quando disponível
   - Retorna zeros temporários se WebSocket sem dados
   - NÃO faz fallback para REST se WebSocket ativo

5. **`get_open_positions_cached()`** - Posições abertas
   - Usa cache do WebSocket quando disponível
   - Retorna lista vazia se WebSocket sem dados
   - NÃO faz fallback para REST se WebSocket ativo

### 3. Sistemas Desabilitados Automaticamente

#### Auto TP/SL Enforcement (Tab 2)
```
⚠️ DESABILITADO quando WebSocket ativo
```
- **Motivo**: Tentava fechar posições a cada 30s no auto-refresh
- **Status**: Desabilitado + mensagem informativa
- **Alternative**: Fechar posições manualmente

#### Bot de Trading Automático (Tab 5)
```
⚠️ Para AUTOMATICAMENTE quando WebSocket ativo
```
- **Motivo**: Faz centenas de REST calls:
  - Coleta dados 15m, 1h, 4h
  - Verifica posições constantemente
  - Executa ordens
- **Status**: Auto-stop + aviso claro
- **Alternative**: Aguardar implementação v18 (bot via WebSocket)

### 4. Snapshot Manual Inicial

**Problema**: WebSocket só envia UPDATES, não estado inicial.

**Solução**: Botão na sidebar:
```
⚡ Carregar Snapshot Inicial (1 REST call)
```
- Executa **1 vez** para popular cache
- Após isso, WebSocket mantém atualizado
- Opcional mas recomendado

### 5. Proteções Adicionais

#### Detecção de Ban com Timestamp
- Extrai timestamp exato do erro: `banned until 1772160460458`
- Mostra countdown preciso: `"Ban expira em 4m 32s às 22:15:30"`
- Auto-limpa flags quando ban expira

#### Cache Inteligente
- Balance: 30s
- Positions: 30s
- Market Data: 60s
- Klines: 60s
- Trades: 30s

#### Auto-Refresh Inteligente
- **Pausa** se banido E WebSocket inativo
- **Continua** se WebSocket ativo (zero REST calls)
- Intervalo configurável: 10-120s (padrão 30s)

---

## Como Usar (Passo a Passo)

### 1. Inicie o Dashboard
```powershell
streamlit run dashboard.py
```

### 2. WebSocket Inicia Automaticamente
```
✅ [WEBSOCKET] Iniciado com sucesso - User Data Stream ativo
✅ [WEBSOCKET] Subscribed: BTCUSDT klines 15m
```

### 3. Carregue Snapshot Inicial (Na Sidebar)
- Clique em **"⚡ Carregar Snapshot Inicial"**
- Executa 1 REST call para popular dados
- Após isso: **ZERO REST calls automáticas!**

### 4. Verifique Status
- Fonte de dados: `🟢 WebSocket (0 REST calls)`
- Sidebar: `✅ Dados atualizados há 5s via WebSocket`

### 5. Limitações com WebSocket Ativo
- ❌ Gráficos de candles desabilitados
- ❌ Bot automático desabilitado
- ❌ Auto TP/SL desabilitado
- ✅ Balance e posições funcionam
- ✅ Trades manuais funcionam
- ✅ Análise de métricas funciona

### 6. Para Ver Gráficos
- Pare WebSocket na sidebar: **"⏹️ Parar WS"**
- Gráficos aparecerão usando REST API (cache 60s)
- ⚠️ Cuidado com refresh muito rápido

---

## Resultados Esperados

### Antes (SEM proteções)
```
❌ Ban a cada 5-10 minutos
❌ 50+ REST calls por minuto (auto-refresh)
❌ Dashboard inutilizável
```

### Depois (COM proteções)
```
✅ ZERO bans automáticos
✅ 0 REST calls com WebSocket ativo
✅ Dashboard 100% funcional
```

---

## Troubleshooting

### "WebSocket não mostra dados"
- WebSocket envia APENAS updates (quando há mudança)
- Clique em "⚡ Carregar Snapshot Inicial"
- Faça um trade na Binance → WebSocket atualizará automaticamente

### "Ainda sendo banido"
1. Verifique se WebSocket está ativo: `🟢 Conectado`
2. Verifique fonte de dados: `🟢 WebSocket (0 REST calls)`
3. Desative auto-refresh temporariamente
4. Aguarde 5-15 minutos para ban expirar

### "Gráficos não aparecem"
- Normal! WebSocket não fornece histórico de candles
- Pare WebSocket para ativar gráficos via REST
- **OU** aguarde implementação futura via WebSocket

### "Bot não funciona"
- Desabilitado propositalmente quando WebSocket ativo
- Bot faz centenas de REST calls → causaria ban imediato
- Aguarde implementação v18 (bot 100% WebSocket)

---

**Última atualização**: 2026-02-26 22:40
**Dashboard Version**: v17.7 + WebSocket
