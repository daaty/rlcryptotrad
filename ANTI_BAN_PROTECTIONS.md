# 🛡️ PROTEÇÕES ANTI-BAN - Dashboard V17.7

## 🚨 PROBLEMA IDENTIFICADO

**Binance Testnet Rate Limit**: 50 chamadas REST / 10 segundos  
**Consequência**: Ban de IP por 10-15 minutos se ultrapassar

### Fontes de Ban Identificadas (26 FEV 2026):

1. **Bot Multi-Par** (CRÍTICO):
   - Faz 3 calls por símbolo (15m, 1h, 4h)
   - 4 pares = 12 calls instantâneas no startup!
   - Se auto-refresh a cada 30s → 24 calls/min → BAN

2. **Multiple Tabs Loading**:
   - Tab 1: Balance + Positions (2 calls)
   - Tab 2: Positions + Auto TP/SL checks (2-4 calls)
   - Tab 3: Recent trades (1-4 calls por símbolo)
   - Tab 4: Gráficos - Klines (3-12 calls)
   - **Total no startup**: 10-25 calls simultâneas

3. **Auto-Refresh Agressivo**:
   - Padrão: 30s
   - Cache: 30-60s
   - Resultado: Cache expira → novas calls a cada 30-60s

4. **Cache Insuficiente**:
   - Balance: 30s (muito curto!)
   - Positions: 30s (muito curto!)
   - Market Data: 60s (insuficiente para 4 pares)

---

## ✅ SOLUÇÕES IMPLEMENTADAS

### 1. **Cache Massivamente Aumentado**

```python
# ANTES → DEPOIS
get_account_balance_cached:    30s → 120s (2 min)
get_open_positions_cached:     30s → 120s (2 min)
get_recent_trades:             30s → 120s (2 min)
collect_market_data:           60s → 180s (3 min)
get_klines:                    60s → 180s (3 min)
```

**Resultado**: Reduz chamadas REST em ~75%

### 2. **Auto-Refresh Desabilitado Por Padrão**

```python
# ANTES
auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)  # ❌ ATIVO
refresh_interval = st.slider("Intervalo (s)", 10, 120, 30)  # ❌ 30s

# DEPOIS
auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)  # ✅ DESLIGADO
refresh_interval = st.slider("Intervalo (s)", 60, 300, 120)  # ✅ MIN 60s, DEFAULT 120s
```

**Aviso na UI**:
```
🔴 IMPORTANTE: Auto-refresh causa MUITAS chamadas REST → BAN!
💡 Recomendação: Deixe DESABILITADO e clique '🔄 Atualizar Agora'
```

### 3. **Limite de Pares: MÁXIMO 2**

```python
# Força limite antes do bot processar
if len(selected_symbols) > 2:
    st.error(f"❌ Máximo 2 pares permitidos! Você selecionou {len(selected_symbols)}")
    selected_symbols = selected_symbols_raw[:2]
```

**Cálculo**:
- 2 pares × 3 calls (15m+1h+4h) = 6 calls
- Com cache de 180s → ~2 calls/min
- Total: **~8 calls/min** (SEGURO!)

### 4. **Rate Limiting Entre Símbolos**

```python
for idx, trade_symbol in enumerate(selected_symbols):
    if idx > 0:
        time.sleep(2)  # 2s delay entre símbolos
    
    market_data = collect_market_data(client, symbol=symbol_binance, limit=200)
    time.sleep(0.5)  # 500ms delay entre calls
    
    df_1h = collect_market_data(client, symbol=symbol_binance, interval='1h', limit=100)
    time.sleep(0.5)
    
    df_4h = collect_market_data(client, symbol=symbol_binance, interval='4h', limit=50)
    time.sleep(0.5)
```

**Resultado**: Espaça chamadas em ~4s por símbolo

### 5. **Bloqueio do Bot se > 2 Pares**

```python
if len(selected_symbols) > 2:
    st.error("❌ BOT BLOQUEADO: {len} pares (máximo: 2)")
    bot_disabled = True
```

---

## 📊 COMPARAÇÃO: ANTES vs DEPOIS

### Cenário: Dashboard + Bot + 4 Pares

| Métrica | ANTES | DEPOIS | Redução |
|---------|-------|--------|---------|
| Calls no Startup | 20-30 | 8-12 | -60% |
| Calls/minuto (bot) | 24-36 | 2-4 | -88% |
| Cache Balance | 30s | 120s | +300% |
| Cache Market Data | 60s | 180s | +200% |
| Intervalo Auto-Refresh | 30s | 120s (padrão OFF) | +300% |
| Pares Simultâneos | 4 (sem limite) | 2 (máximo forçado) | -50% |
| Rate Limit Delays | 0s | 2-4s entre calls | Novo! |

### Taxa de Chamadas REST

```
ANTES (4 pares, auto-refresh 30s):
- Startup: 24 calls (12 bot + 8 tabs + 4 balance/pos)
- A cada 30s: 12 calls (bot update)
- Total: 24 calls/min → BAN a cada 2 minutos!

DEPOIS (2 pares, auto-refresh OFF):
- Startup: 10 calls (6 bot + 2 tabs + 2 balance/pos)
- Cache 2-3min: 0 calls durante cache
- Total: 3-5 calls/min → SEGURO!
```

---

## 🎯 COMO USAR SEM BAN

### ✅ CONFIGURAÇÃO SEGURA

1. **Selecione MAX 2 pares** na sidebar
2. **Deixe auto-refresh DESLIGADO**
3. **Clique "🔄 Atualizar Agora"** quando quiser ver novos dados
4. **Aguarde 2-3 minutos** entre atualizações manuais

### ❌ O QUE EVITAR

1. ❌ **NÃO** selecione 4 pares
2. ❌ **NÃO** ative auto-refresh (especialmente < 60s)
3. ❌ **NÃO** clique "Atualizar Agora" repetidamente
4. ❌ **NÃO** abra múltiplas tabs/janelas do dashboard
5. ❌ **NÃO** rode o bot por longos períodos sem monitorar

### 🟢 OPERAÇÃO RECOMENDADA

**Modo Manual** (ZERO chance de ban):
```
1. Auto-refresh: OFF
2. Pares: 1-2
3. Atualização: Manual a cada 2-3 min
4. Bot: Use apenas para análise, execute trades manualmente
```

**Modo Bot** (risco baixo, mas existente):
```
1. Auto-refresh: OFF ou 120s+
2. Pares: MAX 2
3. Cache: 2-3min (já configurado)
4. Monitorar: Verifique logs para detectar ban early
```

---

## 🔍 DETECTANDO BAN

O dashboard detecta automaticamente e mostra:

```
🚫 IP BANIDO PELA BINANCE API

⏱️ Tempo restante: 8m 32s (expira às 22:45:00)

Por que o ban persiste mesmo após reiniciar modem?
- ⏰ Ban expira AUTOMATICAMENTE no horário acima
- 🌐 Seu provedor pode dar o mesmo IP novamente
- 🔄 Ban é temporário, não permanente!
```

**Durante ban**:
- Auto-refresh pausa automaticamente
- Countdown timer mostra tempo restante
- Dashboard volta a funcionar quando expirar

---

## 📈 RESULTADOS ESPERADOS

Com as proteções implementadas:

- ✅ **Startup sem ban** (10-12 calls, bem abaixo de 50)
- ✅ **Operação contínua segura** (3-5 calls/min, média)
- ✅ **Cache efetivo** (2-3min de TTL)
- ✅ **Rate limiting** automático entre calls
- ✅ **Limites forçados** no UI (max 2 pares)

---

## 🚀 PRÓXIMOS PASSOS (v18)

Para operação 100% livre de bans:

1. **WebSocket Full Implementation**:
   - Subscribe kline streams para todos os TFs
   - Use WebSocket data para inferência do modelo
   - Execute ordens via REST apenas quando necessário

2. **Smart Refresh**:
   - Detecta quando dados mudaram (via diff)
   - Atualiza apenas se necessário
   - Agregação de múltiplas calls em batch

3. **Local Cache Persistence**:
   - Salva dados em SQLite local
   - Carrega do cache local no startup
   - Reduz calls iniciais para ~2-3

---

## 📝 CHANGELOG

### 26 FEV 2026 - v17.7.1 (Anti-Ban Patch)

**Changed**:
- Cache TTL: 30-60s → 120-180s
- Auto-refresh: Default ON → OFF
- Refresh interval: 10-120s → 60-300s (min 60s)
- Max pares: Unlimited → 2 (forçado)

**Added**:
- Rate limiting: 2s entre símbolos, 500ms entre calls
- UI warnings: Avisos agressivos sobre limites
- Bot blocker: Impede iniciar se > 2 pares
- Startup delays: Espaça calls no primeiro load

**Fixed**:
- Calls simultâneas no startup (24 → 10)
- Bot loop infinito de calls
- Tab rendering causando múltiplas calls
- Cache expiry muito rápido

---

**Última atualização**: 26 FEV 2026, 22:50  
**Status**: ✅ Testado e funcionando sem bans
