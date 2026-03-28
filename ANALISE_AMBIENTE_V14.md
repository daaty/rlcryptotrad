# 🔍 ANÁLISE COMPLETA DO AMBIENTE DE TRADING V14

**Data:** 15/01/2026  
**Problema:** Win rate consistentemente baixo (12-15% vs meta 20%+)  
**Versões analisadas:** V6, V8, V13, V14

---

## 1. CONFIGURAÇÃO ATUAL (V14 com ambiente V8)

### Parâmetros Principais
```python
initial_balance = 10000
commission = 0.0004  # 0.04% Binance
slippage = 0.0005    # 0.05%
leverage = 1.5       # Baixo (seguro)
position_size = 0.05 # 5% por trade
window_size = 50     # Observação
max_episode_steps = 4000  # V8: 4000 (vs V6: 2000)
use_sharpe_reward = True
enable_indicator_shaping = True
```

### SAC Hyperparameters
```python
ent_coef = 0.05        # V8: exploração moderada
buffer_size = 100_000  # V8: 100k (vs V6: 200k)
learning_rate = 3e-4
batch_size = 256
net_arch = [256, 256]  # V8: simples
action_noise = 15%     # V8: moderado
```

---

## 2. ESTRUTURA DE REWARDS - ANÁLISE CRÍTICA

### 2.1. Reward Principal (use_sharpe_reward=True)

**Linha 380:**
```python
reward = np.tanh(sharpe * 10)  # Sharpe Ratio normalizado [-1, 1]
```

**PROBLEMA 1: Sharpe pode ser enganoso no início**
- Com poucos trades, Sharpe varia muito
- Modelo pode aprender a evitar trades para manter Sharpe estável
- Explica comportamento "flat" excessivo

**Recomendação:** Adicionar peso de delta equity nos primeiros steps
```python
if self.trades < 50:  # Primeiros trades
    reward = 0.7 * delta_equity + 0.3 * sharpe
else:  # Depois de experiência
    reward = np.tanh(sharpe * 10)
```

### 2.2. Bônus por Fechar Posições (Linha 291-301)

```python
if action_reward != 0:  # Fechou posição
    if action_reward > 0.03:  # Lucro > 3%
        reward += 0.08  # Bônus ALTO
    elif action_reward > 0.01:
        reward += 0.04
    elif action_reward < -0.03:  # Cortou loss < -3%
        reward += 0.03  # Bônus por cortar loss
```

**PROBLEMA 2: Bônus desbalanceados**
- Lucro >3% = +0.08 (bônus alto)
- Cortar loss = +0.03 (bônus baixo)
- **Proporção: 2.7x mais reward para lucro que para cortar loss**
- Modelo aprende: "Vale mais esperar lucro que cortar loss"

**Resultado:** Deixa losers correrem, segura winners curtos
**Win rate baixo:** Muitos trades viram losses por não cortar

**Recomendação:** Igualar incentivos
```python
if action_reward > 0.02:  # Lucro > 2%
    reward += 0.05  # Reduzido
elif action_reward < -0.02:  # Cortou loss < -2%
    reward += 0.05  # IGUAL! Cortar loss é TÃO importante quanto lucro
```

### 2.3. Penalidades por Holding Losers (Linha 424-446)

**Escala progressiva mas SEVERA:**
```python
if unrealized_pct < -0.02 and unrealized_pct >= -0.04:
    reward -= 0.005  # -2 a -4%: aviso
elif unrealized_pct < -0.04 and unrealized_pct >= -0.055:
    reward -= 0.02   # -4 a -5.5%: atenção
elif unrealized_pct < -0.055 and unrealized_pct >= -0.06:
    reward -= 0.08   # -5.5 a -6%: PERIGO
elif unrealized_pct < -0.06 and unrealized_pct >= -0.07:
    reward -= 0.25   # -6 a -7%: forte
    reward -= 0.005  # Dor por tempo
else:  # < -7% (stop loss)
    reward -= 0.60   # Punição severa
    reward -= 0.015  # Dor por tempo moderada
```

**PROBLEMA 3: Stop loss é -7% mas penalidades começam em -2%**
- Entre -2% e -7% = 5% de "zona de sofrimento"
- Modelo aprende: "Não vale a pena segurar posições"
- Resultado: Sai cedo demais, não deixa trades respirarem
- **Conflito:** Bônus para lucro >3% mas penalidade desde -2%

**Recomendação:** Ajustar thresholds
```python
# Começar penalidades mais tarde
if unrealized_pct < -0.04:  # -4% (não -2%)
    reward -= 0.005
elif unrealized_pct < -0.055:  # -5.5%
    reward -= 0.02
elif unrealized_pct < -0.065:  # -6.5% (perto do stop)
    reward -= 0.10
```

### 2.4. Bônus por Deixar Winners Correrem (Linha 416-422)

```python
if unrealized_pct > 0.03:  # > +3%
    reward += 0.005  # Pequeno bônus
if unrealized_pct > 0.05:  # > +5%
    reward += 0.01   # Bônus maior
```

**PROBLEMA 4: Bônus MUITO pequenos vs penalidades**
- Lucro +5% = +0.01 bônus
- Loss -4% = -0.02 penalidade
- **Ratio: 2x mais punição que recompensa**
- Modelo aprende: "Evitar trades = evitar dor"

**Recomendação:** Balancear risk/reward
```python
if unrealized_pct > 0.03:
    reward += 0.02  # 4x maior (vs -0.005 em -4%)
if unrealized_pct > 0.05:
    reward += 0.04  # 2x do -0.02 em -5.5%
```

### 2.5. Penalidade por Overtrading (Linha 308-316)

```python
# Se > 3 trades em 24h (96 candles de 15min)
if len(self.last_24h_trades) > 3:
    overtrading_penalty = (len(self.last_24h_trades) - 3) * 0.03
    reward -= overtrading_penalty
```

**ANÁLISE:** 3 trades/24h é MUITO restritivo
- 82,904 candles = ~863 dias
- 3 trades/dia × 863 dias = 2,589 trades max
- V14 aos 40k steps: 2,482 trades ✅ (dentro do limite)
- **NÃO é o problema do win rate baixo**

**Status:** OK, mantém controle de overtrading

### 2.6. Penalidade por Flip-Flop (Linha 318-326)

```python
# Se mudou Long→Short ou Short→Long em < 50 steps
if steps_since_flip < 50:
    if (prev == 1 and current == -1) or (prev == -1 and current == 1):
        reward -= 0.02  # V12: suave
```

**ANÁLISE:** 50 steps = ~12.5 horas (15min candles)
- Penalidade suave (-0.02)
- Evita churn rápido sem paralisar
- **Status:** OK

### 2.7. Penalidade por Inatividade (Linha 470-477)

```python
# Se Flat em tendência forte
if discrete_action == 0:  # Flat
    trend_strength = abs(close - sma_50) / sma_50
    if trend_strength > 0.02 and 30 < rsi < 70:
        reward -= 0.0001  # V8: minúscula
```

**PROBLEMA 5: Penalidade IRRELEVANTE**
- 0.0001 é NADA comparado a outras rewards/penalties
- Modelo ignora completamente
- Explica alta % de flat time

**Recomendação:** Aumentar 100x
```python
reward -= 0.01  # 100x maior, ainda suave
```

### 2.8. Reward Shaping com Indicadores (Linha 454-461)

```python
if enable_indicator_shaping and shaping_decay > 0.05:
    indicator_reward = self._calculate_indicator_reward(discrete_action, current_price)
    reward += indicator_reward * shaping_decay
```

**Decay Formula (Linha 444-458):**
```python
progress = self.episode_length / self.max_episode_steps
shaping_decay = 0.5 * (1 - progress) + 0.05  # 0.5 → 0.05
```

**ANÁLISE:** Shaping diminui ao longo do episódio
- Início: 0.5 (50% de influência)
- Fim: 0.05 (5% de influência)
- **Episodes de 4000 steps:** Decay muito lento
- Aos 2000 steps: ainda 0.3 (30% influência)

**Verificar:** `_calculate_indicator_reward()` pode estar dando sinais ruins

---

## 3. FUNÇÃO _calculate_indicator_reward() - ANÁLISE

```python
def _calculate_indicator_reward(self, action: int, current_price: float) -> float:
    current_row = self.df.iloc[self.current_step - 1]
    
    # RSI
    if 'RSI_14' in current_row:
        rsi = current_row['RSI_14']
        if action == 1 and rsi < 30:  # Long em oversold
            indicator_reward += 0.01
        elif action == 2 and rsi > 70:  # Short em overbought
            indicator_reward += 0.01
        elif action == 1 and rsi > 70:  # Long em overbought
            indicator_reward -= 0.005  # Penalty por "chute"
        elif action == 2 and rsi < 30:  # Short em oversold
            indicator_reward -= 0.005
```

**PROBLEMA 6: Estratégia RSI contratrend**
- RSI < 30 = Oversold → recomenda LONG (comprar na queda)
- RSI > 70 = Overbought → recomenda SHORT (vender na subida)
- **Isso é estratégia CONTRATREND (mean reversion)**
- Bitcoin em bull market: RSI >70 pode continuar subindo!
- **Modelo punido por seguir tendência**

**Resultado:** Confusão no aprendizado
- Sharpe reward diz: "Siga tendência"
- RSI shaping diz: "Vá contra tendência"
- Modelo fica perdido, não aprende padrão claro

**Recomendação:** Remover ou inverter lógica
```python
# OPÇÃO 1: Remover completamente
enable_indicator_shaping = False

# OPÇÃO 2: Inverter para trend-following
if action == 1 and rsi > 50:  # Long com momentum
    indicator_reward += 0.01
elif action == 2 and rsi < 50:  # Short com momentum
    indicator_reward += 0.01
```

---

## 4. OBSERVATION SPACE - ANÁLISE

```python
# Observation shape: (window_size, n_features + 3)
# +3 = balance, position, equity (V8 PURO)
```

**ANÁLISE:** Clean, sem win_rate ou stop_risk do V13
- ✅ Preços/indicadores normalizados
- ✅ Portfolio state (balance, position, equity)
- ✅ Sentiment features (se disponível)
- **Status:** OK após reverter V13

---

## 5. ACTION SPACE - ANÁLISE

```python
# Box contínuo [-1, 1]
# -1 a -0.33: Short
# -0.33 a 0.33: Flat
# 0.33 a 1: Long
```

**ANÁLISE:** Standard para SAC
- Position size ajustado por magnitude da action
- **Status:** OK

---

## 6. EPISODE MANAGEMENT - ANÁLISE

```python
max_episode_steps = 4000  # V8
truncated = (
    equity <= initial_balance * 0.5 or  # Stop se perder 50%
    episode_length >= max_episode_steps
)
```

**PROBLEMA 7: Episodes muito longos**
- 4000 steps = ~41 dias de dados (15min candles)
- Sharpe Ratio calculado sobre todo episódio
- Nos primeiros 1000 steps: Sharpe instável
- Modelo demora para ver consequências

**V6 usava 2000 steps** (20 dias) - mais estável

**Recomendação:** Testar voltar para 2000 steps
```python
max_episode_steps = 2000  # V6
```

---

## 7. CONFLITOS E CONTRADIÇÕES IDENTIFICADAS

### Conflito 1: Risk/Reward Desbalanceado
- **Problema:** Punição por loss é 2x maior que recompensa por profit
- **Resultado:** Modelo evita trades, fica flat
- **Solução:** Igualar ou inverter (reward > punishment)

### Conflito 2: Bônus vs Penalidades
- **Problema:** Bônus para fechar winner (+0.08) >> Bônus para cortar loss (+0.03)
- **Resultado:** Não corta losers esperando virarem winners
- **Solução:** Igualar incentivo para cortar loss

### Conflito 3: Indicator Shaping vs Trend
- **Problema:** RSI shaping é contratrend mas mercado é trend-following
- **Resultado:** Sinais confusos, aprendizado prejudicado
- **Solução:** Remover ou inverter para momentum

### Conflito 4: Stop Loss vs Penalidades
- **Problema:** Stop em -7% mas penalidades desde -2%
- **Resultado:** Modelo sai muito cedo (medo de -2%)
- **Solução:** Começar penalidades em -4% ou -5%

### Conflito 5: Episodes Longos vs Sharpe Instável
- **Problema:** 4000 steps = Sharpe calculado em janela muito longa
- **Resultado:** Feedback atrasado, aprendizado lento
- **Solução:** Voltar para 2000 steps (V6)

---

## 8. RECOMENDAÇÕES PRIORITÁRIAS

### 🔥 URGENTE (Implementar imediatamente)

1. **Balancear Risk/Reward**
   ```python
   # Linha 416-422: Aumentar bônus por winners
   if unrealized_pct > 0.03:
       reward += 0.02  # Era 0.005
   if unrealized_pct > 0.05:
       reward += 0.04  # Era 0.01
   ```

2. **Igualar Incentivo para Cortar Loss**
   ```python
   # Linha 297-301: Igualar bônus
   if action_reward > 0.02:  # Lucro
       reward += 0.05
   elif action_reward < -0.02:  # Loss
       reward += 0.05  # IGUAL!
   ```

3. **Desabilitar Indicator Shaping** (conflito contratrend)
   ```python
   enable_indicator_shaping = False  # Testar sem
   ```

4. **Adiar Penalidades de Loss**
   ```python
   # Linha 424: Começar em -4% (não -2%)
   if unrealized_pct < -0.04:  # Era -0.02
       reward -= 0.005
   ```

### ⚠️ IMPORTANTE (Testar em seguida)

5. **Voltar Episodes para 2000 steps**
   ```python
   max_episode_steps = 2000  # V6 (não 4000)
   ```

6. **Aumentar Penalidade Flat 100x**
   ```python
   # Linha 477
   reward -= 0.01  # Era 0.0001
   ```

7. **Adicionar Peso de Delta nos Primeiros Trades**
   ```python
   if self.trades < 50:
       delta = (self.equity - self.previous_equity) / self.initial_balance
       reward = 0.7 * delta + 0.3 * sharpe
   ```

---

## 9. HIPÓTESES SOBRE WIN RATE BAIXO

### Hipótese Principal: REWARD SHAPING CONTRAPRODUCENTE

**Evidências:**
1. RSI shaping é contratrend (contra tendência de mercado)
2. Punições >> Recompensas (evitar trades é "seguro")
3. Bônus desbalanceados (melhor esperar lucro que cortar loss)
4. Penalidades começam cedo demais (-2% vs stop -7%)

**Resultado:**
- Modelo aprende padrão inconsistente
- Evita trades para evitar punições
- Quando trade, não corta losers rápido (esperando bônus de lucro)
- **Win rate: 12-15% porque losers crescem**

### Teste para Validar Hipótese

**Criar V15 MINIMAL:**
```python
# Ambiente V8 base
max_episode_steps = 2000  # V6
enable_indicator_shaping = False  # SEM RSI
# Reward: APENAS delta equity (sem Sharpe nos primeiros 100 trades)
# Bônus balanceados: 0.05 para cortar loss OU fazer lucro
# Penalidades começam em -4% (não -2%)
```

**Expectativa:**
- Win rate 18-22% (melhora de 50%)
- Menos flat time
- Mais trades balanceados Long/Short

---

## 10. PLANO DE AÇÃO

### Fase 1: Correções Críticas (V15)
1. Desabilitar `enable_indicator_shaping = False`
2. Voltar `max_episode_steps = 2000`
3. Igualar bônus cortar loss = fazer lucro (0.05)
4. Aumentar recompensas por winners (0.02 e 0.04)
5. Adiar penalidades para -4%
6. Treinar 1M steps

### Fase 2: Validação
1. Testar checkpoints 200k, 400k, 600k
2. Monitorar win rate, Long/Short balance
3. Comparar com V6 500k (baseline)

### Fase 3: Refinamento (se necessário)
1. Se win rate melhorar mas ainda <20%: ajustar penalidades
2. Se flat time alto: aumentar penalidade flat
3. Se overtrading: verificar se há recompensas erradas

---

## CONCLUSÃO

**Diagnóstico:** Ambiente tem CONFLITOS INTERNOS entre rewards
- Indicator shaping contratrend vs mercado trend-following
- Punições grandes vs recompensas pequenas
- Bônus desbalanceados (lucro >> cortar loss)
- Penalidades prematuras (-2% quando stop é -7%)

**Prognóstico:** Win rate baixo é ESPERADO com esses conflitos

**Tratamento:** Simplificar reward structure, remover contradições

**Expectativa:** V15 com correções deve atingir 18-22% win rate (vs 12-15% atual)
