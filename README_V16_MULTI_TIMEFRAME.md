# 🚀 SISTEMA DE TREINAMENTO V16 - MULTI-TIMEFRAME

## 📋 RESUMO DO QUE FOI CRIADO

Sistema completo de treinamento multi-timeframe que mantém todas as configurações do V15 mas adiciona análise em 3 escalas temporais.

---

## 🆕 ARQUIVOS CRIADOS

### 1. **collect_multi_timeframe.py**
- Baixa dados históricos em 3 timeframes: 15m, 1h, 4h
- Aplica mesmos indicadores técnicos em cada timeframe
- Split automático train/test (80/20)
- Gera arquivos: `data/train_btcusdt_36m_{timeframe}_{date}.csv`

### 2. **src/environment/trading_env_multi_tf.py**
- Ambiente de RL com suporte a múltiplos timeframes
- Observation space expandido (~3x maior que V15)
- **Mesma lógica de reward do V15** (100% compatível)
- Processa 15m (tático), 1h (operacional), 4h (estratégico) simultaneamente

### 3. **train_sac_v16.py**
- Script de treinamento idêntico ao V15 mas com multi-timeframe
- **Mantém todos os hiperparâmetros do V15**:
  - Buffer: 100k
  - Ent_coef: 0.05
  - Network: [256, 256]
  - Reward structure: Sharpe + bônus balanceados
- Auto-detecção dos arquivos de dados mais recentes

### 4. **analyze_v15_v16.py**
- Compara performance V15 vs V16
- Gera estatísticas e gráficos
- Valida hipótese de melhoria com multi-timeframe

---

## 📊 ANÁLISE DO V15 (ATUAL)

**Dataset:**
- 82.904 candles de 15m (36 meses)
- Período: 2023-01 a 2025-06
- Features: 17 colunas (OHLCV + indicadores)

**Configurações:**
```yaml
Window size: 50
Episodes: 2000 steps
Leverage: 1.5x
Position size: 5%
Commission: 0.04%
Slippage: 0.05%

Reward:
  - Sharpe Ratio (base)
  - Bônus lucro >2%: +0.05
  - Bônus cortar loss <-2%: +0.05
  - Penalidade overtrading: -0.03
  - Penalidade flip-flop: -0.02
  - Indicator shaping: DESABILITADO

SAC:
  - Buffer: 100k
  - Ent_coef: 0.05
  - Learning rate: 3e-4
  - Network: [256, 256]
```

---

## 🎯 POR QUE MULTI-TIMEFRAME?

### **Vantagens Teóricas:**

1. **Contexto Macro + Micro**
   - 15m: Reage a movimentos imediatos
   - 1h: Filtra ruído, identifica tendências
   - 4h: Confirma direção macro

2. **Menos Falsos Sinais**
   - Se 15m mostra compra mas 4h está em downtrend → evita armadilha

3. **Melhor Timing**
   - Combina velocidade (15m) com confirmação (1h/4h)

4. **Padrões Multi-Escala**
   - Aprende correlações temporais naturais
   - Ex: divergência RSI 15m vs 4h

### **Hipótese V16:**
- Win rate: 22-25%+ (vs 18-22% do V15)
- Sharpe ratio maior (menos volatilidade)
- Menos overtrading (contexto macro filtra ruído)

---

## 🚀 COMO USAR

### **Passo 1: Baixar Dados Multi-Timeframe**
```powershell
python collect_multi_timeframe.py
```

**Isso irá:**
- Baixar 36 meses de BTC/USDT em 15m, 1h, 4h
- Calcular indicadores técnicos
- Fazer split train/test
- Salvar em `data/`

**Output esperado:**
```
data/train_btcusdt_36m_15m_20260125.csv  (~82k candles)
data/train_btcusdt_36m_1h_20260125.csv   (~20k candles)
data/train_btcusdt_36m_4h_20260125.csv   (~5k candles)
```

---

### **Passo 2: Treinar V16**
```powershell
python train_sac_v16.py
```

**Duração:** ~10-20h (AMD DirectML)

**Checkpoints:**
- Salva a cada 5k steps
- Total: 200 checkpoints (1M steps)
- Path: `models/sac_v16_multi_tf_{timestamp}_{steps}_steps.zip`

**Monitorar:**
```powershell
tensorboard --logdir=./tensorboard/
```

---

### **Passo 3: Backtest e Comparação**

**Backtest V16:**
```powershell
python backtest.py models/sac_v16_multi_tf_{timestamp}_1000000_steps.zip
```

**Comparar V15 vs V16:**
```powershell
python analyze_v15_v16.py
```

Gera:
- Estatísticas comparativas
- Gráficos de win rate, return, sharpe
- Validação de hipótese

---

## 📈 MÉTRICAS ESPERADAS

### **V15 (Single-Timeframe Baseline):**
- Win rate: ~18-22%
- Return: +2-5%
- Trades: Moderado
- Balance Long/Short: ~40-50%

### **V16 (Multi-Timeframe Target):**
- Win rate: **22-25%+** ⬆️
- Return: **+5-8%** ⬆️
- Sharpe: **Maior** (menos volatilidade) ⬆️
- Trades: **Menos** (contexto macro filtra) ⬇️

---

## 🔧 OBSERVAÇÕES TÉCNICAS

### **Observation Space:**

**V15 (single):**
```
window_size * (n_features + 3) = 50 * (17 + 3) = 1000 valores
```

**V16 (multi):**
```
15m: 50 * (17 + 3) = 1000
1h:  12 * 17 = 204
4h:  3 * 17 = 51
Total: 1255 valores (~25% maior)
```

### **Alinhamento Temporal:**
- 1 candle 1h = 4 candles 15m
- 1 candle 4h = 16 candles 15m
- Script verifica automaticamente proporção correta

### **Compatibilidade:**
- **100% compatível com V15** (mesma reward, mesmos hiperparâmetros)
- Única diferença: observation space expandido
- Pode comparar diretamente os resultados

---

## ⚠️ TROUBLESHOOTING

### **Erro: "Arquivos de dados não encontrados"**
```
Execute: python collect_multi_timeframe.py
```

### **Erro: "Alinhamento de timeframes incorreto"**
- Verifique se baixou os 3 timeframes do mesmo período
- Re-execute collect_multi_timeframe.py

### **GPU/DirectML não funciona:**
- V16 usa mesma configuração do V15
- Se V15 funcionou, V16 também funciona

### **Memória insuficiente:**
- Observation space é 25% maior
- Se necessário, reduza `window_size` de 50 para 40

---

## 📚 PRÓXIMOS PASSOS

1. ✅ **Baixar dados** → `python collect_multi_timeframe.py`
2. ✅ **Treinar V16** → `python train_sac_v16.py`
3. ⏳ **Aguardar 10-20h** de treinamento
4. ⏳ **Executar backtest** em checkpoints chave (200k, 500k, 1M)
5. ⏳ **Comparar V15 vs V16** → `python analyze_v15_v16.py`
6. ⏳ **Validar hipótese** de win rate 22-25%+

---

## 🎯 EXPECTATIVAS

**Se V16 > V15:**
- ✅ Confirma que multi-timeframe agrega valor
- ✅ Justifica uso em produção
- ✅ Pode expandir para mais timeframes (5m, 12h, 1d)

**Se V16 ≈ V15:**
- 🟡 Multi-timeframe não melhora significativamente
- 🟡 Modelo já extrai informação suficiente de 15m
- 🟡 Considerar outras melhorias (sentiment, volume profile)

**Se V16 < V15:**
- ❌ Observation space muito grande pode causar overfitting
- ❌ Reduzir window_size ou usar técnicas de compressão
- ❌ Considerar feature selection

---

## 📞 SUPORTE

Dúvidas sobre:
- Coleta de dados → Veja `collect_multi_timeframe.py` (bem documentado)
- Ambiente → Veja `src/environment/trading_env_multi_tf.py`
- Treinamento → Veja `train_sac_v16.py`
- Comparação → Veja `analyze_v15_v16.py`

---

**Criado em:** 2026-01-25  
**Versão:** V16 - Multi-Timeframe  
**Baseado em:** V15 (single-timeframe)  
**Objetivo:** Win rate 22-25%+ com análise multi-temporal
