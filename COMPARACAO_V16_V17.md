# 🆚 COMPARAÇÃO V16.3 SAC vs V17 LSTM

## 📋 Resumo Executivo

### V16.3 SAC (MLP) - Status: **CONCLUÍDO E CONGELADO**
- ✅ Treino: 1M steps completo (~12h AMD GPU)
- ❌ Performance: -1.13% return devido a overtrading
- ⚠️ Problema: 784 trades, $313 em comissões destroem potencial lucro de $180

### V17 LSTM (RecurrentPPO) - Status: **EM DESENVOLVIMENTO**
- 🔄 Ambiente: ✅ Criado
- 🔄 Scripts: ✅ Train e backtest prontos
- ⏳ Treino: Pendente (~40h estimado)
- 🎯 Objetivo: Reduzir overtrading através de memória temporal

---

## 🏗️ Arquitetura Comparada

### V16.3 SAC (MLP)

```
┌─────────────────────────────────────────────────────────────┐
│                   FLATTENING DESTROYS STRUCTURE              │
└─────────────────────────────────────────────────────────────┘

15m [50 candles × 26 feat] ─┐
1h  [12 candles × 26 feat] ─┼─→ FLATTEN → [1450] ───┐
4h  [3 candles × 26 feat]  ─┘                       │
                                                     ▼
                                              ┌──────────────┐
                                              │ MLP Network  │
                                              │ [512,512,256]│
                                              └──────────────┘
                                                     │
                                                     ▼
                                              Action [-1,+1]

❌ PROBLEMA: Perde relações temporais entre candles
❌ Não sabe "o que aconteceu ontem" → flip-flop frequente
❌ Overtrading: 784 trades / 2000 steps = trade a cada 2.5 steps
```

#### Hyperparameters SAC:
- Policy: MlpPolicy
- Learning rate: 3e-4
- Buffer size: 300k
- Batch size: 256
- Network: [512, 512, 256]
- Entropy: 0.2 (alta exploração)

#### Performance V16.3:
```
┌──────────────────┬──────────┬──────────┬──────────┐
│    Checkpoint    │  Trades  │ Win Rate │  Return  │
├──────────────────┼──────────┼──────────┼──────────┤
│ 200k steps       │   785    │  26.02%  │  -1.53%  │
│ 500k steps ⭐    │   741    │  29.86%  │  -1.13%  │ ← BEST
│ 1M steps (final) │   744    │  27.86%  │  -1.28%  │
└──────────────────┴──────────┴──────────┴──────────┘

P&L Breakdown (500k checkpoint):
  - Avg Win:  $0.42
  - Avg Loss: $0.41      ← Quase iguais! Seleção é boa
  - Profit Factor: 0.41  ← Problema está no win rate
  - Commissions: $313.60 ← MATA O LUCRO
  - P&L bruto: +$180 (sem comissões)
  - P&L líquido: -$113 (com comissões)

DIAGNÓSTICO:
  ✅ Modelo faz boas previsões (avg win ≈ avg loss)
  ❌ Frequência de trades é ALTA DEMAIS
  ❌ Comissões destroem performance
  ✅ Win rate 30% é aceitável (target: 22-35%)
```

---

### V17 LSTM (RecurrentPPO)

```
┌─────────────────────────────────────────────────────────────┐
│              PRESERVING TEMPORAL STRUCTURE                   │
└─────────────────────────────────────────────────────────────┘

Por cada timestep (50 total):
  15m [26 features]          ─┐
  1h  [1 aggregated feature] ─┼─→ [29 features/timestep]
  4h  [1 aggregated feature] ─┤
  portfolio [1 aggregated]   ─┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Sequence (50,29)  │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  LSTM Layer 1 (256) │  ← Mantém memória
                    │  LSTM Layer 2 (256) │  ← de curto prazo
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   MLP [256, 256]    │
                    └─────────────────────┘
                              │
                              ▼
                        Action [-1,+1]

✅ VANTAGEM: Preserva sequências temporais
✅ LSTM aprende "quando não fazer nada"
✅ Memória de trades recentes → menos flip-flop
✅ Esperado: 400-600 trades (50% menos que V16.3)
```

#### Hyperparameters RecurrentPPO:
- Policy: MlpLstmPolicy
- Learning rate: 3e-4
- N steps: 2048 (PPO batch)
- Batch size: 64 (menor → LSTM usa mais RAM)
- LSTM: 2 layers × 256 neurons
- MLP após LSTM: [256, 256]
- Entropy: 0.01 (PPO precisa menos exploration)

#### Performance V17 (ESPERADA):
```
┌──────────────────┬──────────┬──────────┬──────────┐
│    Checkpoint    │  Trades  │ Win Rate │  Return  │
├──────────────────┼──────────┼──────────┼──────────┤
│ 500k steps       │  400-600 │   32%+   │   +1.5%  │ ← TARGET
│ 1M steps         │  400-600 │   33%+   │   +2.0%  │
│ 1.5M steps       │  400-600 │   35%+   │   +2.5%  │
└──────────────────┴──────────┴──────────┴──────────┘

P&L Target:
  - Avg Win:  $0.45+       ← Ligeiramente melhor seleção
  - Avg Loss: $0.40
  - Profit Factor: > 1.0   ← WINS > LOSSES!
  - Commissions: ~$200     ← 40% menos que V16.3
  - P&L bruto: +$280
  - P&L líquido: +$80-150  ← POSITIVO!

HIPÓTESE:
  ✅ LSTM aprende quando segurar vs sair
  ✅ Menos overtrading → menos comissões
  ✅ Win rate mantido ou melhorado
  ✅ Retorno POSITIVO
```

---

## 🧪 Diferenças Técnicas Chave

### 1. Observation Space

| Aspecto | V16.3 SAC | V17 LSTM |
|---------|-----------|----------|
| **Shape** | `(1450,)` | `(50, 29)` |
| **Tipo** | Flattened | Sequential |
| **15m** | 50 × 26 = 1300 | 50 × 26 = 1300 |
| **1h** | 12 × 26 = 312 | 50 × 1 = 50 (aggregated) |
| **4h** | 3 × 26 = 78 | 50 × 1 = 50 (aggregated) |
| **Portfolio** | Included in features | 50 × 1 = 50 (per timestep) |
| **Estrutura Temporal** | ❌ Perdida | ✅ Preservada |

**Explicação da agregação LSTM:**
- V16.3: Usa 12 candles 1h completos (todas as features de cada candle)
- V17: Por cada timestep 15m, calcula média das features 1h correspondentes
- Vantagem: Reduz dimensionalidade mantendo informação relevante
- LSTM processa 50 timesteps onde cada um tem contexto multi-timeframe

### 2. Network Architecture

**V16.3:**
```
Input (1450) → Dense(512) → Dense(512) → Dense(256) → Output
                  ↓             ↓             ↓
               ReLU          ReLU          ReLU
```

**V17:**
```
Input (50, 29) → LSTM(256) → LSTM(256) → Dense(256) → Dense(256) → Output
                    ↓           ↓            ↓             ↓
                 Hidden      Hidden        ReLU          ReLU
                 States      States
```

### 3. Training Differences

| Aspecto | V16.3 SAC | V17 LSTM |
|---------|-----------|----------|
| **Algorithm** | Off-policy (replay buffer) | On-policy (no buffer) |
| **Updates** | Every step | Every n_steps=2048 |
| **Exploration** | Entropy 0.2 | Entropy 0.01 |
| **Speed** | ~12h / 1M steps | ~40h / 1M steps |
| **Memory** | Moderate | High (LSTM states) |

### 4. Inference Differences

**V16.3 SAC:**
```python
action = model.predict(obs, deterministic=False)
# Cada prediction é independente
# Não há memória entre steps
```

**V17 LSTM:**
```python
lstm_states = None  # Inicializar
episode_start = True

action, lstm_states = model.predict(
    obs, 
    state=lstm_states,  # ← PASS hidden states
    episode_start=episode_start,
    deterministic=False
)
episode_start = False  # Após primeiro step

# LSTM states carregam memória entre predictions
```

---

## 📊 Expectativas de Melhoria

### Por que LSTM deve reduzir overtrading?

#### 1. **Memória de Curto Prazo**
```
V16.3 MLP: "Vejo setup bullish → COMPRA"
           (não lembra que acabou de fechar Long 2 steps atrás)

V17 LSTM:  "Vejo setup bullish → Verifico hidden states"
           "Acabei de fechar Long → Aguardo confirmação maior"
           → MENOS flip-flops
```

#### 2. **Padrões Temporais**
```
V16.3 MLP: Cada candle é processado isoladamente
           Não reconhece "consolidação vs tendência"

V17 LSTM:  Processa sequências de 50 candles
           Aprende: "Últimos 20 steps lateral → AGUARDE breakout"
           → MELHOR timing de entrada
```

#### 3. **Market Regime Recognition**
```
V16.3 MLP: Todas as horas são iguais
           Trade em baixa volatilidade = bad risk/reward

V17 LSTM:  Aprende padrões de volatilidade ao longo de 50 steps
           Identifica quando mercado está "dormindo"
           → SKIP trades em condições ruins
```

---

## 🎯 Métricas de Sucesso para V17

### Mínimo Viável (MVP)
- ✅ Trades: < 700 (redução de 10%+)
- ✅ Win Rate: ≥ 28%
- ✅ Return: ≥ 0% (breakeven)

### Target Ideal
- 🎯 Trades: 400-600 (redução de 40%+)
- 🎯 Win Rate: 32-35%
- 🎯 Return: +1.5% to +2.5%
- 🎯 Profit Factor: > 1.0

### Stretch Goal
- 🚀 Trades: < 400
- 🚀 Win Rate: > 35%
- 🚀 Return: > 3%
- 🚀 Profit Factor: > 1.5

---

## 🔄 Fluxo de Validação

### 1. Treinar V17-LSTM
```bash
python train_recurrent_ppo_v17.py
# Duração: ~40h
# Checkpoints: 10k, 50k, 100k, 200k, 500k, 1M, 1.5M
```

### 2. Testar Checkpoints Críticos
```bash
# Checkpoint 500k
python backtest_recurrent_ppo_v17.py
# Editar MODEL_PATH para cada checkpoint

# Checkpoint 1M
# ...

# Checkpoint 1.5M (final)
# ...
```

### 3. Comparar com V16.3
```
┌───────────────┬──────────┬──────────┬──────────┬────────────────┐
│    Modelo     │  Trades  │ Win Rate │  Return  │ Profit Factor  │
├───────────────┼──────────┼──────────┼──────────┼────────────────┤
│ V16.3 SAC 500k│   741    │  29.86%  │  -1.13%  │     0.41       │
│ V17 LSTM 500k │   ???    │   ???    │   ???    │     ???        │
│ V17 LSTM 1M   │   ???    │   ???    │   ???    │     ???        │
│ V17 LSTM 1.5M │   ???    │   ???    │   ???    │     ???        │
└───────────────┴──────────┴──────────┴──────────┴────────────────┘
```

### 4. Decisão
- **Se V17 LSTM > V16.3**: Adotar LSTM como arquitetura padrão
- **Se V17 LSTM ≈ V16.3**: Retornar a MLP + quality filters (V17-Filters)
- **Se V17 LSTM < V16.3**: Investigar problemas (underfitting? Hyperparams?)

---

## 📝 Hipóteses a Validar

### Hipótese 1: Redução de Overtrading
**V16.3 Issue**: 784 trades (1 a cada 2.5 steps)

**V17 Expectativa**: 400-600 trades (redução 40%)

**Como Validar**:
```python
trades_v17 = len(backtest.trades)
reduction = (784 - trades_v17) / 784 * 100

if reduction > 40:
    print("✅ Hipótese CONFIRMADA")
elif reduction > 20:
    print("⚠️  Melhoria parcial")
else:
    print("❌ Hipótese REJEITADA")
```

### Hipótese 2: Win Rate Estável
**V16.3 Issue**: 42% treino → 30% teste (overfitting)

**V17 Expectativa**: 35% treino → 32% teste (menos overfitting)

**Como Validar**:
- Monitorar win rate no TensorBoard durante treino
- Comparar win rate final TensorBoard vs backtest
- Gap < 5% indica baixo overfitting

### Hipótese 3: Profit Factor > 1.0
**V16.3 Issue**: Profit factor 0.41 (perde $2.44 por cada $1 ganho)

**V17 Expectativa**: Profit factor > 1.0 (ganha mais que perde)

**Como Validar**:
```python
profit_factor = total_wins / total_losses

if profit_factor > 1.5:
    print("🚀 EXCELENTE")
elif profit_factor > 1.0:
    print("✅ TARGET ATINGIDO")
else:
    print("❌ Ainda precisamos melhorar")
```

---

## 🛠️ Troubleshooting

### Se V17 LSTM não funcionar bem:

#### Problema: Ainda há overtrading (> 700 trades)
**Possíveis Causas**:
- LSTM layers muito pequenas (aumentar para 512)
- Penalidade de overtrading insuficiente nas rewards
- N_steps muito baixo (aumentar para 4096)

**Soluções**:
1. Aumentar `lstm_hidden_size` para 512
2. Adicionar penalty específica: `-0.01 * n_trades_recent_window`
3. Aumentar `n_steps` para forçar updates menos frequentes

#### Problema: Win rate baixa (< 25%)
**Possíveis Causas**:
- Underfitting (LSTM não teve tempo de aprender)
- Exploration insuficiente
- Features sequenciais não informativas

**Soluções**:
1. Treinar por mais tempo (2M+ steps)
2. Aumentar `ent_coef` para 0.05
3. Verificar se agregação 1h/4h está correta

#### Problema: Return negativo apesar de menos trades
**Possíveis Causas**:
- Seleção de trades piorou (avg loss > avg win)
- Holding duration muito longo (slippage acumula)
- Timing de entrada/saída ruim

**Soluções**:
1. Analisar distribuição de holding duration
2. Verificar se está segurando losers por muito tempo
3. Adicionar time-based penalties

---

## 🗺️ Roadmap Completo

### ✅ Fase 1: V16.3 SAC (CONCLUÍDA)
- [x] Treino 1M steps
- [x] Backtest e análise
- [x] Diagnóstico: overtrading

### 🔄 Fase 2: V17 LSTM (EM PROGRESSO)
- [x] Criar ambiente LSTM
- [x] Criar script de treino
- [x] Criar script de backtest
- [ ] Treinar modelo (~40h)
- [ ] Validar checkpoints
- [ ] Comparar com V16.3

### ⏳ Fase 3: V17-Filters (SE NECESSÁRIO)
- [ ] Adicionar quality filters a V16.3
- [ ] Backtest filtered V16.3
- [ ] Comparar com V17-LSTM

### ⏳ Fase 4: V18-V19 (FUTURO)
- [ ] V18: Dynamic stops (ATR-based)
- [ ] V19: Ensemble models
- [ ] V20: Meta-learning

---

## 🎓 Lições Aprendidas

### Do V16.3:
1. **MLP funciona** - Win rate 30% é viável
2. **Overtrading mata** - Comissões > lucro potencial
3. **Seleção é boa** - Avg win ≈ avg loss
4. **Frequência é ruim** - 784 trades é demais
5. **Look-ahead bias** - CRÍTICO corrigir
6. **Backtest deterministic** - SAC precisa `False`

### Para V17:
1. **LSTM pode ajudar** - Memória temporal é chave
2. **Paciência é virtude** - ~40h de treino necessário
3. **Checkpoints são importantes** - Testar múltiplos pontos
4. **Comparação justa** - Mesmo test set, mesmos parâmetros
5. **Múltiplas tentativas** - Se falhar, tentar hyperparams diferentes

---

## 📚 Referências

### Papers:
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [LSTM Networks](http://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)

### Stable-Baselines3:
- [SB3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)

---

**Última atualização**: 2026-01-11  
**Próxima revisão**: Após treino V17-LSTM completo
