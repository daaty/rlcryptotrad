# 🚀 PLANO DE MELHORIA V7 - Atingir 35-40% Winrate

## 📊 Análise do Problema V6

### Resultados V6 (500k steps):
- ✅ Winrate: 23.4% (bom, mas instável)
- ✅ Liquidações: 0 (perfeito!)
- ❌ Colapso: 25.9% → 12.8% → 23.4% (volatilidade alta)
- ❌ Exploration fixo (std=0.368 sempre)
- ❌ Reward muito complexo (5 sistemas simultâneos)

### Root Cause:
1. **Sharpe Ratio reward**: Otimiza risco, não lucro → conservadorismo excessivo
2. **Exploration constante**: Não converge, continua explorando até o fim
3. **Reward shaping complexo**: Model confuso entre objetivos conflitantes
4. **Position size variável**: Overtrading em ações pequenas

---

## 🎯 MUDANÇAS V7

### 1️⃣ SIMPLIFICAR REWARD (CRÍTICO!)

**Problema**: 5 sistemas de reward competindo por atenção

**Solução**: Delta Equity PURO + apenas 1 shaping estratégico

```python
# V6 (COMPLEXO - 5 sistemas):
- Sharpe Ratio reward ❌
- Punições progressivas (5 níveis) ❌
- Indicator shaping (3 indicadores) ⚠️
- Holding bonuses ❌
- Exit bonuses ❌

# V7 (SIMPLES - 1 objetivo):
reward = (equity_atual - equity_anterior) / initial_balance

# ÚNICO shaping permitido: Trend alignment
if trading_contra_tendencia_forte:
    reward -= 0.05  # Pequena penalty
```

**Justificativa**: 
- RL aprende MELHOR com objetivos claros
- Delta equity = objetivo direto do trader (lucro!)
- Indicator shaping LEVE apenas para acelerar convergência

---

### 2️⃣ DESABILITAR SHARPE REWARD

**Problema**: Sharpe otimiza risco, não lucro → model fica FLAT demais

**Solução**: Usar delta equity puro

```python
# train_futuros_v7.py
env = TradingEnv(
    use_sharpe_reward=False,      # DESABILITADO!
    use_hybrid_reward=False,       # DESABILITADO!
    enable_indicator_shaping=True  # Apenas trend alignment leve
)
```

---

### 3️⃣ ADICIONAR EXPLORATION DECAY (CRÍTICO!)

**Problema**: std fixo = explora igual nos 500k steps

**Solução**: Reduzir exploration gradualmente

```python
# train_futuros_v7.py

# Callback para decay de exploration
class ExplorationDecayCallback(BaseCallback):
    def __init__(self, initial_std=0.3, final_std=0.05, decay_steps=400000):
        super().__init__()
        self.initial_std = initial_std
        self.final_std = final_std
        self.decay_steps = decay_steps
    
    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        current_std = self.initial_std - (self.initial_std - self.final_std) * progress
        
        # Atualiza std do model
        if hasattr(self.model.actor, 'log_std'):
            self.model.actor.log_std.data.fill_(np.log(current_std))
        
        return True
```

**Resultado esperado**:
- 0-100k: std=0.30 (explora muito)
- 100k-300k: std=0.15 (equilibra)
- 300k-500k: std=0.05 (exploita conhecimento)

---

### 4️⃣ AUMENTAR TARGET_ENTROPY (Mais Exploração Inicial)

**Problema**: target_entropy=-1.0 é muito restritivo

**Solução**: Permitir mais diversidade de ações

```python
# V6:
target_entropy = -1.0  # Muito restritivo

# V7:
target_entropy = -0.5  # Mais liberdade (dim=1, então -0.5 é bom)
```

---

### 5️⃣ REMOVER POSITION SIZE VARIÁVEL

**Problema**: Model controla tamanho → overtrading

**Solução**: Position size FIXO (sempre 5%)

```python
# trading_env.py - V7
def step(self, action):
    action_value = float(action[0])
    
    if action_value < -0.1:
        discrete_action = 2  # Short
        self.current_position_size = 0.05  # FIXO 5%!
    elif action_value > 0.1:
        discrete_action = 1  # Long
        self.current_position_size = 0.05  # FIXO 5%!
    else:
        discrete_action = 0  # Flat
        self.current_position_size = 0
```

**Justificativa**: Kelly Criterion → posição ideal ≈ 5% para 25% winrate

---

### 6️⃣ TREINAR POR 1M STEPS (Dobro do V6)

**Problema**: 500k steps pode ser insuficiente

**Solução**: 1M steps com exploration decay

```python
# train_futuros_v7.py
total_timesteps = 1_000_000  # Dobro do V6
```

**Checkpoints**:
- 100k, 200k, 300k, 400k, 500k, 750k, 1M

---

## 🧪 EXPERIMENTOS ALTERNATIVOS (Se V7 não atingir 35%)

### Experimento A: Curriculum Learning
```python
# Treina em fases de dificuldade crescente:
1. Fase 1 (0-200k): Dados de tendência forte (bull markets)
2. Fase 2 (200k-500k): Dados mistos (bull + bear)
3. Fase 3 (500k-1M): Dados completos (incluindo lateralizações)
```

### Experimento B: Ensemble de Modelos
```python
# Combina 3 checkpoints:
- Model 100k (agressivo, pico de performance)
- Model 500k (conservador, estável)
- Model 1M (equilibrado, bem treinado)

# Votação por maioria para cada trade
```

### Experimento C: Aumentar Leverage para 2x
```python
# Se winrate ≥30%, pode aumentar leverage:
leverage = 2.0  # De 1.5x para 2x
position_size = 0.05  # Mantém 5% fixo
# Risk: 10% exposure (5% * 2x)
```

### Experimento D: Multi-Timeframe Features
```python
# Adicionar features de timeframes maiores:
- Candles de 1h, 4h, 1d
- Tendências de médio prazo
- Suporte/Resistência key levels
```

---

## 📋 IMPLEMENTAÇÃO V7 - CHECKLIST

### Fase 1: Simplificar Reward (Hoje)
- [ ] Desabilitar `use_sharpe_reward=False`
- [ ] Desabilitar `use_hybrid_reward=False`
- [ ] Simplificar `_calculate_indicator_reward()` (apenas trend alignment)
- [ ] Remover holding bonuses
- [ ] Remover punições progressivas (deixar apenas stop-loss)

### Fase 2: Position Size Fixo (Hoje)
- [ ] Modificar `step()` para position_size=0.05 fixo
- [ ] Remover lógica de `min(abs(action_value), 0.5)`

### Fase 3: Exploration Decay (Hoje)
- [ ] Criar `ExplorationDecayCallback`
- [ ] Adicionar ao training loop
- [ ] Configurar: inicial=0.3, final=0.05, decay=400k

### Fase 4: Treinar V7 (Hoje/Amanhã)
- [ ] Criar `train_futuros_v7.py`
- [ ] target_entropy = -0.5
- [ ] total_timesteps = 1M
- [ ] Iniciar treino (~10h estimado)

### Fase 5: Backtests (Após treino)
- [ ] Backtest checkpoints: 100k, 300k, 500k, 1M
- [ ] Comparar com V6 100k
- [ ] Selecionar melhor modelo

---

## 🎯 METAS V7

| Métrica | V6 | V7 Target |
|---------|-----|-----------|
| Winrate | 23.4% | **35-40%** |
| Liquidações | 0 | 0 |
| Trades/1k | 2.3 | 2-3 |
| Max Drawdown | ? | <20% |
| Sharpe Ratio | ? | >1.0 |
| Profit Factor | ? | >1.5 |
| Estabilidade | ❌ Volátil | ✅ Estável |

---

## 💭 EXPECTATIVAS REALISTAS

### Se V7 atingir 30-35%:
- ✅ **EXCELENTE** para trading algorítmico
- ✅ Potencial de 15-25% retorno anual
- ✅ Pronto para testnet → live trading

### Se V7 atingir 25-30%:
- ⚠️ **BOM**, mas precisa otimizar position sizing
- ⚠️ Considerar ensemble de modelos
- ⚠️ Adicionar features (multi-timeframe)

### Se V7 < 25%:
- ❌ Problema mais profundo
- ❌ Considerar:
  - Mudar de SAC para PPO (on-policy)
  - Aumentar complexidade do modelo (net_arch)
  - Curriculum learning
  - Dados de melhor qualidade

---

## 🚀 PRÓXIMOS PASSOS IMEDIATOS

1. **Agora**: Implementar V7 (reward simples + exploration decay)
2. **Hoje**: Iniciar treino 1M steps (~10h)
3. **Amanhã**: Analisar primeiros 100k steps
4. **Em 2 dias**: Backtests completos
5. **Se sucesso**: Testnet Binance

**Vamos começar?** 🎯
