# 🎯 ENT_COEF='AUTO' vs FIXO: ANÁLISE TÉCNICA

## ❓ **A PERGUNTA:**
> "O que você acha de deixar automático? Já que temos um ambiente denso"

---

## ✅ **DECISÃO: HÍBRIDO (AUTO + TARGET CUSTOMIZADO)**

```python
ent_coef='auto',           # SAC ajusta dinamicamente
target_entropy=-0.5        # MAS força target maior (evita colapso)
```

---

## 📊 **COMPARAÇÃO:**

| Configuração | Prós | Contras | Recomendação |
|-------------|------|---------|--------------|
| **`ent_coef=0.5` (FIXO)** | ✅ Previsível<br>✅ Evita colapso<br>✅ Funciona (13% winrate alcançado) | ❌ Não adapta ao reward<br>❌ Pode ser sub-ótimo | ⭐ BOM para baseline |
| **`ent_coef='auto'` (DEFAULT)** | ✅ Adapta dinamicamente<br>✅ Ótimo para ambientes complexos | ❌ **RISCO: Colapso para 0**<br>❌ Target padrão = -1.0 (muito negativo) | ⚠️ ARRISCADO sem ajustes |
| **`ent_coef='auto'` + `target=-0.5`** | ✅ Adapta dinamicamente<br>✅ Target customizado evita colapso<br>✅ Melhor para ambiente denso | ❌ Requer monitoramento | 🏆 **RECOMENDADO** |

---

## 🧮 **MATEMÁTICA POR TRÁS:**

### **SAC Loss Function:**
```
Total Loss = Q_loss + π_loss + α * H(π)
                                 ↑
                            entropia da policy
```

Onde:
- `α` = `ent_coef` (coeficiente de entropia)
- `H(π)` = entropia da policy (quanto é "aleatória")

### **Target Entropy (J):**
```python
# SAC padrão:
target_entropy = -dim(action_space) = -1.0

# Nosso customizado:
target_entropy = -0.5  # Menos negativo = mais entropia
```

### **Alpha (ent_coef) é ajustado para:**
```python
α = α + lr_α * (H(π) - target_entropy)
```

**Exemplo:**
- Se `H(π) = -0.3` e `target = -0.5` → `α` **aumenta** (força mais exploração)
- Se `H(π) = -0.7` e `target = -0.5` → `α` **diminui** (permite mais exploitation)

---

## 🔬 **ANÁLISE DO SEU AMBIENTE:**

### **Por que 'auto' PODE ser melhor:**

1. **Ambiente MUITO denso:**
   - Observation: `(50, 19)` = **950 features**
   - Indicadores: SMA, RSI, MACD, Bollinger
   - Estado: balance, position, equity
   - **SAC precisa explorar MUITO espaço**

2. **Reward shaping complexo:**
   - Delta equity base
   - Bonus progressivo (0.015 → 0.005)
   - Penalty progressivo (0.02 → 0.014)
   - Indicadores (±0.15)
   - Liquidação (-1.0)
   - **Recompensas variam muito = SAC ajusta α automaticamente**

3. **Position size dinâmico:**
   - Action ∈ [-1, 1] controla direção E tamanho
   - SAC precisa aprender nuances (0.3 vs 0.8 é diferente)
   - **Entropia adaptativa ajuda a refinar exploração**

### **Por que 'fixo' TAMBÉM funciona:**

1. **Você já testou:** 13.43% winrate com `ent_coef=0.7`
2. **Previsível:** Exploração constante
3. **Seguro:** Não há risco de colapso

---

## ⚠️ **RISCO DO 'AUTO' (QUE VOCÊ JÁ TEVE):**

### **Caso real - Transfer Learning TD3→SAC:**
```
Entropy colapsou: 0.00000078
Resultado: FLAT MODE (0 trades)
Motivo: target_entropy = -1.0 (padrão) muito negativo
```

### **Solução implementada:**
```python
target_entropy = -0.5  # Força entropia maior
```

**Garantia:** Se entropia cair muito, SAC aumenta α automaticamente.

---

## 📈 **MONITORAMENTO NO TENSORBOARD:**

O script loga:
```python
rollout/entropy       # Entropia da policy
rollout/ent_coef      # Alpha atual (se 'auto')
episode/win_rate      # Performance
episode/liquidations  # Risco
```

**Sinais de alerta:**
- ⚠️ Entropy < -1.5 (muito baixo, quase determinístico)
- ⚠️ ent_coef < 0.001 (colapsando)
- ⚠️ win_rate estagnado + entropy caindo = modelo estagnado

---

## 🎯 **RECOMENDAÇÃO FINAL:**

### **Para PRIMEIRO treino (este):**
```python
ent_coef='auto',
target_entropy=-0.5
```

**Por quê:**
- Ambiente denso beneficia de adaptação
- Target customizado previne colapso
- Você pode comparar com baseline (ent_coef=0.5)

### **Se colapsar (entropy < -1.5 por >100k steps):**
```python
# Interromper e mudar para:
ent_coef=0.5  # Fixo e seguro
```

### **Se funcionar bem (winrate >25%):**
```python
# Próximo treino:
ent_coef='auto',
target_entropy=-0.3  # Ainda mais exploração
```

---

## 🔍 **COMO DECIDIR APÓS TREINO:**

| Cenário | Ação |
|---------|------|
| **Winrate >25% + entropy estável (-0.5 a -0.8)** | ✅ PERFEITO! Usar 'auto' em prod |
| **Winrate 15-25% + entropy variando muito** | ⚠️ Funciona, mas testar fixo (0.4) |
| **Winrate <15% + entropy caindo (-1.2 a -1.5)** | ❌ Colapso! Próximo treino usar fixo |
| **Liquidations >50** | ❌ Exploração demais! Reduzir target (-0.7) |

---

## 📚 **REFERÊNCIAS:**

- **Paper SAC:** [Soft Actor-Critic Algorithms](https://arxiv.org/abs/1812.05905)
- **Entropia em RL:** Controla trade-off exploration/exploitation
- **Ambiente denso:** Mais features = precisa mais exploração

---

## ✅ **CONCLUSÃO:**

**USAR `ent_coef='auto'` com `target_entropy=-0.5` É A MELHOR ESCOLHA** porque:

1. ✅ Ambiente denso (950 features) beneficia de adaptação
2. ✅ Target customizado evita colapso (seu problema anterior)
3. ✅ Logging no TensorBoard permite monitoramento
4. ✅ Se falhar, você sabe que fixo (0.5) funciona
5. ✅ Potencial de performance MAIOR que fixo

**Experimento:** Rodar este treino 'auto' e comparar com baseline 'fixo' (13.43% winrate).

**Expectativa:** Winrate 'auto' > 'fixo' em 2-5% (30-35% vs 25-30%)
