# AMBIENTE COMPLETO TD3 - DOCUMENTAÇÃO PARA MIGRAÇÃO SAC

**Data:** 10/01/2026  
**Modelo:** TD3 (Twin Delayed DDPG)  
**Status:** 3M steps completo (12.7 horas treino)  
**Arquivo:** `models/base_btcusdt_final.zip`  

---

## 1. DATASET

### Dataset 36 Meses (3 anos)
- **Arquivo:** `data/train_btcusdt_36m_20260109.csv`
- **Total candles:** 82,904 (treino)
- **Período:** 2023-01-26 a 2026-01-10 (1,079 dias)
- **Timeframe:** 15 minutos
- **Preço range:** $19,614.20 - $125,986.00
- **Cobertura:** Ciclo completo (bear 2023 + bull 2024-2025)

### Features (17 colunas)
```
timestamp, open, high, low, close, volume,
RSI_14, SMA_20, SMA_50,
BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0,
MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
```

---

## 2. ENVIRONMENT CONFIGURATION

### TradingEnv Parameters
```python
TradingEnv(
    df=df,                          # 82,904 candles
    initial_balance=10000,          # $10,000 inicial
    commission=0.0010,              # 0.10% (Binance Futures taker)
    slippage=0.0005,                # 0.05% slippage
    leverage=3,                     # 3x leverage
    position_size=0.10,             # 10% do balance por trade
    window_size=50,                 # 50 candles de histórico
    max_episode_steps=5000,         # 5000 steps por episódio
    random_start=True,              # Início aleatório do episódio
    persist_balance=False,          # Desabilitado (bug de truncation)
    use_sharpe_reward=True          # Sharpe Ratio como reward
)
```

### Observation Space
- **Shape:** (50, 19)
- **50 timesteps** de histórico
- **19 features:**
  - 16 market features (OHLCV, indicators)
  - 3 portfolio features (balance, position, equity)
- **Normalização:** Z-score (mean=0, std=1)

### Action Space
- **Type:** Box([-1, 1])
- **Discretização:**
  - `action < -0.33` → SHORT
  - `-0.33 ≤ action ≤ 0.33` → FLAT
  - `action > 0.33` → LONG

### Reward Function (Sharpe Ratio)
```python
# Calcular return do step
step_return = (equity - previous_equity) / previous_equity
returns_history.append(step_return)

# Sharpe Ratio (últimos 100 steps)
returns_array = np.array(returns_history[-100:])
mean_return = returns_array.mean()
std_return = returns_array.std() + 1e-8
sharpe = mean_return / std_return

# Normalizar para [-1, 1]
reward = np.tanh(sharpe * 10)

# Reward shaping mínimo
if step_return > 0:
    reward += 0.01      # Bônus pequeno por lucro
elif step_return < -0.01:
    reward -= 0.02      # Penalidade maior por prejuízo
```

---

## 3. HYPERPARAMETERS TD3

### ⚠️ MUDANÇAS ENTRE 2M E 3M STEPS

| Hyperparameter | 2M steps (24 meses) | 3M steps (36 meses) | Mudança |
|----------------|---------------------|---------------------|---------|
| **learning_rate** | 3e-4 (0.0003) | **5e-4 (0.0005)** | ⬆️ +67% |
| **buffer_size** | 500,000 | **1,000,000** | ⬆️ +100% |
| **learning_starts** | 10,000 | **25,000** | ⬆️ +150% |
| **batch_size** | 256 | **512** | ⬆️ +100% |
| **tau** | 0.005 | 0.005 | = |
| **gamma** | 0.995 | 0.995 | = |
| **train_freq** | 1 | 1 | = |
| **gradient_steps** | 1 | 1 | = |
| **action_noise** | σ=0.5 | σ=0.5 | = |
| **timesteps** | 2,000,000 | **3,000,000** | ⬆️ +50% |

### Configuração Completa (3M steps)
```python
from stable_baselines3.common.noise import NormalActionNoise

action_noise = NormalActionNoise(
    mean=np.zeros(1), 
    sigma=0.5 * np.ones(1)  # 50% noise para exploração
)

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=5e-4,         # AUMENTADO de 3e-4
    buffer_size=1000000,        # DOBRADO de 500k
    learning_starts=25000,      # AUMENTADO de 10k
    batch_size=512,             # DOBRADO de 256
    tau=0.005,                  # Target network update rate
    gamma=0.995,                # Discount factor
    train_freq=1,               # Update a cada step
    gradient_steps=1,           # 1 gradient step por update
    action_noise=action_noise,  # Exploration noise
    verbose=1,
    device='privateuseone:0'    # AMD GPU DirectML
)
```

---

## 4. RESULTADOS TD3

### Progressão do Treinamento

| Checkpoint | Dataset | Eval Reward | Trades | Win Rate | Return |
|------------|---------|-------------|--------|----------|--------|
| 1M steps | 9 meses (27k) | 396-537 | 62 | 4.84% | -3.99% |
| 2M steps | 24 meses (55k) | 414.91 | 56 | 19.64% | -1.93% |
| **3M steps** | **36 meses (82k)** | **414.91** | **48** | **8.33%** | **-2.20%** |

### Backtest Final (3M steps)
```
Balance Final: $9,779.71
Total Return: -2.20%
Total Trades: 48
Wins: 4
Losses: 20
Win Rate: 8.33%
Sharpe Ratio: 2.34
Max Drawdown: -24.77%
Profit Factor: 0.01
Score: 3/8 (REGULAR)
```

### Análise Crítica
- ✅ **Sharpe alto (2.34)** - Baixa volatilidade
- ✅ **Drawdown controlado (-24.77%)** - Gestão de risco
- ❌ **Win rate baixo (8.33%)** - Poucas vitórias
- ❌ **Return negativo (-2.20%)** - Ainda perdendo dinheiro
- ⚠️ **Piora de 2M → 3M** - Possível overfitting ou LR alto demais

---

## 5. DIAGNÓSTICO: POR QUE PIOROU?

### Hipóteses

1. **Learning Rate Alto (5e-4)**
   - Dobrou vs baseline (3e-4)
   - Pode ter "desaprendido" boas políticas do 2M
   - Oscilações grandes nos pesos

2. **Batch Size Dobrado (512)**
   - Menos updates por época
   - Gradientes menos precisos
   - Convergência mais lenta

3. **Learning Starts Alto (25k)**
   - Demorou mais para começar a treinar
   - Coletou experiências sub-ótimas no buffer

4. **Overfitting no Treino**
   - Testando no mesmo dataset de treino
   - Não vimos dados "out-of-sample"

5. **Reward Shaping Sharpe**
   - Incentiva baixa volatilidade (Sharpe alto)
   - Pode estar incentivando INAÇÃO (poucos trades)

---

## 6. MIGRAÇÃO PARA SAC

### Por que SAC?
- **Entropy regularization** - Incentiva exploração e diversidade
- **Stochastic policy** - Menos determinístico que TD3
- **Maximum entropy RL** - Balanceia reward e diversidade de ação
- **Menos prone a overfitting** - Entropy evita políticas colapsadas

### Estratégia de Migração

#### Passo 1: Backup TD3
```bash
cp models/base_btcusdt_final.zip models/backup_td3_3M_20260110.zip
```

#### Passo 2: Ambiente Idêntico
```python
# MESMO ambiente do TD3 para comparação justa
env = TradingEnv(
    df=df,
    initial_balance=10000,
    commission=0.0010,
    slippage=0.0005,
    leverage=3,
    position_size=0.10,
    window_size=50,
    max_episode_steps=5000,    # IGUAL TD3
    random_start=True,
    persist_balance=False,
    use_sharpe_reward=True     # TESTAR: True (igual TD3) ou False (delta equity puro)
)
```

#### Passo 3: Hyperparameters SAC (Conservadores)
```python
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=1e-4,         # MENOR que TD3 (5e-4) - não destruir pesos
    buffer_size=1000000,        # IGUAL TD3
    learning_starts=10000,      # MENOR que TD3 (25k) - treinar mais cedo
    batch_size=256,             # MENOR que TD3 (512) - gradientes mais precisos
    tau=0.005,                  # IGUAL TD3
    gamma=0.995,                # IGUAL TD3
    train_freq=1,               # IGUAL TD3
    gradient_steps=1,           # IGUAL TD3
    ent_coef='auto',            # ENTROPY automático (diferencial do SAC)
    target_entropy='auto',      # Entropy target automático
    use_sde=False,              # State Dependent Exploration (testar True depois)
    verbose=1,
    device='privateuseone:0'
)
```

#### Passo 4: Transfer Learning TD3 → SAC
```python
# Carregar pesos do TD3 (critics compartilhados)
td3_model = TD3.load("models/base_btcusdt_final.zip")

# SAC tem 2 Q-networks (como TD3) + actor estocástico
# Copiar pesos manualmente:
sac_model.actor.load_state_dict(td3_model.actor.state_dict())
sac_model.critic.load_state_dict(td3_model.critic.state_dict())
sac_model.critic_target.load_state_dict(td3_model.critic_target.state_dict())
```

#### Passo 5: Fine-tuning SAC
```python
# Treinar 1M-2M steps para adaptar entropy
model.learn(
    total_timesteps=1500000,   # 1.5M steps fine-tuning
    callback=eval_callback,
    progress_bar=True
)
```

---

## 7. EXPERIMENTOS PARALELOS SUGERIDOS

### Variação A: SAC com Sharpe Reward (baseline)
- Ambiente idêntico ao TD3
- Comparação direta de algoritmos

### Variação B: SAC com Delta Equity Puro
```python
# Reward simples sem Sharpe
reward = (equity - previous_equity) / initial_balance

# Reward shaping AUMENTADO
if step_return > 0:
    reward += 0.05  # Bônus maior por lucro
elif step_return < -0.01:
    reward -= 0.10  # Penalidade maior por prejuízo
```

### Variação C: SAC com Episodes Menores
```python
max_episode_steps=3000  # vs 5000 do TD3
# Mais episódios = mais diversidade de experiências
```

### Variação D: SAC com Entropy Fixo
```python
ent_coef=0.2  # vs 'auto'
# Controle manual da exploração
```

---

## 8. MÉTRICAS DE SUCESSO PARA SAC

### Mínimo Aceitável
- Win Rate > 30% (vs 8.33% do TD3)
- Total Trades > 50 (vs 48 do TD3)
- Return > -1% (vs -2.20% do TD3)
- Sharpe > 2.0 (manter controle de risco)

### Objetivo Ideal
- Win Rate > 45%
- Total Trades > 80
- Return > +5%
- Sharpe > 2.5
- Max Drawdown < -20%

---

## 9. CHECKLIST MIGRAÇÃO

- [ ] Backup TD3: `models/backup_td3_3M_20260110.zip`
- [ ] Ambiente SAC idêntico criado
- [ ] Hyperparameters SAC configurados (conservadores)
- [ ] Transfer learning TD3 → SAC implementado
- [ ] Callback de avaliação configurado
- [ ] Treino SAC 1.5M steps iniciado
- [ ] Backtest SAC vs TD3 comparado
- [ ] Experimentos paralelos (A/B/C/D) documentados

---

## 10. ARQUIVOS IMPORTANTES

### Modelos
- `models/base_btcusdt_final.zip` - TD3 3M steps (atual)
- `models/backup_td3_3M_20260110.zip` - Backup TD3
- `models/sac_btcusdt_final.zip` - SAC (a criar)

### Datasets
- `data/train_btcusdt_36m_20260109.csv` - 82,904 candles (treino)
- `data/test_btcusdt_36m_20260109.csv` - 20,727 candles (teste)
- `data/full_btcusdt_36m_20260109.csv` - 103,631 candles (full)

### Logs
- `training_td3_sharpe_5k.log` - Log TD3 2M
- `backtest_report_20260110_112527.txt` - Backtest TD3 3M
- `TD3_AMBIENTE_COMPLETO.md` - Este arquivo

---

## CONCLUSÃO

**TD3 3M steps:** Modelo estável com Sharpe excelente (2.34) mas win rate baixo (8.33%) e return negativo (-2.20%). Hyperparameters agressivos (lr=5e-4, batch=512) podem ter causado piora vs 2M steps.

**Próximo passo:** Migrar para SAC com hyperparameters conservadores (lr=1e-4, batch=256) aproveitando pesos do TD3 via transfer learning. Entropy regularization do SAC deve aumentar exploração e diversidade de trades, melhorando win rate.

**Objetivo:** Win rate >30%, return >-1%, trades >50 em 1.5M steps de fine-tuning SAC.
