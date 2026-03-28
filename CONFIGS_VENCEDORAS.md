# 🏆 CONFIGURAÇÕES VENCEDORAS - MODELO V6

**Data:** 14/01/2026  
**Modelo:** `sac_futuros_v6_final_20260112_012926.zip`  
**Status:** ✅ MELHOR MODELO ATUAL

---

## 📊 Performance Comprovada

### Backtest Results (4 runs estocásticos):
- **Return médio:** -0.96% (vs -10.72% do V8)
- **Trades médio:** 624 (vs 2,318 do V8) → **73% menos trades**
- **Win Rate médio:** 20.21% (vs 17.71% do V8)
- **Gestão de perda:** **11x melhor** que V8

### Características:
- ✅ Adaptativo: Alterna entre conservador (233 trades) e agressivo (1,160 trades)
- ✅ Seletivo: Menos trades = melhor qualidade
- ✅ Resiliente: Perdas controladas mesmo em mercado difícil
- ✅ Balanceado: Varia Long/Short conforme mercado

---

## 🔑 HIPERPARÂMETROS SAC V6

```python
# Core SAC Parameters
ent_coef = 0.1  # FIXO - CRÍTICO! (não usar auto ou 0.05)
target_entropy = -1.0
learning_rate = 3e-4
buffer_size = 200_000
batch_size = 256
tau = 0.005
gamma = 0.99

# SDE (Stochastic Differential Equations)
use_sde = True
sde_sample_freq = 4
use_sde_at_warmup = True

# Network Architecture
policy_kwargs = {
    'net_arch': {'pi': [256, 256], 'qf': [256, 256]},
    'log_std_init': -1.0
}

# Training
learning_starts = 5000
train_freq = 1
gradient_steps = 1
target_update_interval = 1
optimize_memory_usage = False
stats_window_size = 100
```

---

## 🏗️ AMBIENTE V6

```python
# Balance & Trading
initial_balance = 10_000
commission = 0.0004  # Binance taker fee
slippage = 0.0005

# Risk Management (CRÍTICO!)
leverage = 1.5  # REDUZIDO de 2x - mais seguro
position_size = 0.1  # Base 10%, MAX 5% aplicado no código
stop_loss = -5%  # AUTOMÁTICO FORÇADO - não deixa escolha
liquidation_threshold = 0.15  # 15%
maintenance_margin_rate = 0.01  # 1%

# Episode Configuration
window_size = 50  # Lookback de 50 candles
max_episode_steps = 2000  # REDUZIDO de 5000 - aprende mais rápido
random_start = True
persist_balance = False  # Cada episódio independente

# Reward Structure
use_sharpe_reward = False  # Delta equity puro
use_hybrid_reward = False
enable_indicator_shaping = True  # CRÍTICO! USA SMA50, RSI, MACD
```

---

## 🎲 ACTION NOISE

```python
# NormalActionNoise
mean = 0
sigma = 0.2  # 20% - REDUZIDO de 30%
```

**Por que 20%?**
- Menos chaos que 30%
- Exploração suficiente
- Não causa overtrading

---

## 📈 TRAINING SETUP

```python
# Checkpoints
save_freq = 50_000  # A cada 50k steps
save_path = "./models/"
name_prefix = "sac_futuros_v6"
save_replay_buffer = True
save_vecnormalize = True

# TensorBoard
tensorboard_log = "./logs/sac_futuros_v6/"

# Total Training
total_timesteps = 500_000  # Original
# Para continuar: adicionar mais 300k-500k
```

---

## 🎮 CALLBACKS ATIVOS

```python
# 1. TradingMetricsCallback
- Loga métricas no TensorBoard
- win_rate, liquidations, sharpe_ratio, profit_factor

# 2. LiquidationMonitor
max_liquidations = 5  # Para se >5 em 10k steps
check_freq = 10_000

# 3. PerformanceDecayMonitor
min_winrate = 0.05  # 5%
patience = 5  # episódios

# 4. CheckpointCallback
Salva modelo + replay buffer
```

---

## 🚀 DIFERENCIAIS V6 vs V8/V12

| Aspecto | V6 | V8/V12 |
|---------|-----|--------|
| **ent_coef** | 0.1 FIXO | 0.05 FIXO |
| **Leverage** | 1.5x | 1.5x |
| **Position Size** | MAX 5% | 5% base |
| **Episode Steps** | 2000 | 4000 |
| **Action Noise** | 20% | Sem noise |
| **Stop-loss** | -5% automático | -7% |
| **Trades** | 600-1,200 | 2,200-2,400 |
| **Win Rate** | 20% | 17-19% |
| **Return** | -1% | -10% |

**Por que V6 vence:**
- ✅ Menos exploração (0.1 vs 0.05) = mais estável
- ✅ Episódios curtos = aprende gestão de risco rápido
- ✅ Stop-loss automático = corta perdas
- ✅ Noise controlado = não overtrade

---

## 🔧 COMO CONTINUAR TREINAMENTO V6

### Opção 1: Script Dedicado (RECOMENDADO)

Criar `train_futuros_v6_continue.py`:

```python
from stable_baselines3 import SAC
from environment.trading_env import TradingEnv
import torch_directml

# 1. Carregar modelo existente
model = SAC.load(
    "models/sac_futuros_v6_final_20260112_012926.zip",
    device=torch_directml.device()
)

# 2. Continuar treinamento
model.learn(
    total_timesteps=300_000,  # +300k steps
    reset_num_timesteps=False,  # CRÍTICO! Não reseta contador
    tb_log_name="continue_v6",
    callback=callbacks,  # Mesmos callbacks
    progress_bar=True
)

# 3. Salvar
model.save("models/sac_futuros_v6_800k_steps.zip")
```

### Opção 2: Modificar train_futuros_v6.py

Na linha ~337, trocar:
```python
# ANTES:
reset_num_timesteps=True

# DEPOIS:
reset_num_timesteps=False
```

E carregar modelo antes:
```python
# Adicionar após criar ambiente (linha ~169):
if Path("models/sac_futuros_v6_final_20260112_012926.zip").exists():
    model = SAC.load(
        "models/sac_futuros_v6_final_20260112_012926.zip",
        env=env,
        device=dml_device
    )
    print("✅ Modelo V6 carregado! Continuando treinamento...")
else:
    # Criar novo modelo (código atual)
```

---

## 📝 CHECKLIST PRE-TREINO

Antes de iniciar/continuar treinamento:

- [ ] Verificar `ent_coef = 0.1` (FIXO!)
- [ ] Verificar `leverage = 1.5`
- [ ] Verificar `max_episode_steps = 2000`
- [ ] Verificar `action_noise sigma = 0.2`
- [ ] Verificar `stop_loss = -5%` no environment
- [ ] Verificar `save_freq = 50_000`
- [ ] Backup do modelo atual
- [ ] TensorBoard rodando: `tensorboard --logdir=logs/sac_futuros_v6/`

---

## 🎯 METAS PARA PRÓXIMOS 300k STEPS

**Atual (500k):**
- Return: -0.96%
- Win Rate: 20.21%
- Trades: 624 médio

**Meta (800k):**
- Return: +2% a +5%
- Win Rate: 23-25%
- Trades: 600-800 (manter seletividade)
- Sharpe: >1.0

**Sinais de sucesso:**
- Win rate crescendo gradualmente
- Trades mantendo <1,000
- Return virando positivo
- Sharpe ratio subindo

**Sinais de alerta:**
- Win rate caindo abaixo de 18%
- Trades explodindo >2,000
- Liquidações aparecendo
- Return piorando

---

## 🛡️ ERROS A EVITAR

❌ **NÃO mudar:**
- ent_coef (manter 0.1!)
- leverage (manter 1.5x)
- episode_steps (manter 2000)
- stop_loss (manter -5%)

❌ **NÃO fazer:**
- Treinar do zero (perder aprendizado)
- Aumentar noise >30%
- Aumentar leverage >2x
- Remover indicator_shaping

✅ **PODE ajustar:**
- total_timesteps (adicionar mais)
- save_freq (mais/menos frequente)
- tensorboard_log (novo diretório)

---

## 📚 REFERÊNCIAS

- Arquivo de treino: `train_futuros_v6.py`
- Ambiente: `src/environment/trading_env.py`
- Callbacks: `callbacks/trading_metrics.py`
- Modelo atual: `models/sac_futuros_v6_final_20260112_012926.zip`
- Dashboard: `dashboard.py` (linha 114 - já configurado com V6)

---

## 🔄 HISTÓRICO DE VERSÕES

| Versão | Steps | Win Rate | Return | Status |
|--------|-------|----------|--------|--------|
| V3 | 20k | ~10% | N/A | ❌ Muitas liquidações |
| V4 | 20k | ~10% | N/A | ❌ Paralisado por punições |
| V5 | 20k | ~10% | N/A | ❌ Continuou liquidando |
| **V6** | 500k | **20.21%** | **-0.96%** | ✅ **ATUAL - MELHOR** |
| V7 | 700k | ~15% | N/A | ⚠️ Não testado suficiente |
| V8 | 500k | 17.71% | -10.72% | ❌ Inferior ao V6 |

---

## 💡 INSIGHTS IMPORTANTES

1. **Menos é mais**: V6 faz menos trades mas melhores
2. **Stop-loss salva**: Automático em -5% previne desastres
3. **Episódios curtos ensinam**: 2000 steps força aprendizado rápido
4. **Exploration equilibrada**: 0.1 ent_coef é sweet spot
5. **Adaptabilidade vence**: V6 varia estratégia conforme mercado

---

**Última atualização:** 14/01/2026  
**Próxima revisão:** Após atingir 800k steps
