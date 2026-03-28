# 🚀 PLANO V19 — LSTM Multi-Par com Observações Normalizadas

**Data**: 2026-03-01  
**Status**: Em implementação  
**Motivação**: V18 (4.76M/6M steps) não aprendeu — win rate ~43% flat, explained_variance ≈ 0, STD→3.17

---

## 🔴 Problemas Identificados no V18

### Bug Crítico #1 — OHLCV clippado a 100 (maior impacto)

O `clip(-100, 100)` no final da observação destruía as 5 features de preço para BTC/ETH/BNB:

```
BTC close = 27,839 → clip → 100.0  ⚠️ CONSTANTE
BTC open  = 27,823 → clip → 100.0  ⚠️ CONSTANTE
BTC high  = 27,859 → clip → 100.0  ⚠️ CONSTANTE
BTC volume = 1,940 → clip → 100.0  ⚠️ CONSTANTE
SOL close = 83     → clip → 83.4   ✅ ok (por acaso)
```

A LSTM recebia `[100, 100, 100, 100, 100, 0.50, 1.0, ...]` para BTC — as 5 features mais importantes do modelo eram **todas iguais e constantes**. O modelo não conseguia distinguir BTC de "ruído constante = 100".

**Sintomas no TensorBoard V18:**
- `explained_variance ≈ 0` durante todo o treino (crítico não aprendeu)
- `STD = 3.17` e crescendo (exploração aleatória, sin convergência)
- `clip_fraction ≈ 0`, `approx_kl ≈ 0` (política não atualizando)
- Win rate flat 43% = aleatório com viés LONG

### Bug Crítico #2 — `vf_coef=0.1` mata o crítico

| Parâmetro | Recomendado (V17 docs) | V18 (errado) | Impacto |
|---|---|---|---|
| `vf_coef` | 0.5–1.0 | **0.1** | Crítico aprende 5-10x mais devagar |
| `n_steps` | 4096 | 2048 | Menos contexto temporal por update |
| `ent_coef` | 0.05 | 0.03 | Exploração insuficiente para 4 pares |

### Bug #3 — Leverage treino ≠ produção

| Ambiente | Leverage | Stop-loss "real" |
|---|---|---|
| **Treino** | 1.0x | -7% de equity |
| **Produção** | 1.5x | -7% de equity **mas 1.5x mais dólares** |

O modelo aprende risco errado → na produção as perdas são 1.5x o esperado.

### Bug #4 — Estado de portfólio sempre 1.0 na engine

```python
# Engine (bug atual):
port = {'position': 0.0, 'balance_norm': 1.0, 'equity_norm': 1.0}  # NUNCA atualizado com saldo real
```

A feature de portfolio da observação fica sempre 1.0 independente do P&L real → a LSTM nunca sabe se está em drawdown.

---

## ✅ Soluções V19

### Fix #1 — Normalização OHLCV in-situ (sem alterar datasets)

**Ideia**: Converter OHLCV absolutos em **valores relativos ao close** do candle atual. Todos os ativos (BTC=$67k ou SOL=$83) terão a mesma escala após a normalização.

```python
def _normalize_ohlcv(window: np.ndarray) -> np.ndarray:
    """
    Normaliza as 5 colunas OHLCV para escala relativa.
    IDX_CLOSE=3: close de cada candle (não o close do passo atual)
    
    Resultado: todos os valores ficam na escala [-5, +5] aprox.
    BTC $67k e SOL $83 terão o mesmo range de features.
    """
    w = window.copy()
    # Referência: close de cada candle (IDX 3)
    ref = w[:, 3].copy()
    ref[ref == 0] = 1.0
    
    # 0:open → % diff vs close do próprio candle (corpo do candle)
    w[:, 0] = (w[:, 0] / ref - 1.0) * 100
    # 1:high → upper wick % do close
    w[:, 1] = (w[:, 1] / ref - 1.0) * 100
    # 2:low  → lower wick % do close (negativo → sombra abaixo)
    w[:, 2] = (w[:, 2] / ref - 1.0) * 100
    # 3:close → retorno % vs candle anterior (return step)
    prev = np.roll(ref, 1); prev[0] = ref[0]
    w[:, 3] = (ref / (prev + 1e-10) - 1.0) * 100
    # 4:volume → ratio vs Volume_MA_20 (IDX 19)
    vol_ma = w[:, 19].copy(); vol_ma[vol_ma == 0] = 1.0
    w[:, 4] = w[:, 4] / (vol_ma + 1e-8)
    
    return w
```

### Fix #2 — Hiperparâmetros corrigidos

```python
PPO_CONFIG = {
    'learning_rate': 2e-4,
    'n_steps':       4096,      # V18: 2048 → V19: 4096 (mais contexto)
    'batch_size':    256,       # maior batch = gradientes mais estáveis
    'n_epochs':      4,
    'gamma':         0.95,
    'gae_lambda':    0.9,
    'clip_range':    0.2,
    'ent_coef':      0.05,      # V18: 0.03 → V19: 0.05 (mais exploração)
    'vf_coef':       0.5,       # V18: 0.1  → V19: 0.5  (crítico aprende)
    'max_grad_norm': 0.5,
}
```

### Fix #3 — Leverage alinhado

```python
ENV_CONFIG = {
    ...
    'leverage':  1.5,      # V18: 1.0 → V19: 1.5 (igual produção)
    ...
}
# Stop-loss interno ajustado: -7%/1.5 = -4.67% de equity
```

### Fix #4 — Portfolio real na engine (pós-treino)

Após carregar o modelo V19 no motor, atualizar `port` com saldo real do WS:

```python
ws_bal_total = float((ws_bal_data or {}).get('total', 0))
ws_equity = ws_bal_total + sum(float(p.get('unRealizedProfit', 0)) for p in positions)
port['balance_norm'] = ws_bal_total / _initial_balance
port['equity_norm'] = ws_equity / _initial_balance
port['position'] = np.sign(ws_pos_map.get(sym, 0.0))
```

---

## 📊 Expectativa de Resultado

| Métrica | V17.7 | V18 | V19 (previsto) |
|---|---|---|---|
| Win rate treino | ~48% | 43% | 53–60% |
| Explained variance | baixo | ~0 | >0.05 |
| STD ação | estável | 3.17↑ | <2.5 |
| Generalizável (multi-par) | ❌ | ❌ | ✅ |

---

## 🏗 Arquitetura V19 (sem alterações vs V18)

```
RecurrentPPO
├── MlpLstmPolicy
│   ├── LSTM: hidden=256, layers=2
│   └── MLP: [256, 256]
├── Obs: (50, 31)  ← shape idêntico, conteúdo normalizado
├── Action: Box[-1, 1]
├── 4 envs: BTC + ETH + SOL + BNB
└── 6M steps / 732 rollouts
```

---

## 📁 Arquivos Modificados

| Arquivo | Mudança |
|---|---|
| `src/environment/trading_env_v19_lstm.py` | **NOVO** — clone do V18 + normalização OHLCV + leverage 1.5x |
| `train_recurrent_ppo_v19_multipair.py` | **NOVO** — usa env V19 + hiperparâmetros corrigidos |
| `dashboard/trading/observation.py` | fix #1 aplicado também na inferência (consistência treino/produção) |

---

## 🔬 Monitoramento TensorBoard — Sinais de Convergência

Após 50k steps, verificar:

| Métrica | Sinal ruim | Sinal bom (V19 esperado) |
|---|---|---|
| `train/explained_variance` | < 0.01 | > 0.05 após 100k |
| `train/std` | > 3.0 ou crescendo | 1.5–2.5, estável ou decrescendo |
| `rollout/aggregate_winrate` | < 40% ou flat | tendência crescente |
| `train/value_loss` | > 500 ou divergindo | < 200, estável |
| `train/clip_fraction` | ≈ 0 sempre | > 0.01 (política atualizando) |

Se `explained_variance < 0.01` após 200k steps → parar e revisar reward shaping.
