# 🎯 PLANO DE AÇÃO: EVOLUÇÃO V16 → V19 (WIN RATE 20% → 35%+)

**Data:** 26/01/2026  
**Objetivo:** Aumentar win rate de 20% (V15 baseline) para 35%+ através de melhorias incrementais  
**Timeline:** 4-6 semanas  
**Status:** 🚀 EM ANDAMENTO

---

## 📊 HISTÓRICO E LIÇÕES APRENDIDAS

### ✅ O QUE JÁ FUNCIONOU (V6/V15):
1. **Leverage 1.5x** (seguro, quase impossível liquidar)
2. **Episodes 2000 steps** (aprende mais rápido com mais resets)
3. **Buffer 100k** (evita catastrophic forgetting)
4. **Ent_coef 0.05** (exploração moderada)
5. **Net_arch [256, 256]** (simples e efetivo)
6. **Sharpe Reward** (retorno ajustado por risco)
7. **Bônus balanceados** (V15: +0.05 lucro = +0.05 cortar loss)

### ❌ O QUE NÃO FUNCIONOU (V13/V14/V16.0):
1. **Indicator shaping contratrend** (RSI vendendo em alta - DESABILITADO no V15)
2. **Penalidades prematuras** (começavam em -2% quando stop é -7%)
3. **Bônus desbalanceados** (lucro valia 2.7x mais que cortar loss)
4. **Episodes 4000 steps** (muito longo, Sharpe instável)
5. **Buffer 700k** (muito grande, forgetting)
6. **🚨 V16.0 BUG CRÍTICO** (18/02/2026): **Bonificava perdas (+0.05)!**
7. **🚨 V16.0 FREEZE BUG**: Flip-flop penalty + inactivity fraca = modelo congelado
8. **Sharpe reward** (contribuiu para freeze mas não era o único problema)

### 🎯 BASELINE ATUAL:
- **V15:** Win rate 18-22%, Return +2-5%, Sharpe moderado
- **V16 (multi-TF):** Win rate esperado 22-27% (em treinamento)

---

## 🗺️ ROADMAP COMPLETO

```
┌─────────────┬──────────────┬─────────────┬──────────────┐
│   V16       │    V17       │    V18      │     V19      │
│ Multi-TF    │  + Filtros   │ + Stops     │  + Ensemble  │
│  Base       │  Qualidade   │  Dinâmicos  │   (3 models) │
├─────────────┼──────────────┼─────────────┼──────────────┤
│ 22-27%      │   25-31%     │   28-34%    │   31-38%     │
│ (1 semana)  │  (1 semana)  │ (1 semana)  │  (1 semana)  │
└─────────────┴──────────────┴─────────────┴──────────────┘
```

---

# 📋 FASE 1: V16 - MULTI-TIMEFRAME (EM ANDAMENTO)

## ✅ Checklist V16

- [x] Criar `collect_multi_timeframe.py`
- [x] Baixar dados 15m, 1h, 4h (36 meses)
- [x] Criar `trading_env_multi_tf.py`
- [x] Criar `train_sac_v16.py`
- [x] Iniciar treinamento (1M steps, ~10-20h)
- [ ] Monitorar TensorBoard durante treino
- [ ] Checkpoint 200k: Backtest preliminar
- [ ] Checkpoint 500k: Backtest intermediário
- [ ] Checkpoint 1M: Backtest final
- [ ] Análise comparativa V15 vs V16

### 📝 Notas V16:
- **Observation space:** 1255 valores (vs 1000 do V15)
- **Timeframes:** 15m (50 candles), 1h (12 candles), 4h (3 candles)
- **Expectativa:** Win rate 22-27% (multi-temporal reduz falsos sinais)

### 🔍 Métricas para Monitorar:
```bash
# TensorBoard
tensorboard --logdir=./tensorboard/sac_v16_multi_tf_*/

# Observar:
- episode/win_rate (target: >22%)
- episode/return (target: >3%)
- episode/sharpe_ratio (target: >0.5)
- train/entropy (deve permanecer >0.1)
- train/learning_rate (3e-4 constante)
```

### ⚠️ Critérios de Sucesso V16:
- [ ] Win rate >= 22% (vs 18-22% V15)
- [ ] Return >= +3%
- [ ] Sharpe ratio >= 0.5
- [ ] Max drawdown <= 15%
- [ ] Trade count: 400-800 (nem muito nem pouco)
- [ ] Long/Short balance: 40-60% cada

### 📊 Backtest Command:
```bash
python backtest.py models/sac_v16_multi_tf_*_1000000_steps.zip
```

---

# 📋 FASE 2: V17 - FILTROS DE QUALIDADE

## 🎯 Objetivo:
Adicionar filtros para operar APENAS em condições de alta probabilidade.  
**Win rate esperado:** 25-31%

## ✅ Checklist V17

### 1️⃣ Criar Ambiente com Filtros (3h)
- [ ] Copiar `trading_env_multi_tf.py` → `trading_env_v17_filtered.py`
- [ ] Implementar `_calculate_trade_quality()` method
- [ ] Implementar `_check_volume_filter()`
- [ ] Implementar `_check_volatility_filter()`
- [ ] Implementar `_check_trend_clarity_filter()`
- [ ] Implementar `_check_time_filter()`
- [ ] Adicionar penalidade por operar em baixa qualidade
- [ ] Testar ambiente isoladamente

### 2️⃣ Implementação dos Filtros

#### 📊 Filtro 1: Volume (Liquidez)
```python
def _check_volume_filter(self) -> bool:
    """Volume deve estar acima da média para garantir liquidez."""
    current_vol = self.dfs['15m'].iloc[self.current_step]['volume']
    vol_ma = self.dfs['15m']['Volume_MA_20'].iloc[self.current_step]
    
    # Volume deve ser 1.5x a média (movimento real, não fake)
    return current_vol > vol_ma * 1.5
```

#### 📈 Filtro 2: Volatilidade (Range ótimo)
```python
def _check_volatility_filter(self) -> bool:
    """Volatilidade moderada - nem muito nem pouco."""
    atr = self.dfs['15m']['ATR_14'].iloc[self.current_step]
    
    # ATR entre 0.3% e 1.5% (nem flat nem chaos)
    return 0.003 < atr < 0.015
```

#### 🎯 Filtro 3: Trend Clarity (Direção clara)
```python
def _check_trend_clarity_filter(self) -> bool:
    """Tendência clara em timeframe maior (4h)."""
    # EMAs do 4h
    current_4h = self.current_step // 16
    ema9 = self.dfs['4h']['EMA_9'].iloc[current_4h]
    ema21 = self.dfs['4h']['EMA_21'].iloc[current_4h]
    
    # Distância entre EMAs > 1% (tendência forte)
    divergence = abs(ema9 - ema21) / ema21
    return divergence > 0.01
```

#### 🕐 Filtro 4: Time of Day (Horário ideal)
```python
def _check_time_filter(self) -> bool:
    """Evita horários de baixa liquidez."""
    timestamp = self.dfs['15m'].iloc[self.current_step]['timestamp']
    hour = pd.to_datetime(timestamp).hour
    
    # Opera entre 8h e 22h UTC (alta liquidez global)
    return 8 <= hour <= 22
```

#### ✅ Integração no Step
```python
def _calculate_trade_quality(self) -> float:
    """
    Calcula score de qualidade [0, 1].
    1.0 = todas condições ideais
    0.0 = nenhuma condição atendida
    """
    filters = [
        self._check_volume_filter(),
        self._check_volatility_filter(),
        self._check_trend_clarity_filter(),
        self._check_time_filter()
    ]
    
    return sum(filters) / len(filters)

# No método step(), ANTES de executar trade:
def step(self, action):
    # ... código existente ...
    
    quality_score = self._calculate_trade_quality()
    
    # Se qualidade baixa E está tentando operar (não flat)
    if quality_score < 0.5 and discrete_action != 0:
        # Penaliza FORTE por operar em condições ruins
        reward -= 0.15 * (1 - quality_score)  # -0.15 se score=0
        
        # OPCIONAL: Força flat se muito ruim
        if quality_score < 0.25:
            discrete_action = 0  # Força ficar de fora
    
    # ... resto do código ...
```

### 3️⃣ Criar Script de Treinamento (30min)
- [ ] Copiar `train_sac_v16.py` → `train_sac_v17.py`
- [ ] Atualizar imports para usar `trading_env_v17_filtered`
- [ ] Ajustar nome de saves: `sac_v17_filtered_{timestamp}`
- [ ] Manter MESMOS hiperparâmetros V16

### 4️⃣ Treinar V17 (10-20h)
- [ ] Executar: `python train_sac_v17.py`
- [ ] Monitorar win rate (target >25%)
- [ ] Comparar trade count vs V16 (deve reduzir ~30%)
- [ ] Checkpoints: 200k, 500k, 1M

### 5️⃣ Validação V17
- [ ] Backtest checkpoint 1M
- [ ] Comparar com V16 (win rate, trades, return)
- [ ] Analisar: filtros reduziram falsos sinais?
- [ ] Verificar: não está muito conservador (flat >80%)?

### ⚠️ Critérios de Sucesso V17:
- [ ] Win rate >= 25% (vs 22-27% V16)
- [ ] Trade count: 300-600 (redução de ~30% vs V16)
- [ ] Return >= +4%
- [ ] Sharpe ratio >= 0.6
- [ ] Flat time: 60-70% (seletividade)

---

# 📋 FASE 3: V18 - STOP-LOSS & TAKE-PROFIT DINÂMICOS

## 🎯 Objetivo:
Stops adaptativos baseados em volatilidade (ATR).  
**Win rate esperado:** 28-34%

## ✅ Checklist V18

### 1️⃣ Análise Preparatória (2h)
- [ ] Analisar trades perdedores do V17
- [ ] Identificar: quantos stops em -7% poderiam ser evitados?
- [ ] Calcular ATR médio em diferentes regimes de mercado
- [ ] Determinar multiplicadores ótimos (2x, 3x, 4x ATR)

### 2️⃣ Implementação Stops Dinâmicos (3h)

#### 📉 Stop-Loss Dinâmico (ATR-based)
```python
def _calculate_dynamic_stop_loss(self) -> float:
    """
    Calcula stop loss dinâmico baseado em ATR e volatilidade.
    
    Returns:
        Stop loss em % (ex: 0.03 = 3%)
    """
    atr = self.dfs['15m']['ATR_14'].iloc[self.current_step]
    
    # Calcular volatilidade recente (média de 20 ATRs)
    atr_ma = self.dfs['15m']['ATR_14'].rolling(20).mean().iloc[self.current_step]
    
    # Classificar regime de volatilidade
    if atr < atr_ma * 0.8:  # Baixa volatilidade
        multiplier = 2.0  # Stop apertado
    elif atr < atr_ma * 1.2:  # Volatilidade normal
        multiplier = 3.0  # Stop padrão
    else:  # Alta volatilidade
        multiplier = 4.0  # Stop largo (não sair por ruído)
    
    # Stop = multiplier × ATR
    # Limites: mínimo 2%, máximo 10%
    stop_loss = np.clip(multiplier * atr, 0.02, 0.10)
    
    return stop_loss

# Atualizar _check_stop_loss():
def _check_stop_loss(self, current_price: float) -> bool:
    if self.position == 0:
        return False
    
    unrealized_pnl = self._calculate_pnl(current_price)
    unrealized_pct = unrealized_pnl / self.balance
    
    # Stop DINÂMICO (não fixo em -7%)
    dynamic_stop = self._calculate_dynamic_stop_loss()
    
    return unrealized_pct <= -dynamic_stop
```

#### 📈 Take-Profit Dinâmico (Trailing)
```python
def _calculate_dynamic_take_profit(self) -> float:
    """
    TP dinâmico baseado em força de tendência.
    
    Tendência forte = deixa correr
    Tendência fraca = realiza rápido
    """
    # Calcular força de tendência no 4h
    current_4h = self.current_step // 16
    ema9 = self.dfs['4h']['EMA_9'].iloc[current_4h]
    ema21 = self.dfs['4h']['EMA_21'].iloc[current_4h]
    
    trend_strength = abs(ema9 - ema21) / ema21
    
    atr = self.dfs['15m']['ATR_14'].iloc[self.current_step]
    
    if trend_strength > 0.02:  # Tendência FORTE (>2%)
        # Deixa correr - TP distante
        tp_multiplier = 5.0
    elif trend_strength > 0.01:  # Tendência MODERADA
        tp_multiplier = 3.5
    else:  # Tendência FRACA
        # Realiza rápido
        tp_multiplier = 2.5
    
    # TP = multiplier × ATR
    take_profit = tp_multiplier * atr
    
    return take_profit

# Adicionar no step():
def step(self, action):
    # ... após executar action ...
    
    # Se tem posição, verificar TP dinâmico
    if self.position != 0:
        unrealized_pnl = self._calculate_pnl(current_price)
        unrealized_pct = unrealized_pnl / self.balance
        
        dynamic_tp = self._calculate_dynamic_take_profit()
        
        # Se atingiu TP, fecha (mas não força - deixa modelo decidir)
        if unrealized_pct >= dynamic_tp:
            # Bônus EXTRA por realizar em TP ótimo
            reward += 0.03
```

#### 🎯 Trailing Stop
```python
def _update_trailing_stop(self, current_price: float):
    """
    Atualiza trailing stop para proteger lucros.
    """
    if self.position == 0:
        return
    
    unrealized_pct = self._calculate_pnl(current_price) / self.balance
    
    # Ativa trailing em +3%
    if unrealized_pct >= 0.03:
        if not hasattr(self, 'trailing_activated'):
            self.trailing_activated = True
            self.highest_unrealized = unrealized_pct
        else:
            # Atualiza máximo
            self.highest_unrealized = max(self.highest_unrealized, unrealized_pct)
        
        # Trailing stop: fecha se cair 1.5% do máximo
        if unrealized_pct < self.highest_unrealized - 0.015:
            self._close_position(current_price)
            reward += 0.02  # Bônus por proteger lucro
```

### 3️⃣ Criar Ambiente V18 (2h)
- [ ] Copiar `trading_env_v17_filtered.py` → `trading_env_v18_dynamic.py`
- [ ] Implementar `_calculate_dynamic_stop_loss()`
- [ ] Implementar `_calculate_dynamic_take_profit()`
- [ ] Implementar `_update_trailing_stop()`
- [ ] Testar isoladamente com dados sintéticos

### 4️⃣ Criar Script de Treinamento (30min)
- [ ] Copiar `train_sac_v17.py` → `train_sac_v18.py`
- [ ] Atualizar imports
- [ ] Ajustar saves: `sac_v18_dynamic_{timestamp}`

### 5️⃣ Treinar V18 (10-20h)
- [ ] Executar: `python train_sac_v18.py`
- [ ] Monitorar: win rate, avg win vs avg loss
- [ ] Verificar: stops dinâmicos reduzindo losses?
- [ ] Checkpoints: 200k, 500k, 1M

### 6️⃣ Validação V18
- [ ] Backtest checkpoint 1M
- [ ] Analisar distribuição de stops (% de -2%, -5%, -7%)
- [ ] Comparar avg loss: V18 vs V17
- [ ] Verificar risk/reward ratio melhorou

### ⚠️ Critérios de Sucesso V18:
- [ ] Win rate >= 28% (vs 25-31% V17)
- [ ] Avg loss MENOR que V17 (stops mais inteligentes)
- [ ] Risk/reward >= 1:2.5 (ganha 2.5x mais que perde)
- [ ] Max drawdown <= 12%
- [ ] Sharpe ratio >= 0.7

---

# 📋 FASE 4: V19 - ENSEMBLE DE MODELOS

## 🎯 Objetivo:
Combinar V16, V17, V18 em votação para reduzir falsos sinais.  
**Win rate esperado:** 31-38%

## ✅ Checklist V19

### 1️⃣ Análise Preparatória (3h)
- [ ] Comparar backtest V16, V17, V18
- [ ] Identificar: quando cada modelo acerta/erra?
- [ ] Calcular correlação entre previsões
- [ ] Determinar peso de cada modelo na votação

### 2️⃣ Implementação Ensemble (4h)

#### 🗳️ Sistema de Votação
```python
class EnsembleTrader:
    """
    Combina 3 modelos: V16 (multi-TF), V17 (filtered), V18 (dynamic).
    Decisão por votação majoritária ou weighted voting.
    """
    
    def __init__(self):
        self.models = {
            'v16_multi_tf': SAC.load('models/sac_v16_multi_tf_*_1000000_steps.zip'),
            'v17_filtered': SAC.load('models/sac_v17_filtered_*_1000000_steps.zip'),
            'v18_dynamic': SAC.load('models/sac_v18_dynamic_*_1000000_steps.zip')
        }
        
        # Pesos baseados em performance histórica
        self.weights = {
            'v16_multi_tf': 0.30,  # Base sólida
            'v17_filtered': 0.35,  # Melhor win rate
            'v18_dynamic': 0.35    # Melhor risk/reward
        }
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Votação ponderada.
        
        Returns:
            0: Flat, 1: Long, 2: Short
        """
        votes = {}
        
        for model_name, model in self.models.items():
            action, _ = model.predict(observation, deterministic=True)
            
            # Converter action contínuo para discreto
            if action < -0.1:
                vote = 2  # Short
            elif action > 0.1:
                vote = 1  # Long
            else:
                vote = 0  # Flat
            
            weight = self.weights[model_name]
            votes[vote] = votes.get(vote, 0) + weight
        
        # Decisão: maior peso acumulado
        decision = max(votes, key=votes.get)
        
        # CRITÉRIO DE CONFIANÇA: só opera se peso >= 60%
        if votes[decision] < 0.6 and decision != 0:
            return 0  # Discordância = cautela (flat)
        
        return decision
    
    def predict_with_confidence(self, observation: np.ndarray):
        """
        Retorna decisão + nível de confiança.
        """
        votes = {}
        predictions = []
        
        for model_name, model in self.models.items():
            action, _ = model.predict(observation, deterministic=True)
            
            if action < -0.1:
                vote = 2
            elif action > 0.1:
                vote = 1
            else:
                vote = 0
            
            predictions.append(vote)
            weight = self.weights[model_name]
            votes[vote] = votes.get(vote, 0) + weight
        
        decision = max(votes, key=votes.get)
        confidence = votes[decision]
        
        # Unanimidade = confiança máxima
        is_unanimous = len(set(predictions)) == 1
        
        return decision, confidence, is_unanimous
```

#### 📊 Estratégias de Votação

**Estratégia 1: Votação Simples (Maioria)**
```python
def simple_majority(predictions: list) -> int:
    """2 de 3 modelos concordam."""
    from collections import Counter
    vote_counts = Counter(predictions)
    majority = vote_counts.most_common(1)[0]
    
    # Se não há maioria (empate 1-1-1), fica flat
    if majority[1] < 2:
        return 0  # Flat
    
    return majority[0]
```

**Estratégia 2: Votação Ponderada (Baseada em Performance)**
```python
def weighted_voting(predictions: dict) -> int:
    """
    predictions = {'v16': 1, 'v17': 1, 'v18': 0}
    weights = {'v16': 0.30, 'v17': 0.35, 'v18': 0.35}
    """
    scores = {0: 0, 1: 0, 2: 0}
    
    for model, pred in predictions.items():
        scores[pred] += weights[model]
    
    return max(scores, key=scores.get)
```

**Estratégia 3: Conservadora (Unanimidade)**
```python
def unanimous_only(predictions: list) -> int:
    """Só opera se TODOS concordam."""
    if len(set(predictions)) == 1:
        return predictions[0]
    else:
        return 0  # Flat se há discordância
```

### 3️⃣ Criar Sistema de Backtesting Ensemble (3h)
- [ ] Criar `backtest_ensemble.py`
- [ ] Carregar 3 modelos (V16, V17, V18)
- [ ] Implementar votação configurável
- [ ] Simular trading com ensemble
- [ ] Comparar: ensemble vs cada modelo individual

```python
# backtest_ensemble.py
def backtest_ensemble(
    models: dict,
    voting_strategy: str = 'weighted',  # 'simple', 'weighted', 'unanimous'
    confidence_threshold: float = 0.6
):
    """
    Backtest do ensemble.
    
    Args:
        models: {'v16': model, 'v17': model, 'v18': model}
        voting_strategy: tipo de votação
        confidence_threshold: mínimo de confiança para operar
    """
    ensemble = EnsembleTrader(models, strategy=voting_strategy)
    
    # ... loop de backtest normal ...
    
    for step in range(len(test_data)):
        obs = get_observation(step)
        
        action, confidence, unanimous = ensemble.predict_with_confidence(obs)
        
        # Só opera se confiança >= threshold
        if confidence >= confidence_threshold or action == 0:
            execute_action(action)
        else:
            execute_action(0)  # Flat se baixa confiança
```

### 4️⃣ Experimentos Ensemble (8h)
- [ ] Teste 1: Votação simples (maioria)
- [ ] Teste 2: Votação ponderada (performance)
- [ ] Teste 3: Unanimidade (conservadora)
- [ ] Teste 4: Híbrido (ponderada + threshold confiança)
- [ ] Comparar win rate, trades, return de cada estratégia
- [ ] Selecionar melhor estratégia

### 5️⃣ Otimização de Pesos (4h)
- [ ] Grid search de pesos: [0.2, 0.3, 0.4, 0.5]
- [ ] Validação cruzada em múltiplos períodos
- [ ] Determinar pesos ótimos para cada modelo
- [ ] Testar robustez em out-of-sample

### 6️⃣ Criar Script de Trading Live (3h)
- [ ] Criar `trade_ensemble_v19.py`
- [ ] Integrar com Binance API
- [ ] Implementar votação em tempo real
- [ ] Adicionar logging de decisões
- [ ] Safety checks (saldo, posições, etc)

### 7️⃣ Validação Final V19
- [ ] Backtest ensemble vs V18 (melhor individual)
- [ ] Comparar métricas:
  - Win rate (target >31%)
  - Sharpe ratio (target >0.8)
  - Max drawdown (target <10%)
  - Trade quality (false positives)
- [ ] Paper trading por 1 semana
- [ ] Analisar: ensemble vale a complexidade?

### ⚠️ Critérios de Sucesso V19:
- [ ] Win rate >= 31% (vs 28-34% V18)
- [ ] Win rate > MELHOR modelo individual
- [ ] False signals reduzidos em 20%+
- [ ] Sharpe ratio >= 0.8
- [ ] Max drawdown <= 10%
- [ ] Ready for live trading

---

# 🛠️ FERRAMENTAS E SCRIPTS DE SUPORTE

## 1️⃣ Script de Análise Rápida
```python
# analyze_checkpoint.py
"""Análise rápida de checkpoints durante treino."""

import sys
from stable_baselines3 import SAC
from src.environment.trading_env_multi_tf import TradingEnvMultiTF

def quick_analysis(checkpoint_path: str):
    # Carregar modelo
    model = SAC.load(checkpoint_path)
    
    # Criar env de teste
    env = create_test_env()
    
    # Simular 100 episodes
    wins, losses, returns = 0, 0, []
    
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        
        if episode_return > 0:
            wins += 1
        else:
            losses += 1
        
        returns.append(episode_return)
    
    # Métricas
    win_rate = wins / 100 * 100
    avg_return = np.mean(returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    
    print(f"\n{'='*50}")
    print(f"📊 ANÁLISE: {checkpoint_path}")
    print(f"{'='*50}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Return: {avg_return:.4f}")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    quick_analysis(sys.argv[1])
```

## 2️⃣ Comparador de Versões
```python
# compare_versions.py
"""Compara múltiplas versões lado a lado."""

def compare_models(model_paths: list, test_data_path: str):
    results = {}
    
    for path in model_paths:
        version = extract_version(path)  # v16, v17, etc
        
        # Backtest
        metrics = run_backtest(path, test_data_path)
        
        results[version] = {
            'win_rate': metrics['win_rate'],
            'return': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'trades': metrics['total_trades'],
            'max_dd': metrics['max_drawdown']
        }
    
    # Tabela comparativa
    df = pd.DataFrame(results).T
    print("\n📊 COMPARAÇÃO DE VERSÕES")
    print(df.to_string())
    
    # Gráficos
    plot_comparison(df)
    
    return df
```

## 3️⃣ Monitor de Treinamento
```bash
# monitor_training.sh
#!/bin/bash
# Monitora treino e envia alertas

MODEL_DIR="models/"
TELEGRAM_TOKEN="YOUR_TOKEN"
CHAT_ID="YOUR_CHAT_ID"

while true; do
    # Verifica último checkpoint
    LATEST=$(ls -t $MODEL_DIR/*.zip | head -1)
    
    # Extrai steps do nome
    STEPS=$(echo $LATEST | grep -oP '\d+(?=_steps)')
    
    # Análise rápida
    python analyze_checkpoint.py $LATEST > /tmp/analysis.txt
    
    # Envia para Telegram
    WIN_RATE=$(grep "Win Rate" /tmp/analysis.txt | awk '{print $3}')
    
    if [ "$STEPS" -eq "200000" ] || [ "$STEPS" -eq "500000" ] || [ "$STEPS" -eq "1000000" ]; then
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/sendMessage" \
            -d chat_id=$CHAT_ID \
            -d text="🎯 Checkpoint $STEPS: Win Rate = $WIN_RATE"
    fi
    
    sleep 3600  # Checa a cada 1h
done
```

---

# 📈 MÉTRICAS DE ACOMPANHAMENTO

## Dashboard de Progresso

| Versão | Win Rate | Return | Sharpe | Trades | MaxDD | Status |
|--------|----------|--------|--------|--------|-------|--------|
| V15    | 18-22%   | +2-5%  | 0.4    | 600    | 18%   | ✅ Baseline |
| V16    | 22-27%   | +3-6%  | 0.5    | 500    | 15%   | 🔄 Treinando |
| V17    | 25-31%   | +4-7%  | 0.6    | 400    | 13%   | ⏳ Pendente |
| V18    | 28-34%   | +5-9%  | 0.7    | 350    | 12%   | ⏳ Pendente |
| V19    | 31-38%   | +7-12% | 0.8    | 300    | 10%   | ⏳ Pendente |

## KPIs por Fase

### Fase 1 (V16):
- [x] Dados coletados: 103k candles (15m), 25k (1h), 6k (4h)
- [x] Treino inicial V16.0: 11/02 - 12/02 (**FALHOU** - freeze bug)
- [x] **BUGS CRÍTICOS ENCONTRADOS** (18/02):
  - ❌ Bonificava perdas (+0.05 por loss!)
  - ❌ Flip-flop penalty travava modelo
  - ❌ Inactivity/holding penalties muito fracas
  - ❌ Double leverage bug (PnL calculado 2x)
- [x] **CORREÇÕES APLICADAS - V16.1** (18/02):
  - ✅ Perda = -0.03 penalidade
  - ✅ Flip-flop penalty REMOVIDA
  - ✅ Inactivity penalty 20x mais forte
  - ✅ Holding penalty 5x mais forte
  - ✅ Double leverage corrigido
- [x] **TREINO V16.1 INICIADO** (18/02 23:32): 1M steps com reward corrigida
  - ✅ 95k steps: Win rate 40-50%, trades variando, SEM freeze!
  - ✅ Modelo funcionando (não congelou como V16.0)
  - ❌ Checkpoint 200k: CONGELOU no backtest (99.95% holding, 1 trade)
- [x] **ANÁLISE PROFUNDA V16.1** (19/02):
  - ❌ Reward zero durante holding (300-400 steps sem penalty)
  - ❌ Inactivity/holding penalties AINDA muito fracas
  - ❌ Sem penalidade por flat (V15 tinha -0.01)
  - ❌ Bônus/penalties desbalanceados (lucro 1.67x > loss)
- [x] **CORREÇÕES APLICADAS - V16.2** (19/02):
  - ✅ Flat penalty: -0.01 por step (restaurada do V15)
  - ✅ Inactivity: começa em 200 steps (10x mais forte: 0.001)
  - ✅ Holding: começa em 300 steps (10x mais forte: 0.005)
  - ✅ Bônus/penalties balanceadas (0.04 lucro = 0.04 loss)
  - ✅ Bônus por cortar loss cedo: +0.03 se loss < -0.5%
- [ ] **RETREINAR V16.2** (próximo): 1M steps com reward REALMENTE corrigida
- [ ] Checkpoint 200k: Validar trades >100, holding <50%
- [ ] Checkpoint 500k: Confirmar estabilidade
- [ ] Checkpoint 1M: Backtest final vs V15
- [ ] Win rate final: __% (target: 22-27%)

### Fase 2 (V17):
- [ ] Filtros implementados: Volume, Volatilidade, Trend, Time
- [ ] Treino iniciado: __/__/____
- [ ] Win rate final: __%
- [ ] Redução de trades: __%

### Fase 3 (V18):
- [ ] Stops dinâmicos implementados
- [ ] Trailing stop implementado
- [ ] Treino iniciado: __/__/____
- [ ] Win rate final: __%
- [ ] Avg loss reduzido em: __%

### Fase 4 (V19):
- [ ] Ensemble testado com 3 estratégias
- [ ] Melhor estratégia: _______________
- [ ] Pesos otimizados: V16=___, V17=___, V18=___
- [ ] Win rate final: __%
- [ ] Paper trading 1 semana: PASS/FAIL

---

# ⚠️ CRITÉRIOS DE DECISÃO

## Quando PULAR para próxima fase?

### ✅ Critérios para V16 → V17:
- Win rate V16 >= 22% OU
- Return V16 >= +3% OU
- Sharpe V16 >= 0.5

### ✅ Critérios para V17 → V18:
- Win rate V17 >= 25% OU
- Filtros reduziram trades em 20-40% mantendo return

### ✅ Critérios para V18 → V19:
- Win rate V18 >= 28% OU
- Risk/reward ratio >= 1:2.5

### ✅ Critérios para PRODUÇÃO (V19):
- Win rate >= 30%
- Sharpe ratio >= 0.7
- Max drawdown <= 12%
- Paper trading 1 semana sem liquidações
- Backtest em 3+ períodos diferentes (bull, bear, range)

## Quando PARAR e RE-ANALISAR?

### 🛑 Red Flags:
- Win rate CAIR vs versão anterior
- Max drawdown AUMENTAR >20%
- Overfitting (train great, test bad)
- Trade count <100 em 1000 episodes (muito conservador)
- Trade count >1500 em 1000 episodes (overtrading)

### 🔄 Ações Corretivas:
1. Analisar TensorBoard (entropy, learning rate)
2. Verificar distribuição de ações (Long/Short/Flat)
3. Checar se filtros não estão muito restritivos
4. Ajustar pesos de reward shaping
5. Testar com hiperparâmetros diferentes (ent_coef, lr)

---

# 📚 DOCUMENTAÇÃO E ORGANIZAÇÃO

## Estrutura de Arquivos

```
AGENTE_TRANDING/
├── data/
│   ├── train_btcusdt_36m_15m_20260125.csv
│   ├── train_btcusdt_36m_1h_20260125.csv
│   └── train_btcusdt_36m_4h_20260125.csv
│
├── models/
│   ├── v16/
│   │   ├── sac_v16_multi_tf_*_200000_steps.zip
│   │   ├── sac_v16_multi_tf_*_500000_steps.zip
│   │   └── sac_v16_multi_tf_*_1000000_steps.zip
│   ├── v17/
│   ├── v18/
│   └── v19/
│
├── src/
│   └── environment/
│       ├── trading_env.py (V15 original)
│       ├── trading_env_multi_tf.py (V16)
│       ├── trading_env_v17_filtered.py (V17)
│       ├── trading_env_v18_dynamic.py (V18)
│       └── ensemble_trader.py (V19)
│
├── scripts/
│   ├── collect_multi_timeframe.py
│   ├── train_sac_v16.py
│   ├── train_sac_v17.py
│   ├── train_sac_v18.py
│   ├── backtest_ensemble.py
│   ├── analyze_checkpoint.py
│   └── compare_versions.py
│
├── results/
│   ├── v16_backtest_results.csv
│   ├── v17_backtest_results.csv
│   ├── v18_backtest_results.csv
│   ├── v19_backtest_results.csv
│   └── comparison_v15_v19.png
│
└── docs/
    ├── PLANO_ACAO_V16_V19_WINRATE.md (este arquivo)
    ├── V16_RESULTS.md (após treino)
    ├── V17_RESULTS.md
    ├── V18_RESULTS.md
    └── V19_RESULTS.md
```

## Template de Resultado por Versão

```markdown
# RESULTADOS V17 - FILTROS DE QUALIDADE

**Data treino:** DD/MM/AAAA
**Duração:** Xh YYmin
**Checkpoints:** 200k, 500k, 1M

## Configuração
- Base: V16 multi-timeframe
- Novidade: Filtros de volume, volatilidade, trend, time
- Hiperparâmetros: [mesmos V16]

## Métricas Finais (1M steps)
- Win Rate: XX.XX%
- Total Return: +X.XX%
- Sharpe Ratio: X.XX
- Max Drawdown: XX.XX%
- Total Trades: XXX
- Avg Win: +X.XX%
- Avg Loss: -X.XX%
- Risk/Reward: 1:X.XX

## Comparação vs V16
- Win Rate: +X.XX% (vs V16)
- Return: +X.XX%
- Trades: -XX% (filtros funcionaram)

## Análise
- ✅ O que funcionou: [descrever]
- ❌ O que não funcionou: [descrever]
- 💡 Insights: [descrever]

## Decisão
- [ ] APROVADO para produção
- [ ] APROVADO para próxima fase (V18)
- [ ] REQUER ajustes
```

---

# 🎯 TIMELINE ESTIMADO

## Semana 1 (26/01 - 01/02):
- ✅ V16 coleta de dados
- 🔄 V16 treinamento (1M steps)
- ✅ V16 backtest e análise

## Semana 2 (02/02 - 08/02):
- 🔨 Implementar V17 (filtros)
- 🔄 V17 treinamento
- ✅ V17 backtest e comparação

## Semana 3 (09/02 - 15/02):
- 🔨 Implementar V18 (stops dinâmicos)
- 🔄 V18 treinamento
- ✅ V18 backtest e validação

## Semana 4 (16/02 - 22/02):
- 🔨 Implementar V19 (ensemble)
- 🧪 Testes de estratégias de votação
- 🎯 Otimização de pesos
- ✅ Backtest final e comparação completa

## Semana 5-6 (23/02 - 07/03):
- 📝 Paper trading V19
- 🐛 Bug fixes e refinamentos
- 📊 Documentação final
- 🚀 Deploy para produção (se aprovado)

---

# 📞 SUPORTE E TROUBLESHOOTING

## Problemas Comuns

### Problema 1: Win rate não melhora
**Sintomas:** Win rate estagnou ou caiu
**Debug:**
```bash
# Verificar entropy
tensorboard --logdir=tensorboard/

# Analisar distribuição de ações
python analyze_checkpoint.py models/latest.zip

# Verificar overfitting
python compare_train_test.py
```
**Soluções:**
- Aumentar ent_coef (0.05 → 0.08)
- Reduzir penalidades
- Aumentar bônus por winners
- Verificar se filtros não estão muito restritivos

### Problema 2: Overtrading
**Sintomas:** >1500 trades em backtest
**Debug:** Verificar flat time, penalidades
**Soluções:**
- Aumentar penalidade por overtrading
- Filtros mais restritivos
- Confidence threshold mais alto em ensemble

### Problema 3: Muito conservador
**Sintomas:** <100 trades em backtest, >80% flat time
**Debug:** Verificar filtros, penalidades
**Soluções:**
- Relaxar filtros de qualidade
- Reduzir confidence threshold
- Aumentar bônus por operar

### Problema 4: Liquidações frequentes
**Sintomas:** >5 liquidações em backtest
**Debug:** Verificar leverage, stop loss, drawdown
**Soluções:**
- Reduzir leverage (1.5 → 1.2)
- Stop loss mais apertado
- Filtros de volatilidade mais restritivos

---

# ✅ CHECKLIST FINAL PARA PRODUÇÃO

## Pré-Requisitos
- [ ] V19 win rate >= 30%
- [ ] V19 sharpe >= 0.7
- [ ] V19 max drawdown <= 12%
- [ ] Paper trading 1 semana sem problemas
- [ ] Backtest em 3+ períodos diferentes
- [ ] Code review completo
- [ ] Documentação atualizada

## Infraestrutura
- [ ] Servidor com GPU disponível
- [ ] Binance API configurada (testnet → mainnet)
- [ ] Monitoring configurado (Grafana/Prometheus)
- [ ] Alertas configurados (Telegram/Email)
- [ ] Backup automático de modelos
- [ ] Logging centralizado

## Safety Checks
- [ ] Stop loss funcional
- [ ] Liquidation prevention
- [ ] Position size limits
- [ ] Daily loss limit
- [ ] API rate limiting
- [ ] Error handling robusto
- [ ] Restart automático

## Financeiro
- [ ] Capital alocado definido
- [ ] Risk per trade calculado
- [ ] Max daily loss configurado
- [ ] Expectativa de retorno documentada
- [ ] Plano de scaling definido

---

**🎯 META FINAL:** Win Rate 35%+, Sharpe 0.8+, Max DD <10%

**📅 ETA:** 6 semanas (até ~07/03/2026)

**💪 VAMOS NESSA!**
