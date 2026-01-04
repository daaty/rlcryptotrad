# 🔄 Guia Prático: Usando os 2 Sistemas em Conjunto

## 🎯 Estratégia: "Lab + Produção"

A ideia é usar **cada sistema onde ele é mais forte**:

```
SEU SISTEMA (Lab)          →          FREQTRADE (Produção)
   Pesquisa & Treino       →          Trading Real
   Experimentação          →          Execução Robusta
   Inovação                →          Estabilidade
```

---

## 🚀 **Workflow Recomendado**

### **Fase 1: Pesquisa no SEU Sistema (2-4 semanas)**

```bash
# 1. Coleta dados
python -m src.data.data_collector

# 2. Treina modelo
python -m src.training.train --epochs 100000

# 3. Testa performance
python -m src.training.train --mode eval --model models/ppo_v1.zip

# 4. Paper trading local
python -m src.execution.executor --model models/ppo_v1.zip --mode paper
```

**Vantagens:**
- ✅ Você ENTENDE o que está acontecendo
- ✅ Pode modificar TUDO rapidamente
- ✅ Testa ideias malucas sem medo
- ✅ Código limpo e fácil de debugar

**Quando passar para Freqtrade:**
- ✅ Modelo tem win rate > 55%
- ✅ Sharpe ratio > 1.5
- ✅ Passou 1 mês de paper trading
- ✅ Drawdown < 10%

---

### **Fase 2: Port para Freqtrade (1 semana)**

Freqtrade tem um módulo chamado **FreqAI** que permite usar modelos personalizados.

#### **Opção A: Usar seu modelo diretamente (RECOMENDADO)**

Crie uma estratégia que carrega seu modelo PPO:

```python
# user_data/strategies/RLStrategy.py
from freqtrade.strategy import IStrategy
from stable_baselines3 import PPO
import numpy as np
from pandas import DataFrame

class RLStrategy(IStrategy):
    """
    Estratégia que usa SEU modelo RL treinado.
    """
    
    # Configurações básicas
    timeframe = '15m'
    stoploss = -0.10
    can_short = True
    
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Carrega SEU modelo treinado!
        self.model = PPO.load("models/ppo_trading_agent_best.zip")
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Adiciona os mesmos indicadores que você treinou"""
        import pandas_ta as ta
        
        # RSI
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        
        # SMAs
        dataframe['sma_20'] = ta.sma(dataframe['close'], length=20)
        dataframe['sma_50'] = ta.sma(dataframe['close'], length=50)
        
        # Bollinger Bands
        bb = ta.bbands(dataframe['close'], length=20, std=2)
        dataframe['bb_lower'] = bb['BBL_20_2.0']
        dataframe['bb_middle'] = bb['BBM_20_2.0']
        dataframe['bb_upper'] = bb['BBU_20_2.0']
        
        # MACD
        macd = ta.macd(dataframe['close'], fast=12, slow=26, signal=9)
        dataframe['macd'] = macd['MACD_12_26_9']
        dataframe['macd_signal'] = macd['MACDs_12_26_9']
        dataframe['macd_hist'] = macd['MACDh_12_26_9']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Usa SEU modelo para decidir entrada"""
        
        # Prepara observação para o modelo (últimas 50 candles)
        if len(dataframe) < 50:
            return dataframe
            
        # Normaliza features (como você fez no treino)
        features = self._prepare_features(dataframe)
        
        # Pega última observação
        obs = features[-50:].values
        
        # Estado da carteira (simplificado aqui)
        portfolio_state = np.array([1.0, 0, 1.0])  # [saldo_norm, posição, equity_norm]
        portfolio_matrix = np.tile(portfolio_state, (50, 1))
        
        observation = np.concatenate([obs, portfolio_matrix], axis=1)
        observation = observation.astype(np.float32)
        
        # PREVISÃO DO SEU MODELO!
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Traduz ação para sinal Freqtrade
        # 0 = Flat, 1 = Long, 2 = Short
        if action == 1:  # Long
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
        elif action == 2:  # Short
            dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
            
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Modelo também decide saída"""
        # Similar à entrada, mas verifica se deve sair
        # Se modelo prevê Flat (0), sinaliza saída
        
        if len(dataframe) < 50:
            return dataframe
            
        features = self._prepare_features(dataframe)
        obs = features[-50:].values
        portfolio_state = np.array([1.0, 0, 1.0])
        portfolio_matrix = np.tile(portfolio_state, (50, 1))
        observation = np.concatenate([obs, portfolio_matrix], axis=1)
        
        action, _states = self.model.predict(observation, deterministic=True)
        
        if action == 0:  # Flat = sair
            dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
            dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
            
        return dataframe
    
    def _prepare_features(self, dataframe: DataFrame) -> DataFrame:
        """Normaliza features como no treino"""
        # Log returns para preços
        df_norm = dataframe.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_norm[f'{col}_return'] = np.log(df_norm[col] / df_norm[col].shift(1))
        
        df_norm.drop(columns=price_cols, inplace=True)
        df_norm['volume'] = np.log1p(df_norm['volume'])
        
        # Min-max normalização para indicadores
        indicator_cols = [col for col in df_norm.columns 
                         if col not in price_cols + ['volume', 'date']]
        
        for col in indicator_cols:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val - min_val != 0:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        df_norm.fillna(0, inplace=True)
        return df_norm
```

---

#### **Opção B: Usar FreqAI com RL (Mais Complexo)**

```python
# user_data/freqaimodels/MyRLModel.py
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from stable_baselines3 import PPO

class MyRLModel(ReinforcementLearner):
    """
    Modelo customizado baseado no seu sistema
    """
    
    def fit(self, data_dictionary, dk):
        """Usa SEU modelo já treinado ou retreina"""
        
        # Carrega seu modelo existente
        try:
            model = PPO.load("../models/ppo_trading_agent_best.zip")
            print("✅ Modelo pré-treinado carregado!")
        except:
            # Ou treina do zero com dados do Freqtrade
            model = super().fit(data_dictionary, dk)
            
        return model
```

---

### **Fase 3: Backtesting no Freqtrade (2-3 dias)**

```bash
# Backtest completo com seu modelo
freqtrade backtesting \
    --strategy RLStrategy \
    --timeframe 15m \
    --timerange 20240101-20241231 \
    --enable-protections \
    --export trades

# Analisa resultados
freqtrade backtesting-analysis \
    --strategy RLStrategy \
    --analysis-groups 0 1 2
```

**Freqtrade vai adicionar automaticamente:**
- ✅ Slippage realista
- ✅ Custos de corretagem
- ✅ Proteções (StoplossGuard, etc)
- ✅ Métricas avançadas
- ✅ Gráficos interativos

---

### **Fase 4: Dry-run no Freqtrade (1 mês)**

```bash
# Cria config específico
cat > config_rl.json << EOF
{
    "strategy": "RLStrategy",
    "dry_run": true,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "tradable_balance_ratio": 0.99,
    "max_open_trades": 3,
    
    "exchange": {
        "name": "binance",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET"
    },
    
    "telegram": {
        "enabled": true,
        "token": "YOUR_TELEGRAM_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
}
EOF

# Roda em dry-run (sem gastar dinheiro)
freqtrade trade --config config_rl.json --strategy RLStrategy
```

**Agora você tem:**
- ✅ Telegram bot pra monitorar
- ✅ WebUI para ver gráficos
- ✅ Database com todas as decisões
- ✅ Logs estruturados
- ✅ Proteções de mercado

---

### **Fase 5: Live Trading (Depois de 1 mês dry-run)**

```bash
# Muda pra live (CUIDADO!)
# Edita config_rl.json:
{
    "dry_run": false,  # <-- Muda aqui
    "stake_amount": 50,  # <-- Começa pequeno!
    ...
}

# Inicia live trading
freqtrade trade --config config_rl.json --strategy RLStrategy

# Monitor pelo Telegram
/status - Ver trades abertos
/profit - Ver lucro total
/balance - Saldo
```

---

## 🎯 **Ciclo de Melhoria Contínua**

```
┌─────────────────────────────────────────────┐
│  SEU SISTEMA (Desenvolvimento)              │
│  - Testa nova função de recompensa          │
│  - Testa novos indicadores                  │
│  - Testa diferentes algoritmos (A2C, etc)   │
│  - Backtesting rápido                       │
└────────────────┬────────────────────────────┘
                 │
                 ↓ (Se melhorar)
┌─────────────────────────────────────────────┐
│  FREQTRADE (Validação)                      │
│  - Backtest com slippage realista           │
│  - Paper trading 1 mês                      │
│  - Análise de drawdown                      │
└────────────────┬────────────────────────────┘
                 │
                 ↓ (Se validar)
┌─────────────────────────────────────────────┐
│  FREQTRADE (Produção)                       │
│  - Live trading pequeno                     │
│  - Monitoramento 24/7                       │
│  - Coleta métricas reais                    │
└────────────────┬────────────────────────────┘
                 │
                 ↓ (Feedback)
                Volta para SEU SISTEMA
                (Usar dados reais pra retreinar)
```

---

## 💰 **Benefícios Concretos**

### **1. Segurança em Camadas**

```
SEU Sistema:
├─ Kelly Criterion position sizing
├─ Stop Loss validação
└─ Take Profit fixo

    ↓ Integra com ↓

Freqtrade adiciona:
├─ Trailing Stop Loss
├─ Custom ROI table
├─ Stoploss Guard (para de operar após X perdas)
├─ Max Drawdown protection
├─ CoolDown period (espera após perda)
└─ Emergency stop (Telegram /forceexit)
```

**Resultado:** Sistema MUITO mais robusto!

---

### **2. Experimentação Rápida vs Produção Estável**

| Cenário | Seu Sistema | Freqtrade |
|---------|-------------|-----------|
| Nova ideia de reward | ✅ Testa em 1 dia | ❌ Muito complexo |
| Novo algoritmo (SAC, TD3) | ✅ Fácil trocar | ❌ Limitado |
| Add novo indicador | ✅ 5 minutos | ⚠️ Backtest longo |
| Deploy produção | ⚠️ Básico | ✅ Battle-tested |
| Monitoramento 24/7 | ❌ Só logs | ✅ Telegram + WebUI |
| Backtesting realista | ⚠️ Básico | ✅ Slippage, fees |

---

### **3. Workflow de Desenvolvimento**

**Segunda-feira (SEU sistema):**
```python
# Ideia: E se recompensar trades curtos?
def calculate_reward(self, action):
    pnl = ...
    duration_penalty = -0.001 * self.trade_duration
    return pnl + duration_penalty

# Treina
python train.py --epochs 50000

# Testa
python eval.py
# Resultado: +5% melhoria! 🎉
```

**Terça-feira (Freqtrade):**
```bash
# Porta pro Freqtrade
# Atualiza RLStrategy.py com novo modelo

# Backtest
freqtrade backtesting --strategy RLStrategy

# Resultado: Confirma melhoria! ✅
```

**Quarta-feira (Deploy):**
```bash
# Coloca em dry-run
freqtrade trade --config config_rl.json

# Monitor pelo celular via Telegram
# Tudo funcionando! 🚀
```

---

### **4. Exemplo Real de Integração**

#### **Arquivo de Configuração Compartilhado**

```yaml
# shared_config.yaml (usado pelos 2 sistemas)
data:
  symbol: "BTC/USDT"
  timeframe: "15m"
  
indicators:
  - name: "rsi"
    length: 14
  - name: "sma"
    length: 20
  - name: "sma"
    length: 50
    
risk:
  stop_loss_pct: 0.02
  take_profit_pct: 0.04
  max_leverage: 3
```

#### **Sincronização de Dados**

```python
# sync_data.py
"""
Sincroniza dados entre os dois sistemas
"""

def sync_binance_to_both():
    """Baixa dados e salva nos 2 formatos"""
    
    # 1. Baixa via SEU sistema
    from src.data.data_collector import DataCollector
    collector = DataCollector()
    collector.run()
    
    # 2. Converte para formato Freqtrade
    import pandas as pd
    df = pd.read_csv('data/market_data_raw.csv')
    
    # Salva no formato Freqtrade
    df_ft = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df_ft.to_json(
        'freqtrade/user_data/data/binance/BTC_USDT-15m.json',
        orient='values'
    )
    
    print("✅ Dados sincronizados!")

if __name__ == "__main__":
    sync_binance_to_both()
```

---

### **5. Dashboard Unificado**

```python
# dashboard/unified_dashboard.py
"""
Dashboard que mostra métricas dos 2 sistemas
"""

import streamlit as st
import pandas as pd
import sqlite3

st.title("🤖 Trading RL - Dashboard Unificado")

col1, col2 = st.columns(2)

with col1:
    st.header("SEU Sistema (Lab)")
    # Lê logs do seu sistema
    df_lab = pd.read_csv('logs/trading/trading_latest.log', parse_dates=True)
    st.metric("Win Rate Lab", f"{df_lab['wins'].sum() / df_lab['trades'].sum():.2%}")
    st.metric("PnL Lab", f"${df_lab['pnl'].sum():.2f}")
    st.line_chart(df_lab['equity'])

with col2:
    st.header("Freqtrade (Produção)")
    # Lê database do Freqtrade
    conn = sqlite3.connect('freqtrade/user_data/tradesv3.sqlite')
    df_prod = pd.read_sql('SELECT * FROM trades', conn)
    
    win_rate = len(df_prod[df_prod['profit_ratio'] > 0]) / len(df_prod)
    st.metric("Win Rate Prod", f"{win_rate:.2%}")
    st.metric("PnL Prod", f"${df_prod['profit_abs'].sum():.2f}")
    st.line_chart(df_prod['profit_ratio'].cumsum())

# Comparação lado a lado
st.header("📊 Comparação")
comparison = pd.DataFrame({
    'Métrica': ['Win Rate', 'Total Trades', 'Avg Profit'],
    'Lab': [
        f"{df_lab['wins'].sum() / df_lab['trades'].sum():.2%}",
        df_lab['trades'].sum(),
        f"${df_lab['pnl'].mean():.2f}"
    ],
    'Produção': [
        f"{win_rate:.2%}",
        len(df_prod),
        f"${df_prod['profit_abs'].mean():.2f}"
    ]
})
st.table(comparison)
```

---

## 🎓 **Casos de Uso Específicos**

### **Caso 1: Pesquisador / Estudante**
```
✅ Use SEU sistema 90% do tempo
   - Aprenda RL profundamente
   - Publique papers
   - Experimente ideias malucas
   
⚠️ Use Freqtrade 10%
   - Valide resultados
   - Backtest realista
```

### **Caso 2: Trader Profissional**
```
⚠️ Use SEU sistema 30%
   - Desenvolva estratégias
   - Teste modelos novos
   
✅ Use Freqtrade 70%
   - Trading real
   - Monitoramento
   - Proteções
```

### **Caso 3: Empresa / Hedge Fund**
```
Team 1: Pesquisa (SEU sistema)
   - Desenvolve modelos
   - Testa alpha
   
Team 2: Produção (Freqtrade)
   - Deploy modelos validados
   - Risk management
   - Compliance
```

---

## 📋 **Checklist de Integração**

### **Semana 1: Setup**
- [ ] Instalar Freqtrade
- [ ] Configurar .env com API keys
- [ ] Criar estratégia RLStrategy.py
- [ ] Testar carregamento do modelo

### **Semana 2: Backtesting**
- [ ] Backtest 6 meses
- [ ] Comparar com backtest do seu sistema
- [ ] Ajustar stop loss / take profit
- [ ] Validar win rate > 55%

### **Semana 3-6: Paper Trading**
- [ ] Dry-run 1 mês
- [ ] Monitorar diariamente
- [ ] Coletar métricas
- [ ] Verificar drawdown < 10%

### **Semana 7+: Live (Opcional)**
- [ ] Começar com $50-100
- [ ] Aumentar gradualmente
- [ ] Monitorar 24/7 via Telegram
- [ ] Retreinar modelo mensalmente

---

## 🚨 **Erros Comuns a Evitar**

### ❌ **NÃO FAÇA:**
1. Pular direto para live trading
2. Usar modelo diferente no Freqtrade
3. Ignorar o backtesting do Freqtrade
4. Confiar só no seu backtest
5. Não monitorar dry-run

### ✅ **FAÇA:**
1. Validação em múltiplas etapas
2. Usar EXATAMENTE o mesmo modelo
3. Fazer backtest nos 2 sistemas
4. Paper trading por 1 mês mínimo
5. Começar com capital pequeno

---

## 💡 **Próximos Passos Práticos**

Quer que eu crie:

1. **Script de integração automática** → Converte seu modelo para estratégia Freqtrade
2. **Dashboard unificado** → Mostra métricas dos 2 sistemas lado a lado
3. **Pipeline de CI/CD** → Testa modelo no seu sistema → Valida no Freqtrade → Deploy
4. **Sistema de retreino automático** → Treina no seu sistema → Valida → Deploy

Qual te interessa mais? 🚀
