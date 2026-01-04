# 🤖 Agente de Trading: Ensemble RL + LLM

Sistema avançado de trading automatizado combinando **3 algoritmos de Reinforcement Learning** com **análise de sentimento via LLM** para operar futuros de criptomoedas na Binance.

## ✨ Novidades v2.0

### 🆕 **Ensemble de Modelos RL**
- **PPO** (Proximal Policy Optimization) - Conservador e estável
- **SAC** (Soft Actor-Critic) - Agressivo, ideal para trading contínuo
- **TD3** (Twin Delayed DDPG) - Ações contínuas com menor ruído
- **Votação inteligente** combinando previsões dos 3 modelos

### 🧠 **Análise de Sentimento com LLM**
- Coleta automática de notícias (NewsAPI + RSS feeds)
- Análise de sentimento via **GPT-4/GPT-3.5**, **Claude** ou **FinBERT** (local)
- Features temporais (1h, 6h, 24h) com decay
- Detecta tendências e volatilidade de sentimento

### 📊 **Sistema Completo**
- **9+ features de sentimento** integradas ao agente
- **32+ features totais** (mercado + sentimento + portfolio)
- **Gestão de risco** avançada com Kelly Criterion
- **Paper e Live trading** com monitoramento 24/7

---

## 🚀 Quick Start

### 1️⃣ **Instalação**

```powershell
# Clone e navegue
git clone <seu-repositorio>
cd AGENTE_TRANDING

# Ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instale dependências
pip install -r requirements.txt
```

### 2️⃣ **Configuração**

```powershell
# Copie template de variáveis
cp .env.example .env

# Edite .env com suas API keys:
# - BINANCE_API_KEY (obrigatório)
# - OPENAI_API_KEY (recomendado para LLM)
# - NEWSAPI_KEY (opcional, 500 requests/dia grátis)
```

### 3️⃣ **Teste o Sistema**

```powershell
# Verifica se tudo está OK
python test_system.py

# Deve mostrar:
# ✅ Imports OK
# ✅ config.yaml OK
# ✅ Environment OK
# ✅ Notícias coletadas
```

### 4️⃣ **Treinamento**

```powershell
# Coleta dados históricos
python -m src.data.data_collector

# Treina ensemble (PPO + SAC + TD3)
python -m src.training.ensemble_trainer

# Aguarde ~30-60min
# Modelos salvos em: models/ensemble/
```

### 5️⃣ **Trading**

```powershell
# Paper trading (SEM RISCO)
python -m src.execution.ensemble_executor

# Output:
# 📰 Sentimento: 0.654 (23 notícias)
# 🤖 Votos: {'ppo': 1, 'sac': 1, 'td3': 0}
#    Ação Final: 1 (Long)
#    Concordância: 66.7%
# 💵 Preço: $94,523.45
# ✅ Ação executada
```

**⚠️ IMPORTANTE:** Use primeiro a testnet da Binance para testes!

---

## 📚 Documentação Completa

- **[GUIA_ENSEMBLE_LLM.md](GUIA_ENSEMBLE_LLM.md)** - Guia completo de uso (RECOMENDADO)
- **[ARQUITETURA_TECNICA.md](ARQUITETURA_TECNICA.md)** - Detalhes técnicos da arquitetura
- **[COMPARACAO_FREQTRADE.md](COMPARACAO_FREQTRADE.md)** - Comparação com Freqtrade
- **[INTEGRACAO_FREQTRADE.md](INTEGRACAO_FREQTRADE.md)** - Como integrar com Freqtrade

---

## 🎯 Features Principais

### **1. Ensemble de RL** 🤖🤖🤖
```python
# 3 modelos votam em cada decisão
PPO: Long (confiança: 70%)
SAC: Long (confiança: 90%)  
TD3: Flat (confiança: 60%)
→ Resultado: Long (consenso ponderado)
```

**Estratégias de votação:**
- `majority` - Votação simples
- `weighted` - Ponderado por performance (padrão)
- `confidence` - Ponderado por certeza
- `best` - Usa apenas melhor modelo
- `average` - Média das previsões

### **2. Análise de Sentimento** 🧠
```python
# Pipeline completo
Notícias → LLM (GPT/Claude/FinBERT) → Features
↓
sentiment_1h: 0.8 (bullish)
sentiment_6h: 0.6 (bullish)
sentiment_24h: 0.3 (neutral)
trend: +0.5 (melhorando)
volatility: 0.2 (baixa)
```

**Fontes:**
- NewsAPI (500 requests/dia grátis)
- RSS Feeds (CoinTelegraph, CoinDesk, etc)
- Atualizações a cada 1h

**Modelos LLM:**
- OpenAI GPT-3.5/GPT-4 (pago, melhor qualidade)
- Anthropic Claude (alternativa)
- FinBERT local (grátis, offline)

### **3. Gestão de Risco** 🛡️
```python
✅ Kelly Criterion para position sizing
✅ Stop Loss automático (2%)
✅ Take Profit automático (4%)
✅ Max Drawdown protection (15%)
✅ Validação antes de cada trade
```

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────┐
│          Coleta de Dados                    │
├─────────────────────────────────────────────┤
│ Binance API → OHLCV + Indicadores           │
│ NewsAPI/RSS → Notícias → LLM → Sentimento  │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│          Trading Environment                 │
├─────────────────────────────────────────────┤
│ Observation: [Market + Sentiment + Portfolio]│
│ Actions: [Flat, Long, Short]                │
│ Reward: PnL - Costs - Penalties            │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│          Ensemble RL                        │
├─────────────────────────────────────────────┤
│ PPO Model → Vote 1                          │
│ SAC Model → Vote 2  → Combiner → Action    │
│ TD3 Model → Vote 3                          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│          Risk Management                     │
├─────────────────────────────────────────────┤
│ Validate Stop Loss / Take Profit            │
│ Check Max Drawdown                          │
│ Calculate Position Size (Kelly)            │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│          Execution                          │
├─────────────────────────────────────────────┤
│ Paper Trading / Live Trading (Binance)     │
│ Logging & Metrics                          │
└─────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```
AGENTE_TRANDING/
├── src/
│   ├── data/
│   │   └── data_collector.py         # Coleta OHLCV + indicadores
│   ├── environment/
│   │   └── trading_env.py            # Gymnasium environment
│   ├── sentiment/                     # 🆕 Análise de sentimento
│   │   ├── news_collector.py         # Coleta notícias
│   │   ├── llm_analyzer.py           # GPT/Claude/FinBERT
│   │   └── sentiment_processor.py    # Features numéricas
│   ├── models/                        # 🆕 Ensemble
│   │   └── ensemble_model.py         # Votação de modelos
│   ├── training/
│   │   ├── train.py                  # Treina modelo único
│   │   └── ensemble_trainer.py       # 🆕 Treina PPO+SAC+TD3
│   ├── risk/
│   │   └── risk_manager.py           # Kelly, SL, TP
│   └── execution/
│       ├── executor.py               # Executor simples
│       └── ensemble_executor.py      # 🆕 Executor completo
├── data/                              # Dados históricos
├── models/                            # Modelos treinados
│   └── ensemble/                     # 🆕 PPO, SAC, TD3
├── logs/                              # Logs e métricas
├── config.yaml                        # Configuração principal
├── .env.example                       # Template de credenciais
├── requirements.txt                   # Dependências
├── test_system.py                     # 🆕 Teste completo
├── README.md                          # Este arquivo
├── GUIA_ENSEMBLE_LLM.md              # 🆕 Guia de uso
├── ARQUITETURA_TECNICA.md            # 🆕 Docs técnicas
└── COMPARACAO_FREQTRADE.md           # Comparação

🆕 = Novos arquivos v2.0
```

---

## 🚀 Uso

### Fase 1: Coletar Dados

```bash
python -m src.data.data_collector
```

Isso irá:
- Baixar dados OHLCV da Binance
- Calcular indicadores técnicos
- Normalizar os dados
- Dividir em treino/validação/teste

### Fase 2: Treinar o Agente

```bash
python -m src.training.train --mode train
```

O treinamento irá:
- Criar um ambiente de simulação
- Treinar o agente PPO por 100.000 timesteps (configurável)
- Salvar o melhor modelo em `models/`
- Gerar logs em `logs/`

Para visualizar o treinamento no TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

### Fase 3: Avaliar o Modelo

```bash
python -m src.training.train --mode eval --model models/ppo_trading_agent_XXXXXXXX.zip
```

### Fase 4: Executar em Paper Trading

```bash
python -m src.execution.executor --model models/ppo_trading_agent_XXXXXXXX.zip --mode paper
```

### Fase 5: Executar em Live Trading (⚠️ USE COM CAUTELA)

```bash
python -m src.execution.executor --model models/ppo_trading_agent_XXXXXXXX.zip --mode live
```

## 📁 Estrutura do Projeto

```
AGENTE_TRANDING/
├── config.yaml              # Configurações principais
├── requirements.txt         # Dependências Python
├── .env.example            # Template de variáveis de ambiente
├── README.md               # Este arquivo
│
├── src/
│   ├── __init__.py
│   │
│   ├── environment/        # Ambiente Gymnasium
│   │   ├── __init__.py
│   │   └── trading_env.py
│   │
│   ├── data/              # Coleta e processamento de dados
│   │   ├── __init__.py
│   │   └── data_collector.py
│   │
│   ├── risk/              # Gestão de risco
│   │   ├── __init__.py
│   │   └── risk_manager.py
│   │
│   ├── training/          # Treinamento do agente
│   │   ├── __init__.py
│   │   └── train.py
│   │
│   └── execution/         # Execução ao vivo
│       ├── __init__.py
│       └── executor.py
│
├── data/                  # Dados processados (gerado)
├── models/                # Modelos treinados (gerado)
└── logs/                  # Logs de treinamento e trading (gerado)
```

## ⚙️ Configuração

Edite `config.yaml` para ajustar:

- **Símbolo e timeframe** do mercado
- **Indicadores técnicos** a usar
- **Hiperparâmetros do RL** (learning rate, batch size, etc.)
- **Parâmetros de risco** (stop loss, take profit, alavancagem)
- **Tamanho de posição** e capital inicial

## 🧠 Como Funciona

### 1. Ambiente de RL (TradingEnv)

O ambiente simula um mercado de trading onde o agente:
- **Observa:** Preços, indicadores técnicos e estado da carteira
- **Age:** Fica Flat, abre Long ou abre Short
- **Recebe recompensa:** Baseado no PnL e custos de transação

### 2. Função de Recompensa

$$R_t = (\text{Balance}_t - \text{Balance}_{t-1}) - (\text{Trade Cost} \times \text{Action Changed})$$

A recompensa incentiva o agente a:
- Maximizar lucros
- Minimizar custos de transação
- Evitar overtrading

### 3. Gestão de Risco

O Risk Manager aplica regras hardcoded:
- **Kelly Criterion** para tamanho de posição
- **Stop Loss automático** (2%)
- **Take Profit automático** (4%)
- **Controle de Drawdown** (15% máximo)
- **Limite de alavancagem** (3x)

## 📊 Métricas de Avaliação

O sistema rastreia:
- Win Rate (taxa de vitória)
- Total de trades
- PnL (Profit and Loss)
- Drawdown
- Sharpe Ratio (planejado)

## ⚠️ Avisos Importantes

1. **NÃO USE EM PRODUÇÃO SEM TESTES EXTENSIVOS**
2. Comece sempre com a **testnet da Binance**
3. Use **Paper Trading** antes de arriscar capital real
4. O passado **não garante** retornos futuros
5. Trading automatizado envolve **riscos significativos**
6. Nunca invista mais do que pode perder

## 🔧 Troubleshooting

### Erro: "Module not found"
```bash
# Certifique-se de estar no diretório raiz
cd AGENTE_TRANDING
python -m src.data.data_collector
```

### Erro: "API Key inválida"
- Verifique se o `.env` está configurado corretamente
- Certifique-se de usar chaves da testnet primeiro

### Modelo não converge
- Aumente `total_timesteps` no `config.yaml`
- Ajuste `learning_rate` (tente 0.0001 ou 0.0005)
- Verifique se os dados estão normalizados

## 📚 Referências

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)

## 📝 Licença

Este projeto é fornecido "como está" para fins educacionais.

## 🤝 Contribuições

Contribuições são bem-vindas! Abra uma issue ou pull request.

---

**⚠️ DISCLAIMER:** Este software é fornecido para fins educacionais. O uso em produção é por sua conta e risco. Os desenvolvedores não se responsabilizam por perdas financeiras.
