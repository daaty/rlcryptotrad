# 🚀 Guia Rápido: Sistema Ensemble + LLM

## 🎯 O que foi adicionado

Seu sistema agora tem **SUPERPODERES**:

### 1. **Ensemble de 3 Modelos RL** 🤖🤖🤖
- **PPO** (Conservador, estável)
- **SAC** (Agressivo, melhor para trading contínuo)  
- **TD3** (Ações contínuas, menos ruído)

**Como funciona:**
- Cada modelo dá seu voto
- Sistema combina usando votação ponderada
- Decisão final é mais robusta que modelo único

### 2. **Análise de Sentimento com LLM** 🧠
- Coleta notícias de criptomoedas (NewsAPI + RSS)
- Analisa sentimento com GPT/Claude/FinBERT
- Transforma em features numéricas para o agente
- Atualiza automaticamente a cada 1h

---

## 📋 Setup Completo

### **1. Instalar Dependências**

```powershell
# Ative seu ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instale as novas dependências
pip install -r requirements.txt
```

### **2. Configurar API Keys (.env)**

Copie `.env.example` para `.env` e preencha:

```env
# Binance
BINANCE_API_KEY=sua_chave_aqui
BINANCE_SECRET_KEY=sua_secret_aqui

# LLM (escolha um)
OPENAI_API_KEY=sk-...  # Para GPT-3.5/GPT-4
# ou
ANTHROPIC_API_KEY=sk-ant-...  # Para Claude

# News API (opcional, mas recomendado)
NEWSAPI_KEY=sua_key_aqui  # Grátis em https://newsapi.org
```

**Como conseguir as keys:**
- **NewsAPI**: https://newsapi.org (500 requisições/dia grátis)
- **OpenAI**: https://platform.openai.com/api-keys (pague por uso)
- **Anthropic**: https://console.anthropic.com (Claude)

### **3. Configurar config.yaml**

Já está configurado! Mas você pode ajustar:

```yaml
# Habilitar/desabilitar sentimento
sentiment:
  enabled: true  # false para desabilitar

# Escolher provider LLM
llm:
  provider: "openai"  # openai, anthropic, ou local (FinBERT grátis)
  model: "gpt-3.5-turbo"  # ou gpt-4, claude-3-opus

# Estratégia do ensemble
ensemble:
  strategy: "weighted"  # majority, weighted, confidence, best, average
  weights:
    ppo: 0.3
    sac: 0.4  # SAC com maior peso (melhor para trading)
    td3: 0.3
```

---

## 🏋️ Treinamento

### **Opção A: Treinar Ensemble Completo (RECOMENDADO)**

```powershell
# 1. Coleta dados
python -m src.data.data_collector

# 2. Treina os 3 modelos
python -m src.training.ensemble_trainer

# Aguarde ~30-60min (treina PPO, SAC e TD3 sequencialmente)
```

### **Opção B: Treinar Modelo Único**

```powershell
# Treina apenas PPO (mais rápido para testar)
python -m src.training.train
```

---

## 🧪 Testando

### **1. Testar Coleta de Notícias**

```powershell
python -m src.sentiment.news_collector

# Deve mostrar:
# ✅ NewsAPI configurado
# 📰 Coletando notícias das últimas 24h...
# ✅ NewsAPI: 15 notícias
# ✅ RSS Feeds: 32 notícias
# 📊 Total: 47 notícias únicas
```

### **2. Testar Análise de Sentimento**

```powershell
python -m src.sentiment.llm_analyzer

# Teste com notícia de exemplo
# 🧠 Análise de Sentimento:
# Score: 0.8
# Label: bullish
# Confidence: 85%
```

### **3. Testar Ensemble**

```powershell
python -m src.models.ensemble_model

# 🎯 Previsão Ensemble:
# Votos: {'ppo': 1, 'sac': 1, 'td3': 0}
# Ação Final: 1 (Long)
# Concordância: 66%
```

---

## 🚀 Executando

### **Paper Trading (Recomendado para início)**

```powershell
python -m src.execution.ensemble_executor

# Log:
# ✅ EnsembleExecutor inicializado
#    Modo: paper
#    Sentimento: True
#    Ensemble: True
# 🚀 Iniciando trading...
# 📰 Sentimento: 0.654 (23 notícias)
# 🤖 Votos: {'ppo': 1, 'sac': 1, 'td3': 1}
#    Ação Final: 1 (Long)
#    Concordância: 100%
# 💵 Preço: $94,523.45
# ✅ Ação executada
```

### **Live Trading (Depois de validar)**

```yaml
# Mude no config.yaml:
execution:
  mode: "live"  # ⚠️ CUIDADO!

binance:
  testnet: false  # ⚠️ Dinheiro real!
```

```powershell
python -m src.execution.ensemble_executor
```

---

## 🎛️ Modos de Operação

### **1. Ensemble + Sentimento (FULL POWER)**
```yaml
execution:
  use_ensemble: true

sentiment:
  enabled: true
```

### **2. Apenas Ensemble (Sem notícias)**
```yaml
execution:
  use_ensemble: true

sentiment:
  enabled: false
```

### **3. Modelo Único + Sentimento**
```yaml
execution:
  use_ensemble: false

sentiment:
  enabled: true
```

### **4. Modelo Único Tradicional**
```yaml
execution:
  use_ensemble: false

sentiment:
  enabled: false
```

---

## 📊 Monitoramento

### **Durante Execução**

```
Iteração 42 - 2026-01-04 15:30:00
📰 Sentimento: 0.654 (23 notícias)
🤖 Votos: {'ppo': 1, 'sac': 1, 'td3': 0}
   Ação Final: 1 (Long)
   Concordância: 66.7%
💵 Preço: $94,523.45
📈 PAPER LONG: 0.031820 @ $94523.45
✅ Ação executada
💰 Balance: $10,234.56
📊 Posição: Long
💹 PnL Aberto: 1.23%
```

### **Após Sessão**

```
📊 RESUMO DA SESSÃO
Balance Final: $10,567.89
PnL Total: $567.89
Total Trades: 24
Wins: 15 | Losses: 9
Win Rate: 62.5%
💾 Trades salvos: logs/trades_20260104_160000.csv
```

---

## 🔬 Experimentos

### **Teste 1: Qual modelo é melhor?**

```powershell
# Avalia cada modelo individualmente
python -m src.training.ensemble_trainer --mode evaluate

# Resultado:
# PPO: Reward=1234.56 ± 123.45
# SAC: Reward=1456.78 ± 89.01  # 🏆 Melhor!
# TD3: Reward=1345.67 ± 101.23
```

### **Teste 2: Sentimento ajuda?**

```powershell
# Treina SEM sentimento
python -m src.training.ensemble_trainer --no-sentiment

# Treina COM sentimento
python -m src.training.ensemble_trainer --with-sentiment

# Compare resultados!
```

### **Teste 3: Qual estratégia de votação?**

Teste todas no `config.yaml`:
- `majority` - Simples, rápido
- `weighted` - Ponderado por performance
- `confidence` - Ponderado por certeza
- `best` - Usa apenas o melhor
- `average` - Média das ações

---

## 🐛 Troubleshooting

### **Erro: "OPENAI_API_KEY não encontrada"**
```powershell
# Verifique .env
cat .env

# Deve ter:
OPENAI_API_KEY=sk-...
```

### **Erro: "Nenhum modelo encontrado"**
```powershell
# Treine primeiro!
python -m src.training.ensemble_trainer
```

### **Sentimento sempre 0.0**
```powershell
# Teste manualmente
python -m src.sentiment.news_collector

# Se NewsAPI falhar, usa RSS (sempre funciona)
```

### **Modelos demorando muito**
```yaml
# Reduza timesteps para testar
training:
  total_timesteps: 10000  # Ao invés de 100000
```

---

## 📈 Próximos Passos

1. **Teste paper trading por 1 semana**
2. **Ajuste pesos do ensemble** baseado em performance
3. **Experimente diferentes providers LLM** (GPT-4 vs Claude vs FinBERT)
4. **Add mais fontes de notícias** (Twitter, Reddit, etc)
5. **Backtest com dados históricos**
6. **Deploy em servidor 24/7**

---

## 🎓 Entendendo o Sistema

### **Fluxo Completo:**

```
1. Coleta Notícias (NewsAPI + RSS)
          ↓
2. Analisa com LLM (GPT/Claude/FinBERT)
          ↓
3. Extrai Features (sentimento 1h/6h/24h, trend, volatility)
          ↓
4. Coleta Dados Mercado (OHLCV + indicadores)
          ↓
5. Combina tudo → Observação
          ↓
6. PPO + SAC + TD3 fazem previsões
          ↓
7. Ensemble combina → Ação Final
          ↓
8. Risk Manager valida
          ↓
9. Executa Trade
```

### **Por que é melhor?**

| Sistema Tradicional | Nosso Sistema Ensemble + LLM |
|---------------------|------------------------------|
| 1 modelo (PPO) | 3 modelos votam (PPO+SAC+TD3) |
| Só indicadores técnicos | Indicadores + Sentimento de notícias |
| Decisão única | Consenso robusto |
| Ignora notícias | Incorpora contexto de mercado |
| ~55% win rate | **~65%+ win rate** (meta) |

---

## 💡 Dicas Profissionais

1. **Comece com FinBERT (local)** - Grátis e funciona bem
2. **Depois teste GPT-3.5** - Melhor análise, $0.001/requisição
3. **GPT-4 só se tiver budget** - Mais caro mas mais preciso
4. **Monitore logs de sentimento** - Veja se faz sentido
5. **Ajuste pesos do ensemble dinamicamente** - Dá mais peso para quem acerta
6. **Não confie 100% no papel trading** - Slippage real é maior

---

## 🔐 Segurança

- ✅ **SEMPRE teste em testnet primeiro**
- ✅ **Comece com capital pequeno no live**
- ✅ **Monitore 24/7 ou use stop loss**
- ✅ **Nunca compartilhe suas API keys**
- ✅ **Use .env (já está no .gitignore)**

---

## 📞 Ajuda

Algum problema? Verifique:
- [ ] `.env` configurado
- [ ] `pip install -r requirements.txt` rodou
- [ ] Modelos treinados (`models/ensemble/` existe)
- [ ] Dados coletados (`data/*.csv` existem)
- [ ] API keys válidas

---

**Pronto para dominar o mercado! 🚀💰**
