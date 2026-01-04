# Guia Rápido de Uso

## 🚀 Quick Start

### 1️⃣ Configuração Inicial (5 minutos)

```powershell
# Crie e ative o ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instale as dependências
pip install -r requirements.txt

# Configure suas credenciais
copy .env.example .env
# Edite .env com suas chaves da Binance Testnet
```

### 2️⃣ Colete os Dados (2 minutos)

```powershell
python -m src.data.data_collector
```

**O que isso faz:**
- Baixa 1000 candles de BTC/USDT (15m)
- Calcula RSI, SMA, MACD, Bollinger Bands
- Normaliza e divide em train/val/test
- Salva em `data/`

### 3️⃣ Treine o Agente (30-60 minutos)

```powershell
python -m src.training.train --mode train --name meu_bot_v1
```

**O que acontece:**
- O agente aprende a operar através de tentativa e erro
- Progresso salvo automaticamente
- Melhor modelo salvo em `models/`

**Acompanhe em tempo real:**
```powershell
# Em outro terminal
tensorboard --logdir logs/tensorboard
# Abra http://localhost:6006
```

### 4️⃣ Teste o Modelo (5 minutos)

```powershell
python -m src.training.train --mode eval --model models/meu_bot_v1.zip
```

**Você verá:**
- Win Rate
- Total de trades
- PnL final
- Retorno percentual

### 5️⃣ Execute em Paper Trading (contínuo)

```powershell
python -m src.execution.executor --model models/meu_bot_v1.zip --mode paper --interval 60
```

**O bot irá:**
- Verificar o mercado a cada 60 segundos
- Tomar decisões baseadas no modelo treinado
- Simular ordens (não usa dinheiro real)
- Registrar tudo em `logs/trading/`

## 📊 Personalizando

### Mudar o Ativo

Edite [config.yaml](config.yaml):
```yaml
data:
  symbol: "ETH/USDT"  # Mude aqui
  timeframe: "1h"     # Ou aqui
```

### Ajustar Risco

```yaml
risk_management:
  stop_loss_pct: 0.03  # 3% ao invés de 2%
  take_profit_pct: 0.06  # 6% ao invés de 4%
  max_leverage: 5  # Mais agressivo
```

### Treinar por Mais Tempo

```yaml
training:
  total_timesteps: 200000  # Dobro do padrão
```

## ⚡ Comandos Úteis

```powershell
# Ver logs de trading
Get-Content -Path "logs/trading/*.log" -Tail 50 -Wait

# Listar modelos treinados
Get-ChildItem models/*.zip

# Ver dados coletados
Get-ChildItem data/*.csv

# Reinstalar dependências
pip install -r requirements.txt --upgrade
```

## 🎯 Próximos Passos

1. **Otimize os hiperparâmetros** do RL
2. **Adicione mais indicadores** em `config.yaml`
3. **Teste em diferentes timeframes**
4. **Implemente estratégias híbridas** (RL + Regras)
5. **Crie um dashboard Streamlit** para visualização

## 🐛 Problemas Comuns

### Erro: "No module named 'gymnasium'"
```powershell
pip install gymnasium
```

### Erro: "API Key inválida"
- Verifique se o `.env` está na raiz do projeto
- Use chaves da **testnet** primeiro: https://testnet.binancefuture.com

### Modelo não aprende (reward não aumenta)
- Colete mais dados (aumente `limit` no config)
- Reduza o `learning_rate` para `0.0001`
- Aumente `total_timesteps` para `200000+`

### Bot não executa trades em paper
- Verifique se há saldo suficiente na testnet
- Reduza `position_size` no config
- Verifique os logs em `logs/trading/`

## 📞 Suporte

Abra uma issue no repositório com:
- Mensagem de erro completa
- Arquivo `config.yaml`
- Últimas linhas do log

---

**Boa sorte com seu agente de trading! 🚀📈**
