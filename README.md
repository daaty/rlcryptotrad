# 🤖 Trading Bot Dashboard — LSTM V17.7

> Automated Binance Futures trading bot powered by a **RecurrentPPO (LSTM)** model trained with Stable-Baselines3. Built on a modular Streamlit dashboard with real-time WebSocket data, multi-symbol support, and professional risk management.

---

## 📋 Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Bot](#running-the-bot)
7. [File Structure](#file-structure)
8. [Risk Management](#risk-management)
9. [Docker](#docker)
10. [Authentication](#authentication)
11. [Telegram Notifications](#telegram-notifications)
12. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Features

| Feature | Details |
|---|---|
| **Model** | RecurrentPPO (LSTM) — V17.7, 600k steps |
| **Exchange** | Binance Futures (Testnet + Live) |
| **Symbols** | Multi-symbol (BTC, ETH, SOL, …) |
| **Timeframe** | 15-minute candles via WebSocket |
| **Position sizing** | Kelly Criterion (adaptive, based on trade history) |
| **Risk guards** | SL/TP, Trailing Stop, ATR-based SL, breakeven after TP1 |
| **Dashboard** | 5-tab Streamlit (Overview, Positions, Performance, Analysis, Engine) |
| **State persistence** | Engine state saved to disk — survives restarts |
| **Notifications** | Telegram alerts for SL/TP/drawdown/WS issues |
| **Paper mode** | Simulate trades without sending real orders |
| **Auth** | Optional Streamlit login gate (bcrypt hashed passwords) |
| **Logging** | Rotating file logs (10 MB, 7-day retention) |
| **Tests** | 37 unit tests — pytest |

---

## Architecture

```
dashboard_new.py              ← Streamlit entry point
│
├── dashboard/
│   ├── core/
│   │   ├── logging_setup.py  ← RotatingFileHandler + deque for UI
│   │   └── config.py         ← YAML config loader
│   │
│   ├── data/
│   │   ├── websocket_manager.py ← Real-time candle buffers (WS + REST fallback)
│   │   └── account_data.py      ← Balance & positions (WS-first)
│   │
│   ├── trading/
│   │   ├── engine.py          ← Main loop: inference → filters → execute
│   │   ├── executor.py        ← Order placement (Binance API + paper mode)
│   │   ├── entry_filter.py    ← RSI / EMA / Volume / ATR quality filters
│   │   ├── observation.py     ← Feature extraction for LSTM input
│   │   └── state_persistence.py ← Save/restore state to data/engine_state.json
│   │
│   ├── analytics/
│   │   ├── performance.py     ← Sharpe, sortino, drawdown, win-rate
│   │   ├── risk_calculator.py ← Kelly Criterion position sizing
│   │   ├── correlation.py     ← Pearson cross-symbol correlation guard
│   │   └── report_generator.py ← PDF performance reports (fpdf2)
│   │
│   ├── integrations/
│   │   └── telegram_notifier.py ← Async Telegram bot (queue + worker thread)
│   │
│   └── ui/
│       ├── sidebar.py
│       ├── tab_overview.py
│       ├── tab_positions.py
│       ├── tab_performance.py  ← PDF download button
│       ├── tab_analysis.py
│       └── tab_engine.py       ← Start/stop, decisions, config editor, logs
│
├── src/trading/advanced_risk.py ← TrailingStopManager (ATR trailing, breakeven)
├── config.yaml                  ← Central configuration
├── auth_config.yaml             ← Dashboard login credentials (bcrypt)
├── Dockerfile                   ← Production container
└── docker-compose.yml           ← Multi-service orchestration
```

---

## Requirements

- **Python 3.11** (3.10+ should work)
- **Binance API key** (Testnet or Live — Futures enabled)
- **TA-Lib** C library (see [Installation](#installation))
- **CUDA / DirectML** optional (CPU inference works fine)

---

## Installation

### 1. Clone and set up virtual environment

```bash
git clone <your-repo-url>
cd "AGENTE TRANDING"
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install TA-Lib C library

**Windows:** Download the pre-built wheel from [here](https://github.com/cgohlke/talib-build/releases) and install:
```bash
pip install TA_Lib-0.4.x-cpXX-cpXX-win_amd64.whl
```

**Ubuntu / Debian:**
```bash
sudo apt-get install build-essential wget
wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make -j$(nproc) && sudo make install
pip install TA-Lib
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API credentials

```bash
cp .env.example .env
# Edit .env with your Binance API key and secret
```

### 5. Verify installation

```bash
python -m pytest tests/ -v        # All 37 tests should pass
streamlit run dashboard_new.py    # Opens at http://localhost:8501
```

---

## Configuration

All settings are in `config.yaml`. Key sections:

### Trading Mode

```yaml
mode: "testnet"   # "paper" | "testnet" | "live"
```

| Mode | Description |
|---|---|
| `paper` | Simulated fills, no API calls — safest for testing |
| `testnet` | Real API calls on Binance Testnet (fake money) |
| `live` | **Real money** — only after thorough testing |

### Risk Management

```yaml
risk_management:
  stop_loss_pct: 0.02             # 2% fixed SL fallback
  take_profit_pct: 0.04           # 4% TP total (TP1 at 50%, TP2 at 100%)
  trailing_stop_activation: 0.03  # Activate trailing at +3%
  trailing_stop_distance: 0.015   # Trailing distance 1.5%
  max_total_exposure: 0.60        # Max 60% of equity in open positions
  max_exposure_per_asset: 0.25    # Max 25% per symbol
```

Live-edit from the dashboard: **⚙️ Engine → ⚙️ Parâmetros de Risco**.

### Symbols

```yaml
data:
  primary_symbol: "BTC/USDT"
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
```

---

## Running the Bot

### Development (local)

```bash
venv\Scripts\activate             # Windows
source venv/bin/activate          # Linux/macOS
streamlit run dashboard_new.py
```

Open **http://localhost:8501** → select symbols → **▶️ Iniciar Engine**.

### Production (Docker)

```bash
docker compose up -d
docker compose logs -f
```

---

## File Structure

```
AGENTE TRANDING/
├── dashboard_new.py          ← Entry point
├── config.yaml               ← Central config
├── auth_config.yaml          ← Login credentials (never commit to public repos)
├── .env                      ← API secrets (never commit!)
├── requirements.txt          ← Dev dependencies
├── requirements-docker.txt   ← Docker (Linux) dependencies
├── Dockerfile
├── docker-compose.yml
├── dashboard/                ← Modular source
├── src/                      ← Legacy (deprecated)
├── models/                   ← Trained LSTM .zip files
├── data/                     ← Runtime state (engine_state.json)
├── logs/                     ← Rotating log files
├── tests/                    ← pytest suite (37 tests)
└── kline_cache/              ← Disk cache for historical candles
```

---

## Risk Management

Five layers of capital protection:

1. **Fixed SL/TP** — percentage-based, always active
2. **ATR-based SL** — adapts to volatility
3. **Trailing Stop** — activates after `trailing_stop_activation` gain
4. **Breakeven after TP1** — SL moves to entry after partial profit
5. **Exposure guard** — blocks new trades at `max_total_exposure`

Additional guards:
- **Correlation guard** — blocks correlated pairs (Pearson > 0.70, 50 candles)
- **Kelly sizing** — position size adapts to recent win rate (last 30 trades)
- **WS stale guard** — blocks inference if data > 5 minutes old
- **State persistence** — SL/TP/trail state survives bot restarts

---

## Docker

```bash
docker build -t trading-bot .
docker compose up -d
curl http://localhost:8501/_stcore/health
```

**Mounted volumes:** `./data`, `./logs`, `./models`, `./config.yaml`

---

## Authentication

Off by default. To enable:

**`config.yaml`:**
```yaml
auth:
  enabled: true
  cookie_key: "a-random-64-char-secret"
```

**Change password:**
```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt(12)).decode())"
```
Replace the `password` field in `auth_config.yaml`.

---

## Telegram Notifications

1. Create bot via [@BotFather](https://t.me/BotFather) → get `TOKEN`
2. Get `CHAT_ID` via [@userinfobot](https://t.me/userinfobot)
3. Edit `config.yaml`:

```yaml
notifications:
  telegram:
    enabled: true
    token: "123456789:AABBcc..."
    chat_id: "-1001234567890"
    events: [sl, tp, trade, drawdown, ws_down, engine_err]
```

---

## FAQ & Troubleshooting

**`KeyError: 'APIError'` on startup**
Check `BINANCE_API_KEY` in `.env`. Futures must be enabled, IP whitelist correct.

**WS disconnects frequently**
WS manager auto-reconnects. Check internet stability. On VPS, ensure Binance WS endpoints (port 443) are not blocked.

**`ModuleNotFoundError: TA-Lib`**
TA-Lib C library not installed. See [Installation → Step 2](#installation).

**Model not found**
Ensure the `.zip` model is in `models/` and the path matches `config.yaml`.

**Tests fail**
```bash
python -m pytest tests/ -v --tb=short
```

**From Testnet to Live checklist:**
- [ ] ≥ 2 weeks testnet without crashes
- [ ] Set `mode: "live"` in `config.yaml`
- [ ] Add live API keys to `.env`
- [ ] Start with `position_size: 0.005` (0.5%)
- [ ] Monitor Telegram for 48h before increasing size

---

*Private — all rights reserved.*
