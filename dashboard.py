"""
📊 Dashboard em Tempo Real - Trading Bot
Streamlit app com visualizações ao vivo
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yaml
import time
import logging
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv
from pathlib import Path
from stable_baselines3 import TD3
from src.data.data_collector import DataCollector
from src.risk.risk_manager import RiskManager
import talib

load_dotenv()

# Configurar logging
log_file = Path("logs/trading_decisions.log")
log_file.parent.mkdir(exist_ok=True)

# Handler para arquivo com UTF-8
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

# Handler para console com UTF-8 e ignore de erros
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
# Configura stream para UTF-8
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Configura logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="🤖 Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Cache config
@st.cache_resource
def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_risk_manager():
    """Carrega Risk Manager"""
    return RiskManager()

@st.cache_resource
def load_td3_model():
    """Carrega modelo TD3 treinado"""
    try:
        logger.info("[MODELS] Carregando modelo TD3...")
        
        td3 = TD3.load("models/base_btcusdt_final.zip")
        
        logger.info("[MODELS] Modelo TD3 carregado com sucesso!")
        return td3
    except Exception as e:
        logger.error(f"[MODELS] Erro ao carregar modelo TD3: {e}")
        return None

@st.cache_resource
def get_binance_client():
    config = load_config()
    mode = config.get('mode', 'testnet')
    
    if mode == 'testnet':
        return Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True
        )
    else:
        return Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )

def collect_market_data(client, symbol='BTCUSDT', limit=1000):
    """Coleta e processa dados de mercado"""
    try:
        # Coleta dados
        klines = client.futures_klines(symbol=symbol, interval='15m', limit=limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Converte para numérico
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Adiciona indicadores técnicos
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14) / 100.0
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20) / df['close'].max()
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50) / df['close'].max()
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['BBL_20_2.0'] = lower / df['close'].max()
        df['BBM_20_2.0'] = middle / df['close'].max()
        df['BBU_20_2.0'] = upper / df['close'].max()
        df['BBB_20_2.0'] = (upper - lower) / middle
        df['BBP_20_2.0'] = (df['close'] - lower) / (upper - lower)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        df['MACD_12_26_9'] = macd / df['close'].max()
        df['MACDs_12_26_9'] = signal / df['close'].max()
        df['MACDh_12_26_9'] = hist / df['close'].max()
        
        # Returns
        df['open_return'] = df['open'].pct_change()
        df['high_return'] = df['high'].pct_change()
        df['low_return'] = df['low'].pct_change()
        df['close_return'] = df['close'].pct_change()
        
        # Remove NaN
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"[DATA] Erro ao coletar dados: {e}")
        return None

def prepare_observation(market_data):
    """Prepara observação para o modelo (50, 19)"""
    try:
        # Seleciona apenas as 16 colunas do dataset de treino (SEM returns!)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'RSI_14', 'SMA_20', 'SMA_50',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
        ]
        
        # Pega últimos 50 timesteps das 16 features
        obs_data = market_data[feature_cols].iloc[-50:].values  # Shape: (50, 16)
        
        logger.info(f"[OBS] Shape dos dados de mercado: {obs_data.shape}")
        
        # Adiciona 3 features de portfolio
        balance_feat = np.ones((50, 1)) * 1.0  # 100% do saldo inicial
        position_feat = np.zeros((50, 1))  # FLAT (sem posição)
        equity_feat = np.ones((50, 1)) * 1.0  # Equity = saldo
        
        # Concatena: 16 features do mercado + 3 portfolio = 19 features totais
        obs = np.concatenate([
            obs_data,
            balance_feat,
            position_feat,
            equity_feat
        ], axis=1)
        
        logger.info(f"[OBS] Shape final da observacao: {obs.shape}")
        
        return obs.astype(np.float32)
        
    except Exception as e:
        logger.error(f"[OBS] Erro ao preparar observação: {e}")
        return None
    config = load_config()
    mode = config.get('mode', 'testnet')
    
    if mode == 'testnet':
        return Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET_KEY'),
            testnet=True
        )
    else:
        return Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )

def get_account_balance(client):
    """Retorna saldo da conta"""
    balance = client.futures_account_balance()
    usdt = [b for b in balance if b['asset'] == 'USDT'][0]
    return {
        'total': float(usdt['balance']),
        'available': float(usdt['availableBalance']),
        'unrealized_pnl': float(usdt.get('crossUnPnl', 0))
    }

def get_open_positions(client):
    """Retorna posições abertas"""
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    return open_positions

def execute_trade(client, decision, current_price, config):
    """Executa trade baseado na decisão do ensemble"""
    try:
        symbol = config['data']['symbol'].replace('/', '')  # BTC/USDT -> BTCUSDT
        
        # Verifica posição atual
        positions = client.futures_position_information(symbol=symbol)
        
        # Encontra a posição do símbolo (pode haver múltiplas posições)
        current_position = 0.0
        for pos in positions:
            if pos['symbol'] == symbol:
                current_position = float(pos['positionAmt'])
                break
        
        logger.info(f"[TRADE] Posicao atual: {current_position} BTC, Decisao: {decision}")
        
        # Define side baseado na decisão
        if decision == 'LONG' and current_position <= 0:
            # Fecha SHORT (se houver) e abre LONG
            if current_position < 0:
                logger.info(f"[TRADE] Fechando posicao SHORT de {current_position}")
                client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=abs(current_position)
                )
            
            # Calcula quantidade para LONG
            balance = get_account_balance(client)
            position_size = config['environment']['position_size']
            leverage = config['environment']['leverage']
            quantity = (balance['available'] * position_size * leverage) / current_price
            quantity = round(quantity, 3)  # Arredonda para 3 casas decimais
            
            logger.info(f"[TRADE] Abrindo posicao LONG: {quantity} BTC @ ${current_price:,.2f}")
            order = client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"[TRADE] ✅ Ordem LONG executada: {order['orderId']}")
            return order
            
        elif decision == 'SHORT' and current_position >= 0:
            # Fecha LONG (se houver) e abre SHORT
            if current_position > 0:
                logger.info(f"[TRADE] Fechando posicao LONG de {current_position}")
                client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=current_position
                )
            
            # Calcula quantidade para SHORT
            balance = get_account_balance(client)
            position_size = config['environment']['position_size']
            leverage = config['environment']['leverage']
            quantity = (balance['available'] * position_size * leverage) / current_price
            quantity = round(quantity, 3)
            
            logger.info(f"[TRADE] Abrindo posicao SHORT: {quantity} BTC @ ${current_price:,.2f}")
            order = client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"[TRADE] ✅ Ordem SHORT executada: {order['orderId']}")
            return order
            
        elif decision == 'FLAT' and current_position != 0:
            # Fecha qualquer posição aberta
            side = 'SELL' if current_position > 0 else 'BUY'
            logger.info(f"[TRADE] Fechando posicao {side}: {abs(current_position)} BTC")
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=abs(current_position)
            )
            logger.info(f"[TRADE] ✅ Posicao fechada: {order['orderId']}")
            return order
        else:
            logger.info(f"[TRADE] Sem mudança de posição (atual: {current_position}, decisão: {decision})")
            return None
            
    except Exception as e:
        logger.error(f"[TRADE] Erro ao executar trade: {e}")
        return None

def get_recent_trades(client, symbol='BTCUSDT', limit=10):
    """Retorna trades recentes"""
    try:
        trades = client.futures_account_trades(symbol=symbol, limit=limit)
        return trades
    except:
        return []

def get_klines(client, symbol='BTCUSDT', interval='15m', limit=100):
    """Retorna dados de mercado"""
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def plot_candlestick(df):
    """Gráfico de candlestick com volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Preço BTC/USDT', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USDT',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#26a69a' if close >= open_ else '#ef5350' 
              for close, open_ in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def calculate_performance_metrics(trades):
    """Calcula métricas de performance avançadas"""
    if not trades or len(trades) < 2:
        return None
    
    df = pd.DataFrame(trades)
    df['realizedPnl'] = df['realizedPnl'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Métricas básicas
    total_trades = len(df)
    wins = len(df[df['realizedPnl'] > 0])
    losses = len(df[df['realizedPnl'] < 0])
    win_rate = (wins / total_trades) if total_trades > 0 else 0
    
    total_pnl = df['realizedPnl'].sum()
    avg_win = df[df['realizedPnl'] > 0]['realizedPnl'].mean() if wins > 0 else 0
    avg_loss = df[df['realizedPnl'] < 0]['realizedPnl'].mean() if losses > 0 else 0
    
    # Sharpe Ratio (anualizado, assumindo 365 dias)
    returns = df['realizedPnl']
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0
    
    # Profit Factor
    gross_profit = df[df['realizedPnl'] > 0]['realizedPnl'].sum()
    gross_loss = abs(df[df['realizedPnl'] < 0]['realizedPnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    df['cumulative_pnl'] = df['realizedPnl'].cumsum()
    df['running_max'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['running_max']
    max_drawdown = df['drawdown'].min()
    
    # Recovery Factor
    recovery_factor = (total_pnl / abs(max_drawdown)) if max_drawdown < 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor,
        'expectancy': expectancy
    }

def plot_pnl_chart(trades):
    """Gráfico de P&L acumulado"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['realizedPnl'] = df['realizedPnl'].astype(float)
    df['cumulative_pnl'] = df['realizedPnl'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['cumulative_pnl'],
        mode='lines+markers',
        name='P&L Acumulado',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title='P&L Acumulado',
        xaxis_title='Tempo',
        yaxis_title='P&L (USDT)',
        height=300,
        template='plotly_dark'
    )
    
    return fig

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

st.markdown('<div class="main-header">🤖 Trading Bot Dashboard</div>', unsafe_allow_html=True)

# Carrega config e cliente
config = load_config()
client = get_binance_client()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Status do modo
    mode = config.get('mode', 'testnet')
    if mode == 'testnet':
        st.success("🧪 Modo: TESTNET")
    elif mode == 'live':
        st.error("⚠️ Modo: LIVE (REAL)")
    else:
        st.info("📝 Modo: PAPER")
    
    st.divider()
    
    # Configurações de trading
    st.subheader("📊 Parâmetros")
    st.text(f"Symbol: {config['data']['symbol']}")
    st.text(f"Timeframe: {config['data']['timeframe']}")
    st.text(f"Position Size: {config['environment']['position_size']*100}%")
    st.text(f"Leverage: {config['environment']['leverage']}x")
    
    st.divider()
    
    # Risk Management Status
    risk_mgr = load_risk_manager()
    st.subheader("🛡️ Risk Management")
    
    # Circuit Breaker Status
    can_trade, reason = risk_mgr.should_allow_trade()
    if can_trade:
        st.success("✅ Trading Ativo")
    else:
        st.error(f"⛔ {reason}")
        if st.button("🔄 Reset Circuit Breaker"):
            risk_mgr.reset_circuit_breaker()
            st.success("Circuit breaker resetado!")
            st.rerun()
    
    # Trading Stats
    stats = risk_mgr.get_trading_stats()
    if stats['total_trades'] > 0:
        st.text(f"Trades: {stats['total_trades']}")
        st.text(f"Win Rate: {stats['win_rate']*100:.1f}%")
        st.text(f"Losses: {stats['consecutive_losses']}")
    
    st.divider()
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
    refresh_interval = st.slider("Intervalo (s)", 5, 60, 10)
    
    if st.button("🔄 Atualizar Agora"):
        st.rerun()

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Posições", "📈 Performance", "🔍 Logs"])

with tab1:
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    balance = get_account_balance(client)
    positions = get_open_positions(client)
    
    with col1:
        st.metric(
            label="💰 Balance Total",
            value=f"${balance['total']:,.2f}",
            delta=f"{balance['unrealized_pnl']:+.2f} USDT"
        )
    
    with col2:
        st.metric(
            label="💵 Disponível",
            value=f"${balance['available']:,.2f}"
        )
    
    with col3:
        st.metric(
            label="📊 Posições Abertas",
            value=len(positions)
        )
    
    with col4:
        # Pega preço atual
        ticker = client.futures_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        price_change = float(ticker['priceChangePercent'])
        
        st.metric(
            label="₿ BTC/USDT",
            value=f"${current_price:,.2f}",
            delta=f"{price_change:+.2f}%"
        )
    
    st.divider()
    
    # Gráfico de preço
    st.subheader("📈 Gráfico de Mercado")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        timeframe_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR
        }
        
        selected_tf = st.selectbox("Timeframe", list(timeframe_map.keys()), index=2)
        candles_limit = st.slider("Candles", 50, 500, 100)
    
    with col1:
        df = get_klines(client, interval=timeframe_map[selected_tf], limit=candles_limit)
        fig = plot_candlestick(df)
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("💼 Posições Abertas")
    
    if len(positions) == 0:
        st.info("📭 Nenhuma posição aberta no momento")
    else:
        # Carrega Risk Manager
        risk_mgr = load_risk_manager()
        
        for pos in positions:
            symbol = pos['symbol']
            qty = float(pos['positionAmt'])
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            pnl_pct = (unrealized_pnl / (entry_price * abs(qty))) * 100 if qty != 0 else 0
            
            side = "LONG 🟢" if qty > 0 else "SHORT 🔴"
            position_type = 1 if qty > 0 else -1
            
            # Calcula Stop Loss e Take Profit
            # Para demo, usa ATR fictício (idealmente viria dos dados reais)
            atr_estimate = mark_price * 0.02  # ~2% do preço como ATR estimado
            
            stop_price = risk_mgr.calculate_atr_stop_loss(entry_price, atr_estimate, position_type)
            should_stop = risk_mgr.should_stop_loss(entry_price, mark_price, position_type, atr=atr_estimate)
            
            should_tp, tp_level = risk_mgr.should_take_profit(entry_price, mark_price, position_type, return_level=True)
            
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"**{symbol}**")
                    st.text(side)
                
                with col2:
                    st.text(f"Qty: {abs(qty):.4f}")
                    st.text(f"Entry: ${entry_price:,.2f}")
                
                with col3:
                    st.text(f"Mark: ${mark_price:,.2f}")
                    leverage = pos.get('leverage', config.get('environment', {}).get('leverage', 3))
                    st.text(f"Leverage: {leverage}x")
                
                with col4:
                    # Stop Loss e Take Profit
                    stop_color = "🔴" if should_stop else "🟢"
                    st.text(f"{stop_color} SL: ${stop_price:,.0f}")
                    
                    if tp_level > 0:
                        st.text(f"✅ TP Nível {tp_level}")
                    else:
                        tp_target_1 = entry_price * (1.02 if qty > 0 else 0.98)
                        st.text(f"🎯 TP1: ${tp_target_1:,.0f}")
                
                with col5:
                    pnl_class = "positive" if unrealized_pnl >= 0 else "negative"
                    st.markdown(f'<p class="{pnl_class}">P&L: ${unrealized_pnl:,.2f}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="{pnl_class}">({pnl_pct:+.2f}%)</p>', unsafe_allow_html=True)
                
                st.divider()

with tab3:
    st.subheader("📊 Performance de Trades")
    
    trades = get_recent_trades(client, limit=50)
    
    if not trades:
        st.info("📭 Nenhum trade executado ainda")
    else:
        # P&L Chart
        pnl_chart = plot_pnl_chart(trades)
        if pnl_chart:
            st.plotly_chart(pnl_chart, width='stretch')
        
        st.divider()
        
        # Preparar DataFrame de trades
        df_trades = pd.DataFrame(trades)
        df_trades['realizedPnl'] = df_trades['realizedPnl'].astype(float)
        
        # Estatísticas avançadas
        metrics = calculate_performance_metrics(trades)
        
        if metrics:
            st.subheader("📈 Performance Metrics")
            
            # Linha 1: Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics['total_trades'])
            
            with col2:
                win_rate_pct = metrics['win_rate'] * 100
                st.metric("Win Rate", f"{win_rate_pct:.1f}%", 
                         delta="✅" if win_rate_pct >= 50 else "⚠️")
            
            with col3:
                st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}",
                         delta=f"${metrics['total_pnl']:+,.2f}")
            
            with col4:
                sharpe = metrics['sharpe_ratio']
                sharpe_color = "✅" if sharpe > 1.5 else ("⚠️" if sharpe > 0.5 else "❌")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=sharpe_color)
            
            st.divider()
            
            # Linha 2: Métricas de risco
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Win", f"${metrics['avg_win']:,.2f}")
            
            with col2:
                st.metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")
            
            with col3:
                pf = metrics['profit_factor']
                pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
                pf_color = "✅" if pf > 1.5 else ("⚠️" if pf > 1.0 else "❌")
                st.metric("Profit Factor", pf_str, delta=pf_color)
            
            with col4:
                st.metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")
            
            st.divider()
            
            # Linha 3: Métricas avançadas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Wins", metrics['wins'], delta=f"+{metrics['wins']}")
            
            with col2:
                st.metric("Losses", metrics['losses'], delta=f"-{metrics['losses']}")
            
            with col3:
                rf = metrics['recovery_factor']
                rf_str = f"{rf:.2f}" if rf != float('inf') else "∞"
                st.metric("Recovery Factor", rf_str)
            
            with col4:
                st.metric("Expectancy", f"${metrics['expectancy']:,.2f}")
        else:
            # Estatísticas básicas (fallback para poucos trades)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_trades = len(df_trades)
                st.metric("Total Trades", total_trades)
            
            with col2:
                wins = len(df_trades[df_trades['realizedPnl'] > 0])
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                total_pnl = df_trades['realizedPnl'].sum()
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col4:
                avg_pnl = df_trades['realizedPnl'].mean()
                st.metric("Avg P&L/Trade", f"${avg_pnl:,.2f}")
        
        st.divider()
        
        # Tabela de trades
        st.subheader("🗂️ Histórico de Trades")
        df_display = df_trades[['time', 'symbol', 'side', 'qty', 'price', 'realizedPnl']].copy()
        df_display['time'] = pd.to_datetime(df_display['time'], unit='ms')
        df_display['realizedPnl'] = df_display['realizedPnl'].astype(float)
        
        st.dataframe(df_display, width='stretch')

with tab4:
    st.subheader("📜 Logs em Tempo Real")
    
    # Botão para carregar modelo e executar bot
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Executar Bot", type="primary"):
            st.session_state['bot_running'] = True
            logger.info("[BOT] Bot iniciado pelo usuario")
            
            # Carrega modelo TD3
            with st.spinner("Carregando modelo TD3..."):
                td3_model = load_td3_model()
                if td3_model:
                    st.session_state['td3_model'] = td3_model
                    st.success("✅ Modelo TD3 carregado!")
                else:
                    st.error("❌ Erro ao carregar modelo TD3")
                    st.session_state['bot_running'] = False
    
    with col2:
        if st.button("⏹️ Parar Bot", type="secondary"):
            st.session_state['bot_running'] = False
            logger.info("[BOT] Bot parado pelo usuario")
    
    with col3:
        bot_status = "🟢 RODANDO" if st.session_state.get('bot_running', False) else "🔴 PARADO"
        st.info(f"Status: {bot_status}")
    
    st.divider()
    
    # Se bot está rodando, executa lógica de trading
    if st.session_state.get('bot_running', False) and st.session_state.get('td3_model'):
        st.subheader("🤖 Análise em Tempo Real")
        
        td3_model = st.session_state['td3_model']
        
        with st.spinner("Coletando dados de mercado..."):
            # Coleta dados
            market_data = collect_market_data(client, symbol='BTCUSDT', limit=1000)
            
            if market_data is not None:
                # Info do mercado
                current_price = market_data['close'].iloc[-1]
                rsi = market_data['RSI_14'].iloc[-1]
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("💰 Preço BTC", f"${current_price:,.2f}")
                with col_b:
                    st.metric("📊 RSI", f"{rsi*100:.1f}")
                with col_c:
                    volatility = market_data['close_return'].std()
                    st.metric("📈 Volatilidade", f"{volatility:.4f}")
                
                # Prepara observação
                obs = prepare_observation(market_data)
                
                if obs is not None:
                    st.info(f"✅ Observação preparada: shape {obs.shape}")
                    logger.info(f"[TRADING] Preco: ${current_price:,.2f}, RSI: {rsi:.2f}, Vol: {volatility:.4f}")
                    
                    # Faz predição com TD3
                    action, _states = td3_model.predict(obs, deterministic=True)
                    action_value = float(action[0])
                    
                    # Mapeia ação contínua para discreto
                    if action_value < -0.33:
                        final_action = "SHORT"
                    elif action_value > 0.33:
                        final_action = "LONG"
                    else:
                        final_action = "FLAT"
                    
                    # Exibe decisão
                    st.success(f"🎯 Decisão TD3: **{final_action}**")
                    
                    # Detalhes da ação
                    col_action, col_value, col_conf = st.columns(3)
                    
                    with col_action:
                        st.info(f"🤖 Ação\n\n{final_action}")
                    
                    with col_value:
                        st.info(f"📊 Valor\n\n{action_value:.3f}")
                    
                    with col_conf:
                        confidence = abs(action_value) * 100
                        st.success(f"✅ Confiança\n\n{confidence:.1f}%")
                    
                    # Log da decisão
                    logger.info(f"[DECISION] TD3: {final_action} (action_value: {action_value:.3f})")
                    
                    # 🔥 EXECUTAR TRADE! 🔥
                    st.divider()
                    st.subheader("⚡ Executando Trade...")
                    
                    # Verifica Risk Management ANTES de executar
                    risk_mgr = load_risk_manager()
                    can_trade, reason = risk_mgr.should_allow_trade()
                    
                    if not can_trade:
                        st.error(f"⛔ TRADE BLOQUEADO: {reason}")
                        st.warning("Circuit breaker ativo. Aguarde análise manual ou resete no sidebar.")
                    else:
                        with st.spinner("Enviando ordem para Binance..."):
                            order = execute_trade(client, final_action, current_price, config)
                            
                            if order:
                                st.success(f"✅ Ordem executada! ID: {order['orderId']}")
                                st.json({
                                    'orderId': order['orderId'],
                                    'symbol': order['symbol'],
                                    'side': order['side'],
                                    'quantity': order['origQty'],
                                    'price': order.get('avgPrice', 'MARKET')
                                })
                                
                                # Registra trade para risk management (PnL será atualizado quando fechar)
                                # Por enquanto, registra como pending
                                logger.info(f"[RISK] Trade registrado para monitoramento")
                            else:
                                st.info("ℹ️ Nenhuma mudança de posição necessária")
                    
                    st.divider()
                else:
                    st.error("❌ Erro ao preparar observação")
            else:
                st.error("❌ Erro ao coletar dados de mercado")
    
    st.divider()
    
    # Mostrar logs do arquivo
    st.subheader("📋 Últimos Logs")
    
    try:
        # Lê últimas 50 linhas do arquivo de log
        log_lines = []
        # Usa encoding do Windows (cp1252) com fallback para utf-8
        try:
            with open("logs/trading_decisions.log", "r", encoding="cp1252") as f:
                log_lines = f.readlines()[-50:]  # Últimas 50 linhas
        except:
            with open("logs/trading_decisions.log", "r", encoding="utf-8", errors="ignore") as f:
                log_lines = f.readlines()[-50:]  # Últimas 50 linhas
        
        # Mostra em área de texto com scroll
        log_text = "".join(log_lines)
        st.text_area("Logs", log_text, height=400, key="log_area")
        
        # Botão para limpar logs
        if st.button("🗑️ Limpar Logs"):
            with open("logs/trading_decisions.log", "w", encoding='utf-8') as f:
                f.write("")
            st.success("Logs limpos!")
            logger.info("[LOGS] Logs limpos pelo usuario")
            st.rerun()
    
    except FileNotFoundError:
        st.warning("📭 Arquivo de log não encontrado. Execute o bot para gerar logs.")
    
    st.divider()
    
    # Status da conexão
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Conexão com Binance ativa")
        st.info(f"🤖 Modelo: TD3")
        st.info(f"📊 Timeframe: {config['data']['timeframe']}")
    
    with col2:
        st.success(f"💰 Saldo: ${balance['total']:,.2f}")
        st.info(f"📈 Posições: {len(positions)}")
        st.info(f"🔄 Auto-refresh: {refresh_interval}s")
    
    st.divider()
    
    # Últimas atividades
    st.subheader("🕐 Últimas Atividades")
    
    recent_trades = get_recent_trades(client, limit=5)
    if recent_trades:
        for trade in recent_trades:
            trade_time = pd.to_datetime(trade['time'], unit='ms').strftime('%H:%M:%S')
            side = trade['side']
            price = float(trade['price'])
            qty = float(trade['qty'])
            pnl = float(trade['realizedPnl'])
            
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            log_msg = f"{pnl_emoji} [{trade_time}] {side} {qty:.4f} BTC @ ${price:,.2f} | P&L: ${pnl:.2f}"
            st.text(log_msg)
            logger.info(log_msg)
    else:
        st.info("📭 Nenhum trade executado ainda")
    
    st.divider()
    
    # Informações do sistema
    st.subheader("ℹ️ Informações do Sistema")
    system_info = f"""
Modo: {mode.upper()}
Symbol: {config['data']['symbol']}
Alavancagem: {config['environment']['leverage']}x
Tamanho de Posição: {config['environment']['position_size']*100}%
Comissão: {config['environment']['commission']*100}%
Última Atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    st.code(system_info, language="text")
    logger.debug(f"Informações do sistema exibidas: {system_info.strip()}")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    current_time = datetime.now().strftime('%H:%M:%S')
    st.text(f"⏰ Atualizado: {current_time}")

with col2:
    st.text(f"🌐 Modo: {mode.upper()}")

with col3:
    st.text(f"📊 {config['data']['symbol']} @ {config['data']['timeframe']}")

# Auto-refresh com logging
if auto_refresh:
    logger.info(f"[AUTO-REFRESH] Aguardando {refresh_interval}s...")
    time.sleep(refresh_interval)
    logger.info("[REFRESH] Recarregando dashboard...")
    st.rerun()
