"""
Componentes de gráficos Plotly reutilizáveis.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_candlestick(df: pd.DataFrame, symbol: str = 'BTC/USDT') -> go.Figure:
    """Gráfico de candlestick com volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Preço {symbol}', 'Volume'),
    )

    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )

    colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(df['close'], df['open'])
    ]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'], y=df['volume'],
            name='Volume', marker_color=colors, showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark',
    )
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def plot_pnl_chart(trades: list[dict]) -> go.Figure | None:
    """Gráfico de P&L acumulado."""
    if not trades:
        return None

    df = pd.DataFrame(trades)
    if 'realizedPnl' not in df.columns or 'time' not in df.columns:
        return None
    df['time']           = pd.to_datetime(df['time'], unit='ms')
    df['realizedPnl']    = df['realizedPnl'].astype(float)
    df['cumulative_pnl'] = df['realizedPnl'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['cumulative_pnl'],
        mode='lines+markers', name='P&L Acumulado',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
    ))
    fig.update_layout(
        title='P&L Acumulado',
        xaxis_title='Tempo',
        yaxis_title='P&L (USDT)',
        height=300,
        template='plotly_dark',
    )
    return fig
