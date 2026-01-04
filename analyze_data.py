"""
Analisa qualidade dos dados de treinamento
"""
import pandas as pd
import numpy as np

# Carrega dados
df = pd.read_csv('data/train_data.csv')

print('=' * 70)
print('ANALISE DOS DADOS DE TREINAMENTO')
print('=' * 70)

# 1. Informações básicas
print(f'\n📊 SHAPE: {df.shape}')
print(f'   Candles: {df.shape[0]}')
print(f'   Features: {df.shape[1]}')
print(f'   Colunas: {list(df.columns)}')

# 2. Período
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f'\n📅 PERIODO:')
    print(f'   Inicio: {df["timestamp"].min()}')
    print(f'   Fim: {df["timestamp"].max()}')
    print(f'   Duracao: {df["timestamp"].max() - df["timestamp"].min()}')

# 3. Valores faltantes
print(f'\n⚠️ VALORES FALTANTES:')
missing = df.isnull().sum()
if missing.sum() == 0:
    print('   ✅ Nenhum valor faltante!')
else:
    print(missing[missing > 0])

# 4. Estatísticas do preço
print(f'\n💰 ANALISE DE PRECO (close):')
print(f'   Min: ${df["close"].min():,.2f}')
print(f'   Max: ${df["close"].max():,.2f}')
print(f'   Media: ${df["close"].mean():,.2f}')
print(f'   Std: ${df["close"].std():,.2f}')
print(f'   Range: ${df["close"].max() - df["close"].min():,.2f}')
print(f'   Variacao %: {((df["close"].max() / df["close"].min()) - 1) * 100:.2f}%')

# 5. Movimento do mercado
returns = df['close'].pct_change()
print(f'\n📈 MOVIMENTO DO MERCADO:')
print(f'   Return medio: {returns.mean() * 100:.4f}%')
print(f'   Volatilidade: {returns.std() * 100:.4f}%')
print(f'   Max drawdown: {returns.min() * 100:.2f}%')
print(f'   Max pump: {returns.max() * 100:.2f}%')

# Identifica regime
up_days = (returns > 0).sum()
down_days = (returns < 0).sum()
print(f'   Dias alta: {up_days} ({up_days/len(returns)*100:.1f}%)')
print(f'   Dias baixa: {down_days} ({down_days/len(returns)*100:.1f}%)')

if up_days > down_days * 1.5:
    regime = "📈 BULL MARKET"
elif down_days > up_days * 1.5:
    regime = "📉 BEAR MARKET"
else:
    regime = "↔️ LATERAL/CONSOLIDACAO"
print(f'   Regime: {regime}')

# 6. Qualidade dos indicadores
print(f'\n🔍 QUALIDADE DOS INDICADORES:')
indicators = ['RSI_14', 'SMA_20', 'SMA_50', 'MACD_12_26_9']
for ind in indicators:
    if ind in df.columns:
        print(f'   {ind}: Min={df[ind].min():.2f}, Max={df[ind].max():.2f}, NaN={df[ind].isnull().sum()}')

# 7. Diversidade de dados
print(f'\n🎲 DIVERSIDADE:')
price_bins = pd.cut(df['close'], bins=10)
distribution = price_bins.value_counts().sort_index()
print('   Distribuição de preços em 10 bins:')
for bin_range, count in distribution.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f'   {bin_range}: {count:4d} ({pct:5.1f}%) {bar}')

# 8. Recomendações
print(f'\n💡 RECOMENDACOES:')
if len(df) < 500:
    print('   ⚠️ Poucos dados! Recomendado: 1000+ candles')
elif len(df) < 1000:
    print('   ⚠️ Dados suficientes mas limitados. Ideal: 2000+ candles')
else:
    print('   ✅ Quantidade de dados adequada')

if returns.std() < 0.01:
    print('   ⚠️ Baixa volatilidade - modelos podem ter dificuldade em aprender')
else:
    print('   ✅ Volatilidade adequada para treinamento')

if regime == "↔️ LATERAL/CONSOLIDACAO":
    print('   ✅ Dados balanceados - bom para generalização')
else:
    print(f'   ⚠️ Dados tendenciosos ({regime}) - modelos podem overfitar')

print('\n' + '=' * 70)
