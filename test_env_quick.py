import sys
sys.path.append('src')

from environment.trading_env import TradingEnv

env = TradingEnv(data_path='data/train_btcusdt_36m_20260109.csv')
obs, info = env.reset()

print('Environment OK!')
print(f'Info keys: {list(info.keys())}')
print(f'Liquidations (episode): {info["liquidations"]}')
print(f'Total liquidations (global): {info["total_liquidations"]}')
