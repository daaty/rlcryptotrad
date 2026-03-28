"""Diagnóstico isolado de carregamento V19.2 + VecNormalize."""
import numpy as np
import traceback
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from src.environment.trading_env_v19_lstm import TradingEnvV19LSTM

DATA = {
    '15m': 'data/test_btcusdt_36m_15m_20260222.csv',
    '1h':  'data/test_btcusdt_36m_1h_20260222.csv',
    '4h':  'data/test_btcusdt_36m_4h_20260222.csv',
}

ENV_CONFIG = dict(
    window_size=50, max_episode_steps=2000, leverage=1.5,
    commission=0.0004, slippage=0.0005, position_size=0.15,
    use_sharpe_reward=False, enable_indicator_shaping=False,
    random_start=False, persist_balance=False, liquidation_threshold=0.30,
)

STEPS = 280000

try:
    print("1. Criando DummyVecEnv...", flush=True)
    raw_env = DummyVecEnv([lambda: TradingEnvV19LSTM(data_paths=DATA, **ENV_CONFIG)])
    print("   OK", flush=True)

    print("2. Carregando VecNormalize...", flush=True)
    pkl = f"models/recurrent_ppo_v19_multipair_20260306_195359_vecnormalize_{STEPS}_steps.pkl"
    env = VecNormalize.load(pkl, raw_env)
    env.training   = False
    env.norm_reward = False
    print("   OK", flush=True)

    print("3. Carregando RecurrentPPO...", flush=True)
    zip_ = f"models/recurrent_ppo_v19_multipair_20260306_195359_{STEPS}_steps.zip"
    model = RecurrentPPO.load(zip_, env=env, device='cpu')
    print("   OK", flush=True)

    print("4. env.reset()...", flush=True)
    obs = env.reset()
    print(f"   obs shape: {obs.shape}  dtype: {obs.dtype}", flush=True)

    print("5. model.predict()...", flush=True)
    ep_start = np.ones((1,), dtype=bool)
    action, state = model.predict(obs, state=None, episode_start=ep_start, deterministic=True)
    print(f"   action: {action}  shape: {action.shape}", flush=True)

    print("6. env.step()...", flush=True)
    obs2, rew, done, info = env.step(action)
    print(f"   rew: {rew}  done: {done}", flush=True)

    print("\n✅ Tudo OK — backtest deve funcionar")

except Exception as e:
    print(f"\n❌ {e}", flush=True)
    traceback.print_exc()
