"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      🧠 TREINAR RECURRENT PPO V19.3 — LSTM MULTI-PAR MULTI-TF               ║
║                                                                              ║
║  🔧 CORREÇÕES CRÍTICAS vs V18:                                               ║
║                                                                              ║
║  V18 BUG 1 → V19 FIX: OHLCV normalizado no ambiente                        ║
║    BTC close=$27 839 → np.clip(-100,100) = 100 (constante!)                 ║
║    V19: _normalize_ohlcv() converte para % relativos → clip(-10,10) seguro  ║
║                                                                              ║
║  V18 BUG 2 → V19 FIX: vf_coef 0.1 → 0.5                                   ║
║    Critic aprendia 5-10× mais devagar que actor → EV≈0 em 6M steps         ║
║    V19: vf_coef=0.5 (recomendado por SB3 e V17 analysis docs)              ║
║                                                                              ║
║  V18 BUG 3 → V19 FIX: n_steps 2048 → 4096                                 ║
║    Buffer maior → variance menor → gradientes mais estáveis                 ║
║                                                                              ║
║  V18 BUG 4 → V19 FIX: leverage 1.0 → 1.5                                  ║
║    Train/prod mismatch: dashboard usa 1.5, env treinava com 1.0            ║
║    → risco no treino subestimado; SL ativava em momentos errados           ║
║                                                                              ║
║  🔧 CORREÇÕES V19.3 (comprovadas pelo backtest estendido 7 meses):          ║
║                                                                              ║
║  V19.2 BUG A → V19.3 FIX: small-loss bonus no reward shaping               ║
║    elif -0.02 < action_reward < -0.001: reward += 0.05 % pequena perda     ║
║    incentivava overtrading (15 trades/dia) — bug removido                  ║
║                                                                              ║
║  V19.2 BUG B → V19.3 FIX: sem cooldown entre trades                        ║
║    ~15 trades/dia = custo comissão cumulativo destruía retorno              ║
║    V19.3: trade_cooldown=4 (mínimo 1h entre abertura de trades)            ║
║                                                                              ║
║  V19.2 BUG C → V19.3 FIX: PPO instável (clip_fraction=0.35, std→1.47)     ║
║    lr: 2e-4 → 5e-5 | ent_coef: 0.05 → 0.01 | clip_range: 0.2 → 0.1      ║
║                                                                              ║
║  V19.2 BUG D → V19.3 FIX: stop-loss muito largo (-7% balance)              ║
║    MaxDD 25-36% em backtest 7 meses → stop-loss: -7% → -4% do balance     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys

# Forçar UTF-8 no Windows (cmd.exe usa cp1252 por padrão → emojis falham)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from datetime import datetime
from pathlib import Path
import torch
import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("❌ ERRO: stable-baselines3-contrib não instalado!")
    print("   pip install sb3-contrib")
    sys.exit(1)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environment.trading_env_v19_lstm import TradingEnvV19LSTM
from callbacks.trading_metrics import (
    TradingMetricsCallback,
    LiquidationMonitor,
    PerformanceDecayMonitor,
    ValueLossDivergenceMonitor,
)


# ── Pares alvo ────────────────────────────────────────────────────────────────
PAIRS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']

# ── Timesteps ─────────────────────────────────────────────────────────────────
# 4 envs × n_steps=4096 → 16 384 steps/iter
# 6M / 16 384 ≈ 366 rollouts  (metade do V18 em iterações, mas cada rollout
# tem 2× mais diversidade temporal → gradientes mais estáveis)
TOTAL_TIMESTEPS = 6_000_000
SAVE_FREQ       = 10_000
CHECK_FREQ      = 10_000

ENV_CONFIG = {
    'window_size':           50,
    'max_episode_steps':     2000,
    'leverage':              1.5,      # ✅ V19 FIX: alinhado com dashboard (era 1.0)
    'commission':            0.0004,
    'slippage':              0.0005,
    'position_size':         0.15,   # V19.1: 0.05→0.15 (sinal reward 3× maior)
    'use_sharpe_reward':     False,
    'enable_indicator_shaping': False,
    'random_start':          True,
    'persist_balance':       False,
    'liquidation_threshold': 0.30,
    'trade_cooldown':        4,       # V19.3: mínimo 4 steps (1h) entre trades
}

PPO_CONFIG = {
    'learning_rate': 5e-5,      # V19.3: 2e-4→5e-5 (clip_fraction era 0.35 → updates agressivos demais)
    'n_steps':       4096,      # ✅ V19 FIX: 2048→4096 (buffer maior, gradientes estáveis)
    'batch_size':    256,       # proporcional ao n_steps maior
    'n_epochs':      4,
    'gamma':         0.95,
    'gae_lambda':    0.9,
    'clip_range':    0.15,      # V19.4: 0.1→0.15 (clip_fraction subiu 0.005→0.25 = clip bloqueando aprendizado)
    'ent_coef':      0.01,      # V19.3: 0.05→0.01 (std 1.02→1.47 = política divergindo)
    'vf_coef':       0.5,       # ✅ V19 FIX: 0.1→0.5 (critic aprende a mesma taxa que actor)
    'max_grad_norm': 0.5,
}

LSTM_CONFIG = {
    'lstm_hidden_size': 256,
    'n_lstm_layers':    2,
    'net_arch':         [256, 256],
    'activation_fn':    torch.nn.ReLU,
    'ortho_init':       False,
}


# ── Descoberta de dados ────────────────────────────────────────────────────────

def find_pair_data(pair: str) -> dict | None:
    """Procura os arquivos de dados mais recentes para um par."""
    data_dir = Path('data')

    def latest(pattern):
        files = sorted(data_dir.glob(pattern), reverse=True)
        return files[0] if files else None

    f15m = latest(f'train_{pair}_*_15m_*.csv')
    f1h  = latest(f'train_{pair}_*_1h_*.csv')
    f4h  = latest(f'train_{pair}_*_4h_*.csv')

    if f15m and f1h and f4h:
        return {'15m': str(f15m), '1h': str(f1h), '4h': str(f4h)}

    # fallback para btcusdt legado
    if pair == 'btcusdt':
        f15m = latest('train_btcusdt_*_15m_*.csv')
        f1h  = latest('train_btcusdt_*_1h_*.csv')
        f4h  = latest('train_btcusdt_*_4h_*.csv')
        if f15m and f1h and f4h:
            return {'15m': str(f15m), '1h': str(f1h), '4h': str(f4h)}

    return None


def resolve_data_paths(pairs: list) -> list:
    """Retorna lista de data_paths dicts para todos os pares encontrados."""
    available = []

    for pair in pairs:
        paths = find_pair_data(pair)
        if paths:
            available.append((pair.upper(), paths))
            print(f"  ✅ {pair.upper():8s}  15m={Path(paths['15m']).name}")
        else:
            print(f"  ⚠️  {pair.upper():8s}  DADOS NÃO ENCONTRADOS — execute collect_multi_pair_mtf.py")

    if not available:
        print("\n❌ Nenhum dado encontrado. Execute primeiro:")
        print("   python collect_multi_pair_mtf.py")
        sys.exit(1)

    if len(available) < len(pairs):
        print(f"\n⚠️  Treinando com {len(available)}/{len(pairs)} pares disponíveis.")

    return available


# ── Factory de ambiente ────────────────────────────────────────────────────────

def make_env(data_paths: dict, pair_name: str = ''):
    """Factory que captura data_paths para o DummyVecEnv."""
    def _init():
        env = TradingEnvV19LSTM(data_paths=data_paths, **ENV_CONFIG)
        return env
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print("🧠 RECURRENT PPO V19.4 — TREINO MULTI-PAR MULTI-TF")
    print("="*80)
    print(f"  📅 Timestamp : {timestamp}")
    print(f"  🎯 Steps     : {TOTAL_TIMESTEPS:,}")
    print(f"  🪙  Pares     : {', '.join(p.upper() for p in PAIRS)}")
    print(f"  📐 Obs Shape : (50, 31)  —  OHLCV normalizado (V19 fix) + VecNormalize")
    print(f"  🔧 vf_coef   : {PPO_CONFIG['vf_coef']} (era 0.1 no V18 → critic agora aprende)")
    print(f"  🔧 n_steps   : {PPO_CONFIG['n_steps']} (era 2048 no V18 → gradientes mais estáveis)")
    print(f"  🔧 leverage  : {ENV_CONFIG['leverage']} (era 1.0 no V18 → alinhado com produção)")
    print(f"  🔧 lr        : {PPO_CONFIG['learning_rate']} (V19.3: era 2e-4 → clip_fraction estava 0.35)")
    print(f"  🔧 ent_coef  : {PPO_CONFIG['ent_coef']} (V19.3: era 0.05 → std divergindo 1.02→1.47)")
    print(f"  🔧 clip_range: {PPO_CONFIG['clip_range']} (V19.4: era 0.1 → clip_fraction subiu até 0.25)")
    print(f"  🔧 cooldown  : {ENV_CONFIG['trade_cooldown']} steps (V19.3: mínimo 1h entre trades)")
    print("="*80)

    # ── Dados ────────────────────────────────────────────────────────────────
    print("\n🔍 Buscando dados multi-par...")
    available_pairs = resolve_data_paths(PAIRS)

    print(f"\n📂 {len(available_pairs)} par(es) para treino:")
    for name, paths in available_pairs:
        for tf, path in paths.items():
            print(f"   {name} {tf}: {path}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n🖥️  Device: CPU  (LSTM não é suportado pelo DirectML)")

    # ── Envs ──────────────────────────────────────────────────────────────────
    print(f"\n📁 Criando {len(available_pairs)} ambiente(s)...")
    env_fns = [make_env(paths, name) for name, paths in available_pairs]
    env     = DummyVecEnv(env_fns)
    # V19.2: VecNormalize — running z-score em obs + rewards
    # Resolve value_loss explodindo: critic vê retornos de bull/bear com escala uniforme
    env     = VecNormalize(
        env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        clip_reward = 10.0,
        gamma       = PPO_CONFIG['gamma'],
    )
    n_envs  = env.num_envs

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"  ✅ {n_envs} env(s) criados  +  VecNormalize (norm_obs + norm_reward)")
    print(f"  📐 Obs: {obs_shape}  |  Action: {act_shape}")
    print(f"  📊 Rollout por iter: {n_envs} × {PPO_CONFIG['n_steps']} = {n_envs * PPO_CONFIG['n_steps']:,} steps")

    # Quick sanity check: verificar que OHLCV não está constante
    print("\n🔬 Sanity check da normalização V19:")
    test_env = env_fns[0]()
    obs, _ = test_env.reset()
    ohlcv_std = obs[:, :5].std(axis=0)
    print(f"  STD das 5 colunas OHLCV (deve ser > 0.01 para todas):")
    for i, col in enumerate(['open%', 'high%', 'low%', 'close_ret%', 'vol_ratio']):
        ok = "✅" if ohlcv_std[i] > 0.01 else "❌ PROBLEMA!"
        print(f"    col[{i}] {col:12s}: std={ohlcv_std[i]:.4f}  {ok}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("\n🏗️  Criando RecurrentPPO V19...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate = PPO_CONFIG['learning_rate'],
        n_steps       = PPO_CONFIG['n_steps'],
        batch_size    = PPO_CONFIG['batch_size'],
        n_epochs      = PPO_CONFIG['n_epochs'],
        gamma         = PPO_CONFIG['gamma'],
        gae_lambda    = PPO_CONFIG['gae_lambda'],
        clip_range    = PPO_CONFIG['clip_range'],
        ent_coef      = PPO_CONFIG['ent_coef'],
        vf_coef       = PPO_CONFIG['vf_coef'],
        max_grad_norm = PPO_CONFIG['max_grad_norm'],
        policy_kwargs = {
            'lstm_hidden_size': LSTM_CONFIG['lstm_hidden_size'],
            'n_lstm_layers':    LSTM_CONFIG['n_lstm_layers'],
            'net_arch':         LSTM_CONFIG['net_arch'],
            'activation_fn':    LSTM_CONFIG['activation_fn'],
            'ortho_init':       LSTM_CONFIG['ortho_init'],
        },
        verbose       = 1,
        device        = device,
        tensorboard_log = f"./tensorboard/recurrent_ppo_v19_multipair_{timestamp}/",
    )
    print("  ✅ Modelo criado!\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        TradingMetricsCallback(verbose=1),
        LiquidationMonitor(max_liquidations=1000, check_freq=CHECK_FREQ, verbose=1),
        PerformanceDecayMonitor(min_winrate=0.05, patience=5, verbose=1),
        ValueLossDivergenceMonitor(max_value_loss=2500, patience=3, verbose=1),
        CheckpointCallback(
            save_freq  = SAVE_FREQ,
            save_path  = "./models/",
            name_prefix= f"recurrent_ppo_v19_multipair_{timestamp}",
            save_replay_buffer = False,
            save_vecnormalize  = True,
        ),
    ]

    # ── Treino ────────────────────────────────────────────────────────────────
    print("="*80)
    print("🚀 INICIANDO TREINO V19 MULTI-PAR...")
    print("="*80)
    print(f"  Pares ativos  : {', '.join(name for name, _ in available_pairs)}")
    rollouts = TOTAL_TIMESTEPS // (n_envs * PPO_CONFIG['n_steps'])
    print(f"  Total steps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Rollouts      : ~{rollouts:,}  ({n_envs} envs × {PPO_CONFIG['n_steps']} n_steps por iter)")
    print(f"  Batch size    : {PPO_CONFIG['batch_size']}  (proporcional ao n_steps)")
    etaH = (TOTAL_TIMESTEPS // n_envs) / 1_000 * (0.04 if device == 'cuda' else 0.09)
    print(f"  ETA estimado  : ~{etaH:.0f}h ({device.upper()})")
    print(f"  TensorBoard   : tensorboard --logdir=./tensorboard/\n")
    print("  🎯 V19 THRESHOLDS DE QUALIDADE (monitorar no TensorBoard):")
    print("     50k  steps → explained_variance > 0.01  (critic começou a aprender)")
    print("     100k steps → explained_variance > 0.05  (no caminho certo)")
    print("     200k steps → STD deve estabilizar ou cair (não crescer como V18)")
    print("     clip_fraction > 0.01 em todo momento → policy atualizando\n")

    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callbacks,
            progress_bar    = True,
        )

        final_path = f"./models/recurrent_ppo_v19_multipair_{timestamp}_final.zip"
        model.save(final_path)
        vecnorm_path = f"./models/recurrent_ppo_v19_multipair_{timestamp}_vecnorm.pkl"
        env.save(vecnorm_path)

        print("\n" + "="*80)
        print("✅ TREINO V19 CONCLUÍDO!")
        print("="*80)
        print(f"  💾 Modelo  : {final_path}")
        print(f"  📊 TB      : tensorboard --logdir=./tensorboard/recurrent_ppo_v19_multipair_{timestamp}/")
        print("\n  🎯 Próximos passos:")
        print("    1. Verificar explained_variance > 0.15 no TensorBoard")
        print("    2. Verificar win_rate > 0.50 nos últimos 100k steps")
        print("    3. Atualizar MODEL_PATH no dashboard para o modelo V19")
        print("    4. Confirmar que observation.py usa _normalize_ohlcv() igual ao env")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        save_path = f"./models/recurrent_ppo_v19_multipair_{timestamp}_interrupted.zip"
        model.save(save_path)
        env.save(f"./models/recurrent_ppo_v19_multipair_{timestamp}_vecnorm.pkl")
        print(f"\n⏸️  Treino interrompido — modelo salvo: {save_path}")

    except Exception as exc:
        save_path = f"./models/recurrent_ppo_v19_multipair_{timestamp}_error.zip"
        try:
            model.save(save_path)
            print(f"\n❌ Erro: {exc}")
            print(f"   Modelo parcial salvo: {save_path}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
