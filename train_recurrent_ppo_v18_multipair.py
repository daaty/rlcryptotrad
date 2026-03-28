"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         🧠 TREINAR RECURRENT PPO V18 — LSTM MULTI-PAR MULTI-TF              ║
║                                                                              ║
║  📋 ESTRATÉGIA V18: Um único modelo treinado em BTC + ETH + SOL + BNB       ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  🎯 OBJETIVO: Política generalista que opera qualquer par de futuros         ║
║                                                                              ║
║  🔧 ARQUITETURA:                                                             ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  ✨ RecurrentPPO (sb3-contrib) — mesma base do V17.7                        ║
║  ✨ MlpLstmPolicy: LSTM [256, 256] → MLP [256, 256]                         ║
║  ✨ Obs sequenciais (50, 31): scale-invariant → funciona em qualquer par     ║
║  ✨ DummyVecEnv com 4 envs em paralelo (1 por par)                          ║
║                                                                              ║
║  📐 DETALHES TÉCNICOS:                                                       ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • Cada env usa dados do seu par (BTC, ETH, SOL, BNB)                       ║
║  • Rollout buffer = 4 × 2048 = 8 192 steps/iter (4× mais diversidade)      ║
║  • Pesos da LSTM são COMPARTILHADOS — o modelo aprende padrões genéricos    ║
║  • LSTM hidden states são INDEPENDENTES por env (memória por par)           ║
║  • Total: 6M steps  →  ~732 rollouts (idem V17.7 em número de updates)     ║
║                                                                              ║
║  🆕 MELHORIAS VS V17.7:                                                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  ✅ Treinado em 4 ativos → generaliza, não memoriza 1 par                   ║
║  ✅ Dataset 4× maior → menos overfitting                                    ║
║  ✅ Correlação cruzada (BTC↔ETH) vista nos rollouts → aprende regimes macro  ║
║  ✅ Dashboard opera os 4 pares com 1 modelo                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
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

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environment.trading_env_multi_tf_lstm import TradingEnvMultiTFLSTM
from callbacks.trading_metrics import (
    TradingMetricsCallback,
    LiquidationMonitor,
    PerformanceDecayMonitor,
    ValueLossDivergenceMonitor,
)


# ── Pares alvo ────────────────────────────────────────────────────────────────
PAIRS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
# SB3 conta total_timesteps como soma através de TODOS os envs.
# com 4 envs e n_steps=2048: cada rollout = 4×2048=8192 steps.
# 6M / 8192 ≈ 732 rollouts  (igual ao V17.7 com 1.5M / 2048 ≈ 732 rollouts)
# → mesmo número de atualizações de política, mas cada update vê os 4 pares.
TOTAL_TIMESTEPS = 6_000_000
SAVE_FREQ       = 10_000
CHECK_FREQ      = 10_000

ENV_CONFIG = {
    'window_size':        50,
    'max_episode_steps':  2000,
    'leverage':           1.0,
    'commission':         0.0004,
    'slippage':           0.0005,
    'position_size':      0.05,
    'use_sharpe_reward':  False,
    'enable_indicator_shaping': False,
    'random_start':       True,
    'persist_balance':    False,
    'liquidation_threshold': 0.30,
}

PPO_CONFIG = {
    'learning_rate': 2e-4,
    'n_steps':       2048,      # por env → total rollout = 4 × 2048 = 8 192
    'batch_size':    128,       # ligeiramente maior (mais envs)
    'n_epochs':      4,
    'gamma':         0.95,
    'gae_lambda':    0.9,
    'clip_range':    0.2,
    'ent_coef':      0.03,
    'vf_coef':       0.1,
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
    """
    Procura os arquivos de dados mais recentes para um par.

    Tenta primeiro o formato multi-par (collect_multi_pair_mtf.py),
    depois o formato legado (collect_multi_timeframe.py, apenas BTC).

    Returns None se não encontrar dados completos.
    """
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
    """
    Retorna lista de data_paths dicts para todos os pares encontrados.
    Avisa se algum par estiver faltando e treina com os disponíveis.
    """
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
        print("    Para treinar com todos os pares, execute: python collect_multi_pair_mtf.py\n")

    return available


# ── Factory de ambiente ────────────────────────────────────────────────────────

def make_env(data_paths: dict, pair_name: str = ''):
    """Factory que captura data_paths para o DummyVecEnv."""
    def _init():
        env = TradingEnvMultiTFLSTM(data_paths=data_paths, **ENV_CONFIG)
        return env
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print("🧠 RECURRENT PPO V18 — TREINO MULTI-PAR MULTI-TF")
    print("="*80)
    print(f"  📅 Timestamp : {timestamp}")
    print(f"  🎯 Steps     : {TOTAL_TIMESTEPS:,}")
    print(f"  🪙  Pares     : {', '.join(p.upper() for p in PAIRS)}")
    print(f"  📐 Obs Shape : (50, 31)  —  scale-invariant para qualquer par")
    print(f"  🏗️  Envs      : 1 por par via DummyVecEnv")
    print(f"  💡 Inovação  : pesos compartilhados → política generalista")
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
    n_envs  = env.num_envs

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"  ✅ {n_envs} env(s) criados")
    print(f"  📐 Obs: {obs_shape}  |  Action: {act_shape}")
    print(f"  📊 Rollout por iter: {n_envs} × {PPO_CONFIG['n_steps']} = {n_envs * PPO_CONFIG['n_steps']:,} steps")

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("\n🏗️  Criando RecurrentPPO V18...")
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
        tensorboard_log = f"./tensorboard/recurrent_ppo_v18_multipair_{timestamp}/",
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
            name_prefix= f"recurrent_ppo_v18_multipair_{timestamp}",
            save_replay_buffer = False,
            save_vecnormalize  = True,
        ),
    ]

    # ── Treino ────────────────────────────────────────────────────────────────
    print("="*80)
    print("🚀 INICIANDO TREINO V18 MULTI-PAR...")
    print("="*80)
    print(f"  Pares ativos  : {', '.join(name for name, _ in available_pairs)}")
    rollouts = TOTAL_TIMESTEPS // (n_envs * PPO_CONFIG['n_steps'])
    print(f"  Total steps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Rollouts      : ~{rollouts:,}  ({n_envs} envs × {PPO_CONFIG['n_steps']} n_steps por iter)")
    etaH = (TOTAL_TIMESTEPS // n_envs) / 1_000 * (0.04 if device == 'cuda' else 0.09)
    print(f"  ETA estimado  : ~{etaH:.0f}h ({device.upper()})")
    print(f"  TensorBoard   : tensorboard --logdir=./tensorboard/\n")

    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callbacks,
            progress_bar    = True,
        )

        final_path = f"./models/recurrent_ppo_v18_multipair_{timestamp}_final.zip"
        model.save(final_path)

        print("\n" + "="*80)
        print("✅ TREINO V18 CONCLUÍDO!")
        print("="*80)
        print(f"  💾 Modelo  : {final_path}")
        print(f"  📊 TB      : tensorboard --logdir=./tensorboard/recurrent_ppo_v18_multipair_{timestamp}/")
        print("\n  🎯 Próximos passos:")
        print("    1. Avaliar no TensorBoard (win rate por par)")
        print("    2. Atualizar MODEL_PATH no dashboard para o modelo V18")
        print("    3. Testar em paper trading antes de ir para real")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        save_path = f"./models/recurrent_ppo_v18_multipair_{timestamp}_interrupted.zip"
        model.save(save_path)
        print(f"\n⏸️  Treino interrompido — modelo salvo: {save_path}")

    except Exception as exc:
        save_path = f"./models/recurrent_ppo_v18_multipair_{timestamp}_error.zip"
        try:
            model.save(save_path)
            print(f"\n❌ Erro: {exc}")
            print(f"   Modelo parcial salvo: {save_path}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
