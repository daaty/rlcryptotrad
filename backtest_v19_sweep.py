"""
Backtest sweep dos checkpoints V19.2 (VecNormalize).

Testa todos os checkpoints do run 195359 em cada par de treino
(BTC, ETH, SOL, BNB) e exibe ranking por retorno.

Cada checkpoint .zip deve ter um .pkl de VecNormalize correspondente.
"""

import sys
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERRO: pip install sb3-contrib")
    sys.exit(1)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.trading_env_v19_lstm import TradingEnvV19LSTM

# ── Config ambiente (idêntico ao treino V19.2) ────────────────────────────────
ENV_CONFIG = {
    'window_size':           50,
    'max_episode_steps':     2000,
    'leverage':              1.5,
    'commission':            0.0004,
    'slippage':              0.0005,
    'position_size':         0.15,
    'use_sharpe_reward':     False,
    'enable_indicator_shaping': False,
    'random_start':          False,   # backtest sequencial
    'persist_balance':       False,
    'liquidation_threshold': 0.30,
}

INITIAL_BALANCE = 10_000.0
MODELS_DIR      = Path("models")
DATA_DIR        = Path("data")
RUN_ID          = "20260306_195359"   # run V19.2 com VecNormalize

PAIRS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']


# ── Dados ─────────────────────────────────────────────────────────────────────

def find_test_data(pair: str) -> dict | None:
    """Retorna paths de teste ou treino (fallback) para o par."""
    def latest(pattern):
        files = sorted(DATA_DIR.glob(pattern), reverse=True)
        return files[0] if files else None

    # Tenta dados de teste primeiro, cai em treino se não existir
    f15m = latest(f'test_{pair}_*_15m_*.csv') or latest(f'train_{pair}_*_15m_*.csv')
    f1h  = latest(f'test_{pair}_*_1h_*.csv')  or latest(f'train_{pair}_*_1h_*.csv')
    f4h  = latest(f'test_{pair}_*_4h_*.csv')  or latest(f'train_{pair}_*_4h_*.csv')

    if f15m and f1h and f4h:
        return {'15m': str(f15m), '1h': str(f1h), '4h': str(f4h)}
    return None


# ── Checkpoints ───────────────────────────────────────────────────────────────

def find_v19_checkpoints() -> list:
    """Lista todos os checkpoints do run V19.2 com VecNormalize correspondente."""
    zips = sorted(MODELS_DIR.glob(f"recurrent_ppo_v19_multipair_{RUN_ID}_*_steps.zip"))

    checkpoints = []
    for z in zips:
        # Extrair steps do nome
        stem_parts = z.stem.rsplit('_', 2)
        try:
            steps = int(stem_parts[-2])
        except (ValueError, IndexError):
            continue

        # Procurar VecNormalize correspondente
        pkl = MODELS_DIR / f"recurrent_ppo_v19_multipair_{RUN_ID}_vecnormalize_{steps}_steps.pkl"
        if not pkl.exists():
            print(f"  ⚠️  Sem VecNormalize para {z.name} — pulando")
            continue

        checkpoints.append({
            'steps': steps,
            'zip':   z,
            'pkl':   pkl,
        })

    return sorted(checkpoints, key=lambda c: c['steps'])


# ── Backtest único ──────────────────────────────────────────────────────────

def run_backtest(zip_path: Path, pkl_path: Path, data_paths: dict) -> dict | None:
    """Executa backtest de um checkpoint num par. Retorna métricas ou None."""
    try:
        # Criar env de backtest com VecNormalize em modo avaliação (training=False)
        env_fn = lambda: TradingEnvV19LSTM(data_paths=data_paths, **ENV_CONFIG)
        raw_env = DummyVecEnv([env_fn])
        env = VecNormalize.load(str(pkl_path), raw_env)
        env.training = False      # não atualiza estatísticas durante avaliação
        env.norm_reward = False   # não normaliza reward na avaliação

        model = RecurrentPPO.load(str(zip_path), env=env, device='cpu')

        obs = env.reset()
        done = np.array([False])
        lstm_st   = None
        ep_start  = np.ones((1,), dtype=bool)
        trades    = []
        act_hist  = []

        while not done[0]:
            action, lstm_st = model.predict(
                obs, state=lstm_st, episode_start=ep_start, deterministic=True
            )
            obs, _, done, info = env.step(action)
            ep_start = np.zeros((1,), dtype=bool)

            # action shape: (n_envs, action_dim) = (1, 1) com VecEnv
            act_val = float(np.asarray(action).ravel()[0])
            act_hist.append(act_val)
            if info[0].get('trade_executed'):
                trades.append({'pnl': info[0].get('pnl', 0)})

        fin = info[0]
        ret = (fin['equity'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        all_p = [t['pnl'] for t in trades]
        wins  = sum(p for p in all_p if p > 0)
        loss  = abs(sum(p for p in all_p if p < 0))
        pf    = wins / loss if loss > 0 else (999.0 if wins > 0 else 0.0)

        # Distribuição de ações
        act_arr = np.array(act_hist)
        long_pct  = (act_arr > 0.1).mean() * 100
        short_pct = (act_arr < -0.1).mean() * 100
        flat_pct  = 100 - long_pct - short_pct

        env.close()
        return {
            'return':        ret,
            'win_rate':      fin.get('win_rate', 0),
            'trades':        fin.get('trades', 0),
            'profit_factor': pf,
            'long_pct':      long_pct,
            'short_pct':     short_pct,
            'flat_pct':      flat_pct,
            'equity':        fin.get('equity', INITIAL_BALANCE),
        }

    except Exception as exc:
        print(f"    ❌ Erro: {exc}")
        traceback.print_exc()
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(f"backtest_v19_sweep_{ts}.txt")

    print(f"\n{'='*80}")
    print("  BACKTEST SWEEP V19.2 — MULTI-PAR (BTC/ETH/SOL/BNB)")
    print(f"{'='*80}")
    print(f"  Run ID  : {RUN_ID}")
    print(f"  Output  : {out_file}\n")

    # Dados de cada par
    pair_data = {}
    for pair in PAIRS:
        paths = find_test_data(pair)
        if paths:
            tag = '(treino)' if 'train' in paths['15m'] else '(teste)'
            print(f"  ✅ {pair.upper():8s}  {Path(paths['15m']).name}  {tag}")
            pair_data[pair] = paths
        else:
            print(f"  ❌ {pair.upper():8s}  dados não encontrados — par ignorado")

    if not pair_data:
        print("\n❌ Nenhum dado disponível. Abortando.")
        sys.exit(1)

    # Checkpoints
    checkpoints = find_v19_checkpoints()
    if not checkpoints:
        print(f"\n❌ Nenhum checkpoint V19.2 encontrado em models/ (run {RUN_ID})")
        sys.exit(1)

    print(f"\n  Checkpoints encontrados: {len(checkpoints)}")
    print(f"  Steps: {[c['steps'] for c in checkpoints]}\n")

    # ── Executar backtests ───────────────────────────────────────────────────
    all_results = []   # {steps, pair, metrics}

    for ckpt in checkpoints:
        steps = ckpt['steps']
        print(f"\n{'─'*70}")
        print(f"  ▶ {steps:,} steps")
        print(f"{'─'*70}")

        for pair in pair_data:
            print(f"    {pair.upper():8s} ...", end='', flush=True)
            m = run_backtest(ckpt['zip'], ckpt['pkl'], pair_data[pair])
            if m:
                print(f"  ret={m['return']:+.2f}%  wr={m['win_rate']:.1f}%  "
                      f"trades={m['trades']}  pf={m['profit_factor']:.2f}  "
                      f"L{m['long_pct']:.0f}%/S{m['short_pct']:.0f}%/F{m['flat_pct']:.0f}%")
                all_results.append({'steps': steps, 'pair': pair.upper(), 'metrics': m})
            else:
                print("  ERRO")
                all_results.append({'steps': steps, 'pair': pair.upper(), 'metrics': None})

    # ── Relatório ─────────────────────────────────────────────────────────────
    lines = []
    W   = 100
    SEP = "=" * W

    lines.append(f"\n{SEP}")
    lines.append(f"  BACKTEST SWEEP V19.2  —  run {RUN_ID}  —  {ts}".center(W))
    lines.append(SEP)

    # Tabela por par
    for pair in [p.upper() for p in pair_data]:
        pair_results = [r for r in all_results if r['pair'] == pair and r['metrics']]
        pair_results.sort(key=lambda r: r['steps'])

        lines.append(f"\n  PAR: {pair}")
        lines.append(f"  {'Steps':>9}  {'Retorno':>8}  {'WinRate':>8}  {'ProfFact':>9}  "
                     f"{'Trades':>7}  {'Long%':>6}  {'Short%':>7}  {'Flat%':>6}")
        lines.append("  " + "-"*76)

        for r in pair_results:
            m = r['metrics']
            lines.append(
                f"  {r['steps']:>9,}  {m['return']:>+7.2f}%  {m['win_rate']:>7.1f}%  "
                f"{m['profit_factor']:>9.2f}  {m['trades']:>7}  "
                f"{m['long_pct']:>5.0f}%  {m['short_pct']:>6.0f}%  {m['flat_pct']:>5.0f}%"
            )

    # Ranking geral por retorno médio por step
    lines.append(f"\n{SEP}")
    lines.append("  RANKING GERAL — RETORNO MÉDIO ENTRE PARES".center(W))
    lines.append(SEP)
    lines.append(f"  {'#':>3}  {'Steps':>9}  {'Ret médio':>10}  {'Pares positivos':>16}  Detalhes")
    lines.append("  " + "-"*76)

    step_summaries = {}
    for r in all_results:
        steps = r['steps']
        if steps not in step_summaries:
            step_summaries[steps] = []
        if r['metrics']:
            step_summaries[steps].append(r['metrics']['return'])

    ranking = []
    for steps, rets in step_summaries.items():
        if rets:
            avg = np.mean(rets)
            pos = sum(1 for r in rets if r > 0)
            ranking.append({'steps': steps, 'avg_ret': avg, 'pos': pos,
                            'n': len(rets), 'rets': rets})

    ranking.sort(key=lambda x: x['avg_ret'], reverse=True)

    for i, r in enumerate(ranking):
        medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"  {i+1}."
        pair_detail = "  ".join(f"{PAIRS[j].upper()[:3]}={r['rets'][j]:+.2f}%" 
                                for j in range(len(r['rets'])))
        lines.append(
            f"  {medal:>4}  {r['steps']:>9,}  {r['avg_ret']:>+9.2f}%  "
            f"{r['pos']}/{r['n']} positivos  {pair_detail}"
        )

    lines.append(SEP)

    # Melhor checkpoint
    if ranking:
        best = ranking[0]
        lines.append(f"\n  ★ MELHOR CHECKPOINT: {best['steps']:,} steps  "
                     f"(ret médio {best['avg_ret']:+.2f}%, {best['pos']}/{best['n']} pares positivos)")

    lines.append(SEP + "\n")

    report = "\n".join(lines)
    print(report)

    out_file.write_text(report, encoding='utf-8')
    print(f"\n  💾 Relatório salvo: {out_file}")


if __name__ == "__main__":
    main()
