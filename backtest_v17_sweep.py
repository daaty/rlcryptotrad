"""
Backtest sweep dos checkpoints V17 — compara todas as runs.

Testa o checkpoint FINAL de cada run V17 + checkpoints estratégicos
da run principal (030417), exibindo ranking por retorno.
"""

import sys
import io
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERRO: pip install sb3-contrib")
    sys.exit(1)

from src.environment.trading_env_multi_tf_lstm import TradingEnvMultiTFLSTM

# ── Config ambiente (igual ao treino V17) ─────────────────────────────────────
ENV_CONFIG = {
    'window_size':           50,
    'max_episode_steps':     2000,
    'leverage':              1.0,
    'commission':            0.0004,
    'slippage':              0.0005,
    'position_size':         0.05,
    'use_sharpe_reward':     False,
    'enable_indicator_shaping': False,
    'random_start':          False,
    'persist_balance':       False,
    'liquidation_threshold': 0.30,
}

INITIAL_BALANCE = 10_000.0
MODELS_DIR      = Path("models")
DATA_DIR        = Path("data")

# V17 usa apenas BTC (single pair)
PAIR = 'btcusdt'


def find_test_data():
    f15m = sorted(DATA_DIR.glob(f'test_{PAIR}_*_15m_*.csv'), reverse=True)
    f1h  = sorted(DATA_DIR.glob(f'test_{PAIR}_*_1h_*.csv'),  reverse=True)
    f4h  = sorted(DATA_DIR.glob(f'test_{PAIR}_*_4h_*.csv'),  reverse=True)
    if f15m and f1h and f4h:
        return {'15m': str(f15m[0]), '1h': str(f1h[0]), '4h': str(f4h[0])}
    return None


def find_v17_checkpoints_to_test():
    """
    Para cada run V17:
      - Checkpoint final (max steps) → representa o melhor da run
      - Checkpoints estratégicos da run 030417 (referência)
    """
    all_v17 = list(MODELS_DIR.glob("recurrent_ppo_v17_lstm_*_steps.zip"))

    def extract_key(p):
        # nome: recurrent_ppo_v17_lstm_YYYYMMDD_HHMMSS_NNNNNNN_steps.zip
        parts = p.stem.rsplit('_', 2)
        try:
            return parts[0], int(parts[-2])   # (run_prefix, steps)
        except (ValueError, IndexError):
            return None, -1

    # Agrupar por run
    runs = {}
    for p in all_v17:
        prefix, steps = extract_key(p)
        if prefix and steps > 0:
            if prefix not in runs:
                runs[prefix] = []
            runs[prefix].append((steps, p))

    # Ordenar cada run
    for prefix in runs:
        runs[prefix].sort(key=lambda x: x[0])

    checkpoints_to_test = []

    # Para a run principal (030417): testar vários steps
    main_run = "recurrent_ppo_v17_lstm_20260221_030417"
    main_targets = [100_000, 300_000, 600_000, 1_000_000, 1_500_000]

    for prefix, ckpts in sorted(runs.items()):
        available_steps = {s: p for s, p in ckpts}

        if prefix == main_run:
            # Adicionar targets estratégicos + final
            avail = sorted(available_steps.keys())
            used = set()
            for target in main_targets:
                closest = min(avail, key=lambda s: abs(s - target))
                if closest not in used:
                    used.add(closest)
                    checkpoints_to_test.append({
                        'run':   prefix.replace('recurrent_ppo_v17_lstm_', ''),
                        'steps': closest,
                        'path':  available_steps[closest],
                        'note':  '★ main' if closest == 600_000 else '',
                    })
            # Garantir o final também
            max_step = max(avail)
            if max_step not in used:
                checkpoints_to_test.append({
                    'run':   prefix.replace('recurrent_ppo_v17_lstm_', ''),
                    'steps': max_step,
                    'path':  available_steps[max_step],
                    'note':  'final',
                })
        else:
            # Outras runs: só o checkpoint final
            max_step, max_path = ckpts[-1]
            checkpoints_to_test.append({
                'run':   prefix.replace('recurrent_ppo_v17_lstm_', ''),
                'steps': max_step,
                'path':  max_path,
                'note':  f'(apenas final — {len(ckpts)} ckpts)',
            })

    return checkpoints_to_test


def run_backtest_silent(model, data_paths):
    env = TradingEnvMultiTFLSTM(data_paths=data_paths, **ENV_CONFIG)
    obs, _    = env.reset()
    done      = False
    lstm_st   = None
    ep_start  = np.ones((1,), dtype=bool)
    trades    = []
    act_hist  = []

    while not done:
        action, lstm_st = model.predict(obs, state=lstm_st,
                                        episode_start=ep_start,
                                        deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)
        done     = terminated or truncated
        ep_start = np.zeros((1,), dtype=bool)

        act_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        act_hist.append(act_val)
        if info.get('trade_executed'):
            trades.append({'pnl': info.get('pnl', 0)})

    fin   = env._get_info()
    ret   = (fin['equity'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    all_p = [t['pnl'] for t in trades]
    wins  = sum(p for p in all_p if p > 0)
    loss  = abs(sum(p for p in all_p if p < 0))
    pf    = wins / loss if loss > 0 else (999.0 if wins > 0 else 0.0)

    return {
        'return':        ret,
        'win_rate':      fin['win_rate'],
        'trades':        fin['trades'],
        'profit_factor': pf,
    }


def print_table(results, out=None):
    target = out or sys.stdout
    W   = 100
    SEP = "=" * W

    target.write(f"\n{SEP}\n")
    target.write("  RANKING BACKTEST V17 — TODAS AS RUNS (BTC)\n".center(W) + "\n")
    target.write(f"{SEP}\n")
    header = f"  {'#':>3}  {'Run (data_hora)':>22}  {'Steps':>9}  {'Retorno':>8}  " \
             f"{'WinRate':>8}  {'ProfFact':>9}  {'Trades':>7}  Nota\n"
    target.write(header)
    target.write("-" * W + "\n")

    results_sorted = sorted(results, key=lambda r: (r['metrics']['return'] if r.get('metrics') else -999), reverse=True)
    for i, r in enumerate(results_sorted):
        medal = ["1st", "2nd", "3rd"][i] if i < 3 else f" {i+1:2d}."
        m     = r['metrics']
        if m:
            target.write(
                f"  {medal:>4}  {r['run']:>22}  {r['steps']:>9,}  "
                f"{m['return']:>+7.2f}%  {m['win_rate']:>7.1f}%  "
                f"{m['profit_factor']:>9.2f}  {m['trades']:>7}  {r['note']}\n"
            )
        else:
            target.write(f"  {medal:>4}  {r['run']:>22}  {r['steps']:>9,}  ERRO\n")

    target.write(f"{SEP}\n")

    # Recomendacao de limpeza
    target.write("\n  RECOMENDAÇÃO DE LIMPEZA:\n")
    target.write("-" * W + "\n")

    # Top 3
    keep = [r for r in results_sorted[:3] if r.get('metrics') and r['metrics']['trades'] > 50]
    bad  = [r for r in results_sorted if r.get('metrics') and r['metrics']['return'] < 0]

    target.write(f"  MANTER  ({len(keep)} checkpoints testados como top-3):\n")
    for r in keep:
        target.write(f"    + {r['path'].name}  →  {r['metrics']['return']:+.2f}%\n")

    target.write(f"\n  PIORES  (retorno negativo — candidatos a delete):\n")
    if bad:
        for r in bad:
            target.write(f"    - {r['path'].name}  →  {r['metrics']['return']:+.2f}%\n")
    else:
        target.write("    (nenhum com retorno negativo)\n")

    target.write(f"\n{SEP}\n")


def main():
    print("=" * 70)
    print("  BACKTEST SWEEP V17 — COMPARANDO TODAS AS RUNS")
    print("=" * 70)

    data_paths = find_test_data()
    if not data_paths:
        print(f"\nERRO: Dados de teste para {PAIR} não encontrados em data/")
        sys.exit(1)

    print(f"\n  Par de teste: {PAIR.upper()}")
    print(f"  15m: {Path(data_paths['15m']).name}")
    print(f"   1h: {Path(data_paths['1h']).name}")
    print(f"   4h: {Path(data_paths['4h']).name}\n")

    candidates = find_v17_checkpoints_to_test()
    print(f"  {len(candidates)} checkpoints para testar:\n")
    for c in candidates:
        print(f"    {c['run']:>22}  {c['steps']:>9,}  {c['note']}")

    print()
    _real_stdout = sys.stdout
    all_results  = []

    for i, c in enumerate(candidates):
        _real_stdout.write(f"\n[{i+1:2d}/{len(candidates)}] {c['run']}  {c['steps']:>9,} steps  {c['note']}\n")
        _real_stdout.write(f"    Carregando {c['path'].name} ...\n")
        _real_stdout.flush()

        sys.stdout = io.StringIO()
        try:
            model = RecurrentPPO.load(str(c['path']))
        except Exception as e:
            sys.stdout = _real_stdout
            print(f"    ERRO ao carregar: {e}")
            all_results.append({**c, 'metrics': None})
            continue
        finally:
            sys.stdout = _real_stdout

        sys.stdout = io.StringIO()
        try:
            m = run_backtest_silent(model, data_paths)
            sys.stdout = _real_stdout
            _real_stdout.write(
                f"    → retorno={m['return']:+.2f}%  wr={m['win_rate']:.1f}%  "
                f"trades={m['trades']}  pf={m['profit_factor']:.2f}\n"
            )
        except KeyboardInterrupt:
            sys.stdout = _real_stdout
            _real_stdout.write("    INTERROMPIDO\n")
            m = None
        except Exception as e:
            sys.stdout = _real_stdout
            _real_stdout.write(f"    ERRO: {e}\n")
            m = None
        finally:
            sys.stdout = _real_stdout

        all_results.append({**c, 'metrics': m})

    # Tabela final
    print_table(all_results)

    # Salvar
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"backtest_v17_sweep_{ts}.txt"
    with open(out, 'w', encoding='utf-8') as f:
        print_table(all_results, out=f)
    print(f"\n  Resultado salvo: {out}\n")


if __name__ == "__main__":
    main()
