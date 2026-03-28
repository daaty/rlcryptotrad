"""
Backtest sweep dos checkpoints V18 — encontra o melhor save.

Testa uma amostra estratégica (a cada ~400k steps) em todos os pares disponíveis
e exibe uma tabela de ranking ordenada por retorno total médio.
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

# ── Configuracao ambiente (igual ao treino V18) ────────────────────────────────
ENV_CONFIG = {
    'window_size':           50,
    'max_episode_steps':     2000,
    'leverage':              1.0,      # V18 treinou com 1.0
    'commission':            0.0004,
    'slippage':              0.0005,
    'position_size':         0.05,
    'use_sharpe_reward':     False,
    'enable_indicator_shaping': False,
    'random_start':          False,    # deterministico p/ comparacao justa
    'persist_balance':       False,
    'liquidation_threshold': 0.30,
}

INITIAL_BALANCE = 10_000.0

# ── Selecionar checkpoints estrategicos ───────────────────────────────────────
# Amostra a cada ~400k steps + ultimo disponivel
TARGET_STEPS = [
    40_000, 200_000, 400_000, 600_000, 800_000,
    1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000,
    2_000_000, 2_200_000, 2_400_000, 2_600_000, 2_800_000,
    3_000_000, 3_200_000, 3_400_000, 3_600_000, 3_800_000,
    4_000_000, 4_200_000, 4_400_000, 4_600_000, 4_800_000,
    4_880_000,
]

MODELS_DIR = Path("models")
DATA_DIR   = Path("data")

PAIRS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']


def find_v18_checkpoints():
    """Encontra os checkpoints V18 mais proximos dos targets."""
    all_v18 = list(MODELS_DIR.glob("recurrent_ppo_v18_multipair_*_steps.zip"))

    # Extrair numero de steps do nome
    def extract_steps(p):
        parts = p.stem.rsplit('_', 2)
        try:
            return int(parts[-2])
        except (ValueError, IndexError):
            return -1

    all_v18 = [(extract_steps(p), p) for p in all_v18 if extract_steps(p) > 0]
    all_v18.sort(key=lambda x: x[0])

    if not all_v18:
        print("ERRO: Nenhum checkpoint V18 encontrado em models/")
        sys.exit(1)

    all_steps = {s: p for s, p in all_v18}
    available = sorted(all_steps.keys())

    # Para cada target, pega o checkpoint mais proximo
    selected = {}
    for target in TARGET_STEPS:
        closest = min(available, key=lambda s: abs(s - target))
        # Evitar duplicatas
        if closest not in selected:
            selected[closest] = all_steps[closest]

    return sorted(selected.items())  # [(steps, path), ...]


def find_test_data(pair: str):
    """Encontra dados de teste para um par."""
    f15m = sorted(DATA_DIR.glob(f'test_{pair}_*_15m_*.csv'), reverse=True)
    f1h  = sorted(DATA_DIR.glob(f'test_{pair}_*_1h_*.csv'),  reverse=True)
    f4h  = sorted(DATA_DIR.glob(f'test_{pair}_*_4h_*.csv'),  reverse=True)

    if f15m and f1h and f4h:
        return {'15m': str(f15m[0]), '1h': str(f1h[0]), '4h': str(f4h[0])}
    return None


def run_backtest_silent(model, data_paths):
    """
    Roda backtest completo e retorna dict de metricas.
    Suprime prints do ambiente durante o run.
    """
    env = TradingEnvMultiTFLSTM(data_paths=data_paths, **ENV_CONFIG)

    obs, info  = env.reset()
    done       = False
    lstm_states    = None
    episode_start  = np.ones((1,), dtype=bool)
    trades         = []
    actions_hist   = []

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=False,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = np.zeros((1,), dtype=bool)

        act_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        actions_hist.append(act_val)
        if info.get('trade_executed'):
            trades.append({'pnl': info.get('pnl', 0), '_side': info.get('_side', 0)})

    fin_info = env._get_info()
    final    = fin_info['equity']
    ret_pct  = (final - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr       = fin_info['win_rate']
    n_trades = fin_info['trades']

    all_pnl    = [t['pnl'] for t in trades]
    wins_pnl   = sum(p for p in all_pnl if p > 0)
    losses_pnl = abs(sum(p for p in all_pnl if p < 0))
    pf         = wins_pnl / losses_pnl if losses_pnl > 0 else (999.0 if wins_pnl > 0 else 0.0)

    actions  = np.array(actions_hist) if actions_hist else np.array([0.0])
    n_long   = sum(1 for a in actions_hist if a >  0.1)
    n_short  = sum(1 for a in actions_hist if a < -0.1)
    n_flat   = sum(1 for a in actions_hist if -0.1 <= a <= 0.1)
    pct_flat = n_flat / len(actions_hist) * 100 if actions_hist else 100.0

    return {
        'return':        ret_pct,
        'win_rate':      wr,
        'trades':        n_trades,
        'profit_factor': pf,
        'final_equity':  final,
        'pct_flat':      pct_flat,
        'action_std':    float(np.std(actions)),
        'action_mean':   float(np.mean(actions)),
        'liq':           fin_info['liquidations'],
    }


def print_table(results):
    """
    results: list de dicts com 'steps', 'path', 'pairs' (dict par->metricas),
             'avg_return', 'avg_wr', 'avg_pf', 'avg_trades'
    """
    # Ordenar por retorno medio
    results.sort(key=lambda r: r['avg_return'], reverse=True)

    W   = 100
    SEP = "=" * W

    print(f"\n{SEP}")
    print("  RANKING BACKTEST V18 — TODOS OS PARES".center(W))
    print(SEP)
    header = f"  {'Steps':>9}  {'Retorno%':>9}  {'WinRate':>8}  " \
             f"{'ProfFact':>9}  {'Trades':>7}  {'%Flat':>6}  {'Modelo'}"
    print(header)
    print("-" * W)

    for i, r in enumerate(results):
        medal = ["1st", "2nd", "3rd"][i] if i < 3 else f" {i+1:2d}."
        print(f"  {medal:>4}  {r['steps']:>9,}  {r['avg_return']:>+8.2f}%  "
              f"{r['avg_wr']:>7.1f}%  {r['avg_pf']:>9.2f}  "
              f"{r['avg_trades']:>7.0f}  {r['avg_flat']:>5.1f}%  "
              f"{r['path'].name[:55]}")

    print(SEP)

    # Detalhe por par dos top-3
    print(f"\n  DETALHE POR PAR — TOP 3")
    for i, r in enumerate(results[:3]):
        print(f"\n  [{i+1}] {r['steps']:,} steps — {r['path'].name}")
        print(f"  {'Par':<10} {'Retorno':>9} {'WinRate':>8} {'Trades':>8} {'ProfFact':>9}")
        print(f"  {'-'*46}")
        for pair, m in r['pairs'].items():
            if m:
                print(f"  {pair.upper():<10} {m['return']:>+8.2f}%  {m['win_rate']:>7.1f}%  "
                      f"{m['trades']:>7}  {m['profit_factor']:>9.2f}")
            else:
                print(f"  {pair.upper():<10}  SEM DADOS")

    print(f"\n{SEP}\n")


def main():
    print("=" * 70)
    print("  BACKTEST SWEEP V18 — ENCONTRANDO MELHOR CHECKPOINT")
    print("=" * 70)

    # Pares com dados
    test_pairs = {}
    for pair in PAIRS:
        dp = find_test_data(pair)
        if dp:
            test_pairs[pair] = dp
            print(f"  [OK] {pair.upper():8s}  {Path(dp['15m']).name}")
        else:
            print(f"  [--] {pair.upper():8s}  sem dados de teste")

    if not test_pairs:
        print("\nERRO: Nenhum dado de teste encontrado.")
        sys.exit(1)

    checkpoints = find_v18_checkpoints()
    print(f"\n  {len(checkpoints)} checkpoints selecionados para sweep")
    print(f"  {len(test_pairs)} pares de teste: {', '.join(test_pairs)}")
    print(f"  Total de runs: {len(checkpoints) * len(test_pairs)}\n")

    # Suprimir prints do env durante sweep
    _real_stdout = sys.stdout

    all_results = []
    total_runs  = len(checkpoints) * len(test_pairs)
    run_idx     = 0

    for steps, ckpt_path in checkpoints:
        _real_stdout.write(f"\n  [{steps:>9,}]  Carregando {ckpt_path.name} ...\n")
        _real_stdout.flush()

        # Carregar modelo uma vez e testar em todos os pares
        sys.stdout = io.StringIO()  # silenciar
        try:
            model = RecurrentPPO.load(str(ckpt_path))
        except Exception as e:
            sys.stdout = _real_stdout
            print(f"    ERRO ao carregar: {e}")
            continue
        finally:
            sys.stdout = _real_stdout

        pair_metrics = {}
        for pair, dpaths in test_pairs.items():
            run_idx += 1
            _real_stdout.write(f"    [{run_idx:3d}/{total_runs}] {pair.upper()} ... ")
            _real_stdout.flush()

            sys.stdout = io.StringIO()  # silenciar env prints
            try:
                m = run_backtest_silent(model, dpaths)
                pair_metrics[pair] = m
                sys.stdout = _real_stdout
                _real_stdout.write(f"{m['return']:+.2f}%  wr={m['win_rate']:.1f}%  "
                                   f"t={m['trades']}  pf={m['profit_factor']:.2f}\n")
            except KeyboardInterrupt:
                sys.stdout = _real_stdout
                pair_metrics[pair] = None
                _real_stdout.write("INTERROMPIDO (ignorando, continua...)\n")
            except Exception as e:
                sys.stdout = _real_stdout
                pair_metrics[pair] = None
                _real_stdout.write(f"ERRO: {e}\n")

        valid = [m for m in pair_metrics.values() if m is not None]
        if valid:
            avg_return  = np.mean([m['return']        for m in valid])
            avg_wr      = np.mean([m['win_rate']       for m in valid])
            avg_pf      = np.mean([m['profit_factor']  for m in valid])
            avg_trades  = np.mean([m['trades']         for m in valid])
            avg_flat    = np.mean([m['pct_flat']       for m in valid])
        else:
            avg_return = avg_wr = avg_pf = avg_trades = avg_flat = 0.0

        all_results.append({
            'steps':      steps,
            'path':       ckpt_path,
            'pairs':      pair_metrics,
            'avg_return': avg_return,
            'avg_wr':     avg_wr,
            'avg_pf':     avg_pf,
            'avg_trades': avg_trades,
            'avg_flat':   avg_flat,
        })

    print_table(all_results)

    # Salvar resultado em txt
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = f"backtest_v18_sweep_{ts}.txt"
    with open(out, 'w', encoding='utf-8') as f:
        # Redirecionar print para arquivo
        old = sys.stdout
        sys.stdout = f
        print_table(all_results)
        sys.stdout = old
    print(f"  Resultado salvo: {out}\n")


if __name__ == "__main__":
    main()
