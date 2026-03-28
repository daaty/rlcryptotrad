"""
Backtest ESTENDIDO V19.2 — valida qualidade real do checkpoint 320k.

Diferenças vs backtest_v19_sweep.py:
  - max_episode_steps = tamanho total do dataset (epísódio completo, 7 meses)
  - Fecha posição aberta no fim e registra como trade
  - Separa P&L de trades FECHADOS vs unrealized no fim
  - Equity curve detalhada a cada N steps
  - Testa checkpoints: 280k / 320k / 360k / 400k

Hipótese a validar:
  "pf < 1.0 com retorno positivo → retorno vem de posição aberta no fim?"
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERRO: pip install sb3-contrib"); sys.exit(1)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.trading_env_v19_lstm import TradingEnvV19LSTM

# ── Configuração ──────────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
DATA_DIR      = Path("data")
RUN_ID        = "20260306_195359"
INITIAL_BAL   = 10_000.0

# Checkpoints a testar (melhores 4 do sweep normal)
CHECKPOINTS   = [280_000, 320_000, 360_000, 400_000]

PAIRS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']

# Cobre TODO o dataset de teste (20727 candles)
MAX_EP_STEPS  = 20_800

ENV_CONFIG = dict(
    window_size    = 50,
    max_episode_steps = MAX_EP_STEPS,
    leverage       = 1.5,
    commission     = 0.0004,
    slippage       = 0.0005,
    position_size  = 0.15,
    use_sharpe_reward         = False,
    enable_indicator_shaping  = False,
    random_start   = False,
    persist_balance = False,
    liquidation_threshold = 0.30,
)


# ── Dados ─────────────────────────────────────────────────────────────────────

def get_test_data(pair: str) -> dict | None:
    def latest(pat):
        files = sorted(DATA_DIR.glob(pat), reverse=True)
        return files[0] if files else None

    f15m = latest(f'test_{pair}_*_15m_*.csv')
    f1h  = latest(f'test_{pair}_*_1h_*.csv')
    f4h  = latest(f'test_{pair}_*_4h_*.csv')

    if f15m and f1h and f4h:
        return {'15m': str(f15m), '1h': str(f1h), '4h': str(f4h)}
    return None


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_extended_backtest(steps: int, pair: str, data_paths: dict) -> dict | None:
    """Roda episódio completo (~7 meses) e detalha trades fechados vs unrealized."""
    try:
        pkl  = MODELS_DIR / f"recurrent_ppo_v19_multipair_{RUN_ID}_vecnormalize_{steps}_steps.pkl"
        zip_ = MODELS_DIR / f"recurrent_ppo_v19_multipair_{RUN_ID}_{steps}_steps.zip"

        if not pkl.exists() or not zip_.exists():
            print(f"    ⚠️  Checkpoint {steps} não encontrado — pulando")
            return None

        raw_env = DummyVecEnv([lambda: TradingEnvV19LSTM(data_paths=data_paths, **ENV_CONFIG)])
        env = VecNormalize.load(str(pkl), raw_env)
        env.training   = False
        env.norm_reward = False

        model = RecurrentPPO.load(str(zip_), env=env, device='cpu')

        obs      = env.reset()
        done     = np.array([False])
        lstm_st  = None
        ep_start = np.ones((1,), dtype=bool)

        equity_curve   = [INITIAL_BAL]
        actions        = []
        closed_trades  = []        # pnl de cada trade fechado
        step_count     = 0

        while not done[0]:
            action, lstm_st = model.predict(
                obs, state=lstm_st, episode_start=ep_start, deterministic=True
            )
            obs, _, done, info = env.step(action)
            ep_start = np.zeros((1,), dtype=bool)
            step_count += 1

            act_val = float(np.asarray(action).ravel()[0])
            actions.append(act_val)

            i = info[0]
            equity_curve.append(i['equity'])

            if i.get('trade_executed'):
                closed_trades.append(i.get('pnl', 0.0))

        final_info = info[0]

        # Equity final
        final_equity  = final_info['equity']
        total_ret_pct = (final_equity - INITIAL_BAL) / INITIAL_BAL * 100

        # P&L somente de trades FECHADOS
        closed_pnl   = sum(closed_trades)
        closed_ret   = closed_pnl / INITIAL_BAL * 100

        # Unrealized = diferença entre equity final e (initial + closed_pnl)
        unrealized_pnl = final_equity - INITIAL_BAL - closed_pnl
        unrealized_pct = unrealized_pnl / INITIAL_BAL * 100

        # Win/loss dos trades fechados
        wins  = [p for p in closed_trades if p > 0]
        loss  = [p for p in closed_trades if p <= 0]
        wr    = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        avg_w = np.mean(wins)  if wins else 0
        avg_l = np.mean(loss)  if loss else 0
        pf    = sum(wins) / abs(sum(loss)) if loss else (999.0 if wins else 0.0)

        # Drawdown
        peak       = INITIAL_BAL
        max_dd     = 0.0
        for e in equity_curve:
            if e > peak: peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd: max_dd = dd

        # Distribuição de ações
        arr  = np.array(actions)
        long_pct  = (arr >  0.1).mean() * 100
        short_pct = (arr < -0.1).mean() * 100
        flat_pct  = 100 - long_pct - short_pct

        env.close()

        return {
            'steps':          steps,
            'pair':           pair.upper(),
            'total_ret':      total_ret_pct,
            'closed_ret':     closed_ret,
            'unrealized_ret': unrealized_pct,
            'total_trades':   len(closed_trades),
            'win_rate':       wr,
            'profit_factor':  pf,
            'avg_win':        avg_w,
            'avg_loss':       avg_l,
            'max_drawdown':   max_dd,
            'long_pct':       long_pct,
            'short_pct':      short_pct,
            'flat_pct':       flat_pct,
            'final_equity':   final_equity,
            'equity_curve':   equity_curve,
            'ep_steps':       step_count,
        }

    except Exception as e:
        print(f"    ❌ Erro: {e}")
        traceback.print_exc()
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(f"backtest_v19_extended_{ts}.txt")

    print(f"\n{'='*90}")
    print("  BACKTEST ESTENDIDO V19.2 — EPISÓDIO COMPLETO (7 MESES DE DADOS)")
    print(f"{'='*90}")
    print(f"  max_episode_steps = {MAX_EP_STEPS}  (~{MAX_EP_STEPS*15//60//24} dias de 15m)")
    print(f"  Run ID            = {RUN_ID}")
    print(f"  Output            = {out_file}\n")

    pair_data = {}
    for pair in PAIRS:
        paths = get_test_data(pair)
        if paths:
            print(f"  ✅ {pair.upper():8s}  {Path(paths['15m']).name}")
            pair_data[pair] = paths
        else:
            print(f"  ❌ {pair.upper():8s}  dados não encontrados")

    if not pair_data:
        print("\n❌ Sem dados. Abortando."); sys.exit(1)

    all_results = []

    for steps in CHECKPOINTS:
        print(f"\n{'─'*80}")
        print(f"  ▶ CHECKPOINT {steps:,} steps")
        print(f"{'─'*80}")

        for pair in pair_data:
            print(f"\n    {pair.upper():8s} ...", flush=True)
            r = run_extended_backtest(steps, pair, pair_data[pair])
            if r:
                closed_tag = "⚠️ " if abs(r['unrealized_ret']) > abs(r['closed_ret']) * 0.5 else "✅ "
                print(f"    Total: {r['total_ret']:+.2f}%  "
                      f"Fechados: {closed_tag}{r['closed_ret']:+.2f}%  "
                      f"Unrealized: {r['unrealized_ret']:+.2f}%")
                print(f"    WR: {r['win_rate']:.1f}%  "
                      f"PF: {r['profit_factor']:.2f}  "
                      f"Trades: {r['total_trades']}  "
                      f"MaxDD: {r['max_drawdown']:.2f}%")
                print(f"    Ações: L{r['long_pct']:.0f}%/S{r['short_pct']:.0f}%/F{r['flat_pct']:.0f}%")
                all_results.append(r)

    # ── Relatório ─────────────────────────────────────────────────────────────
    W   = 100
    SEP = "=" * W
    lines = []

    lines.append(f"\n{SEP}")
    lines.append(f"  BACKTEST ESTENDIDO V19.2  —  {ts}  —  {MAX_EP_STEPS} steps/ep (~7 meses)".center(W))
    lines.append(SEP)

    for steps in CHECKPOINTS:
        step_r = [r for r in all_results if r['steps'] == steps]
        if not step_r:
            continue

        lines.append(f"\n  CHECKPOINT: {steps:,} steps")
        lines.append(f"  {'Par':8s}  {'Total':>8}  {'Fechados':>10}  {'Unrealiz':>10}  "
                     f"{'WR':>6}  {'PF':>6}  {'MaxDD':>7}  {'Trades':>7}  Ações L/S/F")
        lines.append("  " + "-"*90)

        for r in step_r:
            unrealiz_flag = " ⚠️" if abs(r['unrealized_ret']) > abs(r['closed_ret']) * 0.5 else "   "
            lines.append(
                f"  {r['pair']:8s}  {r['total_ret']:>+7.2f}%  "
                f"{r['closed_ret']:>+9.2f}%  "
                f"{r['unrealized_ret']:>+9.2f}%{unrealiz_flag}  "
                f"{r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
                f"{r['max_drawdown']:>6.2f}%  {r['total_trades']:>7}  "
                f"L{r['long_pct']:.0f}%/S{r['short_pct']:.0f}%/F{r['flat_pct']:.0f}%"
            )

        # Média
        avg_total   = np.mean([r['total_ret']  for r in step_r])
        avg_closed  = np.mean([r['closed_ret'] for r in step_r])
        avg_dd      = np.mean([r['max_drawdown'] for r in step_r])
        avg_wr      = np.mean([r['win_rate'] for r in step_r])
        n_pos_total  = sum(1 for r in step_r if r['total_ret']  > 0)
        n_pos_closed = sum(1 for r in step_r if r['closed_ret'] > 0)
        lines.append("  " + "-"*90)
        lines.append(f"  {'MÉDIA':8s}  {avg_total:>+7.2f}%  {avg_closed:>+9.2f}%  "
                     f"{'':>12}  {avg_wr:>5.1f}%  {'':>5}  {avg_dd:>6.2f}%  "
                     f"{'':>7}  {n_pos_total}/{len(step_r)} totais, {n_pos_closed}/{len(step_r)} fechados")

    # Resumo comparativo
    lines.append(f"\n{SEP}")
    lines.append("  COMPARATIVO — RETORNO MÉDIO (total vs fechado)".center(W))
    lines.append(SEP)
    lines.append(f"  {'Steps':>9}  {'Ret Total':>10}  {'Ret Fechado':>12}  "
                 f"{'Unrealiz':>10}  {'MaxDD':>7}  {'WR':>6}  Qualidade")
    lines.append("  " + "-"*80)

    comparativo = []
    for steps in CHECKPOINTS:
        step_r = [r for r in all_results if r['steps'] == steps]
        if not step_r: continue

        avg_t = np.mean([r['total_ret']      for r in step_r])
        avg_c = np.mean([r['closed_ret']     for r in step_r])
        avg_u = np.mean([r['unrealized_ret'] for r in step_r])
        avg_d = np.mean([r['max_drawdown']   for r in step_r])
        avg_w = np.mean([r['win_rate']       for r in step_r])

        # Qualidade: a maioria do retorno vem de trades fechados?
        if avg_c > 0 and abs(avg_u) < avg_c:
            qualidade = "✅ Real (fechados > unrealiz)"
        elif avg_t > 0 and avg_c < 0:
            qualidade = "⚠️  Só unrealized"
        elif avg_c < 0:
            qualidade = "❌ Trades fechados perdem"
        else:
            qualidade = "⚠️  Misto"

        comparativo.append((steps, avg_t, avg_c, avg_u, avg_d, avg_w, qualidade))
        lines.append(
            f"  {steps:>9,}  {avg_t:>+9.2f}%  {avg_c:>+11.2f}%  "
            f"{avg_u:>+9.2f}%  {avg_d:>6.2f}%  {avg_w:>5.1f}%  {qualidade}"
        )

    lines.append(SEP)

    # Veredito
    best_closed = max(comparativo, key=lambda x: x[2]) if comparativo else None
    if best_closed:
        lines.append(f"\n  VEREDITO:")
        s, t, c, u, d, w, q = best_closed
        lines.append(f"  Melhor checkpoint por trades FECHADOS: {s:,} steps")
        lines.append(f"  Retorno por fechados: {c:+.2f}%  |  Unrealized: {u:+.2f}%  |  MaxDD: {d:.2f}%")
        if c > 0:
            lines.append(f"  ✅ O modelo É LUCRATIVO — retorno vem de trades reais")
        else:
            lines.append(f"  ⚠️  O retorno total é fictício — vem de posição não fechada no fim do episódio")

    lines.append(SEP + "\n")

    report = "\n".join(lines)
    print(report)
    out_file.write_text(report, encoding='utf-8')
    print(f"\n  💾 Salvo: {out_file}")


if __name__ == "__main__":
    main()
