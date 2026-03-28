"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         📊 BACKTEST COMPARATIVO V17.7 - MÚLTIPLOS CHECKPOINTS               ║
║                                                                              ║
║  Roda backtest nos checkpoints-chave e compara com baselines                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from pathlib import Path
import sys
from datetime import datetime

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("❌ sb3-contrib não instalado!")
    sys.exit(1)

from src.environment.trading_env_multi_tf_lstm import TradingEnvMultiTFLSTM

# ============= CHECKPOINTS A TESTAR =============
RUN_ID = "20260221_030417"

CHECKPOINTS = [
    ("V17.7 - 300k",    f"models/recurrent_ppo_v17_lstm_{RUN_ID}_300000_steps.zip"),
    ("V17.7 - 600k",    f"models/recurrent_ppo_v17_lstm_{RUN_ID}_600000_steps.zip"),
    ("V17.7 - 1M",      f"models/recurrent_ppo_v17_lstm_{RUN_ID}_1000000_steps.zip"),
    ("V17.7 - 1.5M",    f"models/recurrent_ppo_v17_lstm_{RUN_ID}_1500000_steps.zip"),
    ("V17.7 - final",   f"models/recurrent_ppo_v17_lstm_{RUN_ID}_final.zip"),
]

# Baselines para comparação no relatório final
BASELINES = {
    "V16.3 SAC 600k":   {"return": -1.13, "trades": 784,  "wr": 0.0,  "commissions": 313},
    "V17.6 LSTM 600k":  {"return": +0.75, "trades": 1002, "wr": 38.4, "commissions": 0.0},
}

# Configuração ambiente (IDÊNTICA ao treino)
ENV_CONFIG = {
    'window_size': 50,
    'max_episode_steps': 2000,
    'leverage': 1.0,
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,
    'use_sharpe_reward': False,
    'enable_indicator_shaping': False,
    'random_start': False,
    'persist_balance': False,
    'liquidation_threshold': 0.30,
}

def find_test_data():
    data_dir = Path('data')
    files_15m = sorted(data_dir.glob('test_btcusdt_*_15m_*.csv'), reverse=True)
    files_1h  = sorted(data_dir.glob('test_btcusdt_*_1h_*.csv'),  reverse=True)
    files_4h  = sorted(data_dir.glob('test_btcusdt_*_4h_*.csv'),  reverse=True)
    if not files_15m or not files_1h or not files_4h:
        print("❌ Dados de teste não encontrados!")
        sys.exit(1)
    return {'15m': str(files_15m[0]), '1h': str(files_1h[0]), '4h': str(files_4h[0])}


def run_backtest(label, model_path, data_paths):
    """Roda um backtest completo e retorna métricas."""
    print(f"\n{'='*70}")
    print(f"🔄 Testando: {label}")
    print(f"   Modelo: {Path(model_path).name}")
    print(f"{'='*70}")

    mp = Path(model_path)
    if not mp.exists():
        print(f"   ⚠️  Arquivo não encontrado — pulando.")
        return None

    # Carregar modelo
    model = RecurrentPPO.load(str(mp))

    # Criar ambiente de teste
    env = TradingEnvMultiTFLSTM(data_paths=data_paths, **ENV_CONFIG)

    # Estado LSTM
    lstm_states = None
    episode_start = np.array([True])

    # Reset
    obs, _ = env.reset()
    obs = obs[np.newaxis]  # (1, 50, 31)

    done = False
    equity_curve = [env.initial_balance]

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=False,
        )
        episode_start = np.array([False])
        # action shape é (1,1) por causa do batch dim — remover antes do env.step
        env_action = action[0] if action.ndim > 1 else action
        obs, reward, terminated, truncated, info = env.step(env_action)
        obs = obs[np.newaxis]
        done = terminated or truncated
        equity_curve.append(env.equity)

    # Calcular métricas
    initial  = env.initial_balance
    final_eq = env.equity
    ret_pct  = (final_eq - initial) / initial * 100
    trades   = env.trades
    wins     = env.wins
    wr       = (wins / trades * 100) if trades > 0 else 0
    commissions = trades * ENV_CONFIG['commission'] * initial * ENV_CONFIG['position_size']
    liquidations = env.liquidations

    # Max drawdown
    peak = initial
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(r for r in getattr(env, 'trade_pnls', []) if r > 0) or 0
    gross_loss   = abs(sum(r for r in getattr(env, 'trade_pnls', []) if r < 0)) or 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    print(f"   💰 Equity final:  ${final_eq:.2f}  ({ret_pct:+.2f}%)")
    print(f"   📈 Trades:        {trades}  |  Wins: {wins}  ({wr:.1f}%)")
    print(f"   📉 Max Drawdown:  {max_dd:.2f}%")
    print(f"   🏦 Commissions:   ${commissions:.2f}")
    print(f"   ⚡ Liquidações:   {liquidations}")

    return {
        "label":        label,
        "return":       ret_pct,
        "final_equity": final_eq,
        "trades":       trades,
        "wr":           wr,
        "max_dd":       max_dd,
        "commissions":  commissions,
        "liquidations": liquidations,
    }


def print_comparison(results):
    """Tabela final de comparação."""
    print("\n\n" + "="*80)
    print("📊 TABELA COMPARATIVA FINAL")
    print("="*80)

    header = f"{'Modelo':<22} {'Retorno':>9} {'Equity':>11} {'Trades':>7} {'WR%':>6} {'MaxDD%':>7} {'Comm$':>7} {'Liq':>4}"
    print(header)
    print("-"*80)

    # Baselines
    print(f"{'V16.3 SAC 600k':<22} {'-1.13%':>9} {'$9,887':>11} {'784':>7} {'---':>6} {'---':>7} {'$313':>7} {'0':>4}")
    print(f"{'V17.6 LSTM 600k':<22} {'+0.75%':>9} {'$10,075':>11} {'1002':>7} {'38.4':>6} {'---':>7} {'---':>7} {'0':>4}")
    print("-"*80)

    best_return = max((r['return'] for r in results if r), default=0)
    for r in results:
        if r is None:
            continue
        marker = " ✅" if r['return'] == best_return and best_return > 0 else ""
        ret_str  = f"{r['return']:+.2f}%"
        eq_str   = f"${r['final_equity']:,.2f}"
        print(f"{r['label']:<22} {ret_str:>9} {eq_str:>11} {r['trades']:>7} {r['wr']:>6.1f} {r['max_dd']:>7.2f} ${r['commissions']:>6.2f} {r['liquidations']:>4}{marker}")

    print("="*80)

    # Veredicto
    positives = [r for r in results if r and r['return'] > 0]
    if positives:
        best = max(positives, key=lambda x: x['return'])
        print(f"\n🏆 MELHOR MODELO: {best['label']}  →  {best['return']:+.2f}%")
        if best['return'] > 0.75:
            print("   ✅ SUPERA V17.6 (+0.75%) — real multi-TF funcionou!")
        elif best['return'] > 0:
            print("   ⚠️  Retorno positivo, mas não supera V17.6 ainda")
    else:
        print("\n⚠️  Nenhum checkpoint com retorno positivo no conjunto de teste")

    print()


def main():
    print("\n" + "="*70)
    print("🧪 BACKTEST COMPARATIVO V17.7 - TODOS OS CHECKPOINTS-CHAVE")
    print(f"   Run: {RUN_ID}")
    print("="*70)

    print("\n🔍 Procurando dados de teste...")
    data_paths = find_test_data()
    for tf, p in data_paths.items():
        print(f"   {tf}: {Path(p).name}")

    results = []
    for label, model_path in CHECKPOINTS:
        result = run_backtest(label, model_path, data_paths)
        results.append(result)

    print_comparison(results)

    # Salvar relatório
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"backtest_report_v177_compare_{ts}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"V17.7 Backtest Comparativo — {ts}\n")
        f.write(f"Run: {RUN_ID}\n\n")
        for r in results:
            if r:
                f.write(f"{r['label']}: {r['return']:+.2f}%  trades={r['trades']}  wr={r['wr']:.1f}%  maxdd={r['max_dd']:.2f}%\n")
    print(f"💾 Relatório salvo: {report_file}")


if __name__ == "__main__":
    main()
