"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       📊 GERAR BASELINE DE FEATURES — V19 DRIFT DETECTION                   ║
║                                                                              ║
║  Computa estatísticas de distribuição (mean, std) das features V19           ║
║  a partir dos CSVs de treinamento e salva em data/feature_baseline.json.    ║
║                                                                              ║
║  Deve ser executado após collect_multi_pair_mtf.py ou retrain_v19_daily.py  ║
║  para manter o baseline atualizado.                                         ║
║                                                                              ║
║  Uso:                                                                        ║
║    python generate_feature_baseline.py                                      ║
║    python generate_feature_baseline.py --pairs btcusdt ethusdt              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dashboard.analytics.drift_detector import (
    compute_baseline_from_csvs,
    save_baseline,
    BASELINE_PATH,
)

PAIRS    = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']
DATA_DIR = Path('data')


def find_15m_csvs(pairs: list[str]) -> list[str]:
    """Encontra os CSVs 15m mais recentes para cada par."""
    paths = []
    for pair in pairs:
        files = sorted(DATA_DIR.glob(f'train_{pair}_*_15m_*.csv'), reverse=True)
        if files:
            paths.append(str(files[0]))
            print(f"  ✅ {pair.upper()}: {files[0].name}")
        else:
            print(f"  ⚠️  {pair.upper()}: nenhum CSV 15m encontrado")
    return paths


def main():
    parser = argparse.ArgumentParser(description='Gerar baseline de features V19')
    parser.add_argument('--pairs', nargs='+', default=PAIRS, help='Pares a usar')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("📊 GERANDO BASELINE DE FEATURES V19")
    print("="*60)
    print(f"  Pares: {', '.join(p.upper() for p in args.pairs)}")
    print()

    csv_paths = find_15m_csvs(args.pairs)

    if not csv_paths:
        print("\n❌ Nenhum CSV encontrado. Execute collect_multi_pair_mtf.py primeiro.")
        sys.exit(1)

    print(f"\n🔧 Computando baseline de {len(csv_paths)} arquivo(s)...")
    baseline = compute_baseline_from_csvs(csv_paths)

    if baseline is None:
        print("\n❌ Falha ao computar baseline.")
        sys.exit(1)

    save_baseline(baseline)

    print(f"\n✅ Baseline salvo: {BASELINE_PATH}")
    print(f"   Amostras       : {baseline['n_samples']:,}")
    print(f"   Features       : {baseline['n_features']}")
    print(f"   Gerado em      : {baseline['generated_at']}")
    print()
    print("  Feature          | Mean      | Std")
    print("  " + "─"*50)
    for name, mean, std in zip(baseline['feature_names'], baseline['mean'], baseline['std']):
        print(f"  {name:<15}  | {mean:>8.4f}  | {std:>8.4f}")
    print()


if __name__ == '__main__':
    main()
