"""
Valida checkpoints V13 automaticamente
Roda 3 backtests stochastic em cada checkpoint e gera relatório
"""

import subprocess
import sys
from pathlib import Path
import re
from datetime import datetime


def run_backtest(model_path, data_path, runs=3):
    """Roda múltiplos backtests e retorna estatísticas"""
    results = []
    
    for i in range(runs):
        print(f"\n   Run {i+1}/{runs}...", end="", flush=True)
        
        cmd = [
            sys.executable,
            "backtest_stochastic.py",
            str(model_path),
            str(data_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        output = result.stdout
        
        # Extrair métricas
        ret_match = re.search(r"Total Return: ([-\d.]+)%", output)
        trades_match = re.search(r"Trades: (\d+)", output)
        long_match = re.search(r"Long: ([\d.]+)%", output)
        short_match = re.search(r"Short: ([\d.]+)%", output)
        wr_match = re.search(r"Win Rate: ([\d.]+)%", output)
        
        if all([ret_match, trades_match, long_match, short_match, wr_match]):
            results.append({
                'return': float(ret_match.group(1)),
                'trades': int(trades_match.group(1)),
                'long': float(long_match.group(1)),
                'short': float(short_match.group(1)),
                'win_rate': float(wr_match.group(1))
            })
            print(f" ✓ (Return: {ret_match.group(1)}%, WR: {wr_match.group(1)}%)")
        else:
            print(f" ✗ Falhou ao parsear")
    
    if not results:
        return None
    
    # Calcular médias
    avg = {
        'return': sum(r['return'] for r in results) / len(results),
        'trades': sum(r['trades'] for r in results) / len(results),
        'long': sum(r['long'] for r in results) / len(results),
        'short': sum(r['short'] for r in results) / len(results),
        'win_rate': sum(r['win_rate'] for r in results) / len(results),
        'std_return': max(r['return'] for r in results) - min(r['return'] for r in results),
        'std_long': max(r['long'] for r in results) - min(r['long'] for r in results)
    }
    
    return avg


def main():
    print("\n" + "="*80)
    print("📊 VALIDAÇÃO CHECKPOINTS V13")
    print("="*80)
    
    models_dir = Path("models")
    data_path = "data/train_btcusdt_36m_20260109.csv"
    
    # Encontrar checkpoints V13
    checkpoints = sorted(models_dir.glob("sac_v13_*_*_steps.zip"))
    
    if not checkpoints:
        print("\n❌ Nenhum checkpoint V13 encontrado!")
        print("   Execute train_sac_v13.py primeiro")
        return
    
    print(f"\n✅ Encontrados {len(checkpoints)} checkpoints V13\n")
    
    # Validar cada checkpoint
    report = []
    
    for i, checkpoint in enumerate(checkpoints, 1):
        # Extrair steps do nome
        steps_match = re.search(r"(\d+)_steps\.zip", checkpoint.name)
        if not steps_match:
            continue
        
        steps = int(steps_match.group(1))
        
        print(f"\n[{i}/{len(checkpoints)}] Validando {checkpoint.name}")
        print(f"   Steps: {steps:,}")
        
        stats = run_backtest(checkpoint, data_path, runs=3)
        
        if stats:
            report.append({
                'checkpoint': checkpoint.name,
                'steps': steps,
                **stats
            })
            
            # Avaliação
            balance_ok = 35 <= stats['long'] <= 55 and 35 <= stats['short'] <= 55
            wr_ok = stats['win_rate'] >= 22
            
            status = "✅ BOM" if (balance_ok and wr_ok) else "⚠️ ATENÇÃO" if wr_ok else "❌ RUIM"
            
            print(f"\n   📊 Resultados (média de 3 runs):")
            print(f"      Return: {stats['return']:+.2f}%")
            print(f"      Win Rate: {stats['win_rate']:.1f}% {status if wr_ok else '❌'}")
            print(f"      Trades: {stats['trades']:.0f}")
            print(f"      Long/Short: {stats['long']:.1f}% / {stats['short']:.1f}% {status if balance_ok else '❌'}")
            print(f"      Variação Return: ±{stats['std_return']:.2f}%")
            print(f"      Variação Long: ±{stats['std_long']:.1f}%")
            print(f"\n   {status}")
        else:
            print(f"   ❌ Falha ao validar")
    
    # Gerar relatório final
    if report:
        print("\n" + "="*80)
        print("📈 RELATÓRIO FINAL V13")
        print("="*80)
        
        # Ordenar por steps
        report.sort(key=lambda x: x['steps'])
        
        print(f"\n{'Steps':<10} {'Return':<10} {'Win Rate':<12} {'Trades':<10} {'Long/Short':<20} {'Status'}")
        print("-" * 80)
        
        best_checkpoint = None
        best_score = -999
        
        for r in report:
            balance_ok = 35 <= r['long'] <= 55 and 35 <= r['short'] <= 55
            wr_ok = r['win_rate'] >= 22
            
            status = "✅ BOM" if (balance_ok and wr_ok) else "⚠️ ATENÇÃO" if wr_ok else "❌ RUIM"
            
            # Score simples: win_rate - abs(return) - abs(long-50) - abs(short-50)
            score = r['win_rate'] - abs(r['return']) - abs(r['long']-50) - abs(r['short']-50)
            
            if score > best_score and balance_ok:
                best_score = score
                best_checkpoint = r
            
            print(f"{r['steps']:<10,} {r['return']:>+7.2f}%  {r['win_rate']:>8.1f}%    {r['trades']:>7.0f}   "
                  f"{r['long']:>6.1f}% / {r['short']:<6.1f}%  {status}")
        
        # Melhor checkpoint
        if best_checkpoint:
            print("\n" + "="*80)
            print("🏆 MELHOR CHECKPOINT V13")
            print("="*80)
            print(f"\nCheckpoint: {best_checkpoint['checkpoint']}")
            print(f"Steps: {best_checkpoint['steps']:,}")
            print(f"Return: {best_checkpoint['return']:+.2f}%")
            print(f"Win Rate: {best_checkpoint['win_rate']:.1f}%")
            print(f"Trades: {best_checkpoint['trades']:.0f}")
            print(f"Long/Short: {best_checkpoint['long']:.1f}% / {best_checkpoint['short']:.1f}%")
            
            # Comparar com V6 500k
            print("\n📊 COMPARAÇÃO COM V6 500k:")
            print(f"{'Métrica':<15} {'V6 500k':<15} {'V13 Melhor':<15} {'Diferença'}")
            print("-" * 60)
            print(f"{'Win Rate':<15} {'20.21%':<15} {f'{best_checkpoint['win_rate']:.1f}%':<15} "
                  f"{best_checkpoint['win_rate'] - 20.21:+.1f}%")
            print(f"{'Return':<15} {'-0.96%':<15} {f'{best_checkpoint['return']:+.2f}%':<15} "
                  f"{best_checkpoint['return'] + 0.96:+.2f}%")
            print(f"{'Long/Short':<15} {'43.7% / 43.2%':<15} "
                  f"{f'{best_checkpoint['long']:.1f}% / {best_checkpoint['short']:.1f}%':<15}")
        
        # Salvar relatório
        report_path = f"reports/v13_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("V13 VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            for r in report:
                f.write(f"Checkpoint: {r['checkpoint']}\n")
                f.write(f"  Steps: {r['steps']:,}\n")
                f.write(f"  Return: {r['return']:+.2f}%\n")
                f.write(f"  Win Rate: {r['win_rate']:.1f}%\n")
                f.write(f"  Trades: {r['trades']:.0f}\n")
                f.write(f"  Long/Short: {r['long']:.1f}% / {r['short']:.1f}%\n\n")
        
        print(f"\n📄 Relatório salvo: {report_path}")


if __name__ == "__main__":
    main()
