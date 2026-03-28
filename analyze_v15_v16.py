"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   📊 ANÁLISE COMPARATIVA: V15 vs V16                         ║
║                                                                              ║
║  Compara performance entre treinamento single-timeframe (V15) e              ║
║  multi-timeframe (V16) para validar a hipótese de melhoria.                 ║
║                                                                              ║
║  📋 MÉTRICAS ANALISADAS:                                                     ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  • Win Rate: % de trades vencedores                                         ║
║  • Return: Retorno total em %                                               ║
║  • Sharpe Ratio: Retorno ajustado por risco                                 ║
║  • Max Drawdown: Pior sequência de perda                                    ║
║  • Trade Count: Número total de operações                                   ║
║  • Long/Short Balance: Distribuição de direções                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class V15V16Analyzer:
    """Analisador comparativo entre V15 e V16."""
    
    def __init__(self):
        self.backtest_dir = Path('.')
        
    def find_backtest_reports(self, version: str = 'v15') -> list:
        """
        Encontra relatórios de backtest para uma versão.
        
        Args:
            version: 'v15' ou 'v16'
            
        Returns:
            Lista de caminhos de arquivos
        """
        pattern = f'backtest_report_*{version}*.txt'
        files = sorted(self.backtest_dir.glob(pattern), reverse=True)
        return [str(f) for f in files]
    
    def parse_backtest_report(self, filepath: str) -> dict:
        """
        Parse de um relatório de backtest.
        
        Returns:
            Dicionário com métricas extraídas
        """
        metrics = {
            'filepath': filepath,
            'timestamp': None,
            'win_rate': None,
            'total_return': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'total_trades': None,
            'long_trades': None,
            'short_trades': None,
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extrair timestamp do nome do arquivo
            filename = Path(filepath).stem
            date_str = filename.split('_')[-2] + '_' + filename.split('_')[-1]
            try:
                metrics['timestamp'] = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            except:
                pass
            
            # Extrair métricas do conteúdo
            lines = content.split('\n')
            
            for line in lines:
                # Win Rate
                if 'Win Rate:' in line or 'win_rate' in line.lower():
                    try:
                        val = line.split(':')[-1].strip().replace('%', '')
                        metrics['win_rate'] = float(val)
                    except:
                        pass
                
                # Total Return
                if 'Total Return:' in line or 'Return:' in line:
                    try:
                        val = line.split(':')[-1].strip().replace('%', '').replace('+', '')
                        metrics['total_return'] = float(val)
                    except:
                        pass
                
                # Sharpe Ratio
                if 'Sharpe Ratio:' in line or 'sharpe' in line.lower():
                    try:
                        val = line.split(':')[-1].strip()
                        metrics['sharpe_ratio'] = float(val)
                    except:
                        pass
                
                # Max Drawdown
                if 'Max Drawdown:' in line or 'drawdown' in line.lower():
                    try:
                        val = line.split(':')[-1].strip().replace('%', '').replace('-', '')
                        metrics['max_drawdown'] = float(val)
                    except:
                        pass
                
                # Total Trades
                if 'Total Trades:' in line or 'trades:' in line.lower():
                    try:
                        val = line.split(':')[-1].strip()
                        metrics['total_trades'] = int(val)
                    except:
                        pass
                
                # Long/Short
                if 'Long:' in line:
                    try:
                        val = line.split(':')[-1].strip()
                        metrics['long_trades'] = int(val)
                    except:
                        pass
                
                if 'Short:' in line:
                    try:
                        val = line.split(':')[-1].strip()
                        metrics['short_trades'] = int(val)
                    except:
                        pass
        
        except Exception as e:
            print(f"⚠️  Erro ao parsear {filepath}: {e}")
        
        return metrics
    
    def analyze_version(self, version: str = 'v15') -> pd.DataFrame:
        """
        Analisa todos os backtests de uma versão.
        
        Args:
            version: 'v15' ou 'v16'
            
        Returns:
            DataFrame com estatísticas agregadas
        """
        print(f"\n{'='*70}")
        print(f"📊 ANALISANDO {version.upper()}")
        print(f"{'='*70}")
        
        files = self.find_backtest_reports(version)
        
        if not files:
            print(f"⚠️  Nenhum relatório encontrado para {version}")
            return pd.DataFrame()
        
        print(f"Encontrados: {len(files)} relatórios")
        
        results = []
        for f in files:
            metrics = self.parse_backtest_report(f)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Remover linhas sem dados
        df = df.dropna(subset=['win_rate', 'total_return'], how='all')
        
        if df.empty:
            print(f"⚠️  Nenhum dado válido encontrado")
            return df
        
        # Estatísticas
        print(f"\n📈 ESTATÍSTICAS {version.upper()}:")
        print(f"   Relatórios válidos: {len(df)}")
        
        if 'win_rate' in df.columns and not df['win_rate'].isna().all():
            print(f"\n   Win Rate:")
            print(f"      Média:  {df['win_rate'].mean():.2f}%")
            print(f"      Mediana: {df['win_rate'].median():.2f}%")
            print(f"      Min:    {df['win_rate'].min():.2f}%")
            print(f"      Max:    {df['win_rate'].max():.2f}%")
        
        if 'total_return' in df.columns and not df['total_return'].isna().all():
            print(f"\n   Total Return:")
            print(f"      Média:  {df['total_return'].mean():.2f}%")
            print(f"      Mediana: {df['total_return'].median():.2f}%")
            print(f"      Min:    {df['total_return'].min():.2f}%")
            print(f"      Max:    {df['total_return'].max():.2f}%")
        
        if 'sharpe_ratio' in df.columns and not df['sharpe_ratio'].isna().all():
            print(f"\n   Sharpe Ratio:")
            print(f"      Média:  {df['sharpe_ratio'].mean():.3f}")
            print(f"      Mediana: {df['sharpe_ratio'].median():.3f}")
        
        if 'total_trades' in df.columns and not df['total_trades'].isna().all():
            print(f"\n   Trades:")
            print(f"      Média:  {df['total_trades'].mean():.0f}")
            print(f"      Mediana: {df['total_trades'].median():.0f}")
        
        return df
    
    def compare_versions(self, df_v15: pd.DataFrame, df_v16: pd.DataFrame):
        """
        Compara V15 vs V16 e gera visualizações.
        
        Args:
            df_v15: DataFrame com métricas V15
            df_v16: DataFrame com métricas V16
        """
        print(f"\n{'='*70}")
        print("🔬 COMPARAÇÃO V15 vs V16")
        print(f"{'='*70}")
        
        if df_v15.empty:
            print("⚠️  V15: Sem dados disponíveis")
        if df_v16.empty:
            print("⚠️  V16: Sem dados disponíveis")
        
        if df_v15.empty or df_v16.empty:
            print("\n⚠️  Comparação não pode ser realizada (faltam dados)")
            return
        
        # Comparação de métricas
        metrics = ['win_rate', 'total_return', 'sharpe_ratio', 'max_drawdown']
        
        print("\n📊 COMPARATIVO:")
        print(f"{'Métrica':<20} {'V15':>12} {'V16':>12} {'Δ':>12}")
        print("-" * 60)
        
        for metric in metrics:
            if metric in df_v15.columns and metric in df_v16.columns:
                v15_mean = df_v15[metric].mean()
                v16_mean = df_v16[metric].mean()
                delta = v16_mean - v15_mean
                delta_pct = (delta / v15_mean * 100) if v15_mean != 0 else 0
                
                print(f"{metric:<20} {v15_mean:>12.2f} {v16_mean:>12.2f} {delta:>+11.2f} ({delta_pct:+.1f}%)")
        
        # Verificar hipótese
        print("\n🎯 VALIDAÇÃO DE HIPÓTESE:")
        
        if 'win_rate' in df_v15.columns and 'win_rate' in df_v16.columns:
            v15_wr = df_v15['win_rate'].mean()
            v16_wr = df_v16['win_rate'].mean()
            
            print(f"\n   Hipótese: Multi-timeframe → Win rate 22-25%+")
            print(f"   V15 (single-TF): {v15_wr:.2f}%")
            print(f"   V16 (multi-TF):  {v16_wr:.2f}%")
            
            if v16_wr >= 22 and v16_wr > v15_wr:
                print("   ✅ HIPÓTESE CONFIRMADA!")
            elif v16_wr > v15_wr:
                print("   🟡 Melhoria detectada, mas abaixo do target 22%")
            else:
                print("   ❌ Hipótese não confirmada (V16 ≤ V15)")
        
        # Gerar gráficos (se matplotlib disponível)
        self._plot_comparison(df_v15, df_v16)
    
    def _plot_comparison(self, df_v15: pd.DataFrame, df_v16: pd.DataFrame):
        """Gera gráficos comparativos."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Comparação V15 (Single-TF) vs V16 (Multi-TF)', fontsize=16, fontweight='bold')
            
            # Win Rate
            if 'win_rate' in df_v15.columns and 'win_rate' in df_v16.columns:
                ax = axes[0, 0]
                data = pd.DataFrame({
                    'V15': df_v15['win_rate'],
                    'V16': df_v16['win_rate']
                })
                data.boxplot(ax=ax)
                ax.set_title('Win Rate (%)')
                ax.set_ylabel('%')
                ax.grid(True, alpha=0.3)
            
            # Total Return
            if 'total_return' in df_v15.columns and 'total_return' in df_v16.columns:
                ax = axes[0, 1]
                data = pd.DataFrame({
                    'V15': df_v15['total_return'],
                    'V16': df_v16['total_return']
                })
                data.boxplot(ax=ax)
                ax.set_title('Total Return (%)')
                ax.set_ylabel('%')
                ax.grid(True, alpha=0.3)
            
            # Sharpe Ratio
            if 'sharpe_ratio' in df_v15.columns and 'sharpe_ratio' in df_v16.columns:
                ax = axes[1, 0]
                data = pd.DataFrame({
                    'V15': df_v15['sharpe_ratio'],
                    'V16': df_v16['sharpe_ratio']
                })
                data.boxplot(ax=ax)
                ax.set_title('Sharpe Ratio')
                ax.grid(True, alpha=0.3)
            
            # Trade Count
            if 'total_trades' in df_v15.columns and 'total_trades' in df_v16.columns:
                ax = axes[1, 1]
                data = pd.DataFrame({
                    'V15': df_v15['total_trades'],
                    'V16': df_v16['total_trades']
                })
                data.boxplot(ax=ax)
                ax.set_title('Total Trades')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'comparison_v15_vs_v16_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n📊 Gráfico salvo: {filename}")
            
        except Exception as e:
            print(f"\n⚠️  Erro ao gerar gráficos: {e}")


def main():
    """Executa análise comparativa completa."""
    print("\n" + "="*70)
    print("📊 ANÁLISE COMPARATIVA: V15 vs V16")
    print("="*70)
    print("\nComparando single-timeframe (V15) vs multi-timeframe (V16)")
    
    analyzer = V15V16Analyzer()
    
    # Analisar V15
    df_v15 = analyzer.analyze_version('v15')
    
    # Analisar V16
    df_v16 = analyzer.analyze_version('v16')
    
    # Comparar
    if not df_v15.empty and not df_v16.empty:
        analyzer.compare_versions(df_v15, df_v16)
    elif df_v15.empty and df_v16.empty:
        print("\n" + "="*70)
        print("⚠️  NENHUM DADO DISPONÍVEL")
        print("="*70)
        print("\n📋 INSTRUÇÕES:")
        print("   1. Execute backtests do V15: python backtest.py <modelo_v15>")
        print("   2. Execute backtests do V16: python backtest.py <modelo_v16>")
        print("   3. Execute esta análise novamente")
    elif df_v16.empty:
        print("\n" + "="*70)
        print("⚠️  V16 AINDA NÃO TREINADO")
        print("="*70)
        print("\n📋 PRÓXIMOS PASSOS:")
        print("   1. Baixar dados: python collect_multi_timeframe.py")
        print("   2. Treinar V16: python train_sac_v16.py")
        print("   3. Executar backtest: python backtest.py <modelo_v16>")
        print("   4. Executar esta análise novamente")
    
    print("\n" + "="*70)
    print("✅ Análise concluída")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
