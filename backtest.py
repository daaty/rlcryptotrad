"""
Framework de Backtesting Profissional
Valida modelos treinados em dados históricos e gera métricas detalhadas.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from stable_baselines3 import PPO, TD3
from src.environment.trading_env import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
from datetime import datetime

class Backtester:
    def __init__(self, model_path: str, data_path: str, config_path: str = "config.yaml"):
        """
        Args:
            model_path: Caminho para o modelo treinado (.zip)
            data_path: Caminho para dados de teste (CSV)
            config_path: Caminho para configuração
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        
        # Carregar config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determinar tipo de modelo
        self.model_type = self._detect_model_type()
        
        # Carregar modelo
        print(f"Carregando modelo {self.model_type}...")
        if self.model_type == "PPO":
            self.model = PPO.load(model_path)
        elif self.model_type == "TD3":
            self.model = TD3.load(model_path)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
        
        # Carregar dados de teste
        print(f"Carregando dados de {data_path}...")
        df = pd.read_csv(data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Pegar apenas colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.df = df[numeric_cols]
        
        print(f"  {len(self.df)} candles carregados")
        
        # Criar ambiente de teste
        env_config = self.config['environment']
        self.env = TradingEnv(
            df=self.df,
            initial_balance=env_config['initial_balance'],
            commission=env_config.get('commission', 0.001),
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size']
        )
        
        # Armazenar histórico
        self.history = {
            'balance': [],
            'equity': [],
            'position': [],
            'trades': [],
            'actions': []
        }
        
    def _detect_model_type(self) -> str:
        """Detecta tipo de modelo pelo nome do arquivo."""
        name = self.model_path.stem.lower()
        if 'ppo' in name:
            return 'PPO'
        elif 'td3' in name:
            return 'TD3'
        else:
            return 'PPO'  # Padrão
    
    def run(self, episodes: int = 1, verbose: bool = True) -> dict:
        """
        Executa backtest por N episódios.
        
        Args:
            episodes: Número de episódios (passadas pelos dados)
            verbose: Se True, mostra progresso
            
        Returns:
            Dicionário com métricas agregadas
        """
        all_results = []
        
        for ep in range(episodes):
            if verbose:
                print(f"\nEpisódio {ep+1}/{episodes}")
            
            results = self._run_episode(verbose=verbose)
            all_results.append(results)
        
        # Agregar resultados de todos os episódios
        aggregated = self._aggregate_results(all_results)
        
        return aggregated
    
    def _run_episode(self, verbose: bool = True) -> dict:
        """Executa um episódio completo de backtest."""
        obs, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        # Resetar histórico
        self.history = {
            'balance': [self.env.balance],
            'equity': [self.env.equity],
            'position': [self.env.position],
            'trades': [],
            'actions': []
        }
        
        while not (done or truncated):
            # Prever ação
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Executar ação
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Armazenar histórico
            self.history['balance'].append(self.env.balance)
            self.history['equity'].append(self.env.equity)
            self.history['position'].append(self.env.position)
            self.history['actions'].append(action)
            
            # Detectar trades (mudança de posição)
            if len(self.history['position']) > 1:
                if self.history['position'][-1] != self.history['position'][-2]:
                    trade = {
                        'step': step,
                        'action': action,
                        'position': self.env.position,
                        'balance': self.env.balance,
                        'equity': self.env.equity
                    }
                    self.history['trades'].append(trade)
            
            step += 1
            
            if verbose and step % 100 == 0:
                print(f"  Step {step}: Balance=${self.env.balance:.2f}, Equity=${self.env.equity:.2f}, Position={self.env.position}")
        
        # Calcular métricas do episódio
        metrics = self._calculate_metrics()
        
        if verbose:
            print(f"\n=== RESULTADOS DO EPISODIO ===")
            print(f"Balance Final: ${metrics['final_balance']:.2f}")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        return metrics
    
    def _calculate_metrics(self) -> dict:
        """Calcula métricas de performance do backtest."""
        initial_balance = self.history['balance'][0]
        final_balance = self.history['balance'][-1]
        
        # Retorno total
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Equity curve
        equity_curve = np.array(self.history['equity'])
        
        # Returns diários (assumindo 15min candles = 96 candles/dia)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Sharpe Ratio (anualizado)
        if len(returns) > 1 and np.std(returns) > 0:
            # 96 períodos/dia * 365 dias = 35040 períodos/ano
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(35040)
        else:
            sharpe_ratio = 0.0
        
        # Max Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Trades
        total_trades = self.env.trades
        wins = self.env.wins
        losses = self.env.losses
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Profit Factor (precisa rastrear wins/losses individuais)
        # Simplificação: usar total_pnl e assumir distribuição
        profit_factor = abs(self.env.total_pnl / initial_balance) if self.env.total_pnl < 0 else (
            self.env.total_pnl / (initial_balance * 0.01) if self.env.total_pnl > 0 else 1.0
        )
        
        # Expectancy (média de lucro por trade)
        expectancy = self.env.total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_pnl': self.env.total_pnl
        }
    
    def _aggregate_results(self, results_list: list) -> dict:
        """Agrega resultados de múltiplos episódios."""
        if len(results_list) == 1:
            return results_list[0]
        
        # Calcular médias
        aggregated = {}
        keys = results_list[0].keys()
        
        for key in keys:
            values = [r[key] for r in results_list]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def plot_results(self, save_path: str = None):
        """
        Gera gráficos de performance.
        
        Args:
            save_path: Se fornecido, salva gráfico neste caminho
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Equity Curve
        axes[0].plot(self.history['equity'], label='Equity', linewidth=2)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity (USDT)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Position Over Time
        axes[1].plot(self.history['position'], label='Position', linewidth=1.5)
        axes[1].set_title('Position (1=Long, 0=Flat, -1=Short)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Position')
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. Drawdown
        equity_curve = np.array(self.history['equity'])
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        
        axes[2].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[2].plot(drawdown, color='red', linewidth=1.5, label='Drawdown')
        axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nGrafico salvo em: {save_path}")
        else:
            plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
            print("\nGrafico salvo em: backtest_results.png")
        
        plt.close()
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Gera relatório detalhado em texto.
        
        Args:
            save_path: Se fornecido, salva relatório neste caminho
            
        Returns:
            String com relatório formatado
        """
        metrics = self._calculate_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           RELATORIO DE BACKTESTING - {datetime.now().strftime('%Y-%m-%d %H:%M')}           ║
╚══════════════════════════════════════════════════════════════╝

MODELO:
  Tipo: {self.model_type}
  Arquivo: {self.model_path.name}

DADOS:
  Arquivo: {self.data_path.name}
  Candles: {len(self.df)}
  Período: {len(self.df) * 15 / 60 / 24:.1f} dias (15min candles)

CONFIGURACAO:
  Balance Inicial: ${metrics['initial_balance']:,.2f}
  Commission: {self.config['environment'].get('commission', 0.001):.4f} ({self.config['environment'].get('commission', 0.001)*100:.2f}%)
  Slippage: {self.config['environment'].get('slippage', 0.0005):.4f} ({self.config['environment'].get('slippage', 0.0005)*100:.2f}%)
  Position Size: {self.config['environment']['position_size']:.1%}
  Leverage: {self.config['environment']['leverage']}x

╔══════════════════════════════════════════════════════════════╗
║                     METRICAS DE PERFORMANCE                  ║
╚══════════════════════════════════════════════════════════════╝

RETORNO:
  Balance Final: ${metrics['final_balance']:,.2f}
  Total Return: {metrics['total_return']:+.2%}
  Total P&L: ${metrics['total_pnl']:+,.2f}
  
TRADING:
  Total Trades: {metrics['total_trades']}
  Wins: {metrics['wins']}
  Losses: {metrics['losses']}
  Win Rate: {metrics['win_rate']:.2%}
  Expectancy: ${metrics['expectancy']:.2f} por trade
  
RISCO:
  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
  Max Drawdown: {metrics['max_drawdown']:.2%}
  Profit Factor: {metrics['profit_factor']:.2f}

AVALIACAO:
"""
        
        # Avaliar performance
        score = 0
        feedback = []
        
        if metrics['total_return'] > 0.05:
            score += 2
            feedback.append("  ✓ Retorno positivo > 5%")
        elif metrics['total_return'] > 0:
            score += 1
            feedback.append("  ~ Retorno positivo mas < 5%")
        else:
            feedback.append("  ✗ Retorno negativo")
        
        if metrics['sharpe_ratio'] > 2.0:
            score += 2
            feedback.append("  ✓ Sharpe excelente (> 2.0)")
        elif metrics['sharpe_ratio'] > 1.0:
            score += 1
            feedback.append("  ~ Sharpe aceitável (> 1.0)")
        else:
            feedback.append("  ✗ Sharpe baixo (< 1.0)")
        
        if metrics['max_drawdown'] > -0.15:
            score += 2
            feedback.append("  ✓ Drawdown controlado (< 15%)")
        elif metrics['max_drawdown'] > -0.25:
            score += 1
            feedback.append("  ~ Drawdown moderado (15-25%)")
        else:
            feedback.append("  ✗ Drawdown alto (> 25%)")
        
        if metrics['win_rate'] > 0.55:
            score += 2
            feedback.append("  ✓ Win rate alto (> 55%)")
        elif metrics['win_rate'] > 0.45:
            score += 1
            feedback.append("  ~ Win rate aceitável (45-55%)")
        else:
            feedback.append("  ✗ Win rate baixo (< 45%)")
        
        report += "\n".join(feedback)
        
        # Score final
        report += f"\n\n  SCORE TOTAL: {score}/8"
        
        if score >= 7:
            report += " - EXCELENTE (Pronto para produção)"
        elif score >= 5:
            report += " - BOM (Considerar refinamentos)"
        elif score >= 3:
            report += " - REGULAR (Precisa melhorar)"
        else:
            report += " - FRACO (Retreinar necessário)"
        
        report += "\n\n" + "="*64 + "\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nRelatorio salvo em: {save_path}")
        
        return report


def main():
    """Exemplo de uso do backtester."""
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python backtest.py <model_path> <data_path>")
        print("Exemplo: python backtest.py models/best_ppo_v2/best_model.zip data/val_data.csv")
        return
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    # Criar backtester
    bt = Backtester(model_path, data_path)
    
    # Rodar backtest
    results = bt.run(episodes=1, verbose=True)
    
    # Gerar relatório
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = bt.generate_report(save_path=f'backtest_report_{timestamp}.txt')
    print(report)
    
    # Gerar gráficos
    bt.plot_results(save_path=f'backtest_plot_{timestamp}.png')
    
    print(f"\nBacktest completo! Resultados salvos.")


if __name__ == "__main__":
    main()
