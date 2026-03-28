"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🔍 BACKTEST MULTI-TIMEFRAME (V16)                                  ║
║                                                                              ║
║  Backtesting adaptado para modelos que usam múltiplos timeframes            ║
║  simultaneamente (15m, 1h, 4h).                                             ║
║                                                                              ║
║  Uso:                                                                        ║
║    python backtest_multi_tf.py                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC
from src.environment.trading_env_multi_tf import TradingEnvMultiTF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime


class BacktesterMultiTF:
    def __init__(
        self,
        model_path: str,
        data_paths: dict,  # {'15m': path, '1h': path, '4h': path}
        window_size: int = 50
    ):
        """
        Args:
            model_path: Caminho para o modelo treinado (.zip)
            data_paths: Dicionário com paths para cada timeframe
            window_size: Janela de observação
        """
        self.model_path = Path(model_path)
        self.data_paths = data_paths
        
        # Carregar modelo SAC
        print(f"📦 Carregando modelo: {model_path}")
        self.model = SAC.load(model_path)
        
        # Verificar dados
        for tf, path in data_paths.items():
            df = pd.read_csv(path)
            print(f"  ✅ {tf}: {len(df)} candles")
        
        # Criar ambiente de teste (V16.3 FIX: TODAS configs do treino!)
        self.env = TradingEnvMultiTF(
            data_paths=data_paths,
            window_size=window_size,
            initial_balance=10000,
            commission=0.0004,
            slippage=0.0005,
            leverage=1.0,                    # V16 CORRETO: Igual ao treino!
            position_size=0.05,
            max_episode_steps=2000,
            random_start=False,              # V16.3 FIX: False para backtest determinístico!
            persist_balance=False,           # Backtest: sem persistência
            use_sharpe_reward=False,         # V16 CORRETO: SEM Sharpe
            enable_indicator_shaping=False,  # V15/V16: desabilitado
            liquidation_threshold=0.30       # V16.3 FIX: FALTAVA! Igual treino (0.30)
        )
        
        # Histórico
        self.history = {
            'balance': [],
            'equity': [],
            'position': [],
            'trades': [],
            'actions': []
        }
    
    def run(self, episodes: int = 1, verbose: bool = True) -> dict:
        """
        Executa backtest por N episódios.
        
        Args:
            episodes: Número de passadas pelos dados
            verbose: Mostrar progresso
            
        Returns:
            Dicionário com métricas agregadas
        """
        all_results = []
        
        for ep in range(episodes):
            if verbose:
                print(f"\n{'='*80}")
                print(f"📊 Episódio {ep+1}/{episodes}")
                print('='*80)
            
            results = self._run_episode(verbose=verbose)
            all_results.append(results)
        
        # Agregar resultados
        aggregated = self._aggregate_results(all_results)
        
        return aggregated
    
    def _run_episode(self, verbose: bool = True) -> dict:
        """Executa um episódio completo de backtest."""
        obs, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        max_steps = self.env.max_episode_steps
        
        # Resetar histórico
        self.history = {
            'balance': [self.env.balance],
            'equity': [self.env.equity],
            'position': [self.env.position],
            'trades': [],
            'actions': []
        }
        
        while not (done or truncated) and step < max_steps:
            # V16.3 FIX: Usar deterministic=FALSE para incluir exploration!
            # Durante treino, SAC amostra estocasticamente da distribuição
            # deterministic=True pega apenas a média → freeze em max Long!
            action, _states = self.model.predict(obs, deterministic=False)
            
            # Executar
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Armazenar
            self.history['balance'].append(self.env.balance)
            self.history['equity'].append(self.env.equity)
            self.history['position'].append(self.env.position)
            self.history['actions'].append(action)
            
            # Detectar trades (mudança de posição)
            if len(self.history['position']) > 1:
                if self.history['position'][-1] != self.history['position'][-2]:
                    trade = {
                        'step': step,
                        'action': float(action[0]),
                        'position': self.env.position,
                        'balance': self.env.balance,
                        'equity': self.env.equity
                    }
                    self.history['trades'].append(trade)
            
            step += 1
            
            if verbose and step % 200 == 0:
                print(f"  Step {step}/{max_steps}: Balance=${self.env.balance:.2f}, "
                      f"Equity=${self.env.equity:.2f}, Position={self.env.position}")
        
        # Calcular métricas
        metrics = self._calculate_metrics()
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> dict:
        """Calcula métricas de performance."""
        balance_series = np.array(self.history['balance'])
        equity_series = np.array(self.history['equity'])
        
        # Return total
        initial_balance = balance_series[0]
        final_balance = balance_series[-1]
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Métricas de trades
        trades = self.history['trades']
        total_trades = len(trades)
        
        # Win rate e análise detalhada de P&L
        winning_trades = 0
        losing_trades = 0
        total_wins_pnl = 0.0
        total_losses_pnl = 0.0
        
        if total_trades > 1:
            for i in range(1, len(trades)):
                pnl = trades[i]['balance'] - trades[i-1]['balance']
                if pnl > 0:
                    winning_trades += 1
                    total_wins_pnl += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_losses_pnl += abs(pnl)
        
        win_rate = winning_trades / max(total_trades - 1, 1)
        
        # V16.3 NOVA MÉTRICA: Average Win vs Average Loss
        avg_win = total_wins_pnl / max(winning_trades, 1)
        avg_loss = total_losses_pnl / max(losing_trades, 1)
        profit_factor = total_wins_pnl / max(total_losses_pnl, 1e-8)
        
        # Sharpe Ratio
        returns = np.diff(balance_series) / balance_series[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 4)  # 15m candles
        
        # Max Drawdown
        cummax = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Tempo em posição
        positions = np.array(self.history['position'])
        time_in_position = np.sum(positions != 0) / len(positions)
        
        # V16.3 NOVA MÉTRICA: Comissões totais estimadas
        commission_rate = 0.0004
        estimated_commissions = total_trades * 2 * commission_rate * initial_balance * 0.05  # 2 sides, 5% position
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'estimated_commissions': estimated_commissions,
            'time_in_position': time_in_position,
            'total_steps': len(balance_series)
        }
    
    def _aggregate_results(self, all_results: list) -> dict:
        """Agrega resultados de múltiplos episódios."""
        if len(all_results) == 1:
            return all_results[0]
        
        # Média das métricas
        aggregated = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def _print_metrics(self, metrics: dict):
        """Imprime métricas formatadas."""
        print(f"\n{'='*80}")
        print(f"📈 RESULTADOS DO BACKTEST")
        print('='*80)
        print(f"Balance Inicial:    ${metrics['initial_balance']:>10,.2f}")
        print(f"Balance Final:      ${metrics['final_balance']:>10,.2f}")
        print(f"Total Return:       {metrics['total_return']:>10.2%}")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
        print('-'*80)
        print(f"Total Trades:       {metrics['total_trades']:>10}")
        print(f"Winning Trades:     {metrics['winning_trades']:>10}")
        print(f"Losing Trades:      {metrics['losing_trades']:>10}")
        print(f"Win Rate:           {metrics['win_rate']:>10.2%}")
        print(f"Time in Position:   {metrics['time_in_position']:>10.2%}")
        print('-'*80)
        print(f"💰 ANÁLISE P&L:")
        print(f"Average Win:        ${metrics['avg_win']:>10.2f}")
        print(f"Average Loss:       ${metrics['avg_loss']:>10.2f}")
        print(f"Profit Factor:      {metrics['profit_factor']:>10.2f}")
        print(f"Est. Commissions:   ${metrics['estimated_commissions']:>10.2f}")
        print('='*80)
    
    def save_report(self, metrics: dict, output_path: str):
        """Salva relatório em arquivo."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BACKTEST REPORT - MULTI-TIMEFRAME V16\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path.name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Balance Inicial:    ${metrics['initial_balance']:,.2f}\n")
            f.write(f"Balance Final:      ${metrics['final_balance']:,.2f}\n")
            f.write(f"Total Return:       {metrics['total_return']:.2%}\n")
            f.write(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown:       {metrics['max_drawdown']:.2%}\n\n")
            
            f.write("TRADING METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Trades:       {metrics['total_trades']}\n")
            f.write(f"Winning Trades:     {metrics['winning_trades']}\n")
            f.write(f"Losing Trades:      {metrics['losing_trades']}\n")
            f.write(f"Win Rate:           {metrics['win_rate']:.2%}\n")
            f.write(f"Time in Position:   {metrics['time_in_position']:.2%}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n💾 Relatório salvo: {output_path}")


def main():
    """Executa backtest no modelo V16 final."""
    
    print("\n" + "="*80)
    print("🚀 BACKTEST MULTI-TIMEFRAME V16")
    print("="*80 + "\n")
    
    # Paths dos dados de TESTE
    data_paths = {
        '15m': 'data/test_btcusdt_36m_15m_20260125.csv',
        '1h': 'data/test_btcusdt_36m_1h_20260125.csv',
        '4h': 'data/test_btcusdt_36m_4h_20260125.csv'
    }
    
    # Path do modelo V16.3 - TESTE CHECKPOINT 500K
    model_path = './models/sac_v16_multi_tf_20260219_023720_500000_steps.zip'
    
    # Verificar se existe
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("\nCheckpoints disponíveis:")
        models_dir = Path('./models')
        v16_models = sorted(models_dir.glob('sac_v16_multi_tf_20260218_233227_*.zip'))
        for i, m in enumerate(v16_models[-10:], 1):
            print(f"  {i}. {m.name}")
        return
    
    # Criar backtester
    backtester = BacktesterMultiTF(
        model_path=model_path,
        data_paths=data_paths,
        window_size=50
    )
    
    # Executar backtest (1 episódio para teste rápido 200k)
    print("\n🎯 Iniciando backtest CHECKPOINT 200K...\n")
    metrics = backtester.run(episodes=1, verbose=True)
    
    # Salvar relatório
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'backtest_v16_report_{timestamp}.txt'
    backtester.save_report(metrics, report_path)
    
    print("\n✅ Backtest concluído!")
    print(f"📊 Relatório: {report_path}\n")


if __name__ == "__main__":
    main()
