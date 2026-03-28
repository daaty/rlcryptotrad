"""
BACKTEST COM DETERMINISTIC=FALSE
Testa se o problema é o deterministic=True removendo exploração
"""
import sys
sys.path.append('.')

from backtest import Backtester

# Patch para usar deterministic=False
class StochasticBacktester(Backtester):
    def _run_episode(self, verbose: bool = True):
        """Executa episódio com EXPLORAÇÃO (deterministic=False)."""
        obs, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        max_steps = self.env.max_episode_steps
        
        self.history = {
            'balance': [self.env.balance],
            'equity': [self.env.equity],
            'position': [self.env.position],
            'trades': [],
            'actions': []
        }
        
        while not (done or truncated) and step < max_steps:
            # *** MUDANÇA: deterministic=FALSE ***
            action, _states = self.model.predict(obs, deterministic=False)
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            self.history['balance'].append(self.env.balance)
            self.history['equity'].append(self.env.equity)
            self.history['position'].append(self.env.position)
            self.history['actions'].append(action)
            
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
            
            if verbose and step % 1000 == 0:
                print(f"  Step {step}: Balance=${self.env.balance:.2f}, Equity=${self.env.equity:.2f}, Position={self.env.position}")
        
        metrics = self._calculate_metrics()
        
        if verbose:
            print(f"\n=== RESULTADOS (STOCHASTIC) ===")
            print(f"Balance Final: ${metrics['final_balance']:.2f}")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Trades: {metrics['total_trades']}")
            print(f"Long: {metrics['long_pct']:.1%} | Short: {metrics['short_pct']:.1%} | Flat: {metrics['flat_pct']:.1%}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
        
        return metrics

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python backtest_stochastic.py <modelo.zip> <dados.csv>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    print("="*70)
    print("BACKTEST STOCHASTIC (deterministic=False)")
    print("="*70)
    
    bt = StochasticBacktester(model_path, data_path)
    results = bt.run(episodes=1, verbose=True)
