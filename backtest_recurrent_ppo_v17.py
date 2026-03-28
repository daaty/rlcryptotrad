"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            📊 BACKTEST RECURRENT PPO V17 - LSTM MULTI-TIMEFRAME              ║
║                                                                              ║
║  🎯 Testa modelo RecurrentPPO treinado com LSTM                              ║
║  🔧 Usa observations sequenciais (50, 29)                                    ║
║  🧠 Mantém LSTM states entre predictions                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from pathlib import Path
import sys
from datetime import datetime

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("❌ ERRO: stable-baselines3-contrib não instalado!")
    print("   Instale com: pip install sb3-contrib")
    sys.exit(1)

from src.environment.trading_env_multi_tf_lstm import TradingEnvMultiTFLSTM

# ============= CONFIGURAÇÃO =============
MODEL_PATH = "models/recurrent_ppo_v17_lstm_20260221_030417_600000_steps.zip"  # V17.7 - 600k steps (melhor checkpoint)

def find_latest_test_data():
    """Encontra dados de teste mais recentes."""
    data_dir = Path('data')
    
    files_15m = sorted(data_dir.glob('test_btcusdt_*_15m_*.csv'), reverse=True)
    files_1h = sorted(data_dir.glob('test_btcusdt_*_1h_*.csv'), reverse=True)
    files_4h = sorted(data_dir.glob('test_btcusdt_*_4h_*.csv'), reverse=True)
    
    if not files_15m or not files_1h or not files_4h:
        print("❌ ERRO: Dados de teste não encontrados!")
        sys.exit(1)
    
    return {
        '15m': str(files_15m[0]),
        '1h': str(files_1h[0]),
        '4h': str(files_4h[0])
    }

# Configuração ambiente (mesma do treino)
ENV_CONFIG = {
    'window_size': 50,
    'max_episode_steps': 2000,
    'leverage': 1.0,
    'commission': 0.0004,
    'slippage': 0.0005,
    'position_size': 0.05,
    'use_sharpe_reward': False,
    'enable_indicator_shaping': False,
    'random_start': False,  # Determinístico para teste
    'persist_balance': False,
    'liquidation_threshold': 0.30,
}

class RecurrentBacktest:
    """Backtest para modelos RecurrentPPO com LSTM."""
    
    def __init__(self, model_path, data_paths, env_config):
        print(f"📂 Carregando modelo: {model_path}")
        self.model = RecurrentPPO.load(model_path)
        
        print("🏗️  Criando ambiente de teste...")
        self.env = TradingEnvMultiTFLSTM(data_paths=data_paths, **env_config)
        
        self.trades = []
        self.episode_rewards = []
        self.actions_history = []
        # action signal counters
        self._action_long_signals  = 0
        self._action_short_signals = 0
        self._action_flat_signals  = 0
        # per-trade tracking
        self._current_side = 0  # direction of open position
        
    def run(self, deterministic=False):
        """
        Roda backtest completo.
        
        Args:
            deterministic: Se True, usa mean da distribuição (não recomendado para SAC/PPO)
                          Se False, samplea da distribuição (recomendado)
        """
        print("\n" + "="*80)
        print("🚀 INICIANDO BACKTEST LSTM")
        print("="*80)
        print(f"🎲 Deterministic: {deterministic}")
        print(f"📊 Max episode steps: {self.env.max_episode_steps}")
        print("="*80 + "\n")
        
        obs, info = self.env.reset()  # FIX: Gymnasium returns (obs, info)
        done = False
        step = 0
        
        # CRÍTICO: LSTM states iniciais
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        while not done and step < self.env.max_episode_steps:
            # Predict com LSTM states
            # RecurrentPPO retorna (action, lstm_states)
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic
            )
            
            # Step no ambiente (Gymnasium retorna 5 valores)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Após primeiro step, episode_start = False
            episode_start = np.zeros((1,), dtype=bool)
            
            # Logging
            self.episode_rewards.append(reward)
            act_val = action[0] if isinstance(action, np.ndarray) else action
            self.actions_history.append(act_val)
            # Count action signals
            if act_val > 0.1:
                self._action_long_signals  += 1
            elif act_val < -0.1:
                self._action_short_signals += 1
            else:
                self._action_flat_signals  += 1

            if info.get('trade_executed'):
                self.trades.append({
                    'pnl':   info.get('pnl', 0),
                    '_side': info.get('_side', 0),
                })
            
            step += 1
            
            if step % 200 == 0:
                print(f"  Step {step:4d} | Action: {action[0]:+.4f} | Reward: {reward:+.4f} | Equity: ${info.get('equity', 0):.2f}")
        
        print("\n" + "="*80)
        print("✅ BACKTEST CONCLUÍDO")
        print("="*80)
        
        self._print_results()
    
    def _print_results(self):
        """Calcula e exibe métricas detalhadas com breakdown LONG/SHORT."""
        info = self.env._get_info()

        # Métricas gerais
        total_trades   = info['trades']
        winning_trades = info['wins']
        losing_trades  = info['losses']
        win_rate       = info['win_rate'] / 100

        long_trades  = info['long_trades']
        short_trades = info['short_trades']
        long_wins    = info['long_wins']
        long_losses  = info['long_losses']
        long_wr      = info['long_wr'] / 100
        short_wins   = info['short_wins']
        short_losses = info['short_losses']
        short_wr     = info['short_wr'] / 100

        initial      = self.env.initial_balance
        final        = info['equity']
        total_return = ((final - initial) / initial) * 100

        # P&L global
        all_pnl    = [t['pnl'] for t in self.trades if 'pnl' in t]
        wins_pnl   = sum(p for p in all_pnl if p > 0)
        losses_pnl = abs(sum(p for p in all_pnl if p < 0))
        avg_win    = wins_pnl  / winning_trades if winning_trades > 0 else 0
        avg_loss   = losses_pnl / losing_trades if losing_trades  > 0 else 0
        profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else float('inf')
        total_pnl  = sum(all_pnl)

        # P&L por lado
        long_pnl  = sum(t['pnl'] for t in self.trades if t.get('_side') ==  1 and 'pnl' in t)
        short_pnl = sum(t['pnl'] for t in self.trades if t.get('_side') == -1 and 'pnl' in t)

        long_avg_win   = (sum(t['pnl'] for t in self.trades if t.get('_side')==1  and t.get('pnl',0)>0) / long_wins)   if long_wins   > 0 else 0
        long_avg_loss  = (abs(sum(t['pnl'] for t in self.trades if t.get('_side')==1  and t.get('pnl',0)<0)) / long_losses)  if long_losses  > 0 else 0
        short_avg_win  = (sum(t['pnl'] for t in self.trades if t.get('_side')==-1 and t.get('pnl',0)>0) / short_wins)  if short_wins  > 0 else 0
        short_avg_loss = (abs(sum(t['pnl'] for t in self.trades if t.get('_side')==-1 and t.get('pnl',0)<0)) / short_losses) if short_losses > 0 else 0

        # Comissões
        position_value        = self.env.position_size * initial
        estimated_commissions = total_trades * 2 * self.env.commission * position_value

        # Rewards / actions
        actions           = np.array(self.actions_history)
        avg_reward        = np.mean(self.episode_rewards)
        cumulative_reward = np.sum(self.episode_rewards)

        W    = 80
        SEP  = "=" * W
        SEP2 = "-" * W

        print(f"\n{SEP}")
        print(f"  V17.7 BACKTEST — {datetime.now().strftime('%Y-%m-%d %H:%M')}".center(W))
        print(SEP)

        print(f"\n{'RESULTADO GERAL':^{W}}")
        print(SEP2)
        print(f"  Balance inicial  : ${initial:>12,.2f}")
        print(f"  Balance final    : ${final:>12,.2f}")
        print(f"  Retorno total    : {total_return:>+12.2f}%")
        print(f"  P&L líquido      : ${total_pnl:>+12.2f}")
        print(f"  Commissions est. : ${estimated_commissions:>12.2f}  ({estimated_commissions/initial*100:.2f}% do capital)")
        print(f"  P&L bruto (s/com): ${total_pnl + estimated_commissions:>+12.2f}")
        print(f"  Liquidations     : {info['liquidations']:>12}")

        print(f"\n{'TRADES — LONG vs SHORT':^{W}}")
        print(SEP2)
        print(f"  {'Métrica':<28} {'Total':>10} {'  LONG':>10} {' SHORT':>10}")
        print(SEP2)
        print(f"  {'Trades abertos':<28} {long_trades+short_trades:>10} {long_trades:>10} {short_trades:>10}")
        print(f"  {'Trades fechados':<28} {total_trades:>10} {long_wins+long_losses:>10} {short_wins+short_losses:>10}")
        print(f"  {'Wins':<28} {winning_trades:>10} {long_wins:>10} {short_wins:>10}")
        print(f"  {'Losses':<28} {losing_trades:>10} {long_losses:>10} {short_losses:>10}")
        print(f"  {'Win Rate':<28} {win_rate:>10.1%} {long_wr:>10.1%} {short_wr:>10.1%}")
        print(f"  {'P&L total':<28} ${total_pnl:>+9.2f} ${long_pnl:>+9.2f} ${short_pnl:>+9.2f}")
        print(f"  {'Avg Win':<28} ${avg_win:>9.2f} ${long_avg_win:>9.2f} ${short_avg_win:>9.2f}")
        print(f"  {'Avg Loss':<28} ${avg_loss:>9.2f} ${long_avg_loss:>9.2f} ${short_avg_loss:>9.2f}")
        print(f"  {'Profit Factor':<28} {profit_factor:>10.2f}")

        print(f"\n{'SINAIS DE AÇÃO DO MODELO':^{W}}")
        print(SEP2)
        total_signals = len(self.actions_history)
        pct_long  = self._action_long_signals  / total_signals * 100 if total_signals else 0
        pct_short = self._action_short_signals / total_signals * 100 if total_signals else 0
        pct_flat  = self._action_flat_signals  / total_signals * 100 if total_signals else 0
        print(f"  Total de steps        : {total_signals:>8}")
        print(f"  Sinais LONG  (> 0.1)  : {self._action_long_signals:>8}  ({pct_long:5.1f}%)")
        print(f"  Sinais SHORT (< -0.1) : {self._action_short_signals:>8}  ({pct_short:5.1f}%)")
        print(f"  Sinais FLAT  (±0.1)   : {self._action_flat_signals:>8}  ({pct_flat:5.1f}%)")
        print(f"  Action mean  : {np.mean(actions):>+.4f}   std: {np.std(actions):.4f}   "
              f"min: {np.min(actions):+.4f}   max: {np.max(actions):+.4f}")
        bias = 'LONG' if np.mean(actions) > 0.02 else ('SHORT' if np.mean(actions) < -0.02 else 'NEUTRO')
        print(f"  Viés do modelo : {bias}")

        print(f"\n{'REWARDS':^{W}}")
        print(SEP2)
        print(f"  Avg Reward        : {avg_reward:>+.4f}")
        print(f"  Cumulative Reward : {cumulative_reward:>+.2f}")

        print(f"\n{SEP}")
        print("  DIAGNÓSTICO".center(W))
        print(SEP2)
        checks = [
            (total_trades < 600,
                "LSTM reduziu overtrading",          f"Overtrading alto ({total_trades} trades > 600)"),
            (win_rate > 0.35,
                f"Win rate OK ({win_rate:.1%})",      f"Win rate baixa ({win_rate:.1%} < 35%)"),
            (total_return > 0,
                f"Retorno POSITIVO ({total_return:+.2f}%)", f"Retorno negativo ({total_return:+.2f}%)"),
            (profit_factor > 1.0,
                f"Profit factor OK ({profit_factor:.2f})", f"Profit factor fraco ({profit_factor:.2f})"),
            (abs(long_wr - short_wr) < 0.15,
                "Equilíbrio LONG/SHORT OK",          f"Desequilíbrio: LONG {long_wr:.1%} vs SHORT {short_wr:.1%}"),
            (0.25 < pct_long / max(pct_short, 0.01) < 4.0,
                "Modelo usa LONG e SHORT",           f"Viés forte: {pct_long:.0f}% LONG vs {pct_short:.0f}% SHORT"),
        ]
        for ok, good, bad in checks:
            print(f"  {'[OK]' if ok else '[!!]'}  {good if ok else bad}")
        print(SEP)
        print()

def main():
    # Verificar se modelo existe
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ ERRO: Modelo não encontrado: {MODEL_PATH}")
        print("\n💡 Modelos disponíveis:")
        models_dir = Path("models")
        lstm_models = sorted(models_dir.glob("recurrent_ppo_v17_lstm_*.zip"))
        if lstm_models:
            for m in lstm_models[-10:]:  # Últimos 10
                print(f"   - {m.name}")
        else:
            print("   (nenhum modelo LSTM encontrado)")
        sys.exit(1)
    
    # Encontrar dados de teste
    print("🔍 Procurando dados de teste...")
    data_paths = find_latest_test_data()
    
    print("\n📂 DADOS DE TESTE:")
    for tf, path in data_paths.items():
        print(f"   {tf:>3}: {Path(path).name}")
    
    # Criar backtest
    print()
    backtest = RecurrentBacktest(
        model_path=str(model_path),
        data_paths=data_paths,
        env_config=ENV_CONFIG
    )
    
    # Rodar com deterministic=False (RECOMENDADO para PPO)
    print("\n💡 Usando deterministic=False (sampling da distribuição)\n")
    backtest.run(deterministic=False)
    
    # Salvar relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"backtest_report_lstm_{timestamp}.txt"
    
    # TODO: Implementar salvamento de relatório
    print(f"📝 Relatório: {report_path}")

if __name__ == "__main__":
    main()
