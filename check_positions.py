"""
Script para verificar posições abertas e testar lógica de Stop Loss
"""

import yaml
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.risk.risk_manager import RiskManager

# Carrega configuração
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

risk_mgr = RiskManager()

print("="*80)
print("VERIFICACAO DE STOP LOSS NAS POSICOES ABERTAS")
print("="*80)

# Dados das posições atuais (do dashboard)
positions = [
    {
        'symbol': 'ETHUSDT',
        'side': 'LONG',
        'entry': 3288.48,
        'mark': 2956.86,
        'sl': 3170,
        'qty': 0.07
    },
    {
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'entry': 95367.00,
        'mark': 89617.30,
        'sl': 91782,
        'qty': 0.004
    }
]

print(f"\nConfiguracao do Risk Manager:")
print(f"   Stop Loss: {risk_mgr.stop_loss_pct * 100}%")
print(f"   Take Profit: {risk_mgr.take_profit_pct * 100}%")

for pos in positions:
    print(f"\n{'='*80}")
    print(f"{pos['symbol']}")
    print(f"{'='*80}")
    
    position_type = 1 if pos['side'] == 'LONG' else -1
    entry_price = pos['entry']
    current_price = pos['mark']
    
    # Calcula PnL
    pnl_pct = ((current_price - entry_price) / entry_price) * position_type * 100
    pnl_usd = (current_price - entry_price) * pos['qty'] * position_type
    
    print(f"\nInformacoes:")
    print(f"   Tipo: {pos['side']}")
    print(f"   Preco Entrada: ${entry_price:,.2f}")
    print(f"   Preco Atual: ${current_price:,.2f}")
    print(f"   Stop Loss Config: ${pos['sl']:,.0f}")
    print(f"   Quantidade: {pos['qty']}")
    
    print(f"\nP&L:")
    print(f"   Percentual: {pnl_pct:+.2f}%")
    print(f"   USD: ${pnl_usd:+.2f}")
    
    # Verifica stop loss
    should_stop = risk_mgr.should_stop_loss(
        entry_price,
        current_price,
        position_type
    )
    
    # Verifica take profit
    should_tp, tp_level = risk_mgr.should_take_profit(
        entry_price,
        current_price,
        position_type,
        return_level=True
    )
    
    print(f"\nVerificacoes:")
    
    # Stop Loss
    stop_loss_pct_config = risk_mgr.stop_loss_pct * 100
    loss_atual = abs(min(0, pnl_pct))
    
    if should_stop:
        print(f"   [!] STOP LOSS ATINGIDO!")
        print(f"      Perda: {loss_atual:.2f}% (limite: {stop_loss_pct_config:.0f}%)")
        print(f"      ACAO: Fechar posicao imediatamente")
    else:
        print(f"   [OK] Stop Loss OK")
        print(f"      Perda: {loss_atual:.2f}% (limite: {stop_loss_pct_config:.0f}%)")
    
    # Take Profit
    if should_tp:
        print(f"   [*] TAKE PROFIT NIVEL {tp_level} ATINGIDO!")
        if tp_level == 1:
            print(f"      ACAO: Fechar 50% da posicao")
        else:
            print(f"      ACAO: Fechar 100% da posicao")
    else:
        print(f"   [ ] Take Profit nao atingido")
        tp_target = risk_mgr.take_profit_pct * 100
        print(f"      Lucro: {max(0, pnl_pct):.2f}% (alvo: {tp_target:.0f}%)")
    
    # Recomendação final
    print(f"\nRecomendacao Final:")
    if should_stop:
        print(f"   [!] FECHAR POSICAO AGORA - Stop Loss ultrapassado")
    elif should_tp:
        if tp_level == 2:
            print(f"   [OK] FECHAR POSICAO COMPLETA - Take Profit atingido")
        else:
            print(f"   [OK] FECHAR 50% - Take Profit Nivel 1")
    elif pnl_pct < -5:
        print(f"   [!] ATENCAO: Perda significativa ({pnl_pct:.2f}%)")
    else:
        print(f"   [i] MANTER posicao - dentro dos limites")

print(f"\n{'='*80}")
print("RESUMO GERAL")
print(f"{'='*80}")

total_stops = sum(1 for pos in positions if risk_mgr.should_stop_loss(
    pos['entry'], pos['mark'], 1 if pos['side'] == 'LONG' else -1
))

total_tps = 0
for pos in positions:
    should_tp, _ = risk_mgr.should_take_profit(
        pos['entry'], pos['mark'], 
        1 if pos['side'] == 'LONG' else -1,
        return_level=True
    )
    if should_tp:
        total_tps += 1

print(f"\n   Total Posicoes: {len(positions)}")
print(f"   [!] Stops Acionados: {total_stops}")
print(f"   [*] Take Profits Acionados: {total_tps}")

if total_stops > 0:
    print(f"\n   [!!!] ACAO URGENTE NECESSARIA!")
    print(f"   {total_stops} posicao(oes) precisa(m) ser fechada(s) pelo stop loss")
elif total_tps > 0:
    print(f"\n   [OK] Parabens! {total_tps} posicao(oes) no take profit")
else:
    print(f"\n   [i] Todas as posicoes dentro dos limites configurados")

print(f"\n{'='*80}\n")
