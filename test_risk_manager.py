"""
Teste das novas funcionalidades de Risk Management

Testa:
1. Stop Loss Dinâmico (ATR-based)
2. Take Profit com níveis parciais
3. Circuit Breaker (3 losses)
"""

import sys
sys.path.append('.')

from src.risk.risk_manager import RiskManager
import numpy as np


def test_atr_stop_loss():
    """Testa Stop Loss dinâmico baseado em ATR"""
    print("\n" + "="*60)
    print("🧪 TESTE 1: Stop Loss Dinâmico (ATR)")
    print("="*60)
    
    rm = RiskManager()
    
    # Cenário 1: LONG com ATR alto (alta volatilidade)
    entry_price = 91_000
    current_price = 90_000  # -1.1% de perda
    atr = 1000  # ATR alto = $1000
    position_type = 1  # LONG
    
    stop_price = rm.calculate_atr_stop_loss(entry_price, atr, position_type, multiplier=2.0)
    should_stop = rm.should_stop_loss(entry_price, current_price, position_type, atr=atr)
    
    print(f"\n📊 Cenário 1: LONG com alta volatilidade")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Current: ${current_price:,.2f} (-1.1%)")
    print(f"   ATR: ${atr:,.2f}")
    print(f"   Stop calculado: ${stop_price:,.2f}")
    print(f"   Should stop? {should_stop}")
    
    # Cenário 2: SHORT com ATR baixo (baixa volatilidade)
    entry_price = 91_000
    current_price = 92_000  # Perdendo no SHORT
    atr = 300  # ATR baixo = $300
    position_type = -1  # SHORT
    
    stop_price = rm.calculate_atr_stop_loss(entry_price, atr, position_type, multiplier=2.0)
    should_stop = rm.should_stop_loss(entry_price, current_price, position_type, atr=atr)
    
    print(f"\n📊 Cenário 2: SHORT com baixa volatilidade")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Current: ${current_price:,.2f} (+1.1%)")
    print(f"   ATR: ${atr:,.2f}")
    print(f"   Stop calculado: ${stop_price:,.2f}")
    print(f"   Should stop? {should_stop}")


def test_partial_take_profit():
    """Testa Take Profit com saída parcial"""
    print("\n" + "="*60)
    print("🧪 TESTE 2: Take Profit Parcial (50%/50%)")
    print("="*60)
    
    rm = RiskManager()
    
    # Cenário 1: +1.5% de lucro (não atinge nenhum TP)
    entry_price = 91_000
    current_price = 92_365  # +1.5%
    position_type = 1  # LONG
    
    should_tp, level = rm.should_take_profit(entry_price, current_price, position_type, return_level=True)
    
    print(f"\n📊 Cenário 1: +1.5% de lucro")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Current: ${current_price:,.2f} (+1.5%)")
    print(f"   Should TP? {should_tp}, Nível: {level}")
    
    # Cenário 2: +2.5% de lucro (atinge TP nível 1 = 50%)
    current_price = 93_275  # +2.5%
    should_tp, level = rm.should_take_profit(entry_price, current_price, position_type, return_level=True)
    
    if should_tp and level == 1:
        close_size = rm.calculate_partial_close_size(0.016, level)  # 0.016 BTC
        print(f"\n📊 Cenário 2: +2.5% de lucro (TP Nível 1)")
        print(f"   Entry: ${entry_price:,.2f}")
        print(f"   Current: ${current_price:,.2f} (+2.5%)")
        print(f"   ✅ TP Nível 1 atingido!")
        print(f"   Fechar: {close_size:.3f} BTC (50% da posição)")
    
    # Cenário 3: +5% de lucro (atinge TP nível 2 = 100%)
    current_price = 95_550  # +5%
    should_tp, level = rm.should_take_profit(entry_price, current_price, position_type, return_level=True)
    
    if should_tp and level == 2:
        close_size = rm.calculate_partial_close_size(0.008, level)  # Restante 50%
        print(f"\n📊 Cenário 3: +5% de lucro (TP Nível 2)")
        print(f"   Entry: ${entry_price:,.2f}")
        print(f"   Current: ${current_price:,.2f} (+5%)")
        print(f"   ✅ TP Nível 2 atingido!")
        print(f"   Fechar: {close_size:.3f} BTC (100% do restante)")


def test_circuit_breaker():
    """Testa Circuit Breaker (3 losses consecutivos)"""
    print("\n" + "="*60)
    print("🧪 TESTE 3: Circuit Breaker (3 Losses)")
    print("="*60)
    
    rm = RiskManager()
    
    # Simula sequência de trades
    trades = [
        ("Trade 1", -50),   # Loss
        ("Trade 2", -30),   # Loss
        ("Trade 3", -20),   # Loss (deve ativar circuit breaker)
        ("Trade 4", 100),   # Win (tentativa, mas deve ser bloqueado)
    ]
    
    for trade_name, pnl in trades:
        print(f"\n📊 {trade_name}: PnL = ${pnl}")
        
        # Verifica se pode tradear
        can_trade, reason = rm.should_allow_trade()
        
        if not can_trade:
            print(f"   ⛔ BLOQUEADO: {reason}")
            continue
        
        # Registra resultado
        rm.record_trade_result(pnl)
        
        # Mostra stats
        stats = rm.get_trading_stats()
        print(f"   Wins: {stats['wins']}, Losses: {stats['losses']}")
        print(f"   Losses consecutivos: {stats['consecutive_losses']}")
        print(f"   Circuit breaker: {'🔴 ATIVO' if stats['circuit_breaker_active'] else '🟢 OK'}")
    
    # Testa reset do circuit breaker
    print(f"\n🔄 Resetando circuit breaker...")
    rm.reset_circuit_breaker()
    can_trade, reason = rm.should_allow_trade()
    print(f"   Status: {reason}")


def test_kelly_criterion():
    """Testa Kelly Criterion Real"""
    print("\n" + "="*60)
    print("🧪 TESTE 4: Kelly Criterion")
    print("="*60)
    
    rm = RiskManager()
    
    # Cenário 1: Sistema lucrativo (win rate 60%, avg win/loss = 1.5)
    balance = 5000
    win_rate = 0.60
    avg_win = 150
    avg_loss = 100
    confidence = 1.0
    
    position_size = rm.calculate_position_size(balance, win_rate, avg_win, avg_loss, confidence)
    
    print(f"\n📊 Cenário 1: Sistema lucrativo")
    print(f"   Balance: ${balance:,.2f}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Kelly Position Size: ${position_size:,.2f} ({position_size/balance*100:.1f}% do capital)")
    
    # Cenário 2: Sistema ruim (win rate 45%)
    win_rate = 0.45
    position_size = rm.calculate_position_size(balance, win_rate, avg_win, avg_loss, confidence)
    
    print(f"\n📊 Cenário 2: Sistema com win rate baixo")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Kelly Position Size: ${position_size:,.2f} ({position_size/balance*100:.1f}% do capital)")


if __name__ == "__main__":
    print("\n" + "🔬 TESTES DE RISK MANAGEMENT".center(60, "="))
    
    test_atr_stop_loss()
    test_partial_take_profit()
    test_circuit_breaker()
    test_kelly_criterion()
    
    print("\n" + "="*60)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print("="*60 + "\n")
