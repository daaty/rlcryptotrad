"""
Execução de ordens — abre, fecha e gerencia posições Binance Futures.
Camada pura sem dependência de Streamlit.
"""
from __future__ import annotations
import math

from binance.client import Client

from dashboard.core.config import get_quantity_precision
from dashboard.core.logging_setup import get_logger
from dashboard.analytics.risk_calculator import kelly_position_size

logger = get_logger()

# ── Precision cache (evita chamar futures_exchange_info() a cada trade) ───────
_qty_precision_cache: dict[str, int] = {}


def _get_qty_precision(client: Client, symbol: str, config: dict) -> int:
    """
    Retorna número de casas decimais para quantidade, baseado no stepSize real
    da exchange (obtido via futures_exchange_info(), cached por símbolo).
    Fallback para get_quantity_precision() do config se a chamada falhar.
    """
    if symbol in _qty_precision_cache:
        return _qty_precision_cache[symbol]
    try:
        info = client.futures_exchange_info()
        for s in info.get('symbols', []):
            if s['symbol'] == symbol:
                for f in s.get('filters', []):
                    if f['filterType'] == 'LOT_SIZE':
                        step = f['stepSize']   # e.g. '1', '0.001', '0.100'
                        # conta decimais significativos do stepSize
                        if '.' in step:
                            precision = len(step.rstrip('0').split('.')[1])
                        else:
                            precision = 0
                        _qty_precision_cache[symbol] = precision
                        logger.debug(f"[PRECISION] {symbol}: stepSize={step} → {precision} decimais")
                        return precision
    except Exception as exc:
        logger.warning(f"[PRECISION] Não foi possível buscar exchange info: {exc}")
    # fallback: mapa hardcoded / config.yaml
    return get_quantity_precision(config, symbol)


def _floor_qty(quantity: float, precision: int) -> float:
    """Arredonda PARA BAIXO (floor) com `precision` decimais — nunca envia mais do que o saldo."""
    factor = 10 ** precision
    return math.floor(quantity * factor) / factor


def execute_trade(
    client: Client,
    decision: str,
    current_price: float,
    config: dict,
    ws_position_amt: float | None = None,
    ws_available_balance: float | None = None,
    symbol: str | None = None,
    closed_trades: list | None = None,
    paper_mode: bool = False,
) -> dict | None:
    """
    Executa trade baseado na decisão do modelo LSTM.

    Args:
        client: binance.Client
        decision: 'LONG', 'SHORT' ou 'FLAT'
        current_price: preço atual de mercado
        config: dict do config.yaml
        ws_position_amt: posição atual via WS (evita REST futures_position_information)
        ws_available_balance: saldo disponível via WS (evita REST futures_account_balance)
        symbol: símbolo explícito (ex: 'BTCUSDT'). Se None, lê de config['data']['primary_symbol'].
        closed_trades: histórico de trades fechados para Kelly sizing.
        paper_mode: Se True, simula fill sem chamar a Binance (modo paper trading).

    Returns:
        dict da ordem Binance ou None se não houve execução.
    """
    try:
        if symbol is None:
            symbol = config['data'].get('primary_symbol', 'BTC/USDT').replace('/', '')

        # ── Paper mode: simula fill imediato sem chamada real à Binance ─────────
        if paper_mode and decision != 'FLAT':
            import time as _t
            side = 'BUY' if decision == 'LONG' else 'SELL'
            avail = ws_available_balance or 0.0
            if avail <= 0:
                logger.warning(f"[PAPER] Saldo WS indisponível — abortando simulação de {decision}")
                return None
            position_size = config['environment']['position_size']
            leverage      = config['environment']['leverage']
            kelly_frac    = config.get('risk_management', {}).get('kelly_fraction', 0.25)
            kelly_usdt    = kelly_position_size(
                balance=avail, closed_trades=closed_trades or [],
                kelly_fraction=kelly_frac, fallback_pct=position_size,
            )
            precision = get_quantity_precision(config, symbol)
            quantity  = _floor_qty(kelly_usdt * leverage / current_price, precision)
            min_notional = config.get('risk_management', {}).get('min_notional_usdt', 20.0)
            if quantity <= 0 or quantity * current_price < min_notional:
                logger.warning(
                    f"[PAPER] Notional ${quantity * current_price:.2f} < mínimo ${min_notional:.2f} "
                    f"— saldo insuficiente para posição útil (balance=${avail:.2f})"
                )
                return None
            paper_order = {
                'orderId':  f'PAPER_{int(_t.time()*1000)}',
                'symbol':   symbol,
                'side':     side,
                'origQty':  str(quantity),
                'avgPrice': str(current_price),
                'status':   'FILLED',
                'paper':    True,
            }
            logger.info(f"[PAPER] {side} {symbol} qty={quantity} @ ${current_price:,.4f}")
            return paper_order

        # ── Posição atual: usa WS cache (zero REST) se disponível ───────────────────
        if ws_position_amt is not None:
            current_position = ws_position_amt
        else:
            positions = client.futures_position_information(symbol=symbol)
            current_position = 0.0
            for pos in positions:
                if pos['symbol'] == symbol:
                    current_position = float(pos['positionAmt'])
                    break

        logger.info(f"[TRADE] Posição atual: {current_position}, decisão: {decision}")

        position_size = config['environment']['position_size']
        leverage      = config['environment']['leverage']
        precision     = _get_qty_precision(client, symbol, config)
        kelly_frac    = config.get('risk_management', {}).get('kelly_fraction', 0.25)

        if decision == 'LONG' and current_position <= 0:
            if current_position < 0:
                close_qty = _floor_qty(abs(current_position), precision)
                logger.info(f"[TRADE] Fechando SHORT de {current_position} → qty={close_qty}")
                client.futures_create_order(
                    symbol=symbol, side='BUY', type='MARKET', quantity=close_qty
                )
            # ── Saldo: usa WS cache se disponível ───────────────────────────────────
            if ws_available_balance is not None:
                avail = ws_available_balance
            else:
                balance_info = client.futures_account_balance()
                usdt  = next((b for b in balance_info if b['asset'] == 'USDT'), None)
                avail = float(usdt['availableBalance']) if usdt else 0.0
            kelly_usdt_long = kelly_position_size(
                balance=avail,
                closed_trades=closed_trades or [],
                kelly_fraction=kelly_frac,
                fallback_pct=position_size,
            )
            quantity = _floor_qty(kelly_usdt_long * leverage / current_price, precision)
            if quantity <= 0:
                logger.warning(f"[TRADE] Quantidade calculada <= 0 ({symbol}), abortando")
                return None
            min_notional = config.get('risk_management', {}).get('min_notional_usdt', 20.0)
            if quantity * current_price < min_notional:
                logger.warning(
                    f"[TRADE] Notional ${quantity * current_price:.2f} < mínimo ${min_notional:.2f} "
                    f"({symbol} LONG), abortando — saldo insuficiente para posição útil"
                )
                return None
            logger.info(f"[TRADE] Abrindo LONG: {quantity} @ ${current_price:,.2f} (kelly=${kelly_usdt_long:.2f})")
            order = client.futures_create_order(
                symbol=symbol, side='BUY', type='MARKET', quantity=quantity
            )
            logger.info(f"[TRADE] ✅ LONG executado: {order['orderId']}")
            return order

        elif decision == 'SHORT' and current_position >= 0:
            if current_position > 0:
                close_qty = _floor_qty(current_position, precision)
                logger.info(f"[TRADE] Fechando LONG de {current_position} → qty={close_qty}")
                client.futures_create_order(
                    symbol=symbol, side='SELL', type='MARKET', quantity=close_qty
                )
            if ws_available_balance is not None:
                avail = ws_available_balance
            else:
                balance_info = client.futures_account_balance()
                usdt  = next((b for b in balance_info if b['asset'] == 'USDT'), None)
                avail = float(usdt['availableBalance']) if usdt else 0.0
            kelly_usdt = kelly_position_size(
                balance=avail,
                closed_trades=closed_trades or [],
                kelly_fraction=kelly_frac,
                fallback_pct=position_size,
            )
            quantity = _floor_qty(kelly_usdt * leverage / current_price, precision)
            if quantity <= 0:
                logger.warning(f"[TRADE] Quantidade calculada <= 0 ({symbol}), abortando")
                return None
            min_notional = config.get('risk_management', {}).get('min_notional_usdt', 20.0)
            if quantity * current_price < min_notional:
                logger.warning(
                    f"[TRADE] Notional ${quantity * current_price:.2f} < mínimo ${min_notional:.2f} "
                    f"({symbol} SHORT), abortando — saldo insuficiente para posição útil"
                )
                return None
            logger.info(f"[TRADE] Abrindo SHORT: {quantity} @ ${current_price:,.2f} (kelly=${kelly_usdt:.2f})")
            order = client.futures_create_order(
                symbol=symbol, side='SELL', type='MARKET', quantity=quantity
            )
            logger.info(f"[TRADE] ✅ SHORT executado: {order['orderId']}")
            return order

        elif decision == 'FLAT' and current_position != 0:
            side = 'SELL' if current_position > 0 else 'BUY'
            logger.info(f"[TRADE] Fechando posição {side}: {abs(current_position)}")
            order = client.futures_create_order(
                symbol=symbol, side=side, type='MARKET', quantity=abs(current_position)
            )
            logger.info(f"[TRADE] ✅ Posição fechada: {order['orderId']}")
            return order

        else:
            logger.info(f"[TRADE] Sem mudança (atual={current_position}, decisão={decision})")
            return None

    except Exception as exc:
        logger.error(f"[TRADE] Erro ao executar trade: {exc}")
        return None


def close_position_direct(
    client: Client,
    symbol: str,
    qty: float,
    config: dict | None = None,
) -> dict | None:
    """
    Fecha uma posição diretamente pelo símbolo e quantidade.
    qty > 0 → posição LONG (precisa SELL); qty < 0 → posição SHORT (precisa BUY).
    Usa reduceOnly=True para nunca abrir nova posição acidentalmente.
    """
    try:
        if qty == 0:
            return None

        precision = _get_qty_precision(client, symbol, config or {})
        quantity  = _floor_qty(abs(qty), precision)

        if quantity == 0:
            logger.warning(
                f"[CLOSE] ⚠️ Quantidade muito pequena após arredondamento: "
                f"{symbol} qty={qty:.6f} → {quantity}"
            )
            return None

        side  = 'SELL' if qty > 0 else 'BUY'
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity,
            reduceOnly=True,
        )
        logger.info(
            f"[CLOSE] ✅ {symbol} qty={qty:.6f} → {quantity} | "
            f"order={order['orderId']}"
        )
        return order

    except Exception as exc:
        logger.error(f"[CLOSE] ❌ Erro ao fechar {symbol}: {exc}")
        return None


def close_all_positions(client: Client, positions: list[dict], config: dict | None = None) -> list[dict]:
    """Fecha todas as posições abertas recebidas como parâmetro."""
    results = []
    for pos in positions:
        sym = pos['symbol']
        qty = float(pos['positionAmt'])
        order = close_position_direct(client, sym, qty, config)
        results.append({'symbol': sym, 'qty': qty, 'order': order})
    return results
