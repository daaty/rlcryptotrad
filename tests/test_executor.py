"""
Testes para dashboard/trading/executor.py — usa mock do binance.Client.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest
from dashboard.trading.executor import execute_trade, close_position_direct


def _make_mock_order(sym="BTCUSDT", side="BUY", qty="0.001", price="95000.0"):
    return {
        "orderId": 123456,
        "symbol":  sym,
        "side":    side,
        "origQty": qty,
        "avgPrice": price,
        "status": "FILLED",
    }


@pytest.fixture()
def mock_client():
    client = MagicMock()
    client.futures_exchange_info.return_value = {
        "symbols": [{
            "symbol": "BTCUSDT",
            "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}],
        }]
    }
    client.futures_create_order.return_value = _make_mock_order()
    return client


class TestExecuteTrade:

    def test_paper_mode_long_returns_fake_order(self, mock_client, minimal_config):
        """Paper mode deve retornar ordem simulada sem chamar a Binance."""
        order = execute_trade(
            client=mock_client,
            decision="LONG",
            current_price=95_000.0,
            config=minimal_config,
            ws_position_amt=0.0,
            ws_available_balance=10_000.0,
            symbol="BTCUSDT",
            paper_mode=True,
        )
        assert order is not None
        assert order["paper"] is True
        assert order["side"] == "BUY"
        assert order["status"] == "FILLED"
        mock_client.futures_create_order.assert_not_called()

    def test_paper_mode_short_returns_sell(self, mock_client, minimal_config):
        order = execute_trade(
            client=mock_client, decision="SHORT",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.0, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=True,
        )
        assert order["side"] == "SELL"

    def test_paper_mode_flat_returns_none(self, mock_client, minimal_config):
        order = execute_trade(
            client=mock_client, decision="FLAT",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.0, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=True,
        )
        assert order is None

    def test_flat_with_no_position_returns_none(self, mock_client, minimal_config):
        """FLAT sem posição aberta → nenhuma ordem."""
        order = execute_trade(
            client=mock_client, decision="FLAT",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.0, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=False,
        )
        assert order is None
        mock_client.futures_create_order.assert_not_called()

    def test_long_opens_buy_order(self, mock_client, minimal_config):
        """LONG com posição flat → abre BUY MARKET."""
        minimal_config["mode"] = "testnet"
        order = execute_trade(
            client=mock_client, decision="LONG",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.0, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=False,
        )
        assert order is not None
        mock_client.futures_create_order.assert_called_once()
        call_kwargs = mock_client.futures_create_order.call_args
        assert call_kwargs.kwargs["side"] == "BUY"
        assert call_kwargs.kwargs["type"] == "MARKET"

    def test_long_no_open_when_already_long(self, mock_client, minimal_config):
        """Já em LONG (positionAmt > 0) → não abre de novo."""
        order = execute_trade(
            client=mock_client, decision="LONG",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.001, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=False,
        )
        assert order is None

    def test_short_closes_long_first(self, mock_client, minimal_config):
        """SHORT com LONG aberto → fecha LONG antes de abrir SHORT."""
        mock_client.futures_create_order.side_effect = [
            _make_mock_order(side="SELL"),  # close LONG
            _make_mock_order(side="SELL"),  # open SHORT
        ]
        order = execute_trade(
            client=mock_client, decision="SHORT",
            current_price=95_000.0, config=minimal_config,
            ws_position_amt=0.001, ws_available_balance=10_000.0,
            symbol="BTCUSDT", paper_mode=False,
        )
        # Deve ter feito 2 chamadas: fechar LONG + abrir SHORT
        assert mock_client.futures_create_order.call_count == 2

    def test_zero_quantity_returns_none(self, mock_client, minimal_config):
        """Saldo muito baixo → qty arredondada para 0 → retorna None."""
        order = execute_trade(
            client=mock_client, decision="LONG",
            current_price=95_000_000.0,  # preço absurdamente alto
            config=minimal_config,
            ws_position_amt=0.0, ws_available_balance=0.01,
            symbol="BTCUSDT", paper_mode=False,
        )
        assert order is None


class TestClosePositionDirect:

    def test_close_long_sends_sell(self, mock_client, minimal_config):
        order = close_position_direct(mock_client, "BTCUSDT", qty=0.001, config=minimal_config)
        assert order is not None
        call_kwargs = mock_client.futures_create_order.call_args.kwargs
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["reduceOnly"] is True

    def test_close_short_sends_buy(self, mock_client, minimal_config):
        order = close_position_direct(mock_client, "BTCUSDT", qty=-0.001, config=minimal_config)
        call_kwargs = mock_client.futures_create_order.call_args.kwargs
        assert call_kwargs["side"] == "BUY"

    def test_zero_qty_returns_none(self, mock_client, minimal_config):
        order = close_position_direct(mock_client, "BTCUSDT", qty=0, config=minimal_config)
        assert order is None
        mock_client.futures_create_order.assert_not_called()
