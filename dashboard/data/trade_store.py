"""
Persistência de trades em SQLite.

Armazena closed_trades e orders do TradingEngine de forma permanente —
sobrevive a restarts do terminal, reloads do Streamlit e reinicializações da engine.

Schema:
  closed_trades(id, ts, symbol, side, qty, entry_price, exit_price,
                realized_pnl, session_id)
  orders(id, ts, symbol, side, qty, price, action, order_id, session_id)

Uso:
    from dashboard.data.trade_store import TradeStore, get_trade_store
    store = get_trade_store()
    store.save_closed_trade({...})
    trades = store.load_closed_trades(limit=500)
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from dashboard.core.logging_setup import get_logger

logger = get_logger()

DB_PATH = Path("data/trades.db")

# ── Session ID: identifica o período de cada execução da engine ──────────────
_SESSION_ID: str = str(uuid.uuid4())[:8]


class TradeStore:
    """
    Thread-safe SQLite store para trades e ordens.

    Características:
      - WAL mode: leituras e escritas simultâneas sem lock total
      - Thread-local connections: cada thread tem sua própria conexão
      - Atomic inserts via parametrized queries (sem SQL injection)
      - Índices em ts e symbol para queries rápidas
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._tl     = threading.local()  # conexões thread-local
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"[STORE] SQLite inicializado: {self.db_path} (session={_SESSION_ID})")

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Retorna conexão thread-local, criando-a se necessário."""
        if not getattr(self._tl, "conn", None):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._tl.conn = conn
        try:
            yield self._tl.conn
        except Exception:
            self._tl.conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Cria tabelas e índices se não existirem."""
        ddl = """
        CREATE TABLE IF NOT EXISTS closed_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,          -- ISO-8601
            ts_epoch    REAL    NOT NULL,           -- epoch float para sort rápido
            symbol      TEXT    NOT NULL,
            side        TEXT    NOT NULL,           -- 'LONG', 'SHORT', 'TP1', 'TP2', 'SL', 'TRAIL'
            qty         REAL    NOT NULL DEFAULT 0,
            entry_price REAL    NOT NULL DEFAULT 0,
            exit_price  REAL    NOT NULL DEFAULT 0,
            realized_pnl REAL   NOT NULL DEFAULT 0,
            session_id  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ct_ts     ON closed_trades(ts_epoch);
        CREATE INDEX IF NOT EXISTS idx_ct_symbol ON closed_trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_ct_session ON closed_trades(session_id);

        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            ts_epoch    REAL    NOT NULL,
            symbol      TEXT    NOT NULL,
            side        TEXT    NOT NULL,          -- 'BUY', 'SELL'
            qty         REAL    NOT NULL DEFAULT 0,
            price       TEXT    NOT NULL DEFAULT 'MKT',
            action      TEXT    NOT NULL DEFAULT '',  -- 'LONG', 'SHORT', 'FLAT'
            order_id    TEXT    NOT NULL DEFAULT '',
            session_id  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ord_ts      ON orders(ts_epoch);
        CREATE INDEX IF NOT EXISTS idx_ord_symbol  ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_ord_session ON orders(session_id);
        """
        with self._conn() as conn:
            conn.executescript(ddl)
            conn.commit()

    # ── Writes ────────────────────────────────────────────────────────────────

    def save_closed_trade(self, trade: dict) -> None:
        """
        Persiste um trade fechado.

        Campos esperados (mesmos do engine.state['closed_trades']):
          symbol, side, realizedPnl, time (ms epoch), entryPrice, exitPrice, qty
        """
        try:
            ts_epoch = float(trade.get('time', time.time() * 1000)) / 1000
            ts_str   = datetime.fromtimestamp(ts_epoch).isoformat(timespec='seconds')
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO closed_trades
                        (ts, ts_epoch, symbol, side, qty, entry_price, exit_price,
                         realized_pnl, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts_str,
                    ts_epoch,
                    str(trade.get('symbol', '')),
                    str(trade.get('side', '')),
                    float(trade.get('qty', 0)),
                    float(trade.get('entryPrice', 0)),
                    float(trade.get('exitPrice', 0)),
                    float(trade.get('realizedPnl', 0)),
                    _SESSION_ID,
                ))
                conn.commit()
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao salvar closed_trade: {exc}")

    def save_order(self, order: dict) -> None:
        """
        Persiste uma ordem.

        Campos esperados (mesmos do engine.state['orders']):
          symbol, side, qty, price, timestamp, action, orderId
        """
        try:
            ts_str   = str(order.get('timestamp', datetime.now().isoformat(timespec='seconds')))
            try:
                ts_epoch = datetime.fromisoformat(ts_str).timestamp()
            except ValueError:
                ts_epoch = time.time()
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO orders
                        (ts, ts_epoch, symbol, side, qty, price, action, order_id, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts_str,
                    ts_epoch,
                    str(order.get('symbol', '')),
                    str(order.get('side', '')),
                    float(str(order.get('qty', 0))),
                    str(order.get('price', 'MKT')),
                    str(order.get('action', '')),
                    str(order.get('orderId', order.get('order_id', ''))),
                    _SESSION_ID,
                ))
                conn.commit()
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao salvar order: {exc}")

    # ── Reads ─────────────────────────────────────────────────────────────────

    def load_closed_trades(
        self,
        limit: int = 500,
        symbol: str | None = None,
        since_epoch: float | None = None,
    ) -> list[dict]:
        """Carrega trades fechados do banco (mais recentes primeiro)."""
        try:
            params: list = []
            where_clauses: list[str] = []
            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol)
            if since_epoch:
                where_clauses.append("ts_epoch >= ?")
                params.append(since_epoch)
            where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            params.append(limit)
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM closed_trades {where} ORDER BY ts_epoch DESC LIMIT ?",
                    params,
                ).fetchall()
            # Converte para o formato que o engine/tabs esperam
            return [
                {
                    'symbol':      r['symbol'],
                    'side':        r['side'],
                    'qty':         r['qty'],
                    'entryPrice':  r['entry_price'],
                    'exitPrice':   r['exit_price'],
                    'realizedPnl': r['realized_pnl'],
                    'time':        int(r['ts_epoch'] * 1000),
                    '_ts':         r['ts'],
                    '_session':    r['session_id'],
                }
                for r in reversed(rows)  # cronológico (mais antigo primeiro)
            ]
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao carregar closed_trades: {exc}")
            return []

    def load_orders(
        self,
        limit: int = 200,
        symbol: str | None = None,
    ) -> list[dict]:
        """Carrega ordens do banco (mais recentes primeiro)."""
        try:
            params: list = []
            where = ""
            if symbol:
                where = "WHERE symbol = ?"
                params.append(symbol)
            params.append(limit)
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM orders {where} ORDER BY ts_epoch DESC LIMIT ?",
                    params,
                ).fetchall()
            return [
                {
                    'symbol':    r['symbol'],
                    'side':      r['side'],
                    'qty':       str(r['qty']),
                    'price':     r['price'],
                    'timestamp': r['ts'],
                    'action':    r['action'],
                    'orderId':   r['order_id'],
                    '_session':  r['session_id'],
                }
                for r in reversed(rows)
            ]
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao carregar orders: {exc}")
            return []

    # ── Stats / Analytics ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Retorna estatísticas globais (todas as sessões)."""
        try:
            with self._conn() as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*)                               AS total_trades,
                        COUNT(DISTINCT session_id)             AS sessions,
                        MIN(ts)                                AS first_trade,
                        MAX(ts)                                AS last_trade,
                        SUM(realized_pnl)                      AS total_pnl,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                        AVG(realized_pnl)                      AS avg_pnl
                    FROM closed_trades
                """).fetchone()
            if row and row['total_trades']:
                return dict(row)
            return {}
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao calcular stats: {exc}")
            return {}

    def get_daily_pnl(self, days: int = 30) -> list[dict]:
        """Retorna PnL agrupado por dia."""
        try:
            since = time.time() - days * 86400
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT
                        DATE(ts) AS day,
                        SUM(realized_pnl) AS daily_pnl,
                        COUNT(*) AS n_trades,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins
                    FROM closed_trades
                    WHERE ts_epoch >= ?
                    GROUP BY DATE(ts)
                    ORDER BY day
                """, [since]).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao calcular daily_pnl: {exc}")
            return []

    def get_symbol_breakdown(self) -> list[dict]:
        """PnL por símbolo (todas as sessões)."""
        try:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT
                        symbol,
                        COUNT(*) AS trades,
                        SUM(realized_pnl) AS total_pnl,
                        AVG(realized_pnl) AS avg_pnl,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses
                    FROM closed_trades
                    GROUP BY symbol
                    ORDER BY total_pnl DESC
                """).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"[STORE] Erro ao calcular symbol_breakdown: {exc}")
            return []

    def current_session_id(self) -> str:
        return _SESSION_ID


# ── Singleton ─────────────────────────────────────────────────────────────────
# Uma instância por processo Python — thread-safe internamente.
_store: TradeStore | None = None
_store_lock = threading.Lock()


def get_trade_store() -> TradeStore:
    """Retorna o singleton do TradeStore (lazy init)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TradeStore()
    return _store
