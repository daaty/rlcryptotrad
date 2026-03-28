"""
Notificações Telegram para eventos críticos do trading bot.

Configuração em config.yaml:
    notifications:
      telegram:
        enabled: false
        token: "BOT_TOKEN_AQUI"
        chat_id: "CHAT_ID_AQUI"
        events:
          - sl         # Stop Loss acionado
          - tp         # Take Profit acionado
          - trade      # Qualquer abertura/fechamento
          - drawdown   # Drawdown > 10%
          - ws_down    # WS stale > 5min
          - engine_err # Erro crítico no engine

Uso (thread-safe; envios são assíncronos via background thread):
    from dashboard.integrations.telegram_notifier import get_notifier
    notifier = get_notifier(config)
    notifier.notify_sl("BTCUSDT", -2.3)
    notifier.notify_trade("BTCUSDT", "LONG", 0.001, 95000.0)
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Singleton global ──────────────────────────────────────────────────────────
_instance: Optional["TelegramNotifier"] = None
_instance_lock = threading.Lock()


def get_notifier(config: dict | None = None) -> "TelegramNotifier":
    """Retorna singleton do TelegramNotifier, criando se necessário."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TelegramNotifier(config or {})
    elif config is not None:
        # Atualiza config se fornecida novamente (ex: reload)
        _instance._reload_config(config)
    return _instance


class TelegramNotifier:
    """
    Envia mensagens Telegram de forma assíncrona (fila + worker thread).
    Nunca bloqueia o loop de trading.
    """

    def __init__(self, config: dict) -> None:
        self._tg_cfg: dict = {}
        self._enabled = False
        self._events: set[str] = set()
        self._token: str = ""
        self._chat_id: str = ""
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._worker: threading.Thread | None = None
        self._reload_config(config)

    # ── Config ────────────────────────────────────────────────────────────────

    def _reload_config(self, config: dict) -> None:
        tg = config.get("notifications", {}).get("telegram", {})
        self._tg_cfg   = tg
        self._enabled  = bool(tg.get("enabled", False))
        self._token    = str(tg.get("token", ""))
        self._chat_id  = str(tg.get("chat_id", ""))
        self._events   = set(tg.get("events", [
            "sl", "tp", "trade", "drawdown", "ws_down", "engine_err"
        ]))

        if self._enabled and self._token and self._chat_id:
            self._ensure_worker()
            logger.info("[TELEGRAM] Notificações ativadas")
        elif self._enabled:
            logger.warning("[TELEGRAM] enabled=true mas token/chat_id ausente — desabilitado")
            self._enabled = False

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._send_loop, daemon=True, name="TelegramWorker"
            )
            self._worker.start()

    def _send_loop(self) -> None:
        while True:
            try:
                msg = self._queue.get(timeout=5)
                self._do_send(msg)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                logger.debug(f"[TELEGRAM] Worker error: {exc}")

    def _do_send(self, text: str) -> None:
        """Faz a chamada HTTP real ao Telegram API."""
        try:
            import urllib.request, urllib.parse, json as _json
            url  = f"https://api.telegram.org/bot{self._token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id":    self._chat_id,
                "text":       text,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = _json.loads(resp.read())
                if not result.get("ok"):
                    logger.warning(f"[TELEGRAM] API error: {result}")
        except Exception as exc:
            logger.warning(f"[TELEGRAM] Falha ao enviar: {exc}")

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def send(self, text: str, event: str = "trade") -> None:
        """Enfileira mensagem se event está ativo e notifier está habilitado."""
        if not self._enabled:
            return
        if event not in self._events:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.debug("[TELEGRAM] Fila cheia — mensagem descartada")

    # ── Formatadores de evento ────────────────────────────────────────────────

    def notify_trade(
        self,
        sym: str,
        side: str,
        qty: float,
        price: float,
        pnl: float | None = None,
    ) -> None:
        icon = "📈" if side in ("LONG", "BUY") else "📉"
        pnl_str = f" | P&L: <b>{pnl:+.4f} USDT</b>" if pnl is not None else ""
        self.send(
            f"{icon} <b>{side} {sym}</b>\n"
            f"Qty: {qty} @ ${price:,.4f}{pnl_str}\n"
            f"⏱ {_now()}",
            event="trade",
        )

    def notify_sl(self, sym: str, pnl_pct: float) -> None:
        self.send(
            f"🛑 <b>STOP LOSS</b> — {sym}\n"
            f"P&L: <b>{pnl_pct:+.2f}%</b>\n"
            f"⏱ {_now()}",
            event="sl",
        )

    def notify_tp(self, sym: str, level: int, pnl_pct: float) -> None:
        self.send(
            f"🎯 <b>TAKE PROFIT L{level}</b> — {sym}\n"
            f"P&L: <b>+{pnl_pct:.2f}%</b>\n"
            f"⏱ {_now()}",
            event="tp",
        )

    def notify_drawdown(self, drawdown_pct: float) -> None:
        self.send(
            f"⚠️ <b>DRAWDOWN ALTO</b>\n"
            f"Drawdown atual: <b>{drawdown_pct:.2f}%</b>\n"
            f"⏱ {_now()}",
            event="drawdown",
        )

    def notify_ws_down(self, sym: str, age_secs: int) -> None:
        self.send(
            f"📡 <b>WS STALE</b> — {sym}\n"
            f"Último candle há {age_secs//60:.0f}min {age_secs%60:.0f}s\n"
            f"⏱ {_now()}",
            event="ws_down",
        )

    def notify_engine_error(self, error: str) -> None:
        self.send(
            f"❌ <b>ERRO ENGINE</b>\n"
            f"<code>{error[:300]}</code>\n"
            f"⏱ {_now()}",
            event="engine_err",
        )


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%d/%m %H:%M:%S")
