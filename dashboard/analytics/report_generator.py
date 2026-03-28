"""
dashboard/analytics/report_generator.py
----------------------------------------
Generates a PDF performance report from closed trades.

Usage:
    from dashboard.analytics.report_generator import generate_monthly_report
    pdf_bytes = generate_monthly_report(closed_trades, start_date, end_date)
    # then: st.download_button("Baixar PDF", data=pdf_bytes, ...)
"""
from __future__ import annotations

import io
from datetime import datetime, date
from typing import Sequence

try:
    from fpdf import FPDF, XPos, YPos

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

    # Dummy stubs so class _TradePDF(FPDF) doesn't raise NameError at import time.
    # The real code path is only reached when _FPDF_AVAILABLE is True.
    class FPDF:  # type: ignore[no-redef]
        pass

    class XPos:  # type: ignore[no-redef]
        LMARGIN = "LMARGIN"

    class YPos:  # type: ignore[no-redef]
        NEXT = "NEXT"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_monthly_report(
    closed_trades: Sequence[dict],
    start_date: date | None = None,
    end_date: date | None = None,
) -> bytes:
    """Return PDF bytes for the given trade list and optional date range.

    Falls back to a minimal plain-text PDF if fpdf2 is unavailable.
    """
    trades = _filter_by_date(list(closed_trades), start_date, end_date)
    metrics = _compute_metrics(trades)

    if not _FPDF_AVAILABLE:
        return _text_fallback(metrics, trades)

    return _build_pdf(metrics, trades, start_date, end_date)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _filter_by_date(
    trades: list[dict],
    start: date | None,
    end: date | None,
) -> list[dict]:
    if not start and not end:
        return trades
    result = []
    for t in trades:
        ts = t.get("timestamp") or t.get("close_time") or t.get("time")
        if ts is None:
            result.append(t)
            continue
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts).date()
            except ValueError:
                result.append(t)
                continue
        elif isinstance(ts, datetime):
            ts = ts.date()
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        result.append(t)
    return result


def _compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
            "profit_factor": 0.0, "avg_hold_hours": 0.0,
            "by_symbol": {},
        }

    pnls = [float(t.get("pnl", 0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # By-symbol breakdown
    by_sym: dict[str, dict] = {}
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        pnl = float(t.get("pnl", 0))
        rec = by_sym.setdefault(sym, {"trades": 0, "pnl": 0.0, "wins": 0})
        rec["trades"] += 1
        rec["pnl"] += pnl
        if pnl > 0:
            rec["wins"] += 1

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "total_pnl": sum(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "largest_win": max(wins, default=0.0),
        "largest_loss": min(losses, default=0.0),
        "profit_factor": profit_factor,
        "by_symbol": by_sym,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF builder
# ─────────────────────────────────────────────────────────────────────────────

class _TradePDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    _title: str = "Performance Report"

    def header(self) -> None:  # type: ignore[override]
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self._title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(2)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Página {self.page_no()} — gerado em {datetime.now():%Y-%m-%d %H:%M}",
                  align="C")


def _build_pdf(
    metrics: dict,
    trades: list[dict],
    start_date: date | None,
    end_date: date | None,
) -> bytes:
    pdf = _TradePDF()
    period_label = _period_str(start_date, end_date)
    pdf._title = f"Trading Performance Report — {period_label}"
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)

    # ── 1. Executive summary ──────────────────────────────────────────────────
    _section_title(pdf, "1. Resumo Executivo")
    _kv_grid(pdf, [
        ("Período",           period_label),
        ("Total de Trades",   str(metrics["total"])),
        ("Ganhos",            str(metrics["wins"])),
        ("Perdas",            str(metrics["losses"])),
        ("Win Rate",          f"{metrics['win_rate']:.1%}"),
        ("PnL Total",         f"${metrics['total_pnl']:+,.2f}"),
        ("Média Ganho",       f"${metrics['avg_win']:,.2f}"),
        ("Média Perda",       f"${metrics['avg_loss']:,.2f}"),
        ("Maior Ganho",       f"${metrics['largest_win']:,.2f}"),
        ("Maior Perda",       f"${metrics['largest_loss']:,.2f}"),
        ("Profit Factor",     f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"),
    ], cols=2)

    # ── 2. Per-symbol breakdown ───────────────────────────────────────────────
    if metrics["by_symbol"]:
        pdf.ln(4)
        _section_title(pdf, "2. Breakdown por Símbolo")
        headers = ["Símbolo", "Trades", "Win Rate", "PnL Total"]
        col_w   = [50, 30, 40, 60]
        _table_header(pdf, headers, col_w)
        for sym, rec in sorted(metrics["by_symbol"].items(),
                               key=lambda x: x[1]["pnl"], reverse=True):
            wr = rec["wins"] / rec["trades"] if rec["trades"] else 0.0
            _table_row(pdf, [
                sym,
                str(rec["trades"]),
                f"{wr:.1%}",
                f"${rec['pnl']:+,.2f}",
            ], col_w)

    # ── 3. Trade log (last 50) ────────────────────────────────────────────────
    if trades:
        pdf.add_page()
        _section_title(pdf, "3. Últimos 50 Trades")
        last50 = trades[-50:]
        headers = ["Data/Hora", "Par", "Lado", "PnL", "Razão Fechamento"]
        col_w   = [42, 28, 18, 32, 60]
        _table_header(pdf, headers, col_w)
        for t in reversed(last50):
            ts = str(t.get("timestamp", t.get("time", "")))[:16]
            sym = t.get("symbol", "")
            side = t.get("side", t.get("position_type", ""))
            pnl  = float(t.get("pnl", 0))
            reason = str(t.get("reason", t.get("exit_reason", "")))[:30]
            _table_row(pdf, [ts, sym, side, f"${pnl:+,.2f}", reason], col_w)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_title(pdf: "FPDF", text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(30, 30, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _kv_grid(pdf: "FPDF", items: list[tuple[str, str]], cols: int = 2) -> None:
    """Render a key-value grid with `cols` columns."""
    cell_w = 190 // cols
    for i, (k, v) in enumerate(items):
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(cell_w // 2, 7, f"{k}:", border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(cell_w // 2, 7, v, border=0,
                 new_x=XPos.LMARGIN if (i + 1) % cols == 0 else XPos.RIGHT,
                 new_y=YPos.NEXT     if (i + 1) % cols == 0 else YPos.TMARGIN)
    # Ensure we're on a new line
    if len(items) % cols != 0:
        pdf.ln()


def _table_header(pdf: "FPDF", headers: list[str], col_w: list[int]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 240)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()


def _table_row(pdf: "FPDF", values: list[str], col_w: list[int]) -> None:
    pdf.set_font("Helvetica", "", 8)
    for v, w in zip(values, col_w):
        pdf.cell(w, 6, str(v)[:30], border=1)
    pdf.ln()


def _period_str(start: date | None, end: date | None) -> str:
    if start and end:
        return f"{start:%d/%m/%Y} – {end:%d/%m/%Y}"
    if start:
        return f"desde {start:%d/%m/%Y}"
    if end:
        return f"até {end:%d/%m/%Y}"
    return "todos os períodos"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (no fpdf2)
# ─────────────────────────────────────────────────────────────────────────────

def _text_fallback(metrics: dict, trades: list[dict]) -> bytes:
    """Return a trivial 'PDF' (actually plain text bytes) when fpdf2 is absent."""
    lines = [
        b"%PDF-1.4",
        f"% Trading Report — generated {datetime.now():%Y-%m-%d %H:%M}".encode(),
        f"% Total trades : {metrics['total']}".encode(),
        f"% Win rate     : {metrics['win_rate']:.1%}".encode(),
        f"% PnL total    : ${metrics['total_pnl']:+,.2f}".encode(),
        b"% Install fpdf2 to get a proper PDF: pip install fpdf2",
    ]
    return b"\n".join(lines)
