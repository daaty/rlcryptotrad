"""
Componentes compartilhados para NiceGUI.
Helpers de estilo que replicam a estética de um trading terminal.
"""
from __future__ import annotations
from nicegui import ui


# ── Paleta ────────────────────────────────────────────────────────────────────
GREEN  = 'text-green-400'
RED    = 'text-red-400'
YELLOW = 'text-yellow-400'
GRAY   = 'text-gray-400'
WHITE  = 'text-white'


def metric_card(title: str, value_ref: tuple, delta_ref: tuple | None = None,
                icon: str = '', width: str = 'w-44') -> ui.card:
    """
    Cria um card de métrica no estilo Streamlit st.metric.

    Args:
        title: Rótulo do metric.
        value_ref: (obj, 'attr') para bind_text_from.
        delta_ref: (obj, 'attr') para o delta (opcional).
        icon: Emoji ou ícone prefixo.
        width: Classe CSS de largura.
    Returns:
        ui.card reference.
    """
    with ui.card().classes(f'{width} p-3 bg-gray-800 rounded-lg border border-gray-700') as card:
        ui.label(f'{icon} {title}'.strip()).classes(f'{GRAY} text-xs uppercase tracking-wide mb-1')
        obj, attr = value_ref
        ui.label().classes(f'{WHITE} text-xl font-bold font-mono').bind_text_from(obj, attr)
        if delta_ref:
            d_obj, d_attr = delta_ref
            ui.label().classes(f'{GRAY} text-xs mt-0.5').bind_text_from(d_obj, d_attr)
    return card


def status_badge(label_ref: tuple, color_fn=None) -> ui.badge:
    """Cria um badge de status com texto dinâmico."""
    obj, attr = label_ref
    badge = ui.badge('').classes('text-xs font-mono')
    badge.bind_text_from(obj, attr)
    return badge


def section_title(text: str) -> ui.label:
    return ui.label(text).classes('text-base font-semibold text-gray-200 mt-4 mb-2')


def divider() -> ui.separator:
    return ui.separator().classes('my-3 border-gray-700')


def colored_pnl(value: float) -> str:
    """Retorna string formatada de P&L com sinal."""
    sign = '+' if value >= 0 else ''
    return f'{sign}${value:,.2f}'


def pnl_color(value: float) -> str:
    return 'text-green-400' if value >= 0 else 'text-red-400'


def side_pill(side: str) -> ui.badge:
    """Badge colorido LONG/SHORT/FLAT."""
    color = 'bg-green-700' if side == 'LONG' else ('bg-red-700' if side == 'SHORT' else 'bg-gray-600')
    return ui.badge(side).classes(f'{color} text-white text-xs font-bold px-2 py-0.5 rounded-full')
