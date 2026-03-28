"""
Tab Posições — NiceGUI version.
Lista todas as posições abertas com botão de fechar.
"""
from __future__ import annotations
from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider


def render_positions_tab(state: LiveState) -> None:
    _positions_panel(state)


@ui.refreshable
def _positions_panel(state: LiveState) -> None:
    section_title(f'💼 Posições Abertas ({state.n_positions})')

    if not state.open_positions:
        with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-6 text-center'):
            ui.label('📭 Sem posições abertas').classes('text-gray-400 text-lg')
        return

    # Tabela de posições com ações
    for pos in state.open_positions:
        amt    = float(pos.get('positionAmt', 0))
        entry  = float(pos.get('entryPrice', 0))
        mark   = float(pos.get('markPrice', 0))
        upnl   = float(pos.get('unRealizedProfit', 0))
        sym    = pos.get('symbol', '—')
        side   = 'LONG' if amt > 0 else 'SHORT'
        pnl_pct = ((mark - entry) / entry * 100 * (1 if amt > 0 else -1)) if entry else 0
        pnl_color = 'text-green-400' if upnl >= 0 else 'text-red-400'
        side_color = 'bg-green-700' if side == 'LONG' else 'bg-red-700'
        sym_clean  = sym.replace('USDT', '')

        with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-4 mb-3'):
            with ui.row().classes('items-center justify-between flex-wrap gap-3'):
                # Left: symbol + side
                with ui.row().classes('items-center gap-3'):
                    ui.label(sym_clean).classes('text-white font-bold text-lg w-16')
                    ui.badge(side).classes(f'{side_color} text-white text-xs font-bold px-2 rounded-full')

                # Middle: price info
                with ui.row().classes('gap-6'):
                    _pos_metric('Entrada',  f'${entry:,.4f}')
                    _pos_metric('Mark',     f'${mark:,.4f}')
                    _pos_metric('Qty',      f'{abs(amt):.4f}')
                    _pos_metric('P&L Não-Realizado',
                                f'{"+"if upnl>=0 else ""}${upnl:,.2f} ({pnl_pct:.2f}%)',
                                color=pnl_color)

                # Right: close button
                async def close_pos(s=sym, a=amt):
                    from dashboard.resources_ng import get_binance_client, get_config
                    from dashboard.trading.executor import close_position_direct
                    try:
                        client = get_binance_client()
                        cfg    = get_config()
                        close_position_direct(client, s, a, cfg)
                        ui.notify(f'{s} fechado.', type='positive')
                        _positions_panel.refresh(state)
                    except Exception as exc:
                        ui.notify(f'Erro: {exc}', type='negative')

                ui.button('✕ Fechar', on_click=close_pos).props(
                    'color=negative size=sm flat').classes('text-xs')


def _pos_metric(label: str, value: str, color: str = 'text-white') -> None:
    with ui.column().classes('items-center gap-0'):
        ui.label(label).classes('text-gray-400 text-xs')
        ui.label(value).classes(f'{color} font-mono text-sm font-semibold')
