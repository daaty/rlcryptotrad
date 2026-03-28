"""
Tab 🏆 Champion vs Challenger — NiceGUI version.
Lista modelos V19, permite promover Challenger → Champion e lançar re-treino.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from nicegui import ui
from dashboard.state_ng import LiveState
from dashboard.ui_ng.components import section_title, divider

# ── Constantes ────────────────────────────────────────────────────────────────
MODELS_DIR  = Path('models')
RETRAIN_LOG = Path('data') / 'retrain_log.json'
CONFIG_PATH = Path('config.yaml')
_V19_RE = re.compile(r'recurrent_ppo_v19_(?P<tag>[^_]+(?:_[^_]+)*?)_(?P<steps>\d{5,10})_steps')


# ── Modelo helpers ─────────────────────────────────────────────────────────────

def _parse_model(path: Path) -> dict:
    m     = _V19_RE.search(path.stem)
    steps = int(m.group('steps'))  if m else 0
    tag   = m.group('tag').replace('_', ' ') if m else path.stem[:40]
    stat  = path.stat()
    return {
        'path'      : str(path),
        'name'      : path.name,
        'stem'      : path.stem,
        'steps'     : steps,
        'tag'       : tag,
        'is_retrain': 'retrain' in path.stem,
        'is_final'  : 'final'   in path.stem,
        'date'      : datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        'size_mb'   : round(stat.st_size / 1024 / 1024, 1),
    }


def _all_v19_models() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    paths  = sorted(MODELS_DIR.glob('recurrent_ppo_v19_*.zip'), reverse=True)
    models = [_parse_model(p) for p in paths]
    models.sort(key=lambda x: (x['is_retrain'], x['steps']), reverse=True)
    return models


def _load_retrain_log() -> list[dict]:
    if RETRAIN_LOG.exists():
        try:
            return json.loads(RETRAIN_LOG.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []


def _set_champion(new_path: str) -> None:
    """Atualiza config.yaml com novo caminho ativo."""
    import yaml
    raw = CONFIG_PATH.read_text(encoding='utf-8')
    try:
        data = yaml.safe_load(raw)
    except Exception:
        data = {}
    models = data.setdefault('models', {})
    active = models.setdefault('lstm_active', {})
    active['path'] = new_path
    CONFIG_PATH.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False),
                           encoding='utf-8')


def _get_champion_path() -> str:
    from dashboard.resources_ng import get_config
    cfg = get_config()
    from dashboard.core.config import get_lstm_model_path
    return get_lstm_model_path(cfg)


# ── Entry point ────────────────────────────────────────────────────────────────

def render_challenger_tab(state: LiveState) -> None:
    _challenger_panel(state)


@ui.refreshable
def _challenger_panel(state: LiveState) -> None:
    section_title('🏆 Champion vs Challenger')

    champion_path = _get_champion_path()
    models        = _all_v19_models()

    # ── Champion Card ─────────────────────────────────────────────────────────
    with ui.card().classes('w-full bg-green-900 border border-green-600 p-4 mb-4'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('emoji_events').classes('text-yellow-300 text-2xl')
            ui.label('Champion Ativo').classes('text-white font-bold text-lg')
        ui.label(Path(champion_path).name if champion_path else '—'
                 ).classes('text-green-200 font-mono text-sm mt-1')

    # ── Modelos disponíveis ───────────────────────────────────────────────────
    section_title(f'📦 Modelos V19 ({len(models)})')

    if not models:
        ui.label('Nenhum modelo V19 encontrado em models/').classes('text-gray-400')
    else:
        rows = []
        for m in models:
            is_champ = (m['path'] == champion_path or
                        Path(m['path']).name == Path(champion_path).name)
            rows.append({
                'name'      : m['name'],
                'steps'     : m['steps'],
                'tag'       : m['tag'],
                'date'      : m['date'],
                'size_mb'   : m['size_mb'],
                'is_retrain': '✅' if m['is_retrain'] else '—',
                'champion'  : '👑' if is_champ else '',
                '_path'     : m['path'],
            })

        grid = ui.aggrid({
            'columnDefs': [
                {'field': 'champion',   'headerName': '',       'width': 50},
                {'field': 'name',       'headerName': 'Modelo', 'width': 340, 'flex': 1},
                {'field': 'steps',      'headerName': 'Steps',  'width': 100,
                 'valueFormatter': "x.toLocaleString()"},
                {'field': 'date',       'headerName': 'Data',   'width': 140},
                {'field': 'size_mb',    'headerName': 'MB',     'width': 70},
                {'field': 'is_retrain', 'headerName': 'Retrain','width': 85},
            ],
            'rowData': rows,
            'rowSelection': 'single',
            'domLayout': 'autoHeight',
        }).classes('w-full ag-theme-alpine-dark')

        async def promote_selected():
            selected = await grid.get_selected_rows()
            if not selected:
                ui.notify('Selecione um modelo na tabela.', type='warning')
                return
            sel_path = selected[0].get('_path', '')
            if not sel_path:
                ui.notify('Caminho inválido.', type='negative')
                return
            try:
                _set_champion(sel_path)
                from dashboard.resources_ng import reload_config
                reload_config()
                ui.notify(f'✅ Champion atualizado: {Path(sel_path).name}', type='positive')
                _challenger_panel.refresh(state)
            except Exception as exc:
                ui.notify(f'Erro ao promover: {exc}', type='negative')

        ui.button('👑 Promover como Champion', on_click=promote_selected
                  ).props('color=positive').classes('mt-2')

    divider()

    # ── Lançar Re-treino ──────────────────────────────────────────────────────
    section_title('🔄 Iniciar Re-treino')

    steps_ref = ui.number('Steps', value=500_000, min=10_000, step=50_000
                          ).props('outlined dense').classes('w-40')

    _retrain_log_ref: list = []

    async def start_retrain():
        steps = int(steps_ref.value or 500_000)
        cmd   = [sys.executable, 'retrain_v19_daily.py', '--steps', str(steps)]
        try:
            subprocess.Popen(cmd, cwd=str(Path.cwd()))
            ui.notify(f'🚀 Re-treino iniciado ({steps:,} steps). Veja retrain_log.json.', type='info')
        except Exception as exc:
            ui.notify(f'Erro: {exc}', type='negative')

    ui.button('🚀 Iniciar Re-treino', on_click=start_retrain
              ).props('color=primary').classes('mt-1')

    divider()

    # ── Histórico de re-treinos ───────────────────────────────────────────────
    section_title('📜 Histórico de Re-treinos')
    log = _load_retrain_log()

    if not log:
        ui.label('Nenhum re-treino registrado ainda.').classes('text-gray-400')
    else:
        log_rows = []
        for entry in reversed(log[-50:]):
            log_rows.append({
                'Data'    : entry.get('date',         '—'),
                'Modelo'  : Path(entry.get('model_path', '—')).name,
                'Steps'   : entry.get('steps',        0),
                'Pares'   : ', '.join(entry.get('pairs', [])),
                'Erro'    : entry.get('error',        ''),
            })
        ui.aggrid({
            'columnDefs': [
                {'field': 'Data',   'width': 140},
                {'field': 'Modelo', 'width': 290, 'flex': 1},
                {'field': 'Steps',  'width': 90},
                {'field': 'Pares',  'width': 200},
                {'field': 'Erro',   'width': 150},
            ],
            'rowData': log_rows,
            'domLayout': 'autoHeight',
        }).classes('w-full ag-theme-alpine-dark')
