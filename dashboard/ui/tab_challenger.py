"""
Tab: 🏆 Champion vs Challenger
──────────────────────────────
Compara o modelo em produção (Champion) com novos checkpoints (Challenger)
gerados pelo retrain_v19_daily.py.

Funcionalidades:
  • Listagem de todos os modelos V19 em models/ com metadados
  • Histórico de re-treinos (data/retrain_log.json)
  • Promoção de Challenger → Champion (atualiza config.yaml)
  • Botão de Backtest rápido por par (lança subprocess)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from dashboard.core.config import get_lstm_model_path, load_config_raw

# ── Constantes ────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
RETRAIN_LOG  = Path("data") / "retrain_log.json"
CONFIG_PATH  = Path("config.yaml")

_V19_RE = re.compile(r'recurrent_ppo_v19_(?P<tag>[^_]+(?:_[^_]+)*?)_(?P<steps>\d{5,10})_steps')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_model(path: Path) -> dict:
    """Extrai metadados do nome do arquivo do modelo."""
    m = _V19_RE.search(path.stem)
    if m:
        steps = int(m.group('steps'))
        tag   = m.group('tag').replace('_', ' ')
    else:
        steps = 0
        tag   = path.stem[:40]

    stat       = path.stat()
    is_retrain = 'retrain' in path.stem
    is_final   = 'final' in path.stem
    date_str   = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    size_mb    = stat.st_size / 1024 / 1024

    return {
        'path'      : str(path),
        'name'      : path.name,
        'stem'      : path.stem,
        'steps'     : steps,
        'tag'       : tag,
        'is_retrain': is_retrain,
        'is_final'  : is_final,
        'date'      : date_str,
        'size_mb'   : round(size_mb, 1),
    }


def _all_v19_models() -> list[dict]:
    """Lista todos os modelos V19 ordenados por steps (desc)."""
    if not MODELS_DIR.exists():
        return []
    paths = sorted(MODELS_DIR.glob('recurrent_ppo_v19_*.zip'), reverse=True)
    models = [_parse_model(p) for p in paths]
    # Ordenar: retrain final (mais steps) primeiro
    models.sort(key=lambda x: (x['is_retrain'], x['steps']), reverse=True)
    return models


def _load_retrain_log() -> list[dict]:
    if RETRAIN_LOG.exists():
        try:
            return json.loads(RETRAIN_LOG.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []


def _get_champion_path(config: dict) -> str:
    return get_lstm_model_path(config)


def _set_champion(new_model_path: str):
    """
    Atualiza config.yaml com o novo caminho de modelo ativo.
    Adiciona bloco `models.lstm_active.path` se não existir.
    Usa substituição de texto segura para não alterar outros campos.
    """
    import yaml
    text = CONFIG_PATH.read_text(encoding='utf-8')

    # Verificar se já existe o bloco models:
    if 'models:' in text and 'lstm_active:' in text:
        # Substituir o path existente
        text = re.sub(
            r'(models:.*?lstm_active:.*?path:\s*)\"[^\"]*\"',
            rf'\1"{new_model_path}"',
            text,
            flags=re.DOTALL,
        )
        CONFIG_PATH.write_text(text, encoding='utf-8')
    else:
        # Adicionar bloco ao final
        models_block = f"""
# Modelo ativo (gerenciado pelo Champion/Challenger)
models:
  lstm_active:
    path: "{new_model_path}"
"""
        CONFIG_PATH.write_text(text.rstrip() + '\n' + models_block, encoding='utf-8')

    # Invalidar cache do Streamlit para reload
    st.cache_resource.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Render principal
# ──────────────────────────────────────────────────────────────────────────────

def render_tab_challenger(tab, config: dict):
    with tab:
        st.title("🏆 Champion vs Challenger")
        st.caption(
            "Gerencie checkpoints V19. Promova um Challenger a Champion "
            "(requer reinício do engine para carregar o novo modelo)."
        )

        champion_path = _get_champion_path(config)
        v19_models    = _all_v19_models()
        retrain_log   = _load_retrain_log()

        # ── Seção Champion ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("👑 Champion (produção)")

        champion_name = Path(champion_path).name
        champion_meta = next((m for m in v19_models if m['name'] == champion_name), None)

        col_champ, col_actions = st.columns([3, 1])
        with col_champ:
            if champion_meta:
                st.success(f"**{champion_meta['name']}**")
                st.markdown(
                    f"📅 Modificado: `{champion_meta['date']}`  |  "
                    f"🔢 Steps: `{champion_meta['steps']:,}`  |  "
                    f"💾 Tamanho: `{champion_meta['size_mb']} MB`"
                )
                badge = "🔄 Retrain" if champion_meta['is_retrain'] else ("✅ Final" if champion_meta['is_final'] else "🔵 Checkpoint")
                st.markdown(f"Tipo: {badge}")
            else:
                # Modelo V17 ou V18 em produção
                st.warning(f"**{champion_name}**")
                st.caption("(Modelo V17/V18 — re-treino V19 ainda não promovido)")

        with col_actions:
            if champion_meta and champion_meta.get('steps', 0) > 0:
                st.metric("Steps treinados", f"{champion_meta['steps']:,}")
            total_retrains = sum(1 for e in retrain_log if e.get('status') == 'success')
            st.metric("Re-treinos bem-sucedidos", total_retrains)

        # ── Seção Challengers ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🥊 Challengers disponíveis")

        if not v19_models:
            st.info("Nenhum modelo V19 encontrado em `models/`. Execute `retrain_v19_daily.py` primeiro.")
        else:
            # Tabela de modelos
            df_models = pd.DataFrame(v19_models)
            df_models['ativo'] = df_models['name'] == champion_name
            df_models['tipo']  = df_models.apply(
                lambda r: '👑 Champion' if r['ativo'] else ('🔄 Retrain' if r['is_retrain'] else ('✅ Final' if r['is_final'] else '📦 Checkpoint')),
                axis=1
            )

            # Mostrar tabela resumida
            st.dataframe(
                df_models[['tipo', 'name', 'steps', 'date', 'size_mb']].rename(columns={
                    'tipo'    : 'Tipo',
                    'name'    : 'Arquivo',
                    'steps'   : 'Steps',
                    'date'    : 'Modificado',
                    'size_mb' : 'MB',
                }),
                use_container_width=True,
                hide_index=True,
                height=min(300, 45 + len(df_models) * 35),
            )

            # Seletor de challenger
            st.markdown("#### Promover um challenger")
            non_champion = [m for m in v19_models if m['name'] != champion_name]

            if not non_champion:
                st.success("✅ Apenas um modelo disponível — já é o Champion.")
            else:
                challenger_names = [m['name'] for m in non_champion]
                selected = st.selectbox(
                    "Selecionar Challenger",
                    options=challenger_names,
                    key="challenger_select",
                    help="Escolha o modelo para promover a Champion.",
                )
                selected_meta = next(m for m in non_champion if m['name'] == selected)

                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f"📄 **{selected_meta['name']}**  \n"
                        f"🔢 Steps: `{selected_meta['steps']:,}`  |  "
                        f"📅 `{selected_meta['date']}`"
                    )
                with col_btn:
                    if st.button("🚀 Promover a Champion", type="primary", key="promote_btn"):
                        _set_champion(selected_meta['path'])
                        st.success(f"✅ **{selected_meta['name']}** promovido a Champion!")
                        st.warning("⚠️ Reinicie o Engine para carregar o novo modelo.")
                        time.sleep(0.5)
                        st.rerun()

        # ── Seção Retrain History ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Histórico de Re-treinos")

        if not retrain_log:
            st.info("Nenhum re-treino registrado ainda. Execute `python retrain_v19_daily.py`.")
        else:
            rows = []
            for entry in reversed(retrain_log):
                training = entry.get('training', {})
                rows.append({
                    'Data'       : entry.get('today', '?'),
                    'Status'     : '✅' if entry.get('status') == 'success' else ('⬜' if entry.get('status') == 'data-only' else '❌'),
                    'Steps ++'   : f"{training.get('added_steps', 0):,}" if training else '—',
                    'Total steps': f"{training.get('total_steps', 0):,}" if training else '—',
                    'Tempo'      : f"{training.get('elapsed_s', 0) / 60:.0f} min" if training else '—',
                    'Device'     : training.get('device', '—').upper() if training else '—',
                    'Checkpoint' : Path(training['new_checkpoint']).name if training.get('new_checkpoint') else '—',
                    'Erros'      : len(entry.get('errors', [])),
                })

            df_log = pd.DataFrame(rows)
            st.dataframe(df_log, use_container_width=True, hide_index=True)

            # Estatísticas
            n_ok  = sum(1 for e in retrain_log if e.get('status') == 'success')
            n_err = sum(1 for e in retrain_log if e.get('status') == 'failed')
            n_dat = sum(1 for e in retrain_log if e.get('status') == 'data-only')

            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Sucessos",      n_ok)
            col2.metric("❌ Erros",          n_err)
            col3.metric("⬜ Só dados",      n_dat)

        # ── Seção Executar Re-treino ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔧 Executar Re-treino Agora")

        with st.expander("⚙️ Configurações do re-treino"):
            col_s, col_skip = st.columns(2)
            with col_s:
                steps = st.number_input(
                    "Steps adicionais",
                    min_value=100_000,
                    max_value=3_000_000,
                    value=500_000,
                    step=100_000,
                    key="retrain_steps",
                    format="%d",
                )
            with col_skip:
                skip_data = st.checkbox(
                    "⏭️ Pular coleta de dados (usar CSVs existentes)",
                    key="skip_data_check",
                )

        col_btn_rt, col_dry = st.columns(2)

        with col_btn_rt:
            if st.button("🚀 Iniciar Re-treino", key="start_retrain_btn"):
                cmd = [sys.executable, "retrain_v19_daily.py", "--steps", str(int(steps))]
                if skip_data:
                    cmd.append("--skip-data")
                cmd_str = " ".join(cmd)
                st.info(f"Iniciando em background:\n```\n{cmd_str}\n```")
                try:
                    subprocess.Popen(
                        cmd,
                        cwd=str(Path.cwd()),
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                    )
                    st.success("✅ Re-treino iniciado em nova janela do terminal.")
                    st.caption("Acompanhe o progresso na nova janela e retorne aqui para "
                               "ver o novo checkpoint no Histórico.")
                except Exception as exc:
                    st.error(f"❌ Falha ao iniciar: {exc}")

        with col_dry:
            if st.button("🧪 Dry-run (verificar config)", key="dryrun_btn"):
                try:
                    result = subprocess.run(
                        [sys.executable, "retrain_v19_daily.py", "--dry-run"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(Path.cwd()),
                    )
                    output = result.stdout + result.stderr
                    st.text_area("Output do dry-run", value=output, height=300)
                except subprocess.TimeoutExpired:
                    st.error("Timeout após 30s.")
                except Exception as exc:
                    st.error(f"Erro: {exc}")

        # ── Nota de agendamento ──────────────────────────────────────────────
        st.markdown("---")
        with st.expander("🕐 Agendar re-treino automático (Windows Task Scheduler)"):
            st.markdown("""
**Criar tarefa agendada no Windows (PowerShell como admin):**

```powershell
$action  = New-ScheduledTaskAction -Execute "python" `
             -Argument "retrain_v19_daily.py --steps 500000" `
             -WorkingDirectory "C:\\Users\\arcti\\OneDrive\\Área de Trabalho\\AGENTE TRANDING"
$trigger = New-ScheduledTaskTrigger -Daily -At "02:00AM"
Register-ScheduledTask -TaskName "AgentV19Retrain" `
    -Action $action -Trigger $trigger -RunLevel Highest
```

Isso executará o re-treino toda madrugada às 02:00, adicionando 500k steps novos.
O checkpoint ficará disponível aqui para promoção.
""")
