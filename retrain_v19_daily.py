"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       🔄 RETRAIN V19 DAILY — PIPELINE DE RE-TREINAMENTO INCREMENTAL         ║
║                                                                              ║
║  Executa Shadow Training (Gemini strategy):                                 ║
║   1. Lê os CSVs de treinamento mais recentes para BTC/ETH/SOL/BNB           ║
║   2. Busca novos candles desde o último timestamp dos CSVs (delta fetch)    ║
║   3. Recalcula indicadores técnicos no dataset expandido                    ║
║   4. Salva CSVs atualizados com timestamp do dia                            ║
║   5. Carrega o checkpoint V19 mais recente                                  ║
║   6. Continua treinamento por RETRAIN_STEPS (padrão: 500k)                 ║
║   7. Salva novo checkpoint: recurrent_ppo_v19_retrain_YYYYMMDD_Xsteps.zip  ║
║   8. Grava relatório em data/retrain_log.json                               ║
║                                                                              ║
║  Uso:                                                                        ║
║    python retrain_v19_daily.py              # 500k steps padrão             ║
║    python retrain_v19_daily.py --steps 1000000                              ║
║    python retrain_v19_daily.py --skip-data  # não busca dados novos         ║
║    python retrain_v19_daily.py --dry-run    # só valida sem treinar         ║
║                                                                              ║
║  Agendar no Windows Task Scheduler:                                         ║
║    Trigger: Daily 02:00                                                     ║
║    Action: python retrain_v19_daily.py --steps 500000                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Forçar UTF-8 no Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Dependências opcionais ─────────────────────────────────────────────────
try:
    import ccxt
except ImportError:
    print("❌ ERRO: ccxt não instalado.  pip install ccxt")
    sys.exit(1)

try:
    import talib
except ImportError:
    print("❌ ERRO: TA-Lib não instalado.  pip install TA-Lib")
    sys.exit(1)

try:
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SB3/torch não disponível: {e}")
    print("    Modo --dry-run ou --skip-training funcionará sem SB3.")
    SB3_AVAILABLE = False

from src.environment.trading_env_v19_lstm import TradingEnvV19LSTM
from callbacks.trading_metrics import TradingMetricsCallback

# ── Configuração ──────────────────────────────────────────────────────────
PAIRS            = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']
TIMEFRAMES       = ['15m', '1h', '4h']
RETRAIN_STEPS    = 500_000        # steps adicionados por rodada diária
SAVE_FREQ        = 10_000
DATA_DIR         = Path('data')
MODELS_DIR       = Path('models')
RETRAIN_LOG      = DATA_DIR / 'retrain_log.json'
MONTHS_HISTORY   = 36             # meses de histórico mantidos nos CSVs

# Hiperparâmetros idênticos ao treino original V19 (não muda na continuação)
ENV_CONFIG = {
    'window_size':           50,
    'max_episode_steps':     2000,
    'leverage':              1.5,
    'commission':            0.0004,
    'slippage':              0.0005,
    'position_size':         0.15,   # V19.1: 0.05→0.15 (sinal reward 3× maior)
    'use_sharpe_reward':     False,
    'enable_indicator_shaping': False,
    'random_start':          True,
    'persist_balance':       False,
    'liquidation_threshold': 0.30,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCOBERTA DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_v19_checkpoint() -> Path | None:
    """
    Encontra o checkpoint V19 mais recente em models/.
    Prioridade: retrain > final > stepN (maior número de steps).
    Exclui modelos V17 e V18.
    """
    if not MODELS_DIR.exists():
        return None

    candidates = list(MODELS_DIR.glob('recurrent_ppo_v19_*.zip'))
    if not candidates:
        return None

    def sort_key(p: Path):
        name = p.stem
        # Extrair número de steps do nome (ex: _500000_steps → 500000)
        steps = 0
        for part in name.split('_'):
            if part.isdigit():
                steps = int(part)
        # retrain > final > outros
        bonus = 0
        if 'retrain' in name:
            bonus = 10_000_000_000
        elif 'final' in name:
            bonus = 5_000_000_000
        return bonus + steps

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. DESCOBERTA DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_csv(pair: str, tf: str) -> Path | None:
    """Encontra o CSV de treino mais recente para um par/timeframe."""
    files = sorted(DATA_DIR.glob(f'train_{pair}_*_{tf}_*.csv'), reverse=True)
    return files[0] if files else None


def get_last_timestamp(csv_path: Path) -> datetime | None:
    """Lê o último timestamp do CSV sem carregar todas as linhas."""
    try:
        df_tail = pd.read_csv(csv_path, usecols=['timestamp'])
        last_str = df_tail['timestamp'].iloc[-1]
        return pd.to_datetime(last_str)
    except Exception as exc:
        print(f"  ⚠️  Não conseguiu ler timestamp de {csv_path.name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. COLETA DELTA (novos candles desde último timestamp)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaFetcher:
    """Busca apenas os candles novos desde o último timestamp nos CSVs."""

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        print("✅ CCXT Binance Futures inicializado")

    def fetch_delta(self, symbol: str, tf: str, since_dt: datetime) -> pd.DataFrame | None:
        """
        Busca candles desde since_dt até agora.
        Retorna DataFrame com colunas [timestamp, open, high, low, close, volume].
        """
        since_ms = int(since_dt.timestamp() * 1000) + 1   # +1ms para não repetir
        now      = datetime.now()
        symbol_ccxt = symbol.upper().replace('USDT', '/USDT')   # btcusdt → BTC/USDT

        print(f"  📥 {symbol_ccxt} {tf}  desde {since_dt:%Y-%m-%d %H:%M} → {now:%Y-%m-%d %H:%M}")

        all_candles = []
        page = 1
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol_ccxt, timeframe=tf, since=since_ms, limit=1000
                )
                if not candles:
                    break

                all_candles.extend(candles)
                last_ts = candles[-1][0]
                last_dt = datetime.fromtimestamp(last_ts / 1000)
                print(f"    [pág {page}] +{len(candles):4d} candles → total {len(all_candles):,}", end='\r')

                if last_dt >= now - timedelta(minutes=15):
                    break

                since_ms = last_ts + 1
                page += 1
                time.sleep(0.35)

            except Exception as exc:
                print(f"\n  ⚠️  Erro pág {page}: {exc}")
                time.sleep(2)
                break

        if not all_candles:
            print(f"\n  ⚪ Sem novos candles para {symbol_ccxt} {tf}")
            return None

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        print(f"\n  ✅ +{len(df):,} candles novos  ({since_dt:%Y-%m-%d} → {df['timestamp'].iloc[-1]:%Y-%m-%d})")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. RECOMPUTE INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula todos os 20 indicadores técnicos no estilo do collect_multi_pair_mtf.py.
    Espera colunas: timestamp, open, high, low, close, volume.
    Retorna DataFrame com 21 colunas (timestamp + 20 features numéricas).
    """
    df = df.set_index('timestamp').copy()

    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)

    # RSI normalizado (0-1)
    df['RSI_14'] = talib.RSI(close, timeperiod=14) / 100.0

    # SMAs relativas ao close
    df['SMA_20'] = talib.SMA(close, timeperiod=20) / (close + 1e-8)
    df['SMA_50'] = talib.SMA(close, timeperiod=50) / (close + 1e-8)

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BBL_20_2.0'] = lower  / (close + 1e-8)
    df['BBM_20_2.0'] = middle / (close + 1e-8)
    df['BBU_20_2.0'] = upper  / (close + 1e-8)
    df['BBB_20_2.0'] = (upper - lower) / (middle + 1e-8)
    df['BBP_20_2.0'] = (close - lower) / (upper - lower + 1e-8)

    # MACD relativo ao close
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_12_26_9']  = macd   / (close + 1e-8)
    df['MACDs_12_26_9'] = signal / (close + 1e-8)
    df['MACDh_12_26_9'] = hist   / (close + 1e-8)

    # EMAs relativas ao close
    df['EMA_9']  = talib.EMA(close, timeperiod=9)  / (close + 1e-8)
    df['EMA_21'] = talib.EMA(close, timeperiod=21) / (close + 1e-8)

    # ATR relativo ao close
    df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14) / (close + 1e-8)

    # Volume relativo à média
    vol_ma = talib.SMA(volume, timeperiod=20)
    df['Volume_MA_20'] = volume / (vol_ma + 1e-8)

    df.dropna(inplace=True)
    df = df.reset_index()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ATUALIZAÇÃO DOS CSVs
# ─────────────────────────────────────────────────────────────────────────────

def update_csv(pair: str, tf: str, fetcher: DeltaFetcher, today_str: str) -> str | None:
    """
    Atualiza o CSV de treino para um par/timeframe:
      1. Lê CSV existente
      2. Busca novos candles (delta)
      3. Concatena e recalcula indicadores no dataset completo
      4. Trunca para MONTHS_HISTORY mais recentes
      5. Salva novo CSV com data de hoje
    Retorna o path do novo CSV ou None se falhou.
    """
    existing = find_latest_csv(pair, tf)
    new_path  = DATA_DIR / f'train_{pair}_{MONTHS_HISTORY}m_{tf}_{today_str}.csv'

    if existing is None:
        print(f"  ⚠️  {pair} {tf}: nenhum CSV base encontrado — execute collect_multi_pair_mtf.py primeiro")
        return None

    # Se já atualizamos hoje, reusar
    if existing == new_path:
        print(f"  ✅ {pair} {tf}: CSV já atualizado hoje ({existing.name})")
        return str(existing)

    # Ler CSV base (só colunas raw: timestamp, open, high, low, close, volume)
    print(f"\n  📂 Lendo {existing.name} ...")
    df_base = pd.read_csv(existing, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                          parse_dates=['timestamp'])
    last_ts = df_base['timestamp'].iloc[-1]
    print(f"  📅 Último candle existente: {last_ts:%Y-%m-%d %H:%M}")

    # Buscar candles novos
    df_new = fetcher.fetch_delta(pair, tf, last_ts)

    if df_new is not None and len(df_new) > 0:
        df_combined = pd.concat([df_base, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    else:
        df_combined = df_base.copy()

    # Truncar para MONTHS_HISTORY mais recentes (evitar CSVs infinitamente crescentes)
    cutoff = datetime.now() - timedelta(days=MONTHS_HISTORY * 30)
    df_combined = df_combined[df_combined['timestamp'] >= cutoff].reset_index(drop=True)

    print(f"  🔧 Recalculando indicadores ({len(df_combined):,} candles totais)...")
    df_final = compute_indicators(df_combined)

    df_final.to_csv(new_path, index=False)
    n_new = len(df_new) if df_new is not None else 0
    print(f"  💾 Salvo: {new_path.name}  ({len(df_final):,} candles, +{n_new} novos)")
    return str(new_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FACTORY DE AMBIENTE
# ─────────────────────────────────────────────────────────────────────────────

def make_env(data_paths: dict):
    def _init():
        return TradingEnvV19LSTM(data_paths=data_paths, **ENV_CONFIG)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# 7. RETRAIN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_retrain(
    pair_paths: list,           # [(pair_name, {tf: path}), ...]
    retrain_steps: int,
    timestamp: str,
) -> dict:
    """
    Carrega o checkpoint V19 mais recente e continua o treino por retrain_steps.
    Retorna dict com metadados do checkpoint produzido.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("SB3/torch não disponível — instale stable-baselines3-contrib")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = find_latest_v19_checkpoint()

    if checkpoint is None:
        raise RuntimeError(
            "Nenhum checkpoint V19 encontrado em models/. "
            "Execute train_recurrent_ppo_v19_multipair.py primeiro."
        )

    print(f"\n🤖 Checkpoint base: {checkpoint.name}")
    print(f"🖥️  Device: {device}")

    # Criar envs com dados atualizados
    print(f"\n📁 Criando {len(pair_paths)} ambiente(s)...")
    env_fns = [make_env(paths) for _, paths in pair_paths]
    env     = DummyVecEnv(env_fns)
    print(f"  ✅ {env.num_envs} env(s) criados  |  obs={env.observation_space.shape}")

    # Carregar modelo e substituir env
    print(f"\n📦 Carregando modelo de {checkpoint.name}...")
    model = RecurrentPPO.load(
        str(checkpoint),
        env    = env,
        device = device,
        # force_reset=True para garantir que buffers sejam reiniciados
    )
    # Atualizar env caso load não tenha substituído
    model.set_env(env)

    # Extrair steps já treinados a partir do nome do arquivo
    prev_steps = 0
    for part in checkpoint.stem.split('_'):
        if part.isdigit():
            prev_steps = int(part)
    total_after = prev_steps + retrain_steps

    new_name      = f"recurrent_ppo_v19_retrain_{timestamp}"
    new_model_dir = str(MODELS_DIR)

    callbacks = [
        TradingMetricsCallback(verbose=0),
        CheckpointCallback(
            save_freq   = SAVE_FREQ,
            save_path   = new_model_dir,
            name_prefix = new_name,
            save_replay_buffer = False,
            save_vecnormalize  = False,
        ),
    ]

    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO RE-TREINO V19 ({retrain_steps:,} steps adicionais)")
    print(f"   Base checkpoint : {checkpoint.name}")
    print(f"   Steps estimados : {prev_steps:,} → {total_after:,}")
    print(f"   Novos steps     : {retrain_steps:,}")
    print(f"   Pares ativos    : {', '.join(n for n, _ in pair_paths)}")
    print(f"{'='*70}\n")

    start_time = time.time()

    model.learn(
        total_timesteps  = retrain_steps,
        callback         = callbacks,
        reset_num_timesteps = False,   # preserva contador global de steps
        progress_bar     = True,
    )

    elapsed = time.time() - start_time

    # Salvar checkpoint final do retrain
    final_path = MODELS_DIR / f"{new_name}_{total_after}_steps.zip"
    model.save(str(final_path))
    print(f"\n💾 Checkpoint final salvo: {final_path.name}")

    return {
        'base_checkpoint': str(checkpoint),
        'new_checkpoint' : str(final_path),
        'prev_steps'     : prev_steps,
        'added_steps'    : retrain_steps,
        'total_steps'    : total_after,
        'elapsed_s'      : round(elapsed, 1),
        'device'         : device,
        'n_envs'         : env.num_envs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. RELATÓRIO JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_retrain_log() -> list:
    if RETRAIN_LOG.exists():
        try:
            return json.loads(RETRAIN_LOG.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []


def save_retrain_log(entry: dict):
    log = load_retrain_log()
    log.append(entry)
    # Manter apenas os últimos 90 dias de entradas
    cutoff = (datetime.now() - timedelta(days=90)).isoformat()
    log = [e for e in log if e.get('timestamp', '') >= cutoff]
    RETRAIN_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"📝 Log atualizado: {RETRAIN_LOG}  ({len(log)} entradas)")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Re-treino incremental V19 diário')
    p.add_argument('--steps',       type=int,  default=RETRAIN_STEPS,
                   help=f'Steps adicionais de treino (padrão: {RETRAIN_STEPS:,})')
    p.add_argument('--skip-data',   action='store_true',
                   help='Não busca novos dados — usa CSVs existentes')
    p.add_argument('--skip-training', action='store_true',
                   help='Apenas atualiza dados, sem treinar')
    p.add_argument('--dry-run',     action='store_true',
                   help='Valida configuração sem executar nada')
    p.add_argument('--pairs',       nargs='+', default=PAIRS,
                   help=f'Pares a usar (padrão: {PAIRS})')
    return p.parse_args()


def main():
    args   = parse_args()
    today  = datetime.now().strftime('%Y%m%d')
    ts_iso = datetime.now().isoformat(timespec='seconds')

    print("\n" + "="*70)
    print("🔄 RETRAIN V19 DAILY — PIPELINE DE RE-TREINAMENTO INCREMENTAL")
    print("="*70)
    print(f"  📅 Data         : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  🎯 Steps extra  : {args.steps:,}")
    print(f"  📊 Pares        : {', '.join(p.upper() for p in args.pairs)}")
    print(f"  🔧 Modo         : {'DRY-RUN' if args.dry_run else 'TREINO COMPLETO'}")
    print("="*70)

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    log_entry: dict = {
        'timestamp'  : ts_iso,
        'today'      : today,
        'steps'      : args.steps,
        'pairs'      : args.pairs,
        'status'     : 'started',
        'data_update': {},
        'training'   : {},
        'errors'     : [],
    }

    # ── Verificar checkpoint disponível ──────────────────────────────────────
    checkpoint = find_latest_v19_checkpoint()
    if checkpoint is None:
        print("\n❌ Nenhum checkpoint V19 encontrado em models/")
        print("   Execute primeiro: python train_recurrent_ppo_v19_multipair.py")
        sys.exit(1)
    print(f"\n✅ Checkpoint base encontrado: {checkpoint.name}")

    if args.dry_run:
        print("\n[DRY-RUN] Validando dados...")

    # ── Fase 1: Atualização de Dados ──────────────────────────────────────────
    pair_paths = []   # [(pair, {tf: path})]

    if not args.skip_data and not args.dry_run:
        print("\n" + "─"*70)
        print("📊 FASE 1 — ATUALIZAÇÃO DE DADOS")
        print("─"*70)

        fetcher = DeltaFetcher()

        for pair in args.pairs:
            pair_data = {}
            ok = True
            for tf in TIMEFRAMES:
                print(f"\n  [{pair.upper()} {tf}]")
                try:
                    path = update_csv(pair, tf, fetcher, today)
                    if path:
                        pair_data[tf] = path
                        log_entry['data_update'][f'{pair}_{tf}'] = path
                    else:
                        ok = False
                except Exception as exc:
                    err = f"{pair} {tf}: {exc}"
                    print(f"  ❌ {err}")
                    log_entry['errors'].append(err)
                    ok = False

            if len(pair_data) == 3:
                pair_paths.append((pair.upper(), pair_data))
            else:
                print(f"  ⚠️  {pair.upper()}: dados incompletos ({len(pair_data)}/3 TFs) — par excluído")

    else:
        # Usar CSVs existentes
        print(f"\n{'─'*70}")
        print("📂 Usando CSVs existentes (--skip-data ou --dry-run)...")
        print(f"{'─'*70}")
        for pair in args.pairs:
            pair_data = {}
            for tf in TIMEFRAMES:
                csv = find_latest_csv(pair, tf)
                if csv:
                    pair_data[tf] = str(csv)
                    print(f"  ✅ {pair.upper()} {tf}: {csv.name}")
                else:
                    print(f"  ❌ {pair.upper()} {tf}: CSV não encontrado")
            if len(pair_data) == 3:
                pair_paths.append((pair.upper(), pair_data))

    if not pair_paths:
        msg = "Nenhum par disponível para treino — verifique os dados."
        print(f"\n❌ {msg}")
        log_entry['status'] = 'failed'
        log_entry['errors'].append(msg)
        save_retrain_log(log_entry)
        sys.exit(1)

    print(f"\n✅ {len(pair_paths)} par(es) prontos: {', '.join(n for n, _ in pair_paths)}")

    # ── Fase 2: Re-Treinamento ────────────────────────────────────────────────
    if args.skip_training or args.dry_run:
        mode_label = "DRY-RUN" if args.dry_run else "--skip-training"
        print(f"\n[{mode_label}] Pulando re-treino.")
        print("\n✅ Pipeline concluído (sem treino)")
        log_entry['status'] = 'data-only'
        save_retrain_log(log_entry)
        return

    print(f"\n{'─'*70}")
    print("🧠 FASE 2 — RE-TREINAMENTO")
    print(f"{'─'*70}")

    try:
        result = run_retrain(pair_paths, args.steps, today)
        log_entry['training'] = result
        log_entry['status']   = 'success'

        print(f"\n{'='*70}")
        print("✅ RE-TREINO CONCLUÍDO!")
        print(f"{'='*70}")
        print(f"  Checkpoint base   : {Path(result['base_checkpoint']).name}")
        print(f"  Novo checkpoint   : {Path(result['new_checkpoint']).name}")
        print(f"  Steps adicionados : {result['added_steps']:,}")
        print(f"  Total steps       : {result['total_steps']:,}")
        print(f"  Tempo decorrido   : {result['elapsed_s']:.0f}s  ({result['elapsed_s']/3600:.1f}h)")
        print(f"  Device            : {result['device'].upper()}")
        print(f"\n🎯 Próximos passos:")
        print(f"   1. Verificar métricas no TensorBoard")
        print(f"   2. Rodar backtest: python backtest_recurrent_ppo_v17.py \\")
        print(f"        --model {result['new_checkpoint']}")
        print(f"   3. Se win_rate > atual → promover via Champion/Challenger")

    except Exception as exc:
        err_str = traceback.format_exc()
        print(f"\n❌ ERRO no re-treino:\n{err_str}")
        log_entry['status'] = 'failed'
        log_entry['errors'].append(str(exc))
        log_entry['errors'].append(err_str)

    finally:
        save_retrain_log(log_entry)


if __name__ == '__main__':
    main()
