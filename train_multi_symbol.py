"""
SISTEMA DE TREINAMENTO MULTI-SYMBOL COM TRANSFER LEARNING
Treina modelo base no BTC, depois fine-tune para ETH/BNB/SOL
Economia de 70-90% no tempo de treinamento
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.environment.trading_env import TradingEnv
import torch


class MultiSymbolTrainer:
    """Treinamento multi-symbol com transfer learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_model_path = None
        self.models = {}
        
        # Auto-detectar GPU AMD (DirectML) ou NVIDIA (CUDA)
        self.device = self._detect_gpu()
        print(f"\n[DEVICE] {self.device}")
    
    def _detect_gpu(self) -> str:
        """Detecta GPU disponível (AMD DirectML ou NVIDIA CUDA)."""
        try:
            # Tenta importar torch_directml (AMD GPU no Windows)
            import torch_directml
            dml_device = torch_directml.device()
            print("[OK] GPU AMD detectada via DirectML")
            return dml_device
        except ImportError:
            pass
        
        # Tenta CUDA (NVIDIA)
        if torch.cuda.is_available():
            print(f"[OK] GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        
        # Fallback para CPU
        print("[INFO] Usando CPU (nenhuma GPU detectada)")
        return 'cpu'
        
    def train_base_model(
        self,
        symbol: str,
        train_data_path: str,
        model_class,
        timesteps: int = 2000000,  # AUMENTADO: 1M -> 2M steps
        model_name: str = "base",
        continue_training: bool = False  # NOVO: permite continuar de checkpoint
    ):
        """
        Treina modelo BASE no símbolo principal (geralmente BTC).
        Este é o modelo que será usado para transfer learning.
        
        Args:
            symbol: Símbolo do ativo (ex: 'BTC/USDT')
            train_data_path: Caminho dos dados de treino
            model_class: PPO ou TD3
            timesteps: Timesteps de treinamento (default: 1M)
            model_name: Nome para salvar
            continue_training: Se True, continua de checkpoint existente
        """
        print("\n" + "="*64)
        print(f"[TRAIN] TREINANDO MODELO BASE: {symbol}")
        print("="*64)
        print(f"Modelo: {model_class.__name__}")
        print(f"Timesteps: {timesteps:,}")
        print(f"Dados: {train_data_path}")
        
        if continue_training:
            print(f"[CONTINUE] Continuando treinamento de checkpoint existente")
        
        # Carregar dados
        df = pd.read_csv(train_data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"[OK] {len(df):,} candles carregados")
        
        # Criar ambiente
        env_config = self.config['environment']
        env = TradingEnv(
            df=df,
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size'],
            max_episode_steps=5000,  # AUMENTADO: 1500 -> 5000
            random_start=True,  # Random start mantido
            persist_balance=False,  # DESABILITADO: Causava truncation instantânea
            use_sharpe_reward=True  # Usa Sharpe Ratio como reward
        )
        env = DummyVecEnv([lambda: env])
        
        # Criar modelo ou carregar checkpoint
        symbol_clean = symbol.replace('/', '').lower()
        checkpoint_path = f'models/{model_name}_{symbol_clean}_final.zip'
        
        if continue_training and Path(checkpoint_path).exists():
            print(f"[LOAD] Carregando checkpoint: {checkpoint_path}")
            if model_class == PPO:
                model = PPO.load(checkpoint_path, env=env, device=self.device)
                print(f"[OK] Modelo PPO carregado, continuando treinamento...")
            else:  # TD3
                model = TD3.load(checkpoint_path, env=env, device=self.device)
                print(f"[OK] Modelo TD3 carregado, continuando treinamento...")
        else:
            print(f"[NEW] Criando novo modelo do zero...")
            if model_class == PPO:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=4096,
                    batch_size=256,
                    n_epochs=15,
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.1,
                    verbose=1,
                    device=self.device
                )
            else:  # TD3
                from stable_baselines3.common.noise import NormalActionNoise
                
                # AUMENTAR NOISE DRASTICAMENTE para forçar exploração
                # Action space é Box(1,) então noise shape = (1,)
                action_noise = NormalActionNoise(
                    mean=np.zeros(1), 
                    sigma=0.5 * np.ones(1)  # 0.5 = 50% de noise (vs 0.1 padrão)
                )
                
                model = TD3(
                    "MlpPolicy",
                env,
                learning_rate=5e-4,  # AUMENTADO: 3e-4 -> 5e-4 (mais dados = lr maior)
                buffer_size=1000000,  # DOBRADO: 500k -> 1M (mais dados históricos)
                learning_starts=25000,  # AUMENTADO: 10k -> 25k (aprende melhor antes de treinar)
                batch_size=512,  # DOBRADO: 256 -> 512 (aproveita 82k candles)
                tau=0.005,
                gamma=0.995,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,  # NOISE ALTO - força exploração caótica
                verbose=1,
                device=self.device  # GPU AMD/NVIDIA ou CPU
            )
        
        # Callbacks
        symbol_clean = symbol.replace('/', '').lower()
        # checkpoint_dir = f'models/checkpoints_{model_name}_{symbol_clean}'
        best_model_dir = f'models/best_{model_name}_{symbol_clean}'
        
        # Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(best_model_dir).mkdir(parents=True, exist_ok=True)
        
        # DESABILITADO: Checkpoints intermediarios consomem muito espaco
        # checkpoint_cb = CheckpointCallback(
        #     save_freq=400000,
        #     save_path=checkpoint_dir,
        #     name_prefix=f'{model_class.__name__.lower()}_model'
        # )
        
        # Criar ambiente de validação com Monitor
        eval_env = TradingEnv(
            df=df,  # TODO: usar dados de validação separados
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size'],
            max_episode_steps=5000,  # Mesmo tamanho do treino
            random_start=True,
            persist_balance=False,  # DESABILITADO
            use_sharpe_reward=True
        )
        eval_env = Monitor(eval_env)  # Wrap com Monitor para métricas corretas
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=best_model_dir,
            eval_freq=10000,  # Aumentado: a cada 10k steps (vs 25k)
            n_eval_episodes=1,  # 1 episódio completo já tem 27k steps
            deterministic=True,
            render=False
        )
        
        # Treinar
        print(f"\n⏳ Iniciando treinamento ({timesteps/1e6:.1f}M timesteps)...")
        print(f"Tempo estimado: {timesteps/1e6 * 2:.1f}-{timesteps/1e6 * 3:.1f} horas")
        
        if continue_training:
            print(f"[RESUME] Treinamento adicional de {timesteps:,} steps")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=timesteps,
            callback=eval_cb,  # Apenas eval callback (salva o melhor modelo)
            progress_bar=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Salvar modelo final
        final_path = f'models/{model_name}_{symbol_clean}_final.zip'
        model.save(final_path)
        
        print(f"\n✅ TREINAMENTO BASE COMPLETO!")
        print(f"Tempo: {duration:.2f} horas")
        print(f"Modelo salvo em: {final_path}")
        
        # Armazenar caminho do modelo base
        self.base_model_path = final_path
        self.models[symbol] = {
            'path': final_path,
            'type': model_class.__name__,
            'timesteps': timesteps,
            'duration': duration
        }
        
        return model, final_path
    
    def fine_tune_model(
        self,
        base_model_path: str,
        symbol: str,
        train_data_path: str,
        model_class,
        timesteps: int = 100000,  # 10% do base
        model_name: str = None
    ):
        """
        Fine-tune do modelo base para novo símbolo (TRANSFER LEARNING).
        
        Args:
            base_model_path: Caminho do modelo base treinado
            symbol: Novo símbolo (ex: 'ETH/USDT')
            train_data_path: Dados de treino do novo símbolo
            model_class: PPO ou TD3
            timesteps: Timesteps de fine-tuning (10-20% do original)
            model_name: Nome para salvar
        """
        if model_name is None:
            model_name = symbol.replace('/', '').lower()
        
        print("\n" + "="*64)
        print(f"🔧 FINE-TUNING PARA: {symbol}")
        print("="*64)
        print(f"Modelo base: {base_model_path}")
        print(f"Timesteps: {timesteps:,} (fine-tuning)")
        print(f"Dados: {train_data_path}")
        
        # Carregar dados do novo símbolo
        df = pd.read_csv(train_data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✅ {len(df):,} candles carregados")
        
        # Criar ambiente
        env_config = self.config['environment']
        env = TradingEnv(
            df=df,
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size']
        )
        env = DummyVecEnv([lambda: env])
        
        # 🔑 CARREGAR MODELO BASE (Transfer Learning)
        print(f"\n🔄 Carregando modelo base...")
        if model_class == PPO:
            model = PPO.load(base_model_path, env=env)
        else:
            model = TD3.load(base_model_path, env=env)
        
        print(f"✅ Modelo base carregado com sucesso!")
        print(f"🎓 Conhecimento do BTC será adaptado para {symbol}")
        
        # Callbacks
        symbol_clean = symbol.replace('/', '').lower()
        checkpoint_dir = f'models/checkpoints_finetune_{symbol_clean}'
        best_model_dir = f'models/best_finetune_{symbol_clean}'
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(best_model_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_cb = CheckpointCallback(
            save_freq=50000,  # Mais frequente em fine-tuning
            save_path=checkpoint_dir,
            name_prefix=f'{model_class.__name__.lower()}_finetune'
        )
        
        eval_cb = EvalCallback(
            env,
            best_model_save_path=best_model_dir,
            log_path=best_model_dir,
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Fine-tuning
        print(f"\n⏳ Iniciando fine-tuning ({timesteps/1e3:.0f}k timesteps)...")
        print(f"Tempo estimado: {timesteps/1e6 * 2:.1f}-{timesteps/1e6 * 3:.1f} horas")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
            reset_num_timesteps=False  # Continua contagem do modelo base
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Salvar modelo fine-tuned
        final_path = f'models/{model_name}_finetune.zip'
        model.save(final_path)
        
        print(f"\n✅ FINE-TUNING COMPLETO!")
        print(f"Tempo: {duration:.2f} horas")
        print(f"Modelo salvo em: {final_path}")
        print(f"💾 Economia vs treino do zero: ~{100*(1-timesteps/1000000):.0f}%")
        
        self.models[symbol] = {
            'path': final_path,
            'type': model_class.__name__,
            'base_model': base_model_path,
            'timesteps': timesteps,
            'duration': duration
        }
        
        return model, final_path
    
    def train_pipeline_multi_symbol(
        self,
        symbols: list,
        model_class=PPO,
        base_timesteps: int = 1000000,
        finetune_timesteps: int = 100000
    ):
        """
        Pipeline completo de treinamento multi-symbol:
        1. Treina modelo base no primeiro símbolo (BTC)
        2. Fine-tune para demais símbolos
        
        Args:
            symbols: Lista de símbolos (primeiro é base)
            model_class: PPO ou TD3
            base_timesteps: Timesteps para modelo base
            finetune_timesteps: Timesteps para fine-tuning
        """
        print("\n" + "="*70)
        print(f"🚀 PIPELINE MULTI-SYMBOL COM TRANSFER LEARNING")
        print("="*70)
        print(f"Símbolos: {', '.join(symbols)}")
        print(f"Modelo: {model_class.__name__}")
        print(f"Base timesteps: {base_timesteps:,}")
        print(f"Fine-tune timesteps: {finetune_timesteps:,}")
        
        base_symbol = symbols[0]
        other_symbols = symbols[1:]
        
        # 1. Treinar modelo BASE
        print(f"\n📍 FASE 1: Treinar modelo BASE em {base_symbol}")
        
        # Buscar arquivo de dados mais recente
        symbol_clean = base_symbol.replace('/', '').lower()
        data_files = list(Path('data').glob(f'train_{symbol_clean}_*m_*.csv'))
        
        if not data_files:
            raise FileNotFoundError(f"Dados de treino não encontrados para {base_symbol}")
        
        base_train_data = str(sorted(data_files)[-1])  # Mais recente
        
        base_model, base_model_path = self.train_base_model(
            symbol=base_symbol,
            train_data_path=base_train_data,
            model_class=model_class,
            timesteps=base_timesteps,
            model_name=f"{model_class.__name__.lower()}_base"
        )
        
        # 2. Fine-tune para outros símbolos
        if other_symbols:
            print(f"\n📍 FASE 2: Fine-tuning para {len(other_symbols)} símbolos")
            
            for i, symbol in enumerate(other_symbols, 1):
                print(f"\n[{i}/{len(other_symbols)}] {symbol}")
                
                # Buscar dados
                symbol_clean = symbol.replace('/', '').lower()
                data_files = list(Path('data').glob(f'train_{symbol_clean}_*m_*.csv'))
                
                if not data_files:
                    print(f"⚠️  Dados não encontrados para {symbol}, pulando...")
                    continue
                
                train_data = str(sorted(data_files)[-1])
                
                try:
                    self.fine_tune_model(
                        base_model_path=base_model_path,
                        symbol=symbol,
                        train_data_path=train_data,
                        model_class=model_class,
                        timesteps=finetune_timesteps
                    )
                except Exception as e:
                    print(f"❌ ERRO em {symbol}: {e}")
                    continue
        
        # 3. Resumo final
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumo de todos os modelos treinados."""
        print("\n" + "="*70)
        print("📊 RESUMO DO TREINAMENTO MULTI-SYMBOL")
        print("="*70)
        
        total_time = sum(m['duration'] for m in self.models.values())
        
        for symbol, info in self.models.items():
            print(f"\n{symbol}:")
            print(f"  Tipo: {info['type']}")
            print(f"  Timesteps: {info['timesteps']:,}")
            print(f"  Duração: {info['duration']:.2f}h")
            print(f"  Modelo: {info['path']}")
            
            if 'base_model' in info:
                print(f"  Transfer Learning de: {info['base_model']}")
        
        print(f"\n⏱️  Tempo total: {total_time:.2f} horas")
        print(f"✅ {len(self.models)} modelos treinados")
        
        # Estimar economia
        if len(self.models) > 1:
            no_transfer_time = len(self.models) * (1000000 / 1e6 * 2.5)  # Média
            savings = (no_transfer_time - total_time) / no_transfer_time * 100
            print(f"💰 Economia vs treino do zero: ~{savings:.0f}%")


def main():
    """Exemplo de uso do treinamento multi-symbol."""
    import sys
    
    # Símbolos para treinar (BTC primeiro = base model)
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    
    # Configuração
    trainer = MultiSymbolTrainer()
    
    if len(sys.argv) > 1:
        # Modo custom
        if sys.argv[1] == 'base':
            # Treinar apenas TD3 base (mais estável que PPO para trading)
            print("Modo: Treinar TD3 BASE (BTC)")
            print("Dataset: 36 meses (ciclo completo: $19k -> $125k)")
            print("Duração estimada: 8-9 horas (3M steps)\n")
            # Treinar TD3 base (1M steps, continuando de 200k)
            print("\n" + "="*70)
            print("[TD3] TREINANDO TD3 BASE")
            print("="*70)
            print("[OK] TD3 e mais adequado para trading:")
            print("   - Action space continuo (melhor para size/timing)")
            print("   - Twin Q-networks (previne overestimation)")
            print("   - Target policy smoothing (decisoes mais estaveis)\n")
            
            trainer.train_base_model(
                symbol='BTC/USDT',
                train_data_path='data/train_btcusdt_36m_20260109.csv',  # NOVO: 82k candles (36 meses - ciclo completo!)
                model_class=TD3,
                timesteps=3000000,  # AUMENTADO: 3M steps (50% mais dados = mais treinamento)
                model_name='base',
                continue_training=True  # Fine-tune do modelo 2M steps com dataset maior
            )
            
            print("\n" + "="*70)
            print("✅ TREINAMENTO TD3 COMPLETO!")
            print("="*70)
            print("Modelo salvo:")
            print("  - models/base_btcusdt_final.zip")
            print("\nPróximo passo: Executar backtest")
            print("  python backtest.py models/base_btcusdt_final.zip")
        elif sys.argv[1] == 'finetune':
            # Fine-tune específico
            if len(sys.argv) < 3:
                print("Uso: python train_multi_symbol.py finetune <SYMBOL>")
                return
            
            symbol = sys.argv[2]
            print(f"Modo: Fine-tuning para {symbol}")
            
            trainer.fine_tune_model(
                base_model_path='models/ppo_base_btcusdt_final.zip',
                symbol=symbol,
                train_data_path=f'data/train_{symbol.replace("/", "").lower()}_12m_20260105.csv',
                model_class=PPO,
                timesteps=100000
            )
    else:
        # Modo automático: pipeline completo
        print("Modo: Pipeline COMPLETO multi-symbol")
        trainer.train_pipeline_multi_symbol(
            symbols=symbols,
            model_class=PPO,
            base_timesteps=1000000,  # 1M para base
            finetune_timesteps=100000  # 100k para cada fine-tune
        )


if __name__ == "__main__":
    main()
