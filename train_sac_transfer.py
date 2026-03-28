"""
MIGRAÇÃO TD3 → SAC COM TRANSFER LEARNING
Aproveita pesos do TD3 3M steps e fine-tune com SAC
Entropy regularization para aumentar exploração e win rate
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from src.environment.trading_env import TradingEnv
import torch
import subprocess
import os


class TD3toSACMigrator:
    """Migração de TD3 para SAC com transfer learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._detect_gpu()
        print(f"\n[DEVICE] {self.device}")


class CheckpointCallback(BaseCallback):
    """
    Callback que salva modelo a cada N steps e executa backtest automático.
    """
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        experiment_name: str,
        train_data_path: str,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.experiment_name = experiment_name
        self.train_data_path = train_data_path
        self.checkpoint_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.checkpoint_count += 1
            checkpoint_path = f"{self.save_path}/sac_{self.experiment_name}_checkpoint_{self.checkpoint_count}_{self.n_calls}.zip"
            
            print(f"\n{'='*70}")
            print(f"💾 CHECKPOINT {self.checkpoint_count} - {self.n_calls:,} steps")
            print(f"{'='*70}")
            
            # Salvar modelo
            self.model.save(checkpoint_path)
            print(f"✅ Modelo salvo: {checkpoint_path}")
            
            # Executar backtest automático
            print(f"\n🔬 Executando backtest automático...")
            try:
                result = subprocess.run(
                    ['python', 'backtest.py', checkpoint_path, self.train_data_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutos timeout
                )
                
                if result.returncode == 0:
                    # Extrair métricas do output
                    output = result.stdout
                    if "Return:" in output and "Winrate:" in output:
                        for line in output.split('\n'):
                            if any(metric in line for metric in ["Return:", "Winrate:", "Profit Factor:", "Sharpe:", "Total Trades:"]):
                                print(f"  {line.strip()}")
                    else:
                        print("  ✅ Backtest concluído")
                else:
                    print(f"  ⚠️ Backtest falhou: {result.stderr[:200]}")
            except Exception as e:
                print(f"  ⚠️ Erro no backtest: {str(e)[:100]}")
            
            print(f"{'='*70}\n")
        
        return True


class TD3toSACMigrator:
    """Migração de TD3 para SAC com transfer learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._detect_gpu()
        print(f"\n[DEVICE] {self.device}")
    
    def _detect_gpu(self) -> str:
        """Detecta GPU disponível."""
        try:
            import torch_directml
            dml_device = torch_directml.device()
            print("[OK] GPU AMD detectada via DirectML")
            return dml_device
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            print(f"[OK] GPU NVIDIA: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        
        print("[INFO] Usando CPU")
        return 'cpu'
    
    def create_environment(
        self,
        df: pd.DataFrame,
        use_sharpe_reward: bool = True,
        use_hybrid_reward: bool = False,
        max_episode_steps: int = 5000
    ):
        """
        Cria ambiente idêntico ao TD3 para comparação justa.
        
        Args:
            df: Dataset
            use_sharpe_reward: True = Sharpe (igual TD3), False = delta equity
            use_hybrid_reward: True = 0.6*delta + 0.4*sharpe (agressivo + estável)
            max_episode_steps: Tamanho do episódio
        """
        env_config = self.config['environment']
        
        env = TradingEnv(
            df=df,
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size'],
            max_episode_steps=max_episode_steps,
            random_start=True,
            persist_balance=False,
            use_sharpe_reward=use_sharpe_reward,
            use_hybrid_reward=use_hybrid_reward
        )
        
        return env
    
    def train_sac_from_td3(
        self,
        td3_model_path: str,
        train_data_path: str,
        timesteps: int = 1500000,
        use_sharpe_reward: bool = True,
        use_hybrid_reward: bool = False,
        max_episode_steps: int = 5000,
        use_sde: bool = False,
        experiment_name: str = "baseline"
    ):
        """
        Treina SAC aproveitando pesos do TD3.
        
        Args:
            td3_model_path: Caminho do modelo TD3
            train_data_path: Dataset de treino
            timesteps: Steps de fine-tuning (1.5M recomendado)
            use_sharpe_reward: Usar Sharpe ou delta equity
            max_episode_steps: Tamanho episódio
            use_sde: State Dependent Exploration (mais trades/exploração)
            experiment_name: Nome do experimento (baseline, delta_equity, etc)
        """
        print("\n" + "="*70)
        print(f"[SAC] MIGRAÇÃO TD3 → SAC")
        print("="*70)
        print(f"Experimento: {experiment_name}")
        print(f"TD3 base: {td3_model_path}")
        print(f"Timesteps: {timesteps:,}")
        print(f"Reward: {'Sharpe Ratio' if use_sharpe_reward else 'Hybrid (60% Delta + 40% Sharpe)' if use_hybrid_reward else 'Delta Equity'}")
        print(f"Episode steps: {max_episode_steps}")
        
        # Carregar dados
        df = pd.read_csv(train_data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"[OK] {len(df):,} candles carregados")
        
        # Criar ambiente
        env = self.create_environment(
            df=df,
            use_sharpe_reward=use_sharpe_reward,
            use_hybrid_reward=use_hybrid_reward,
            max_episode_steps=max_episode_steps
        )
        env = DummyVecEnv([lambda: env])
        
        # Criar modelo SAC com NET_ARCH IGUAL AO TD3 para transfer learning
        print(f"\n[SAC] Criando modelo SAC...")
        print("[CONFIG] Hyperparameters + Transfer Learning:")
        print("  learning_rate: 1e-4 (vs 5e-4 do TD3)")
        print("  batch_size: 256 (vs 512 do TD3)")
        print("  learning_starts: 10k (vs 25k do TD3)")
        print(f"  ent_coef: 0.5 (fixo, alta explora\u00e7\u00e3o)")
        print(f"  use_sde: {use_sde} (State Dependent Exploration)")
        print("  net_arch: [400, 300] (IGUAL TD3 - permite transfer learning!)")
        
        # Policy kwargs com net_arch igual ao TD3
        policy_kwargs = dict(
            net_arch=[400, 300],  # IGUAL TD3 para transfer learning!
            activation_fn=torch.nn.ReLU
        )
        
        # Action noise para exploração adicional
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.3 * np.ones(n_actions)  # 30% de noise
        )
        
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,  # net_arch=[400, 300]
            learning_rate=1e-4,         # MENOR que TD3 (5e-4) - conservador
            buffer_size=1000000,        # IGUAL TD3
            learning_starts=10000,      # MENOR que TD3 (25k) - treina mais cedo
            batch_size=256,             # MENOR que TD3 (512) - gradientes precisos
            tau=0.005,                  # IGUAL TD3
            gamma=0.995,                # IGUAL TD3
            train_freq=1,               # IGUAL TD3
            gradient_steps=1,           # IGUAL TD3
            ent_coef=0.5,               # FIXO - alta explora\u00e7\u00e3o (vs 'auto')
            target_entropy='auto',      # Target entropy autom\u00e1tico
            use_sde=use_sde,            # State Dependent Exploration
            action_noise=action_noise,  # Noise gaussiano 30%
            verbose=1,
            device=self.device
        )
        
        # Transfer Learning Manual: TD3 Actor → SAC Actor
        # Agora possível porque net_arch=[400, 300] é igual!
        print(f"\n[TRANSFER] Copiando pesos do Actor TD3 → SAC...")
        try:
            td3_model = TD3.load(td3_model_path, device=self.device)
            
            with torch.no_grad():
                if use_sde:
                    # Com SDE: Transfer apenas latent_pi (feature extractor)
                    # SDE adiciona layers extras que serão treinadas do zero
                    print(f"[SDE MODE] Copiando apenas feature extractor (latent_pi)...")
                    
                    # TD3 mu.0 → SAC latent_pi.0
                    model.policy.actor.latent_pi[0].weight.copy_(td3_model.policy.actor.mu[0].weight)
                    model.policy.actor.latent_pi[0].bias.copy_(td3_model.policy.actor.mu[0].bias)
                    
                    # TD3 mu.2 → SAC latent_pi.2
                    model.policy.actor.latent_pi[2].weight.copy_(td3_model.policy.actor.mu[2].weight)
                    model.policy.actor.latent_pi[2].bias.copy_(td3_model.policy.actor.mu[2].bias)
                    
                    print("[OK] Feature extractor copiado! Camadas SDE treinarão do zero")
                    
                else:
                    # Sem SDE: Transfer completo do actor
                    # TD3 Actor: mu.0 (input→400), mu.2 (400→300), mu.4 (300→1)
                    # SAC Actor: latent_pi.0 (input→400), latent_pi.2 (400→300), mu (300→1)
                    
                    # Copiar camadas do TD3 para SAC
                    # TD3 mu.0 → SAC latent_pi.0
                    model.policy.actor.latent_pi[0].weight.copy_(td3_model.policy.actor.mu[0].weight)
                    model.policy.actor.latent_pi[0].bias.copy_(td3_model.policy.actor.mu[0].bias)
                    
                    # TD3 mu.2 → SAC latent_pi.2
                    model.policy.actor.latent_pi[2].weight.copy_(td3_model.policy.actor.mu[2].weight)
                    model.policy.actor.latent_pi[2].bias.copy_(td3_model.policy.actor.mu[2].bias)
                    
                    # TD3 mu.4 → SAC mu (camada final)
                    model.policy.actor.mu.weight.copy_(td3_model.policy.actor.mu[4].weight)
                    model.policy.actor.mu.bias.copy_(td3_model.policy.actor.mu[4].bias)
                    
                    # Inicializar log_std pequeno (policy determinística inicialmente)
                    # log_std é uma Linear layer, não tensor direto
                    model.policy.actor.log_std.weight.data.fill_(0.0)
                    model.policy.actor.log_std.bias.data.fill_(np.log(0.2))
                    
                    print("[OK] Transfer learning completo! SAC começa com policy do TD3")
                
            print("     Actor: latent_pi[400, 300] + mu + log_std inicializado")
            print("     Critics: [400, 300] - treinarão do zero com entropy")
        except Exception as e:
            print(f"[AVISO] Transfer learning falhou: {e}")
            print("        SAC treinará do zero")
        
        # Callbacks
        best_model_dir = f'models/best_sac_{experiment_name}'
        Path(best_model_dir).mkdir(parents=True, exist_ok=True)
        
        # Checkpoint callback - salva a cada 200k steps + backtest automático
        checkpoint_callback = CheckpointCallback(
            save_freq=200000,
            save_path='models',
            experiment_name=experiment_name,
            train_data_path=train_data_path,
            verbose=1
        )
        
        # Ambiente de avaliação
        eval_env = self.create_environment(df, use_sharpe_reward, use_hybrid_reward, max_episode_steps)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=None,  # DESABILITADO: DirectML tem bug com gradients
            log_path=None,  # DESABILITADO: Evita disco cheio (evaluations.npz cresce muito)
            eval_freq=10000,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Combinar callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = CallbackList([checkpoint_callback, eval_cb])
        
        # Treinar SAC
        print(f"\n⏳ Iniciando fine-tuning SAC ({timesteps/1e6:.1f}M timesteps)...")
        print(f"💾 Checkpoint automático a cada 200k steps")
        print(f"🔬 Backtest automático após cada checkpoint")
        print(f"Tempo estimado: {timesteps/1e6 * 3:.1f}-{timesteps/1e6 * 4:.1f} horas")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False  # Continua contador do TD3 (comparação direta)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        # Salvar modelo final
        final_path = f'models/sac_{experiment_name}_final.zip'
        model.save(final_path)
        
        print(f"\n✅ TREINAMENTO SAC COMPLETO!")
        print(f"Tempo: {duration:.2f} horas")
        print(f"Modelo salvo em: {final_path}")
        
        return model, final_path


def main():
    """Executa migração TD3 → SAC."""
    import sys
    
    migrator = TD3toSACMigrator()
    
    # Configurações
    td3_model = 'models/base_btcusdt_final.zip'
    train_data = 'data/train_btcusdt_36m_20260109.csv'
    
    # Timesteps configurável via argumento
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        timesteps = int(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == 'quick':
        timesteps = 200000  # 200k steps para teste rápido (~40 min)
        experiment = 'baseline'  # Quick test usa baseline
    else:
        timesteps = 1500000  # 1.5M steps fine-tuning padrão
    
    if len(sys.argv) > 1 and sys.argv[1] != 'quick':
        experiment = sys.argv[1]
    elif len(sys.argv) > 1 and sys.argv[1] == 'quick':
        experiment = 'baseline'  # Quick usa baseline
    else:
        experiment = 'baseline'
    
    # Experimentos disponíveis
    experiments = {
        'baseline': {
            'use_sharpe_reward': True,
            'use_hybrid_reward': False,
            'max_episode_steps': 5000,
            'use_sde': False,
            'desc': 'Sharpe puro (conservador, herda TD3)',
            'priority': 3  # Menor prioridade
        },
        'delta': {
            'use_sharpe_reward': False,
            'use_hybrid_reward': False,
            'max_episode_steps': 5000,
            'use_sde': True,  # SDE para aumentar exploração com delta equity
            'desc': 'Delta equity + SDE (AGRESSIVO, prioridade ALTA)',
            'priority': 1  # MÁXIMA PRIORIDADE
        },
        'hybrid': {
            'use_sharpe_reward': False,
            'use_hybrid_reward': True,
            'max_episode_steps': 5000,
            'use_sde': True,
            'desc': 'Hybrid 60% delta + 40% sharpe + SDE (EQUILIBRADO)',
            'priority': 2  # Segunda prioridade
        },
        'short': {
            'use_sharpe_reward': True,
            'use_hybrid_reward': False,
            'max_episode_steps': 3000,
            'use_sde': True,  # SDE para mais trades em episódios curtos
            'desc': 'Sharpe + SDE, episódios curtos (mais trades)',
            'priority': 4
        },
        'long': {
            'use_sharpe_reward': True,
            'use_hybrid_reward': False,
            'max_episode_steps': 7000,
            'use_sde': False,
            'desc': 'Sharpe, episódios longos (holds longos)',
            'priority': 5
        }
    }
    
    if experiment not in experiments:
        print(f"Experimento '{experiment}' não encontrado!")
        print(f"Disponíveis: {', '.join(experiments.keys())}")
        return
    
    config = experiments[experiment]
    
    print("\n" + "="*70)
    print(f"EXPERIMENTO SAC: {experiment}")
    print("="*70)
    print(f"Descrição: {config['desc']}")
    print(f"Prioridade: {'⭐' * config.get('priority', 3)}")
    print(f"Sharpe Reward: {config['use_sharpe_reward']}")
    print(f"Hybrid Reward: {config.get('use_hybrid_reward', False)}")
    print(f"Episode Steps: {config['max_episode_steps']}")
    print(f"SDE (State Dependent Exploration): {config['use_sde']}")
    print(f"Timesteps: {timesteps:,}")
    if timesteps == 200000:
        print("\n[QUICK TEST] Treino rápido para validar ensemble TD3+SAC")
        print("             Duração: ~40 minutos")
        print("             Após validar, execute treinamento completo (1.5M)")
    print("="*70)
    
    # Executar migração
    migrator.train_sac_from_td3(
        td3_model_path=td3_model,
        train_data_path=train_data,
        timesteps=timesteps,
        use_sharpe_reward=config['use_sharpe_reward'],
        use_hybrid_reward=config.get('use_hybrid_reward', False),
        max_episode_steps=config['max_episode_steps'],
        use_sde=config['use_sde'],
        experiment_name=experiment
    )
    
    print("\n" + "="*70)
    print("✅ MIGRAÇÃO COMPLETA!")
    print("="*70)
    print(f"Modelo SAC: models/sac_{experiment}_final.zip")
    print("\nPróximo passo: Executar backtest")
    print(f"  python backtest.py models/sac_{experiment}_final.zip data/train_btcusdt_36m_20260109.csv")


if __name__ == "__main__":
    main()
