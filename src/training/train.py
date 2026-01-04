"""
Script de Treinamento do Agente de RL
Usa Stable Baselines3 com algoritmo PPO.
"""

import yaml
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append(str(Path(__file__).parent.parent))

from environment.trading_env import TradingEnv


class CustomCallback(EvalCallback):
    """Callback personalizado para logging detalhado."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Log a cada 1000 steps
        if self.n_calls % 1000 == 0:
            print(f"\nStep {self.n_calls}:")
            print(f"  Mean Reward: {np.mean(self.episode_rewards[-10:]):.2f}")
            print(f"  Best Mean Reward: {self.best_mean_reward:.2f}")
        
        return result


class TradingTrainer:
    """Gerencia o treinamento do agente de trading."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Caminho para arquivo de configuração
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.env_config = self.config['environment']
        
        # Diretórios
        self.models_dir = Path("models")
        self.logs_dir = Path("logs")
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def create_env(self, data_path: str, is_eval: bool = False) -> DummyVecEnv:
        """
        Cria um ambiente de trading vetorizado.
        
        Args:
            data_path: Caminho para o arquivo CSV com dados
            is_eval: Se True, cria ambiente de avaliação
            
        Returns:
            Ambiente vetorizado
        """
        # Carrega dados
        df = pd.read_csv(data_path, index_col=0)
        
        # Cria ambiente base
        env = TradingEnv(
            df=df,
            initial_balance=self.env_config['initial_balance'],
            commission=self.env_config['commission'],
            leverage=self.env_config['leverage'],
            position_size=self.env_config['position_size']
        )
        
        # Envelopa com Monitor para logging
        log_file = self.logs_dir / ("eval" if is_eval else "train")
        env = Monitor(env, str(log_file))
        
        # Vetoriza (necessário para SB3)
        env = DummyVecEnv([lambda: env])
        
        return env
    
    def create_model(self, env: DummyVecEnv) -> PPO:
        """
        Cria o modelo PPO com configurações otimizadas.
        
        Args:
            env: Ambiente de treinamento
            
        Returns:
            Modelo PPO
        """
        print("\n🧠 Criando modelo PPO...")
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.training_config['learning_rate'],
            n_steps=self.training_config['n_steps'],
            batch_size=self.training_config['batch_size'],
            n_epochs=self.training_config['n_epochs'],
            gamma=self.training_config['gamma'],
            gae_lambda=self.training_config['gae_lambda'],
            clip_range=self.training_config['clip_range'],
            ent_coef=self.training_config['ent_coef'],
            verbose=1,
            tensorboard_log=str(self.logs_dir / "tensorboard")
        )
        
        print("✅ Modelo criado")
        
        return model
    
    def train(
        self,
        train_data_path: str = "data/train_data.csv",
        val_data_path: str = "data/val_data.csv",
        model_name: str = None
    ):
        """
        Treina o agente de trading.
        
        Args:
            train_data_path: Caminho para dados de treino
            val_data_path: Caminho para dados de validação
            model_name: Nome do modelo a salvar (auto-gerado se None)
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO TREINAMENTO DO AGENTE DE TRADING")
        print("="*60)
        
        # Cria ambientes
        print("\n📊 Carregando dados...")
        train_env = self.create_env(train_data_path, is_eval=False)
        eval_env = self.create_env(val_data_path, is_eval=True)
        
        # Cria modelo
        model = self.create_model(train_env)
        
        # Callbacks
        eval_callback = CustomCallback(
            eval_env=eval_env,
            best_model_save_path=str(self.models_dir),
            log_path=str(self.logs_dir),
            eval_freq=5000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        callbacks = CallbackList([eval_callback])
        
        # Treina
        print(f"\n🏋️ Treinando por {self.training_config['total_timesteps']} timesteps...")
        print("Pressione Ctrl+C para interromper o treinamento.\n")
        
        try:
            model.learn(
                total_timesteps=self.training_config['total_timesteps'],
                callback=callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Treinamento interrompido pelo usuário")
        
        # Salva modelo final
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"ppo_trading_agent_{timestamp}"
        
        model_path = self.models_dir / f"{model_name}.zip"
        model.save(str(model_path))
        
        print("\n" + "="*60)
        print(f"✅ Treinamento concluído!")
        print(f"💾 Modelo salvo em: {model_path}")
        print(f"📊 Logs em: {self.logs_dir}")
        print("="*60)
        
        return model
    
    def evaluate(self, model_path: str, test_data_path: str = "data/test_data.csv"):
        """
        Avalia o modelo treinado em dados de teste.
        
        Args:
            model_path: Caminho para o modelo .zip
            test_data_path: Caminho para dados de teste
        """
        print("\n" + "="*60)
        print("📈 AVALIANDO MODELO")
        print("="*60)
        
        # Carrega modelo
        model = PPO.load(model_path)
        
        # Cria ambiente de teste
        test_env = self.create_env(test_data_path, is_eval=True)
        
        # Roda episódio completo
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print("\n🎮 Executando no conjunto de teste...\n")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            steps += 1
            
            # Mostra progresso a cada 100 steps
            if steps % 100 == 0:
                print(f"Step {steps} | Reward acumulado: {total_reward:.2f}")
        
        # Resultados finais
        final_info = info[0]
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DO TESTE")
        print("="*60)
        print(f"Reward Total: {total_reward:.2f}")
        print(f"Balance Final: ${final_info['balance']:.2f}")
        print(f"Equity Final: ${final_info['equity']:.2f}")
        print(f"Trades: {final_info['trades']}")
        print(f"Wins: {final_info['wins']} | Losses: {final_info['losses']}")
        print(f"Win Rate: {final_info['win_rate']:.2%}")
        print(f"Total PnL: ${final_info['total_pnl']:.2f}")
        print(f"Return: {(final_info['equity'] / self.env_config['initial_balance'] - 1) * 100:.2f}%")
        print("="*60)
        
        return final_info


def main():
    """Função principal para execução via CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar agente de trading com RL")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Modo de operação")
    parser.add_argument("--model", type=str, help="Caminho do modelo (para avaliação)")
    parser.add_argument("--name", type=str, help="Nome do modelo a salvar")
    
    args = parser.parse_args()
    
    trainer = TradingTrainer()
    
    if args.mode == "train":
        trainer.train(model_name=args.name)
    elif args.mode == "eval":
        if not args.model:
            print("❌ Erro: Especifique o modelo com --model")
            return
        trainer.evaluate(args.model)


if __name__ == "__main__":
    main()
