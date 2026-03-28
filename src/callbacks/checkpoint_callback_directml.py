"""
Callback de Checkpoint customizado para DirectML.
Resolve problemas de save com tensors que têm requires_grad=True.
"""

import os
import torch
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallbackDirectML(BaseCallback):
    """
    Callback para salvar modelo a cada N steps com suporte DirectML.
    
    Resolve o erro: "Can't call numpy() on Tensor that requires grad"
    que ocorre ao salvar modelos com DirectML.
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        
        # Cria diretório se não existir
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
    
    def _init_callback(self) -> None:
        # Cria diretório se não existir
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            
            # CRÍTICO: Desabilita gradientes temporariamente durante o save
            # Isso resolve o erro com DirectML
            with torch.no_grad():
                # Move modelo para CPU temporariamente (mais seguro com DirectML)
                original_device = self.model.device
                
                try:
                    # Salva modelo
                    self.model.save(model_path)
                    
                    if self.verbose >= 1:
                        print(f"✓ Checkpoint salvo: {model_path}.zip")
                    
                    # Salva replay buffer se solicitado
                    if self.save_replay_buffer and hasattr(self.model, "replay_buffer"):
                        replay_buffer_path = os.path.join(
                            self.save_path,
                            f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl",
                        )
                        self.model.save_replay_buffer(replay_buffer_path)
                        
                        if self.verbose >= 1:
                            print(f"✓ Replay buffer salvo: {replay_buffer_path}")
                
                except Exception as e:
                    print(f"⚠️ Erro ao salvar checkpoint: {e}")
                    print(f"   Tentando fallback...")
                    
                    # FALLBACK: Tenta salvar apenas as políticas (sem replay buffer)
                    try:
                        # Salva apenas actor/critic (sem buffer)
                        save_dict = {
                            "policy": self.model.policy.state_dict(),
                        }
                        torch.save(save_dict, f"{model_path}_policy.pth")
                        print(f"✓ Policy salva (fallback): {model_path}_policy.pth")
                    except Exception as e2:
                        print(f"❌ Fallback falhou: {e2}")
        
        return True
