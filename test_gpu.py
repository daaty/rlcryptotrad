"""
🚀 Teste rápido de treinamento com GPU AMD
"""

import logging
from src.training.ensemble_trainer import EnsembleTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("="*60)
print("🎮 TESTE DE GPU AMD RX 7700")
print("="*60)

trainer = EnsembleTrainer()

# Treina apenas PPO por 2000 timesteps (rápido)
print("\n🏋️ Treinando PPO com GPU...")
models = trainer.train_all(
    train_data_path='data/train_data.csv',
    val_data_path='data/val_data.csv',
    total_timesteps=2000  # Apenas 2k para teste rápido
)

print("\n✅ Teste concluído!")
print("Se viu 'Using privateuseone:0 device' = GPU está funcionando! 🚀")
