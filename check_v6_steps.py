"""
Verifica número de steps dos modelos V6
"""
from stable_baselines3 import SAC

models = [
    "models/sac_futuros_v6_final_20260112_012926.zip",
    "models/sac_futuros_v6_500000_steps.zip",
    "models/sac_futuros_v6_continue_600000_steps.zip",
    "models/sac_futuros_v6_continue_700000_steps.zip",
]

print("=" * 60)
print("VERIFICANDO STEPS DOS MODELOS V6")
print("=" * 60)

for model_path in models:
    try:
        model = SAC.load(model_path)
        print(f"\n{model_path.split('/')[-1]}")
        print(f"  Steps: {model.num_timesteps:,}")
    except Exception as e:
        print(f"\n{model_path.split('/')[-1]}")
        print(f"  Erro: {e}")

print("\n" + "=" * 60)
