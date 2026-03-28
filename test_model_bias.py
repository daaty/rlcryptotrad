"""
Teste diagnóstico: Verifica se o modelo tem bias para short
"""
from stable_baselines3 import SAC
import numpy as np

print("Carregando modelo...")
model = SAC.load('models/sac_v8_100000_steps.zip')

print("\nTestando com observações aleatórias:")
print("=" * 60)

# Criar observação aleatória com shape correto (50 timesteps, 58 features)
test_obs = np.random.randn(50, 58).astype(np.float32)

actions = []
for i in range(100):
    # Randomizar a observação a cada teste
    test_obs = np.random.randn(50, 58).astype(np.float32)
    
    action, _ = model.predict(test_obs, deterministic=True)
    action_value = float(action[0])
    actions.append(action_value)
    
    # Classificar ação
    if action_value < -0.1:
        position = "Short"
    elif action_value > 0.1:
        position = "Long"
    else:
        position = "Flat"
    
    if i < 10:  # Mostra primeiros 10
        print(f"Test {i+1:3d}: action={action_value:+.4f} → {position}")

# Análise estatística
actions = np.array(actions)
short_count = np.sum(actions < -0.1)
long_count = np.sum(actions > 0.1)
flat_count = np.sum((actions >= -0.1) & (actions <= 0.1))

print("\n" + "=" * 60)
print("ESTATÍSTICAS (100 testes com obs aleatórias):")
print(f"  Long:  {long_count:3d} ({long_count}%)")
print(f"  Short: {short_count:3d} ({short_count}%)")
print(f"  Flat:  {flat_count:3d} ({flat_count}%)")
print(f"\n  Action média: {actions.mean():.4f}")
print(f"  Action std:   {actions.std():.4f}")
print(f"  Action min:   {actions.min():.4f}")
print(f"  Action max:   {actions.max():.4f}")

# DIAGNÓSTICO
print("\n" + "=" * 60)
print("DIAGNÓSTICO:")
if short_count > 80:
    print("❌ MODELO TEM BIAS EXTREMO PARA SHORT!")
    print("   Isso explica o backtest mostrando só short.")
elif long_count > 80:
    print("❌ MODELO TEM BIAS EXTREMO PARA LONG!")
elif flat_count > 80:
    print("⚠️  MODELO ESTÁ PARALISADO (SÓ FAZ FLAT)")
else:
    print("✅ Modelo parece balanceado nos testes aleatórios.")
    print("   O problema deve estar no BACKTEST, não no modelo.")
