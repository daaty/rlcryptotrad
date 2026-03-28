"""
Teste diagnóstico do modelo V6 em modo estocástico
Verifica se o modelo varia ações com noise
"""

from stable_baselines3 import SAC
import numpy as np

# Carrega modelo V6
model = SAC.load("models/sac_futuros_v6_final_20260112_012926.zip")

print("=" * 60)
print("TESTE: V6 em modo estocástico com 10 predições")
print("=" * 60)

# Cria observação sintética (50, 19) simulando mercado atual
obs = np.random.randn(50, 19).astype(np.float32) * 0.1

# Testa 10 predições consecutivas (modo estocástico)
print("\n🎲 MODO ESTOCÁSTICO (deterministic=False):")
actions_stochastic = []
for i in range(10):
    action, _ = model.predict(obs, deterministic=False)
    actions_stochastic.append(float(action[0]))
    label = "LONG" if action[0] > 0.33 else ("SHORT" if action[0] < -0.33 else "FLAT")
    print(f"  Run {i+1:2d}: {action[0]:+.4f} → {label}")

print(f"\n📊 Estatísticas Estocásticas:")
print(f"  Média: {np.mean(actions_stochastic):+.4f}")
print(f"  Desvio: {np.std(actions_stochastic):.4f}")
print(f"  Min: {np.min(actions_stochastic):+.4f}")
print(f"  Max: {np.max(actions_stochastic):+.4f}")

# Testa 10 predições consecutivas (modo determinístico)
print("\n🎯 MODO DETERMINÍSTICO (deterministic=True):")
actions_deterministic = []
for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    actions_deterministic.append(float(action[0]))
    label = "LONG" if action[0] > 0.33 else ("SHORT" if action[0] < -0.33 else "FLAT")
    print(f"  Run {i+1:2d}: {action[0]:+.4f} → {label}")

print(f"\n📊 Estatísticas Determinísticas:")
print(f"  Média: {np.mean(actions_deterministic):+.4f}")
print(f"  Desvio: {np.std(actions_deterministic):.4f}")
print(f"  Min: {np.min(actions_deterministic):+.4f}")
print(f"  Max: {np.max(actions_deterministic):+.4f}")

# Análise
stoch_range = np.max(actions_stochastic) - np.min(actions_stochastic)
det_range = np.max(actions_deterministic) - np.min(actions_deterministic)

print("\n" + "=" * 60)
print("ANÁLISE:")
print("=" * 60)
print(f"Variação estocástica: {stoch_range:.4f}")
print(f"Variação determinística: {stoch_range:.4f}")

if stoch_range > 0.1:
    print("✅ Modelo explora corretamente com noise")
else:
    print("❌ Modelo não varia (possível colapso ou cache)")

# Conta distribuição de ações
long_count_stoch = sum(1 for a in actions_stochastic if a > 0.33)
flat_count_stoch = sum(1 for a in actions_stochastic if -0.33 <= a <= 0.33)
short_count_stoch = sum(1 for a in actions_stochastic if a < -0.33)

print(f"\nDistribuição estocástica:")
print(f"  LONG: {long_count_stoch}/10 ({long_count_stoch*10}%)")
print(f"  FLAT: {flat_count_stoch}/10 ({flat_count_stoch*10}%)")
print(f"  SHORT: {short_count_stoch}/10 ({short_count_stoch*10}%)")
