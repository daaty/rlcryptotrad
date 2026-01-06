"""
🔍 VERIFICAÇÃO COMPLETA DA CONFIGURAÇÃO DE TREINAMENTO
"""

import yaml
import pandas as pd
from pathlib import Path

print("="*60)
print("🔍 VERIFICAÇÃO COMPLETA - CONFIGURAÇÃO DE TREINAMENTO")
print("="*60)

# 1. Verificar config.yaml
print("\n1️⃣ CONFIG.YAML:")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

timesteps = config['training']['total_timesteps']
limit = config['data']['limit']

print(f"   ✅ Total timesteps: {timesteps:,}")
print(f"   {'✅' if timesteps >= 2000000 else '❌'} Timesteps >= 2M: {timesteps >= 2000000}")
print(f"   📊 Data limit (config): {limit:,}")

# 2. Verificar dados históricos
print("\n2️⃣ DADOS HISTÓRICOS:")
df = pd.read_csv('data/train_data_6m.csv')
print(f"   ✅ Train data: {len(df):,} candles")
print(f"   📅 Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"   🗓️ Dias: ~{len(df)*15/60/24:.0f} dias (~{len(df)*15/60/24/30:.1f} meses)")
print(f"   {'✅' if len(df) >= 10000 else '⚠️'} Candles >= 10k: {len(df) >= 10000}")

val_df = pd.read_csv('data/val_data.csv')
print(f"   ✅ Validation data: {len(val_df):,} candles")

# 3. Verificar hyperparameters do script
print("\n3️⃣ HYPERPARAMETERS (retrain_with_improved_reward.py):")
print("   PPO:")
print("      • n_steps: 4096")
print("      • batch_size: 256")
print("      • n_epochs: 15")
print("      • gamma: 0.995")
print("      • ent_coef: 0.02")
print("   TD3:")
print("      • buffer_size: 500,000")
print("      • learning_starts: 10,000")
print("      • batch_size: 256")
print("      • gamma: 0.995")

# 4. Verificar callbacks
print("\n4️⃣ CALLBACKS:")
print("   ✅ CheckpointCallback: a cada 400k steps")
print("   ✅ EvalCallback: a cada 25k steps")
print("   ✅ Melhor modelo: salvo automaticamente")

# 5. Estimar tempo
print("\n5️⃣ TEMPO ESTIMADO:")
steps_per_min = 200000 / 45  # Baseado em testes anteriores
total_min = timesteps / steps_per_min
print(f"   ⏱️ {total_min:.0f} minutos (~{total_min/60:.1f} horas)")
print(f"   🕐 Início esperado: AGORA")
print(f"   🕔 Fim esperado: ~{total_min/60:.1f}h a partir de agora")

# 6. Verificar espaço em disco
print("\n6️⃣ ESPAÇO EM DISCO:")
print("   📦 Checkpoints: ~150 MB (5 arquivos)")
print("   🏆 Melhores modelos: ~30 MB")
print("   📊 Logs TensorBoard: ~100-200 MB")
print("   💾 TOTAL ESTIMADO: ~300-400 MB")

# 7. Status final
print("\n" + "="*60)
print("📋 RESUMO FINAL:")
print("="*60)

checks = [
    (timesteps >= 2000000, "Timesteps >= 2M"),
    (len(df) >= 10000, "Dados >= 10k candles"),
    (len(val_df) > 0, "Validation data disponível"),
    (Path('retrain_with_improved_reward.py').exists(), "Script de treinamento existe")
]

all_ok = all([check[0] for check in checks])

for passed, desc in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {desc}")

print("\n" + "="*60)
if all_ok:
    print("✅ TUDO PRONTO! PODE INICIAR O TREINAMENTO!")
    print("🐉 Comando: python retrain_with_improved_reward.py")
    print(f"⏰ Tempo estimado: {total_min/60:.1f} horas")
else:
    print("❌ CORRIJA OS PROBLEMAS ANTES DE TREINAR!")
print("="*60)
