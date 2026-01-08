"""
Validação PRÉ-TREINAMENTO: Testa todos os componentes antes de iniciar
"""

import sys
import yaml
import pandas as pd
from pathlib import Path
import torch

def validate_dependencies():
    """Valida todas as dependências necessárias"""
    print("=" * 70)
    print("📦 VALIDANDO DEPENDÊNCIAS")
    print("=" * 70)
    
    errors = []
    
    # 1. Imports críticos
    try:
        from stable_baselines3 import PPO, TD3
        print("✅ stable-baselines3 (PPO, TD3)")
    except Exception as e:
        errors.append(f"❌ stable-baselines3: {e}")
    
    try:
        from src.environment.trading_env import TradingEnv
        print("✅ TradingEnv")
    except Exception as e:
        errors.append(f"❌ TradingEnv: {e}")
    
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"✅ GPU AMD DirectML: {dml_device}")
    except ImportError:
        print("⚠️  torch-directml não encontrado (usará CPU)")
    
    # 2. GPU
    if torch.cuda.is_available():
        print(f"✅ GPU NVIDIA: {torch.cuda.get_device_name(0)}")
    
    return errors


def validate_data_files():
    """Valida arquivos de dados"""
    print("\n" + "=" * 70)
    print("📊 VALIDANDO DADOS")
    print("=" * 70)
    
    errors = []
    required_file = 'data/train_btcusdt_12m_20260105.csv'
    
    # 1. Arquivo existe?
    if not Path(required_file).exists():
        errors.append(f"❌ Arquivo não encontrado: {required_file}")
        return errors
    
    print(f"✅ Arquivo encontrado: {required_file}")
    
    # 2. CSV válido?
    try:
        df = pd.read_csv(required_file)
        print(f"✅ CSV válido: {len(df):,} linhas")
        
        # 3. Colunas necessárias
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(f"❌ Colunas faltando: {missing_cols}")
        else:
            print(f"✅ Colunas OK: {len(df.columns)} features")
        
        # 4. Dados válidos (sem NaN/Inf)
        nan_count = df.isnull().sum().sum()
        inf_count = df.isin([float('inf'), float('-inf')]).sum().sum()
        
        if nan_count > 0:
            errors.append(f"❌ {nan_count} valores NaN encontrados")
        else:
            print("✅ Sem valores NaN")
        
        if inf_count > 0:
            errors.append(f"❌ {inf_count} valores Inf encontrados")
        else:
            print("✅ Sem valores Inf")
        
        # 5. Quantidade suficiente
        if len(df) < 10000:
            errors.append(f"⚠️  Poucos dados: {len(df)} linhas (recomendado > 10k)")
        else:
            print(f"✅ Quantidade suficiente: {len(df):,} linhas")
    
    except Exception as e:
        errors.append(f"❌ Erro ao ler CSV: {e}")
    
    return errors


def validate_config():
    """Valida config.yaml"""
    print("\n" + "=" * 70)
    print("⚙️  VALIDANDO CONFIGURAÇÃO")
    print("=" * 70)
    
    errors = []
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ config.yaml carregado")
        
        # Verificar seções necessárias
        required_sections = ['environment', 'training']
        for section in required_sections:
            if section not in config:
                errors.append(f"❌ Seção '{section}' faltando no config.yaml")
            else:
                print(f"✅ Seção '{section}' presente")
        
        # Verificar parâmetros críticos
        env_config = config.get('environment', {})
        required_params = ['initial_balance', 'commission', 'leverage', 'window_size']
        
        for param in required_params:
            if param not in env_config:
                errors.append(f"❌ Parâmetro 'environment.{param}' faltando")
            else:
                print(f"✅ {param}: {env_config[param]}")
    
    except Exception as e:
        errors.append(f"❌ Erro ao ler config.yaml: {e}")
    
    return errors


def validate_environment():
    """Testa criação do ambiente de trading"""
    print("\n" + "=" * 70)
    print("🏭 VALIDANDO AMBIENTE DE TRADING")
    print("=" * 70)
    
    errors = []
    
    try:
        from src.environment.trading_env import TradingEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        import yaml
        
        # Carregar config e dados
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        df = pd.read_csv('data/train_btcusdt_12m_20260105.csv')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Criar ambiente
        env_config = config['environment']
        env = TradingEnv(
            df=df,
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size']
        )
        
        print("✅ TradingEnv criado")
        
        # Testar reset e step
        obs, info = env.reset()
        print(f"✅ Reset OK: observation shape = {obs.shape}")
        
        action = 0  # Ação neutra
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step OK: reward = {reward:.4f}")
        
        # Vectorizar
        vec_env = DummyVecEnv([lambda: env])
        print("✅ DummyVecEnv OK")
        
    except Exception as e:
        errors.append(f"❌ Erro no ambiente: {e}")
        import traceback
        errors.append(traceback.format_exc())
    
    return errors


def validate_model_creation():
    """Testa criação dos modelos PPO e TD3"""
    print("\n" + "=" * 70)
    print("🤖 VALIDANDO CRIAÇÃO DE MODELOS")
    print("=" * 70)
    
    errors = []
    
    try:
        from stable_baselines3 import PPO, TD3
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.environment.trading_env import TradingEnv
        import yaml
        
        # Preparar ambiente dummy pequeno
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        df = pd.read_csv('data/train_btcusdt_12m_20260105.csv').head(1000)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        env_config = config['environment']
        env = TradingEnv(
            df=df,
            initial_balance=env_config['initial_balance'],
            commission=env_config['commission'],
            slippage=env_config.get('slippage', 0.0005),
            leverage=env_config['leverage'],
            position_size=env_config['position_size'],
            window_size=env_config['window_size']
        )
        vec_env = DummyVecEnv([lambda: env])
        
        # Detectar device
        device = 'cpu'
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"✅ Device: {device} (DirectML)")
        except:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✅ Device: {device} (CUDA)")
            else:
                print("✅ Device: cpu")
        
        # Testar PPO
        try:
            ppo_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                verbose=0,
                device=device
            )
            print("✅ PPO criado com sucesso")
        except Exception as e:
            errors.append(f"❌ Erro ao criar PPO: {e}")
        
        # Testar TD3
        try:
            td3_model = TD3(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=64,
                verbose=0,
                device=device
            )
            print("✅ TD3 criado com sucesso")
        except Exception as e:
            errors.append(f"❌ Erro ao criar TD3: {e}")
        
    except Exception as e:
        errors.append(f"❌ Erro na criação de modelos: {e}")
        import traceback
        errors.append(traceback.format_exc())
    
    return errors


def validate_directories():
    """Valida que diretórios necessários existem/podem ser criados"""
    print("\n" + "=" * 70)
    print("📁 VALIDANDO DIRETÓRIOS")
    print("=" * 70)
    
    errors = []
    
    # Diretórios que serão criados
    dirs_to_check = [
        'models',
        'logs',
        'data'
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Diretório criado: {dir_path}")
            except Exception as e:
                errors.append(f"❌ Não pode criar diretório {dir_path}: {e}")
        else:
            print(f"✅ Diretório existe: {dir_path}")
    
    return errors


def main():
    """Executa todas as validações"""
    print("\n" + "="*70)
    print("🔍 VALIDAÇÃO PRÉ-TREINAMENTO")
    print("="*70)
    print("Validando todos os componentes antes do treinamento overnight...\n")
    
    all_errors = []
    
    # Executar validações
    all_errors.extend(validate_dependencies())
    all_errors.extend(validate_data_files())
    all_errors.extend(validate_config())
    all_errors.extend(validate_environment())
    all_errors.extend(validate_model_creation())
    all_errors.extend(validate_directories())
    
    # Resultado final
    print("\n" + "="*70)
    if all_errors:
        print("❌ VALIDAÇÃO FALHOU")
        print("="*70)
        print("\n🚨 ERROS ENCONTRADOS:")
        for error in all_errors:
            print(f"  {error}")
        print("\n⚠️  CORRIJA OS ERROS ANTES DE TREINAR!")
        sys.exit(1)
    else:
        print("✅ VALIDAÇÃO COMPLETA - TUDO OK!")
        print("="*70)
        print("\n🚀 Sistema pronto para treinamento overnight!")
        print("\nComando para iniciar:")
        print("  python train_multi_symbol.py base")
        print("\nDuração estimada: 3-5 horas com GPU AMD")
        print("Modelos que serão criados:")
        print("  - models/ppo_base_btcusdt_final.zip")
        print("  - models/td3_base_btcusdt_final.zip")
        sys.exit(0)


if __name__ == "__main__":
    main()
