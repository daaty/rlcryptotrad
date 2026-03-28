"""
Ambiente de Trading para Reinforcement Learning usando Gymnasium.
Este é o coração do sistema - onde o agente aprende a operar.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Ambiente de Trading personalizado seguindo a interface do Gymnasium.
    
    Observation Space:
        - Preços OHLCV normalizados
        - Indicadores técnicos
        - Estado da carteira (saldo, posição, PnL)
        
    Action Space:
        0: Flat (Ficar de fora)
        1: Long (Comprar)
        2: Short (Vender)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame = None,
        data_path: str = None,
        config: Dict = None,
        initial_balance: float = 10000,
        commission: float = 0.0004,  # 0.04% - taxa REALISTA da Binance (taker fee)
        slippage: float = 0.0005,  # 0.05% - slippage realista em mercado líquido
        leverage: int = 1.5,  # V6: 1.5x (seguro, quase impossível liquidar)
        position_size: float = 0.05,  # V6: 5% base (action limita max 5%)
        window_size: int = 50,
        max_episode_steps: int = 2000,  # V6: 2000 steps (episódios curtos, mais resets)
        random_start: bool = True,  # NOVA: Começa em pontos aleatórios do dataset
        persist_balance: bool = True,  # NOVA: Balance persiste entre episódios
        use_sharpe_reward: bool = True,  # V6: Usa Sharpe Ratio como reward principal
        use_hybrid_reward: bool = False,  # V6: Só Sharpe + delta
        sentiment_features: np.ndarray = None,
        # FUTUROS BINANCE: Simulação realista
        maintenance_margin_rate: float = 0.005,  # 0.5% margin de manutenção (1.5x leverage)
        liquidation_threshold: float = 0.10,  # Liquida se equity cair 10% (margin call)
        enable_indicator_shaping: bool = True  # V6 CRÍTICO: Ativa reward shaping com 6 técnicas
    ):
        """
        Args:
            df: DataFrame com dados OHLCV e indicadores
            data_path: Caminho alternativo para carregar dados
            config: Dicionário de configuração
            initial_balance: Saldo inicial em USDT
            commission: Taxa de corretagem (0.0004 = 0.04%)
            leverage: Alavancagem máxima
            position_size: Fração do saldo por trade
            window_size: Janela de observação (candles)
            max_episode_steps: Máximo de steps por episódio (truncação para exploration)
            random_start: Se True, cada episódio começa em ponto aleatório do dataset
            persist_balance: Se True, balance persiste entre episódios (continua aprendendo)
            use_sharpe_reward: Se True, usa Sharpe Ratio ao invés de delta equity puro
            sentiment_features: Features de sentimento (opcional)
            maintenance_margin_rate: Taxa de margin de manutenção (0.005 = 0.5% para 3x leverage)
            liquidation_threshold: % de perda de equity que causa liquidação (0.10 = 10%)
            enable_indicator_shaping: Se True, usa EMA/RSI/MACD no reward (RECOMENDADO)
        """
        super().__init__()
        
        # Carrega dados
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif data_path is not None:
            df_loaded = pd.read_csv(data_path)
            # Remove colunas não-numéricas (como timestamp)
            numeric_cols = df_loaded.select_dtypes(include=[np.number]).columns
            self.df = df_loaded[numeric_cols].reset_index(drop=True)
        else:
            raise ValueError("Forneça df ou data_path")
        
        # Carrega config
        if config:
            initial_balance = config.get('initial_balance', initial_balance)
            commission = config.get('commission', commission)
            slippage = config.get('slippage', slippage)
            leverage = config.get('leverage', leverage)
            position_size = config.get('position_size', position_size)
            window_size = config.get('window_size', window_size)
            max_episode_steps = config.get('max_episode_steps', max_episode_steps)
            random_start = config.get('random_start', random_start)
        
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage  # Slippage em fração (0.0005 = 0.05%)
        self.leverage = leverage
        self.position_size = position_size
        self.current_position_size = position_size  # Position size dinâmico (varia com action)
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.persist_balance = persist_balance
        self.use_sharpe_reward = use_sharpe_reward
        self.use_hybrid_reward = use_hybrid_reward
        
        # FUTUROS BINANCE: Parâmetros realistas
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_threshold = liquidation_threshold
        self.enable_indicator_shaping = enable_indicator_shaping
        self.liquidations = 0  # Contador GLOBAL de liquidações (todas)
        self.episode_liquidations = 0  # Contador POR EPISÓDIO
        
        # V8: Sistema de decaimento LENTO com piso mínimo (nunca vai a zero)
        self.total_timesteps_trained = 0  # Rastreia steps totais (setado externamente pelo callback)
        self.last_24h_trades = []  # NOVO: Rastreia trades nas últimas 24h (anti-overtrading)
        
        # Balance persistente entre episódios
        self.persistent_balance = initial_balance
        self.persistent_equity = initial_balance
        
        # Histórico de returns para Sharpe Ratio
        self.returns_history = []
        
        # Features de sentimento (opcional)
        self.sentiment_features = sentiment_features
        self.n_sentiment_features = 0
        if sentiment_features is not None:
            self.n_sentiment_features = sentiment_features.shape[1] if len(sentiment_features.shape) > 1 else 1
        
        # CACHE: DataFrame numérico (evita select_dtypes a cada step!)
        self.df_numeric = self.df.select_dtypes(include=[np.number])
        self.df_values = self.df_numeric.values  # Cache dos valores como numpy array
        
        # Espaço de Ações: Box contínuo [-1, 1] para compatibilidade com SAC/TD3
        # -1 a -0.33: Short | -0.33 a 0.33: Flat | 0.33 a 1: Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Espaço de Observações: [preços, indicadores, carteira, sentimento]
        # Conta apenas colunas numéricas (exclui timestamp/strings)
        self.n_features = len(self.df_numeric.columns)
        # +3 = balance, position, equity (V8 PURO)
        obs_shape = (window_size, self.n_features + 3 + self.n_sentiment_features)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Estado inicial
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reinicia o ambiente para um novo episódio."""
        super().reset(seed=seed)
        
        # Carregar balance persistente se habilitado
        if self.persist_balance:
            # CRÍTICO: Reset forçado se balance caiu muito (evita morte por liquidação)
            if self.persistent_balance < self.initial_balance * 0.6:  # CORRIGIDO: 10%→60%!
                print(f"⚠️ Balance crítico (${self.persistent_balance:.2f}), resetando para ${self.initial_balance}")
                self.persistent_balance = self.initial_balance
                self.persistent_equity = self.initial_balance
            
            self.balance = self.persistent_balance
            self.equity = self.persistent_equity
        else:
            self.balance = self.initial_balance
            self.equity = self.initial_balance
        
        # Estado da conta (não sobrescrever se persist_balance está ativo)
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0
        self.position_value = 0
        
        # Métricas
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.episode_liquidations = 0  # Reseta a cada episódio
        self.last_trade_step = 0  # NOVO: Rastreia quando entrou na posição atual
        # V9: Métricas de direção (Long vs Short)
        self.long_trades = 0  # Contador de trades Long
        self.short_trades = 0  # Contador de trades Short
        # V10: Tracking de flip-flop (mudanças rápidas Long<->Short)
        self._prev_position = 0
        self._last_position_change_step = 0
        # Não reseta liquidations (métrica global entre episódios)
        
        # Equity anterior para reward (delta equity)
        self.previous_equity = self.equity
        
        # Histórico de returns para Sharpe Ratio (limpa a cada episódio)
        self.returns_history = []
        
        # NOVA: Random start position para diversificar exploração
        # Cada episódio vê uma parte diferente do dataset
        if self.random_start:
            # Garante espaço suficiente: [window_size, len(df) - max_episode_steps]
            max_start = max(self.window_size, len(self.df) - self.max_episode_steps - 1)
            self.episode_start = np.random.randint(self.window_size, max_start)
        else:
            self.episode_start = self.window_size
        
        # Índice temporal
        self.current_step = self.episode_start
        self.episode_length = 0  # Contador de steps do episódio atual
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa uma ação no ambiente.
        
        Args:
            action: Array [-1, 1] contínuo:
                - [-1.0, -0.1]: Short com position_size variável (10% a 100%)
                - (-0.1, 0.1): Flat (zona neutra)
                - [0.1, 1.0]: Long com position_size variável (10% a 100%)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Converte action contínuo para posição + intensidade
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # NOVO: Action determina direção E tamanho da posição
        if action_value < -0.1:
            discrete_action = 2  # Short
            # Intensidade: -1.0 = 100%, -0.1 = 10%
            # V6: Limita a MÁXIMO 5% (50% do position_size base de 10%)
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        elif action_value > 0.1:
            discrete_action = 1  # Long
            # Intensidade: +1.0 = 100%, +0.1 = 10%
            # V6: Limita a MÁXIMO 5% (50% do position_size base de 10%)
            self.current_position_size = min(abs(action_value), 0.5) * self.position_size
        else:
            discrete_action = 0  # Flat
            self.current_position_size = 0
        
        # Preço atual (CORRIGIDO: usa .iloc[] ao invés de .loc[])
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calcular PnL da posição anterior
        pnl = self._calculate_pnl(current_price)
        
        # ===== STOP-LOSS AUTOMÁTICO (FORÇADO) - V11 =====
        # NÃO DEIXA O AGENTE ESCOLHER - FECHA AUTOMATICAMENTE!
        # V11: Stop MUITO largo (-8%) para não cortar winners
        stop_loss_triggered = False
        if self.position != 0:
            unrealized_pnl = self._calculate_pnl(current_price)
            unrealized_pct = unrealized_pnl / self.balance  # CORRIGIDO: usa balance ATUAL!
            
            # V12: Stop-loss em -7% (equilibrado)
            if unrealized_pct <= -0.07:
                self._close_position(current_price)
                stop_loss_triggered = True
                discrete_action = 0  # Force Flat após stop
                # V12: Punição simples (7% loss + 2% penalty)
                reward = -0.09  # Perdeu 7% + punição 2%
        
        # Executar ação (V2: retorna reward adicional se fechou posição crítica)
        # Pula se stop-loss já fechou
        if not stop_loss_triggered:
            action_reward = self._execute_action(discrete_action, current_price)
            reward = action_reward  # Inicia com reward de fechar posição crítica (se houver)
        else:
            action_reward = 0.0
            reward = -0.08  # Já definido acima
        
        # Atualizar estado
        self.equity = self.balance + self.position_value
        
        # === PENALIDADE POR HOLDING PROLONGADO (ANTI-"BUY & HOLD") - V8: REDUZIDA ===
        # V8: Penalidade MUITO mais suave (overtrading era o problema real)
        if self.position != 0:
            holding_time = self.episode_length - self.last_trade_step
            
            # ESCALA PROGRESSIVA: quanto mais tempo segura, maior a penalidade
            if holding_time > 300:  # Mais de 300 steps (~75 horas) - AUMENTADO
                reward -= 0.0005 * (holding_time - 300) * 0.0001  # Dor MUITO suave
            elif holding_time > 150:  # 150-300 steps (37-75 horas) - AUMENTADO
                reward -= 0.0002 * (holding_time - 150) * 0.0001  # Aviso mínimo
        
        # === V10: BONIFICAÇÃO POR REALIZAR LUCROS (INCENTIVO ATIVO) ===
        # Se fechou posição com lucro, bonuço EXTRA além do PnL
        # V15: Bônus IGUAIS para lucro e cortar loss (balanceado!)
        if action_reward != 0:  # Significa que fechou posição
            # action_reward já contém PnL % da posição fechada
            if action_reward > 0.02:  # V15: Lucro > 2% (era >3%)
                reward += 0.05  # V15: Balanceado (era 0.08)
            elif action_reward < -0.02:  # V15: Cortou loss < -2% (era <-3%)
                reward += 0.05  # V15: IGUAL ao lucro! Cortar loss é TÃO importante
        
        # === V12: PENALIDADE POR OVERTRADING (ANTI-CHURN) - MODERADA ===
        # OBJETIVO: Reduzir trades mas não paralisar (permitir Long E Short)
        # 24h = 96 candles de 15min
        # V12: Threshold MODERADO - 3 trades/24h (era 1 em V11)
        # V12: Penalty MODERADA -0.03 por extra (era -0.10 em V11)
        if discrete_action != 0 or (self.position == 0 and hasattr(self, '_prev_position') and self._prev_position != 0):
            self.last_24h_trades.append(self.episode_length)
        # Remove trades antigos (>96 steps atrás)
        self.last_24h_trades = [t for t in self.last_24h_trades if self.episode_length - t <= 96]
        
        # V12: Penaliza se >3 trades em 24h (MODERADO - permite explorar)
        if len(self.last_24h_trades) > 3:
            overtrading_penalty = (len(self.last_24h_trades) - 3) * 0.03  # V12: -0.03 (SUAVE!)
            reward -= overtrading_penalty
        
        # === V12: PENALIDADE POR FLIP-FLOP (CHURN RÁPIDO) - SUAVE ===
        # Se entrou Long depois de Short (ou vice-versa) em <50 steps → -0.02
        if hasattr(self, '_prev_position') and hasattr(self, '_last_position_change_step'):
            steps_since_flip = self.episode_length - self._last_position_change_step
            # V12: Detecta flip em <50 steps (era <100 em V11) com penalty SUAVE
            if steps_since_flip < 50:
                if (self._prev_position == 1 and self.position == -1) or \
                   (self._prev_position == -1 and self.position == 1):
                    reward -= 0.02  # V12: -0.02 (era -0.05 em V11)
        
        # Atualiza tracking de flip-flop
        if self.position != getattr(self, '_prev_position', 0):
            self._last_position_change_step = self.episode_length
        
        # === V12: REMOVIDOS BÔNUS POR SAIR DE SHORT/LONG ===
        # MOTIVO: Criavam ASSIMETRIA - modelo achava "minas" em uma direção
        # V12: Deixa modelo aprender naturalmente quando sair via PnL puro
        
        # Salvar posição atual para próximo step
        self._prev_position = self.position
        
        # NOVA: Calcular return do step atual
        step_return = (self.equity - self.previous_equity) / self.previous_equity
        self.returns_history.append(step_return)
        
        # REWARD COM MÚLTIPLAS ESTRATÉGIAS
        if self.use_hybrid_reward and len(self.returns_history) > 1:
            # REWARD HÍBRIDO: Combina delta equity (60%) + sharpe (40%)
            # Mantém Sharpe alto mas incentiva lucro bruto
            returns_array = np.array(self.returns_history[-100:])
            mean_return = returns_array.mean()
            std_return = returns_array.std() + 1e-8
            sharpe = mean_return / std_return
            sharpe_norm = np.tanh(sharpe * 10)  # [-1, 1]
            
            # Delta equity normalizado
            delta_equity = (self.equity - self.previous_equity) / self.initial_balance
            delta_norm = np.tanh(delta_equity * 100)  # [-1, 1]
            
            # Combinar: 60% delta (agressivo) + 40% sharpe (estável)
            reward = 0.6 * delta_norm + 0.4 * sharpe_norm
            
            # REWARD SHAPING PROGRESSIVO: Agressivo no início, equilibra depois
            # Progresso: 0 (início) → 1 (fim do episódio)
            progress = min(self.episode_length / self.max_episode_steps, 1.0)
            bonus = 0.03 * (1 - progress * 0.5)    # 0.03 → 0.015 (bônus alto no início)
            penalty = 0.01 * (1 - progress * 0.3)  # 0.01 → 0.007 (punição baixa sempre)
            
            if step_return > 0.01:
                reward += bonus  # Recompensa MAIOR que punição (incentiva ação)
            elif step_return < -0.01:
                reward -= penalty
                
        elif self.use_sharpe_reward and len(self.returns_history) > 1:
            # SHARPE RATIO: Lucro ajustado por risco
            # Recompensa retornos consistentes, penaliza volatilidade
            returns_array = np.array(self.returns_history[-100:])  # Últimos 100 steps
            mean_return = returns_array.mean()
            std_return = returns_array.std() + 1e-8  # Evita divisão por zero
            sharpe = mean_return / std_return
            
            # Normalizar Sharpe para escala [-1, 1] aproximada
            reward = np.tanh(sharpe * 10)  # tanh comprime para [-1, 1]
            
            # REWARD SHAPING PROGRESSIVO: Agressivo no início, equilibra depois
            progress = min(self.episode_length / self.max_episode_steps, 1.0)
            bonus = 0.03 * (1 - progress * 0.5)    # 0.03 → 0.015
            penalty = 0.01 * (1 - progress * 0.3)  # 0.01 → 0.007
            
            if step_return > 0.01:
                reward += bonus  # Recompensa MAIOR que punição
            elif step_return < -0.01:
                reward -= penalty
        else:
            # Delta equity COM transição suave de penalty (não choca o modelo)
            reward = (self.equity - self.previous_equity) / self.initial_balance
            
            # TRANSIÇÃO SUAVE: Penalty 2x maior (vs anterior 1x), mas não 3x ainda
            progress = min(self.episode_length / self.max_episode_steps, 1.0)
            bonus = 0.015 * (1 - progress * 0.65)   # 0.015 → 0.005 (meio termo)
            penalty = 0.02 * (1 - progress * 0.3)   # 0.02 → 0.014 (2x vs 1x, não 3x)
            
            # Aplica reward shaping
            if step_return > 0.01:
                reward += bonus
            elif step_return < -0.01:
                reward -= penalty
            
            # ===== SISTEMA DE PUNIÇÃO PROGRESSIVA V3 (EQUILIBRADO) =====
            # Pune o modelo ANTES da liquidação, ensinando a sair de posições ruins
            # MELHORIAS V3 (vs V2 que paralisou o modelo):
            # - Punições MODERADAS nos níveis 4-5 (meio termo entre V2 e V3)
            # - Penalidade por tempo REDUZIDA (não paralisa)
            # - Recompensa por SAIR de posições ruins (mantido)
            # - 🆕 RECOMPENSA por HOLDING em LUCRO (deixa winners correrem!)
            if self.position != 0:
                unrealized_pnl = self._calculate_pnl(current_price)
                unrealized_pct = unrealized_pnl / self.initial_balance
                
                # 🎁 RECOMPENSA POR HOLDING EM LUCRO (NOVO V3!)
                # Incentiva o modelo a DEIXAR POSIÇÕES VENCEDORAS CORREREM
                # V15: Bônus AUMENTADO 4x para balancear punições
                if unrealized_pct > 0.02:  # Lucro > 2%
                    reward += 0.02  # V15: 4x maior (era 0.005)
                elif unrealized_pct > 0.05:  # Lucro > 5%
                    reward += 0.04  # V15: 4x maior (era 0.01)
                
                # PUNIÇÕES POR LOSS (apenas se estiver em perda)
                # V15: Começa em -4% (era -3%) - dá mais espaço para respirar
                if unrealized_pct < 0:
                    # NÍVEL 1: ALERTA (perda 4-5%) - V15: adiado de 3% para 4%
                    if unrealized_pct > -0.05 and unrealized_pct <= -0.04:
                        reward -= 0.005  # Punição mínima (aviso gentil)
                    
                    # NÍVEL 2: ATENÇÃO (perda 5-6%) - V15: adiado
                    elif unrealized_pct > -0.06:
                        reward -= 0.02  # Punição leve (atenção!)
                    
                    # NÍVEL 3: PERIGO (perda 6-8%) - V15: adiado
                    elif unrealized_pct > -0.08:
                        reward -= 0.08  # Punição média (PERIGO! Saia!)
                    
                    # NÍVEL 4: CRÍTICO (perda 8-12%) - MODERADO: 0.25 (meio termo)
                    elif unrealized_pct > -0.12:
                        reward -= 0.25  # Punição forte mas não paralisa
                        reward -= 0.005  # Dor por tempo REDUZIDA (não trava)
                    
                    # NÍVEL 5: CATASTRÓFICO (perda 12-15%) - MODERADO: 0.60
                    elif unrealized_pct >= -0.15:
                        reward -= 0.60  # Punição severa mas não extrema
                        reward -= 0.015  # Dor por tempo moderada
                
                # NOTA: Se chegar a -15%, a liquidação acontece e aplica -5.0 (TRAUMA)
            
            # ===== REWARD SHAPING COM INDICADORES TÉCNICOS (O "NORTE" QUE FALTAVA) =====
            if self.enable_indicator_shaping:
                # V8: Decaimento LENTO com PISO MÍNIMO (nunca vai a zero!)
                # 1M→2.5M: 100% → 20% (mantém 20% de guidance sempre)
                shaping_decay = 1.0
                if self.total_timesteps_trained > 1000000:
                    # Linear decay de 1.0 → 0.20 entre 1M e 2.5M steps
                    decay_progress = (self.total_timesteps_trained - 1000000) / 1500000
                    shaping_decay = max(0.20, 1.0 - (0.80 * min(decay_progress, 1.0)))
                
                indicator_reward = self._calculate_indicator_reward(discrete_action, current_price)
                reward += indicator_reward * shaping_decay
                
                # C) PENALIDADE POR INATIVIDADE EM TENDÊNCIAS CLARAS
                # Se está Flat mas mercado tem tendência forte, penaliza levemente
                if discrete_action == 0:  # Flat
                    current_row = self.df.iloc[self.current_step - 1]
                    if 'SMA_50' in current_row and 'RSI_14' in current_row:
                        close_price = current_row['close']
                        sma_50 = current_row['SMA_50']
                        rsi = current_row['RSI_14']
                        
                        # Tendência forte: preço 2%+ acima/abaixo da SMA + RSI não extremo
                        trend_strength = abs(close_price - sma_50) / sma_50
                        is_strong_trend = trend_strength > 0.02 and 30 < rsi < 70
                        
                        if is_strong_trend:
                            reward -= 0.01  # V15: 100x maior (era 0.0001) - força ação!
        
        # Atualizar equity anterior
        self.previous_equity = self.equity
        
        self.current_step += 1
        self.episode_length += 1
        
        # ===== FUTUROS BINANCE: VERIFICAR LIQUIDAÇÃO (MARGIN CALL) =====
        liquidated = self._check_liquidation(current_price)
        if liquidated:
            # PENALIDADE MASSIVA: Liquidação é o pior cenário possível
            # V6: -10.0 (TRAUMA EXTREMO - não deveria acontecer com stop-loss!)
            reward -= 10.0  # -1000% do equity normalizado (TRAUMA MÁXIMO!)
            self.liquidations += 1  # Global
            self.episode_liquidations += 1  # Episódio
            terminated = True
            truncated = False
        else:
            # Verificar se terminou
            # NOVA: Trunca episódio após max_episode_steps (força exploration via resets)
            terminated = self.current_step >= len(self.df) - 1
            truncated = (
                self.equity <= self.initial_balance * 0.5 or  # Stop se perder 50%
                self.episode_length >= self.max_episode_steps  # Trunca após N steps
            )
        
        # Fechar posição aberta ao terminar episódio (para métricas precisas)
        if (terminated or truncated) and self.position != 0:
            final_price = self.df.iloc[self.current_step - 1]['close']
            self._close_position(final_price)
            self.equity = self.balance  # Atualiza equity final
        
        # NOVA: Persistir balance entre episódios se habilitado
        if self.persist_balance:
            self.persistent_balance = self.balance
            self.persistent_equity = self.equity
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Executa a ação de trading SEM reward shaping.
        
        A reward agora é PURAMENTE baseada em delta equity (calculada no step()),
        permitindo que o agente aprenda QUALQUER estratégia viável.
        
        V2: Adiciona RECOMPENSA por SAIR de posições ruins (incentiva cortar loss)
        
        Lógica:
        - action 0: Fecha posição se houver
        - action 1: Abre/mantém Long
        - action 2: Abre/mantém Short
        """
        # Mapeia ação para posição: 0 -> 0, 1 -> 1, 2 -> -1
        target_position = 0 if action == 0 else (1 if action == 1 else -1)
        
        # RECOMPENSA POR SAIR DE POSIÇÃO RUIM (V2) + TOMAR LUCRO (V3) + SAIR ANTES DO STOP (V13)
        # Ensina o modelo que SAIR antes da catástrofe é BOM
        # E que REALIZAR LUCROS também é BOM!
        # 🆕 V13: BÔNUS especial por sair ANTES do stop loss (-7%)
        if self.position != 0 and target_position == 0:
            unrealized_pnl = self._calculate_pnl(current_price)
            unrealized_pct = unrealized_pnl / self.initial_balance
            
            # 🆕 V13: BÔNUS INVERTIDO - Quanto MAIS CEDO sair, MAIOR o bônus!
            # Ensina: "Não deixe a perda crescer, saia LOGO"
            # MAS: Bônus NÃO compensa a perda (só alivia) - ainda aprende a NÃO ENTRAR ERRADO
            if -0.035 <= unrealized_pct < -0.02:
                return 0.03  # Bônus MÁXIMO: saiu com perda pequena (-2% a -3.5%)
            elif -0.05 <= unrealized_pct < -0.035:
                return 0.02  # Bônus médio: perda moderada (-3.5% a -5%)
            elif -0.065 <= unrealized_pct < -0.05:
                return 0.01  # Bônus mínimo: esperou demais (-5% a -6.5%)
            # Abaixo de -6.5% = SEM bônus (deveria ter saído antes!)
            
            # RECOMPENSA POR CORTAR LOSS (mantido de V2)
            elif -0.12 < unrealized_pct <= -0.08:
                return 0.10  # Bonus por cortar loss crítico!
            elif unrealized_pct < -0.12:
                return 0.15  # Bonus REDUZIDO (0.20→0.15) por sair de catástrofe
            
            # 🆕 V13: RECOMPENSA POR REALIZAR LUCRO (INVERTIDO!)
            # Ensina: "Realize lucros CEDO, mercado pode virar!"
            # Bônus MAIOR para lucros MENORES (incentiva realizar cedo)
            elif 0.015 <= unrealized_pct < 0.025:
                return 0.06  # Bônus MÁXIMO: realizou lucro pequeno mas SEGURO (+1.5-2.5%)
            elif 0.025 <= unrealized_pct < 0.04:
                return 0.05  # Bônus médio: lucro moderado (+2.5-4%)
            elif 0.04 <= unrealized_pct < 0.06:
                return 0.03  # Bônus menor: lucro bom MAS arriscado (+4-6%)
            elif unrealized_pct >= 0.06:
                return 0.02  # Bônus mínimo: ganancioso demais (+6%+), pode perder tudo!
        
        # Se a ação mudou
        if target_position != self.position:
            self.trades += 1  # Incrementa contador de trades
            
            # Fecha posição atual se existir
            if self.position != 0:
                self._close_position(current_price)
                
            # Abre nova posição se não for Flat
            if target_position != 0:
                self._open_position(target_position, current_price)
                
                # V9: Rastreia direção dos trades
                if target_position == 1:
                    self.long_trades += 1
                elif target_position == -1:
                    self.short_trades += 1
        
        return 0.0  # Sem reward adicional se não fechou posição crítica
    
    def _open_position(self, position_type: int, price: float):
        """
        Abre uma posição aplicando slippage e fees realistas.
        
        Args:
            position_type: 1 (Long) ou -1 (Short)
            price: Preço de mercado base
        """
        # Aplicar slippage: Long paga mais, Short recebe menos
        if position_type == 1:  # Long
            execution_price = price * (1 + self.slippage)
        else:  # Short
            execution_price = price * (1 - self.slippage)
        
        self.position = position_type
        self.entry_price = execution_price
        
        # Tamanho da posição em USDT (usa position_size DINÂMICO)
        position_usdt = self.balance * self.current_position_size * self.leverage
        
        # Fees: cobrado sobre o valor da posição
        fee = position_usdt * self.commission
        self.balance -= fee  # Desconta fee do saldo
        
        self.position_value = position_usdt * position_type
        
        # NOVO: Marca o momento que entrou na posição
        self.last_trade_step = self.episode_length
        
    def _close_position(self, current_price: float) -> float:
        """
        Fecha a posição atual aplicando slippage e fees realistas.
        Retorna PnL realizado (já descontado balance e fees).
        """
        if self.position == 0:
            return 0
        
        # Aplicar slippage ao fechar: Long recebe menos, Short paga mais
        if self.position == 1:  # Long (vende)
            execution_price = current_price * (1 - self.slippage)
        else:  # Short (compra)
            execution_price = current_price * (1 + self.slippage)
        
        # Calcular PnL com preço de execução ajustado
        pnl = self._calculate_pnl(execution_price)
        
        # Cobrar fee ao fechar
        fee = abs(self.position_value) * self.commission
        pnl -= fee  # Desconta fee do PnL
        
        # Atualiza saldo e métricas
        self.balance += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        # Reseta posição
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
        
        return pnl
    
    def _calculate_pnl(self, current_price: float) -> float:
        """
        Calcula o PnL não realizado da posição atual.
        """
        if self.position == 0 or self.entry_price == 0:
            return 0
        
        # PnL = (preço_atual - preço_entrada) * valor_posição / preço_entrada
        price_change = (current_price - self.entry_price) / self.entry_price
        
        # Long: ganha se subir | Short: ganha se cair
        pnl = price_change * abs(self.position_value) * self.position
        
        return pnl
    
    def _get_observation(self) -> np.ndarray:
        """
        Retorna a observação atual do ambiente.
        Inclui: histórico de preços/indicadores NORMALIZADOS + estado da carteira + sentimento.
        OTIMIZADO: Usa cache de df_values para evitar memory leak.
        """
        # Janela de dados históricos (usa cache!)
        start = self.current_step - self.window_size
        end = self.current_step
        
        # Usa valores cacheados (MUITO mais rápido e sem memory leak)
        historical_data = self.df_values[start:end].copy()
        
        # NORMALIZAÇÃO ROBUSTA: Combina Z-Score + Clipping
        # Evita amplificação de ruído em períodos de baixa volatilidade
        # NOTA: Mean/Std calculados APENAS na janela atual (sem look-ahead bias)
        # Janela de 50 candles = ~12.5h de dados (15min cada)
        mean = historical_data.mean(axis=0, keepdims=True)
        std = historical_data.std(axis=0, keepdims=True)
        
        # CRÍTICO: Usa std mínimo baseado em percentil (mais robusto que 1e-8)
        # Se std < 1% do mean, usa 1% do mean como std mínimo
        # Previne divisão por zero EM PERÍODO DE BAIXA VOLATILIDADE (não é ruído, é real!)
        std_min = np.maximum(np.abs(mean) * 0.01, 1e-8)
        std = np.maximum(std, std_min)
        
        # Z-Score normalização
        historical_data = (historical_data - mean) / std
        
        # CLIPPING: Limita outliers extremos a [-5, +5] sigmas
        # Evita valores absurdos que confundem o modelo
        # NOTA: 5 sigmas = 99.9999% dos dados (raramente ativado)
        historical_data = np.clip(historical_data, -5, 5)
        
        # Estado da carteira (normalizado) - V8 PURO
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Saldo normalizado
            self.position,  # -1, 0 ou 1
            self.equity / self.initial_balance  # Equity normalizado
        ], dtype=np.float32)
        
        # Replica o estado da carteira para cada timestep
        portfolio_matrix = np.tile(portfolio_state, (self.window_size, 1))
        
        # Concatena dados históricos + estado da carteira
        observation = np.concatenate([historical_data, portfolio_matrix], axis=1)
        
        # Adiciona features de sentimento se disponíveis
        if self.sentiment_features is not None and len(self.sentiment_features) > end:
            sentiment_window = self.sentiment_features[start:end]
            
            # Garante que tem a forma correta
            if len(sentiment_window.shape) == 1:
                sentiment_window = sentiment_window.reshape(-1, 1)
            
            # Replica para cada timestep se necessário
            if sentiment_window.shape[0] != self.window_size:
                # Repete primeiro valor conhecido (mais realista que zeros)
                n_missing = self.window_size - sentiment_window.shape[0]
                first_value = sentiment_window[0] if len(sentiment_window) > 0 else np.zeros(sentiment_window.shape[1])
                padding = np.tile(first_value, (n_missing, 1))
                sentiment_window = np.vstack([padding, sentiment_window])
            
            observation = np.concatenate([observation, sentiment_window], axis=1)
        
        return observation.astype(np.float32)
    
    def _check_liquidation(self, current_price: float) -> bool:
        """
        Verifica se a posição deve ser liquidada (margin call).
        Simula comportamento realista de Futuros Binance.
        
        Regras:
        - Maintenance Margin = posição_usdt * maintenance_margin_rate
        - Available Margin = equity - Maintenance Margin
        - Liquidação ocorre se: Available Margin <= 0 OU equity cai > liquidation_threshold
        
        Returns:
            True se liquidado, False caso contrário
        """
        if self.position == 0:
            return False
        
        # Calcula margin ratio atual
        unrealized_pnl = self._calculate_pnl(current_price)
        current_equity = self.balance + unrealized_pnl
        
        # Valor da posição em USDT
        position_usdt = abs(self.position_value)
        
        # Maintenance margin required
        maintenance_margin = position_usdt * self.maintenance_margin_rate
        
        # Available margin (o que sobra após subtrair maintenance)
        available_margin = current_equity - maintenance_margin
        
        # Condição 1: Margin call (available margin negativo)
        if available_margin <= 0:
            self._force_liquidation(current_price)
            return True
        
        # Condição 2: Perda drástica de equity (>10% do balance ATUAL - CORRIGIDO!)
        equity_loss_pct = (self.balance - current_equity) / self.balance if self.balance > 0 else 1.0
        if equity_loss_pct >= self.liquidation_threshold:
            self._force_liquidation(current_price)
            return True
        
        return False
    
    def _force_liquidation(self, current_price: float):
        """
        Executa liquidação forçada da posição.
        Em Futuros Binance, você perde TUDO na liquidação (não apenas unrealized PnL).
        """
        # Liquidação: Perde a posição inteira + fees adicionais
        liquidation_fee = abs(self.position_value) * 0.005  # 0.5% fee de liquidação
        
        # Fecha posição com penalidade adicional
        pnl = self._calculate_pnl(current_price)
        self.balance += pnl - liquidation_fee
        
        # Garante que equity não fica negativo (Binance zera conta)
        if self.balance < 0:
            self.balance = 0
        
        self.equity = self.balance
        self.total_pnl += pnl - liquidation_fee
        
        # Marca como perda (liquidação SEMPRE é perda)
        self.losses += 1
        
        # Reseta posição
        self.position = 0
        self.entry_price = 0
        self.position_value = 0
    
    def _calculate_indicator_reward(self, action: int, current_price: float) -> float:
        """
        V11: REWARD SHAPING MINIMALISTA - SEM COMBOS!
        
        Objetivo: Parar de incentivar "entrar sempre que tem sinal"
        Mantém APENAS:
        1. Punição por entrada aleatória (sem sinal claro)
        
        REMOVIDOS (causavam churn):
        - Combos long/short
        - Bônus por crossover EMA/MACD
        - Bônus por RSI extremo
        - Bônus por Bollinger Bands
        - Bônus por ADX/ATR
        
        Args:
            action: 0 (Flat), 1 (Long), 2 (Short)
            current_price: Preço atual
            
        Returns:
            Reward adicional (APENAS negativo - punir aleatório)
        """
        indicator_reward = 0.0
        
        # Obtém indicadores do candle atual
        current_row = self.df.iloc[self.current_step - 1]
        
        # Verifica se indicadores existem
        if 'RSI_14' not in current_row:
            return 0.0  # Dataset sem indicadores
        
        rsi = current_row.get('RSI_14', 50)
        macd = current_row.get('MACD_12_26_9', 0)
        macd_signal = current_row.get('MACDs_12_26_9', 0)
        adx = current_row.get('ADX_14', 20)
        
        # ===== V11: APENAS PUNIÇÃO POR ENTRADA ALEATÓRIA =====
        # Se ADX baixo E RSI neutro E MACD neutro → entrada sem base
        is_rsi_neutral = 40 < rsi < 60
        is_macd_neutral = abs(macd - macd_signal) < 0.001 if macd != 0 else True
        
        if adx < 20 and is_rsi_neutral and is_macd_neutral:
            if action != 0:
                indicator_reward -= 0.005  # V11: Penalty por "chute" (sem sinal)
        
        return indicator_reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Retorna informações adicionais sobre o estado atual."""
        win_rate = self.wins / self.trades if self.trades > 0 else 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'current_step': self.current_step,
            'liquidations': self.episode_liquidations,  # Episódio atual
            'total_liquidations': self.liquidations  # Histórico global
        }
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas do episódio para logging no TensorBoard.
        Chamado ao final de cada episódio pelo callback customizado.
        
        Métricas incluem:
        - Performance: equity_return, total_pnl, win_rate
        - Atividade: trades, wins, losses
        - Risco: liquidations, max_drawdown
        - Eficiência: sharpe_ratio, profit_factor
        """
        win_rate = self.wins / self.trades if self.trades > 0 else 0
        equity_return = (self.equity - self.initial_balance) / self.initial_balance
        
        # Sharpe Ratio do episódio (se houver dados)
        sharpe_ratio = 0.0
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            mean_return = returns_array.mean()
            std_return = returns_array.std() + 1e-8
            sharpe_ratio = mean_return / std_return * np.sqrt(252 * 96)  # Anualizado (15min candles)
        
        # Profit Factor (total_wins / total_losses)
        total_wins = sum([r for r in self.returns_history if r > 0])
        total_losses = abs(sum([r for r in self.returns_history if r < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # V9: Calcula % de Long vs Short
        long_pct = (self.long_trades / self.trades * 100) if self.trades > 0 else 0
        short_pct = (self.short_trades / self.trades * 100) if self.trades > 0 else 0
        flat_pct = 100 - long_pct - short_pct
        
        return {
            'episode/equity_return': equity_return,
            'episode/total_pnl': self.total_pnl,
            'episode/win_rate': win_rate,
            'episode/trades': float(self.trades),
            'episode/wins': float(self.wins),
            'episode/losses': float(self.losses),
            'episode/liquidations': float(self.episode_liquidations),  # POR EPISÓDIO!
            'episode/total_liquidations': float(self.liquidations),  # GLOBAL (histórico)
            'episode/sharpe_ratio': sharpe_ratio,
            'episode/profit_factor': profit_factor,
            'episode/final_balance': self.balance,
            'episode/final_equity': self.equity,
            # V9: Métricas de direção
            'episode/long_trades': float(self.long_trades),
            'episode/short_trades': float(self.short_trades),
            'episode/long_pct': long_pct,
            'episode/short_pct': short_pct,
            'episode/flat_pct': flat_pct,
        }
    
    def render(self):
        """Renderiza o estado atual (modo texto)."""
        info = self._get_info()
        print(f"\n=== Step {info['current_step']} ===")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Equity: ${info['equity']:.2f}")
        print(f"Position: {['Flat', 'Long', 'Short'][self.position + 1]}")
        print(f"Trades: {info['trades']} | Wins: {info['wins']} | Losses: {info['losses']}")
        print(f"Win Rate: {info['win_rate']:.2%}")
        print(f"Total PnL: ${info['total_pnl']:.2f}")
