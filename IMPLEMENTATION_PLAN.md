# Texas Hold'em AI 학습 프로젝트 - 구현 계획

## 🎯 프로젝트 목표

**최종 목표**: 커스텀 POKERENGINE을 사용하여 헤즈업 노리밋 텍사스 홀덤 AI를 강화학습(Reinforcement Learning)으로 훈련

**핵심 목표**:
- GTO(Game Theory Optimal) 전략에 근접하는 AI 개발
- Self-play를 통한 자가 학습
- 다양한 상황에서 적응 가능한 전략 학습
- 인간 플레이어와 대결 가능한 수준

**범위**:
- 2인 헤즈업 노리밋 홀덤에 집중
- 멀티플레이어는 1단계 완료 후 고려

---

## 🛠️ 기술 스택

### 핵심 프레임워크
- **Python 3.10+** - 개발 언어
- **Ray/RLlib 2.x** - 강화학습 프레임워크
- **PyTorch** - 신경망 백엔드
- **Gymnasium** - 환경 인터페이스
- **POKERENGINE** - 커스텀 포커 엔진 (TDA 규칙 준수)

### 모니터링 & 분석
- **TensorBoard** - 학습 메트릭 시각화
- **MLflow** - 실험 관리 (선택)

### 개발 도구
- **Git** - 버전 관리
- **pytest** - 테스트

---

## 🎮 Gymnasium 환경 인터페이스 설계

### 기술 스택
- **Gymnasium 0.29+** - 표준 RL 환경 인터페이스
- **NumPy 1.24+** - 효율적인 배열 연산 및 데이터 처리
- **POKERENGINE** - 커스텀 TDA 준수 게임 로직

### 환경 클래스 구조

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import numpy as np
from poker_engine import PokerGame, Action, ActionType

class PokerMultiAgentEnv(MultiAgentEnv):
    """
    2-player Heads-up No-Limit Texas Hold'em Multi-Agent Environment
    
    핵심 설계:
    - MultiAgentEnv 사용 (gym.Env 아님!)
    - Self-Play와 분산 학습 지원
    - 각 플레이어가 독립적인 정책 사용 가능
    
    Compatible with:
    - RLlib multi-agent training
    - Self-play (Phase 3)
    - League training (Phase 2)
    - NumPy-based observations
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 1}
    
    def __init__(self, config=None):
        super().__init__()
        
        config = config or {}
        
        # ⭐ BB 고정 (절대값은 무의미, BB 단위만 중요)
        self.small_blind = 50.0
        self.big_blind = 100.0
        
        # ⭐ 스택 깊이 분포 (BB 단위)
        # 중요도 기반 샘플링
        self.stack_distribution = {
            'standard': (80, 120, 0.40),   # 100BB ±20, 40% 확률
            'middle': (20, 50, 0.30),      # 20-50BB, 30% 확률
            'short': (5, 20, 0.20),        # 5-20BB, 20% 확률
            'deep': (150, 250, 0.10)       # 150-250BB, 10% 확률
        }
        
        # Observation: NumPy array (338 floats)
        # 119: 카드 one-hot
        # 20: 게임 상태 + Legal Actions Mask
        # 20: 스트리트별 요약 (Context) ← NEW!
        # 179: 스트리트별 액션 히스토리 (4×4×11)
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.5,
            shape=(338,),
            dtype=np.float32
        )
        
        # Action: Discrete(8)
        self.action_space = spaces.Discrete(8)
        
        # Multi-agent 설정
        self._agent_ids = {"player_0", "player_1"}
    
    def _sample_stack_depth(self) -> float:
        """중요도 기반 스택 깊이 샘플링"""
        categories = ['standard', 'middle', 'short', 'deep']
        probs = [0.40, 0.30, 0.20, 0.10]
        
        # 카테고리 선택
        category = np.random.choice(categories, p=probs)
        min_bb, max_bb, _ = self.stack_distribution[category]
        
        # 범위 내 랜덤
        stack_bb = np.random.uniform(min_bb, max_bb)
        
        # 칩 수로 변환
        return stack_bb * self.big_blind
    
    def reset(self, *, seed=None, options=None):
        """
        매 핸드마다 호출!
        스택과 버튼을 랜덤화하여 다양한 상황 학습
        """
        if seed is not None:
            np.random.seed(seed)
        
        # ⭐ 스택 랜덤 샘플링 (중요도 기반)
        self.chips = [
            self._sample_stack_depth(),
            self._sample_stack_depth()
        ]
        
        # ⭐ 버튼 위치 랜덤화 (50:50)
        self.button = np.random.randint(0, 2)
        
        # 핸드 시작
        self.game = PokerGame(
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        self.game.start_hand(
            players_info=[(0, self.chips[0]), (1, self.chips[1])],
            button=self.button  # ⭐ 랜덤 버튼!
        )
        
        # 액션 히스토리 초기화
        self.action_history = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }
        
        # 현재 플레이어 관찰 반환
        current_player = self.game.get_current_player()
        obs_dict = {f"player_{current_player}": self._get_observation(current_player)}
        info_dict = {
            f"player_{current_player}": {
                'legal_actions': self._get_legal_actions_mask(current_player),
                'button': self.button,
                'stacks': self.chips
            }
        }
        
        return obs_dict, info_dict
```

---

### 1. 관찰 공간 (Observation Space)

**타입**: `spaces.Box(shape=(150,), dtype=np.float32)`

**NumPy 배열 구조** (150 floats):

#### 카드 One-Hot 인코딩 (0-118)
**비공개 카드 표현**: All-zero vector (One-hot의 자연스러운 "없음" 표현)
- Preflop 액션이 핸드 범위 결정

**River 단계 전체 관찰 예시**:
```
River 단계에서:
- obs[150:190]: Preflop 액션 + 누가 + 정확한 베팅 비율 ✅
- obs[190:230]: Flop 액션 + 누가 + 정확한 베팅 비율 ✅
- obs[230:270]: Turn 액션 + 누가 + 정확한 베팅 비율 ✅
- obs[270:310]: River 현재 액션 + 누가 + 정확한 베팅 비율

→ AI가 "스토리", "공격/방어", "Bet Sizing Tell"을 모두 이해!
```

#### 확장 영역



**구현 예시** (NumPy One-Hot):
```python
def _encode_card_onehot(self, card) -> np.ndarray:
    """
    카드를 17차원 one-hot 벡터로 인코딩
    
    [Rank: 13차원] + [Suit: 4차원] = 17차원
    
    예: ♠3 = [0,0,1,0,0,0,0,0,0,0,0,0,0, 1,0,0,0]
             └─────── Rank 3 ────────┘ └─ Spade ─┘
    """
    encoding = np.zeros(17, dtype=np.float32)
    
    # Rank one-hot (0-12): 2=0, 3=1, ..., A=12
    rank_idx = Card.RANKS.index(card.rank)
    encoding[rank_idx] = 1.0
    
    # Suit one-hot (13-16): S=13, H=14, D=15, C=16
    suit_idx = Card.SUITS.index(card.suit)
    encoding[13 + suit_idx] = 1.0
    
    return encoding

def _get_observation(self, player_id: int) -> np.ndarray:
    obs = np.zeros(150, dtype=np.float32)
    
    # === 카드 One-Hot 인코딩 (0-118) ===
    
    # 1. 내 홀 카드 (0-33)
    for i, card in enumerate(self.game.players[player_id].hand[:2]):
        obs[i*17:(i+1)*17] = self._encode_card_onehot(card)
    
    # 2. 커뮤니티 카드 (34-118)
    for i, card in enumerate(self.game.community_cards):
        obs[34+i*17:34+(i+1)*17] = self._encode_card_onehot(card)
    
    # 없는 카드는 all-zero (one-hot의 자연스러운 표현)
    # 별도 처리 불필요!
    
    # === 게임 상태 (119-149): 이중 정규화 ===
    player = self.game.players[player_id]
    opponent = self.game.players[1 - player_id]
    pot = self.game.get_pot_size()
    to_call = self.game.current_bet - player.bet_this_round
    
    bb = self.big_blind
    max_bb = self.starting_chips / bb  # 500BB
    
    obs[119:150] = [  # 게임 상태는 119-149 (31차원)
        (player.chips / bb) / max_bb,
        (opponent.chips / bb) / max_bb,
        (pot / bb) / max_bb,
        (self.game.current_bet / bb) / max_bb,
        (player.bet_this_round / bb) / max_bb,
        (to_call / bb) / max_bb,
        1.0 if self.game.button_position == player_id else 0.0,
        {'preflop': 0.0, 'flop': 0.33, 'turn': 0.66, 'river': 1.0}[self.game.street.value],
        to_call / (pot + to_call) if to_call > 0 and pot > 0 else 0.0,
        np.clip((player.chips / pot) / 10.0, 0, 1.0) if pot > 0 else 1.0,
        self.hand_count / self.max_hands,
        len(self.game.community_cards) / 5.0,
        (self.game.min_raise / bb) / max_bb,
        (opponent.bet_this_round / bb) / max_bb,
        (opponent.bet_this_hand / bb) / max_bb,
        bb / (self.starting_chips / bb)
    ]
    
    # === 액션 히스토리 (150-309): 스트리트별 보존 + 베팅 비율 ===
    
    # 각 스트리트별 액션 히스토리 (최근 4개)
    street_actions = {
        'preflop': self.action_history.get('preflop', []),
        'flop': self.action_history.get('flop', []),
        'turn': self.action_history.get('turn', []),
        'river': self.action_history.get('river', [])
    }
    
    offset = 150
    for street in ['preflop', 'flop', 'turn', 'river']:
        actions = street_actions[street][-4:]  # 최근 4개
        
        # 각 액션을 10차원으로 인코딩
        for i in range(4):
            if i < len(actions):
                action_idx, player_id, bet_ratio = actions[i]
                # Action one-hot (7차원)
                obs[offset + i*10 : offset + i*10 + 7] = np.eye(7)[action_idx]
                # Player one-hot (2차원): [나, 상대]
                obs[offset + i*10 + 7 : offset + i*10 + 9] = np.eye(2)[player_id]
                # Bet ratio (1차원): 팟 대비 베팅 비율
                obs[offset + i*10 + 9] = bet_ratio
            # else: all-zero (액션 없음)
        
        offset += 40  # 다음 스트리트로 (4 actions × 10 dims)
    
    return obs
```

**액션 히스토리 추적 (업데이트)**:
```python
def _record_action(self, action_idx: int, player_id: int, bet_amount: float, pot_before: float):
    """액션을 현재 스트리트 히스토리에 기록"""
    current_street = self.game.street.value
    
    # 베팅 비율 계산 (팟 대비)
    if pot_before > 0:
        bet_ratio = bet_amount / pot_before
    else:
        bet_ratio = 0.0
    
    # Clip to reasonable range
    bet_ratio = np.clip(bet_ratio, 0.0, 2.5)
    
    self.action_history[current_street].append((action_idx, player_id, bet_ratio))

def _map_bet_to_bucket(self, bet_ratio: float) -> int:
    """
    베팅 비율을 가장 가까운 액션 버킷으로 매핑 (Nearest Neighbor)
    
    Args:
        bet_ratio: 팟 대비 베팅 비율 (0.0 ~ 2.5)
    
    Returns:
        action_idx: 2-5 (Bet33%, Bet75%, Bet100%, Bet150%)
    
    매핑 규칙:
        - 0.33 근처 → 2 (Bet33%)
        - 0.75 근처 → 3 (Bet75%)
        - 1.00 근처 → 4 (Bet100%)
        - 1.50 근처 → 5 (Bet150%)
        - 2.0+ → 6 (All-in)
    """
    # 벳 버킷 중심값
    buckets = [0.33, 0.75, 1.0, 1.5]
    
    # All-in 특수 케이스
    if bet_ratio >= 2.0:
        return 6
    
    # Euclidean distance로 가장 가까운 버킷 찾기
    distances = [abs(bet_ratio - bucket) for bucket in buckets]
    nearest_idx = np.argmin(distances)
    
    return nearest_idx + 2  # 2-5 (Bet33%, Bet75%, Bet100%, Bet150%)

# 예시:
# bet_ratio = 0.45
# distances = [|0.45-0.33|=0.12, |0.45-0.75|=0.30, |0.45-1.0|=0.55, |0.45-1.5|=1.05]
# nearest_idx = 0 (0.12가 최소)
# return 0 + 2 = 2 (Bet33%)
```

**사용 예시**:
```python
# 액션 히스토리 기록 시
pot_before = self.game.get_pot_size()
bet_amount = 22.5  # 예: 팟이 50, 22.5 베팅
bet_ratio = bet_amount / pot_before  # 0.45

# One-hot 카테고리 결정
action_bucket = self._map_bet_to_bucket(bet_ratio)  # 2 (Bet33%)

# 히스토리 저장: (카테고리, 플레이어, 정확한 비율)
self._record_action(action_bucket, player_id, bet_ratio, pot_before)

**액션 히스토리 추적 (별도 구현 필요)**:
```python
def __init__(self, ...):
    # ...
    self.action_history = {
        'preflop': [],
        'flop': [],
        'turn': [],
        'river': []
    }

def _record_action(self, action_idx: int, player_id: int):
    """액션을 현재 스트리트 히스토리에 기록"""
    current_street = self.game.street.value
    self.action_history[current_street].append((action_idx, player_id))

def _start_new_hand(self):
    # ...
    # 핸드 시작 시 히스토리 초기화
    self.action_history = {
        'preflop': [],
        'flop': [],
        'turn': [],
        'river': []
    }
```

---

### 2. 액션 공간 (Action Space)

**타입**: `spaces.Discrete(7)`

**액션 인덱스 → POKERENGINE 매핑**:

```python
0: Fold       → Action.fold()
1: Check/Call → Action.check() or Action.call(to_call)
2: Bet 33%    → Action.bet/raise_to(pot * 0.33)
3: Bet 75%    → Action.bet/raise_to(pot * 0.75)
4: Bet 100%   → Action.bet/raise_to(pot * 1.0)  # Pot bet
5: Bet 150%   → Action.bet/raise_to(pot * 1.5)  # Overbet
6: All-in     → Action.all_in(chips)
```

**매핑 로직** (NumPy로 계산):
```python
def _map_action(self, action_idx: int, player_id: int) -> Action:
    player = self.game.players[player_id]
    pot = self.game.get_pot_size()
    to_call = self.game.current_bet - player.bet_this_round
    
    if action_idx == 0:
        return Action.fold()
    elif action_idx == 1:
        return Action.check() if to_call == 0 else Action.call(to_call)
    elif action_idx == 6:
        return Action.all_in(player.chips)
    else:
        # Percentage bets (2-5)
        pcts = np.array([0.33, 0.75, 1.0, 1.5])
        pct = pcts[action_idx - 2]
        bet_amount = pot * pct
        
        if self.game.current_bet > 0:
            # Raise
            target = max(
                self.game.current_bet + bet_amount,
                self.game.current_bet + self.game.min_raise
            )
            max_bet = player.chips + player.bet_this_round
            return Action.all_in(player.chips) if target > max_bet else Action.raise_to(target)
        else:
            # Bet
            bet_amount = max(bet_amount, self.big_blind)
            return Action.all_in(player.chips) if bet_amount > player.chips else Action.bet(bet_amount)
```

---

### 3. 보상 함수 (Reward Function)

**타입**: Dense Reward (매 핸드마다)

**공식** (NumPy 클리핑):
```python
def _calculate_reward(self, player_id: int, stack_before: float, stack_after: float) -> float:
    chip_change = stack_after - stack_before
    bb_change = chip_change / self.big_blind
    reward = bb_change / 100.0  # Normalization factor
    return float(np.clip(reward, -5.0, 5.0))
```

**특징**:
- 칩 EV 최대화 = 최적 포커 전략
- 범위: -1.0 ~ +1.0 (일반적)
- 클리핑으로 극단값 방지
    legal = self.game.get_legal_actions(self.game.get_current_player())
    mask = np.zeros(7, dtype=np.int8)
    
    if ActionType.FOLD in legal: mask[0] = 1
    if ActionType.CHECK in legal or ActionType.CALL in legal: mask[1] = 1
    if ActionType.BET in legal or ActionType.RAISE in legal: mask[2:6] = 1
    if ActionType.ALL_IN in legal: mask[6] = 1
    
    return mask
```

---

### 5. 에피소드 구조

**Tournament 방식**:
- 에피소드 = 한 토너먼트 (한 명의 칩이 0이 될 때까지)
- 시작 칩: 1000 (블라인드 1/2 기준 500BB)
- 최대 핸드: 500 (무한 루프 방지)
- 딜러 로테이션: 매 핸드 교대

**핸드 구조**:
- 딜러 로테이션 후 `game.start_hand(chips, button)` 호출
- 핸드 종료 시 베팅 리셋, 칩 누적
- POKERENGINE이 자동으로 스트리트 진행

---

### 6. NumPy 최적화 팁

```python
# ✅ Good: Pre-allocate
obs = np.zeros(60, dtype=np.float32)

# ✅ Good: Vectorized slicing
obs[14:30] = state_features

# ✅ Good: NumPy clip
reward = np.clip(raw_reward, -5.0, 5.0)

# ❌ Bad: List append + convert
obs = []
obs.append(...)
obs = np.array(obs)  # Slow!
```

---


---

## 🧠 모델 아키텍처


### Phase 1: FC + LSTM (시작) - 권장!

> [!WARNING] **Transformer 전환 신중론**
> Transformer는 강력하지만 초기 단계에서는 다음과 같은 이유로 **비권장**됩니다:
> 1. **데이터 효율성**: MLP/LSTM 대비 5~10배 많은 데이터 필요
> 2. **추론 속도**: 시뮬레이션(Rollout) 속도 저하 → 전체 학습 속도 감소
> 3. **짧은 컨텍스트**: 포커 히스토리(10~20)는 LSTM으로 충분히 커버 가능
> 
> **결론**: 초기에는 **FC + LSTM**에 집중하고, Transformer는 Phase 3 이후 실험적으로 고려하십시오.

**⚠️ 중요**: 순수 MLP는 시퀀스 이해 불가

**MLP의 한계**:
```python
# MLP가 보는 방식:
obs = [action1, action2, ..., action_n]
→ "310개의 독립 변수" (순서 무시!)

# MLP가 못 보는 것:
"Preflop: 조심스럽게 call → Flop: 빠르게 raise"
→ 시간적 패턴 포착 불가
```

**해결책: FC (Feature Extraction) + LSTM (Temporal)**

**✅ 올바른 구조**:
```
입력 (310) - One-hot 카드 + 게임 상태 + 액션 히스토리
  → FC(256) + ReLU  [특징 추출: "이 패는 강하다"]
  → FC(256) + ReLU  [특징 압축: 310 → 256]
  → LSTM(256)       [시퀀스 이해: "패턴 파악"]
  → FC(7)           [액션 확률]
```

**왜 FC가 먼저?**
```python
# ❌ 잘못된 순서
Input(310) → LSTM(256) → FC
문제:
- 310차원 sparse input을 LSTM 직접 = 느림
- 연산량 폭발
- 수렴 어려움

# ✅ 올바른 순서
Input(310) → FC(256) → LSTM(256)
장점:
- FC가 특징 추출: One-hot → Abstract features
- LSTM은 압축된 특징의 시간적 흐름만 처리
- 효율적이고 빠름
```

**RLlib 구현** (자동 FC 추가!):
            "fcnet_hiddens": [256, 256],  # ⭐ FC(256) → FC(256)
            "fcnet_activation": "relu",
            
            # 2. LSTM 설정
            "use_lstm": True,              # ⭐ LSTM 활성화
            "lstm_cell_size": 256,         # Hidden state 크기
            
            # 3. 시퀀스 길이
            #    포커 한 핸드 = 보통 10~30 액션
            #    20이면 충분하고 효율적
            "max_seq_len": 20,
            
            # 4. 이전 정보 활용
            "lstm_use_prev_action": True,   # 이전 액션 입력에 추가
            "lstm_use_prev_reward": True    # 이전 보상도 추가 (Sparse지만 유용)
        }
    )
)
```

**실제 구조 (RLlib 자동 생성)**:
```
Input(310)
  ↓
FC(310 → 256) + ReLU  [fcnet_hiddens[0]]
  ↓
FC(256 → 256) + ReLU  [fcnet_hiddens[1]]
  ↓
LSTM(256)  [use_lstm=True]
  ↓  
Policy Head: FC(256 → 7)   [액션 확률]
Value Head:  FC(256 → 1)   [상태 가치]
```

**💡 내부 작동 원리** (특징 추출 과정)

**데이터 변환 단계**:
```
Step 1: Raw Input (310차원)
[0,1,0,0,0,...,0.75,0.45,...]
↓ "이해하기 힘든 0과 1의 나열"

Step 2: FC Layer 1 (256차원)
[0.23, 0.87, 0.12, ...]
↓ "1차 패턴 인식"
AI 내부: "A와 K가 있네? 스페이드가 3장?"

Step 3: FC Layer 2 (256차원)  
[0.91, 0.15, 0.68, ...]
↓ "추상적 특징 완성"
AI 내부: 
- 강도:0.91, 위험:0.15, 팟오즈:0.68
→ **310개 → 256개 '상황 요약 벡터'**

Step 4: LSTM
시간축 처리 + 메모리
AI 내부:
"Preflop 강했는데 → Flop 위험 증가 → Turn 더 위험"
→ "상황 악화 중!"
```

**구체적 예시**:
```python
# Raw Input (310차원)
[
  0,0,1,0,...,0,  # ♠3 (one-hot)
  1,0,0,0,...,0,  # ♠2
  0.95,           # My stack
  0.25,           # Pot
  ...
  0.75,           # Bet ratio
]
↓
# FC1 학습 패턴:
"스페이드 여러 개 → Flush 가능성"
"Bet 0.75 반복 → 공격적 스타일"
↓
# FC2 추상화:
feature[0] = 0.91  # "내 핸드 강도"
feature[1] = 0.15  # "보드 위험도"  
feature[2] = 0.33  # "상대 공격성"
feature[3] = 0.68  # "팟 오즈"
↓
# LSTM 시퀀스:
t-3: [강:0.9, 위:0.2] "안전"
t-2: [강:0.9, 위:0.5] "조금 위험"
t-1: [강:0.7, 위:0.8] "위험"
now: [강:0.6, 위:0.9] "매우 위험!"
→ "폴드 고려"
```

**왜 FC 2개?**
- **FC1**: 단순 패턴 ("이 위치=Ace")
- **FC2**: 복합 패턴 ("AK+Flush+상대약함=공격")
  → 추상적 *포커 개념* 형성


**장점**:
- ✅ **효율적**: FC가 차원 축소 (310 → 256)
- ✅ **특징 추출**: One-hot → 추상적 특징
- ✅ **시퀀스 이해**: LSTM이 temporal pattern 학습
- ✅ **가벼움**: Transformer보다 훨씬 작음
- ✅ **RLlib 자동화**: `fcnet_hiddens`로 자동 구성
- ✅ **포커 최적**: 액션 시퀀스 + 특징 조합

**파라미터 수**: ~600K
- FC layers: ~300K
- LSTM: ~250K
- Heads: ~50K

**시퀀스 처리 예시**:
```python
# FC가 추출한 특징:
feature_t1 = [강한패: 0.9, 위험보드: 0.3, ...]
feature_t2 = [강한패: 0.9, 위험보드: 0.7, ...]
feature_t3 = [강한패: 0.6, 위험보드: 0.9, ...]

# LSTM이 이해:
"처음엔 강했지만 점점 약해짐" → Bluff 가능성 감지
```

---

### 🎮 Inference 시 State 관리 (중요!)

**MLP vs LSTM 차이**:

```python
# MLP (Stateless):
action = algo.compute_single_action(obs)
# 간단! 상태 관리 불필요

# LSTM (Stateful):
# 핸드 시작 시 초기화
state = algo.get_initial_state()  
# 또는
state = [np.zeros([256], np.float32), np.zeros([256], np.float32)]

# 매 턴마다
action, state, _ = algo.compute_single_action(
    obs, 
    state=state  # ⭐ 이전 state 전달!
)
# state를 업데이트하고 다음 턴에 재사용

# 핸드 종료 시 state 리셋
```

**실전 예시** (`play_vs_ai.py`):
```python
# 핸드 시작
lstm_state_0 = algo.get_initial_state()  # P0 상태
lstm_state_1 = algo.get_initial_state()  # P1 상태

while not hand_over:
    current_player = game.get_current_player()
    obs = env._get_observation(current_player)
    
    if current_player == 0:
        action, lstm_state_0, _ = algo.compute_single_action(
            obs, state=lstm_state_0
        )
    else:
        action, lstm_state_1, _ = algo.compute_single_action(
            obs, state=lstm_state_1
        )
    
    game.process_action(current_player, action)

# 다음 핸드: state 리셋!
```

**주의사항**:
- ✅ 핸드마다 state 초기화
- ✅ 플레이어별로 state 분리
- ✅ state는 LSTM hidden state (2개 tensor)


### 🛡️ Action Masking (필수!)

**중요**: Action Masking은 **선택이 아닌 필수**

**문제 - 불법 액션 강제 변환 방식**:
```python
# ❌ 나쁜 방법 (현재 계획)
if not legal:
    action = check_or_call  # 강제 변환

문제점:
1. 신경망이 계속 불법 액션 학습
2. 확률 분포 왜곡 (Fold 30% → Check로 변환)
3. 학습 비효율
4. 수렴 느림
```

**해결 - Action Masking**:
```python
# ✅ 올바른 방법
legal_mask = [1, 1, 0, 1, 1, 0, 1]  # 0 = 불법
logits[~legal_mask] = -inf
probs = softmax(logits)

장점:
1. 불법 액션 확률 = 0
2. 합법 액션만 학습
3. 안정적 학습
4. 빠른 수렴
```

**RLlib 구현 - ParametricActionModel**:

환경에서 마스크 제공:
```python
# env.py
def step(self, action):
    obs = self._get_observation(...)
    info = {
        'action_mask': self._get_legal_actions_mask()  # 필수!
    }
    return obs, reward, terminated, truncated, info

def _get_legal_actions_mask(self) -> np.ndarray:
    """7차원 binary mask"""
    legal = self.game.get_legal_actions(self.game.get_current_player())
    mask = np.zeros(7, dtype=np.int8)
    
    if ActionType.FOLD in legal: mask[0] = 1
    if ActionType.CHECK in legal or ActionType.CALL in legal: mask[1] = 1
    if ActionType.BET in legal or ActionType.RAISE in legal: mask[2:6] = 1
    if ActionType.ALL_IN in legal: mask[6] = 1
    
    return mask
```

커스텀 모델 (RLlib ParametricActionModel):
```python
# models/masked_mlp.py
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch.nn as nn

class MaskedMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.fc1 = nn.Linear(310, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.logits = nn.Linear(256, 7)
        self.value = nn.Linear(256, 1)
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        action_mask = input_dict["obs"]["action_mask"]  # 마스크 추출
        
        # Forward pass
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        logits = self.logits(x)
        
        # ⭐ Action Masking 적용
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MIN)
        masked_logits = logits + inf_mask
        
        self._value = self.value(x).squeeze(1)
        return masked_logits, state
    
    def value_function(self):
        return self._value
```

RLlib 설정:
```python
# train.py
from ray.rllib.algorithms.ppo import PPOConfig
from models.masked_mlp import MaskedMLP

config = (
    PPOConfig()
    .training(
        model={
            "custom_model": MaskedMLP,  # 커스텀 모델 사용
        }
    )
)
```

**⚠️ 중요: step()에서 절대 금지!**

```python
# ❌ 절대 하지 말 것! (사후 처리)
def step(self, action_dict):
    success, error = self.game.process_action(player, action)
    
    if not success:
        # ❌ 강제 변환 금지!
        action = Action.check()  
        self.game.process_action(player, action)
        
    # 이 방식은:
    # 1. 신경망이 학습 못 함
    # 2. 확률 분포 왜곡
    # 3. 비효율적
```

**✅ 올바른 step() 구현**:

```python
def step(self, action_dict):
    current_player = self.game.get_current_player()
    action = action_dict[f"player_{current_player}"]
    
    # ⭐ Action Masking으로 이미 합법 액션만 옴
    # 그냥 실행!
    engine_action = self._map_action(action, current_player)
    self.game.process_action(current_player, engine_action)
    
    # 사후 처리 없음!
    # 마스킹이 제대로 되었다면 항상 성공!
    
    # (디버깅용으로만 체크)
    # assert success, "Action masking failed!"
```

**핵심**:
- ✅ 환경: `action_mask`를 info에 제공
- ✅ 모델: Logit 레벨에서 마스킹 적용
- ✅ step(): 받은 액션을 그대로 실행
- ❌ **절대**: 사후 처리 금지!


**참고 자료**:
- RLlib Parametric Actions: https://docs.ray.io/en/latest/rllib/rllib-models.html#parametric-action-space
- Action Masking 예제: https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py


---

### 💰 보상 함수 (Sparse Reward) - 필수 주의!

**⚠️ 용어 정정**: **Sparse Reward** (Dense 아님!)

**중요한 구분**:
- ❌ Dense Reward = 매 스텝마다 보상 (우리는 X)
- ✅ Sparse Reward = 핸드 종료 시에만 보상 (우리!)

**절대 원칙**:
```python
# ✅ 올바른 구현
핸드 진행 중 (betting, calling, raising):
    reward = 0.0  # 절대 보상 없음!
    
핸드 종료 시 (showdown or all folded):
    reward = (stack_after - stack_before) / BB / 100.0
```

**위험한 착각 - 절대 금지!**:
```python
# ❌ 절대 하면 안 되는 것들
if action == BET:
    reward = -bet_amount  # 베팅 = 즉시 손실?
    → AI가 베팅 회피 학습! (체크만 함)

if action == FOLD:
    reward = -(chips_invested)  # 폴드 = 손실 확정?
    → AI가 무조건 폴드 학습!
    
if action == CALL:
    reward = -to_call  # 콜 = 돈 나감?
    → AI가 폴드만 함!

# 핸드 중간에는 무조건 reward = 0.0!
```

**올바른 구현**:
```python
def _calculate_reward(self, player_id: int, stack_before: float, stack_after: float) -> float:
    """
    핸드 종료 시에만 호출!
    
    보상 = 최종 칩 변화량 (BB 정규화)
    - 이긴 경우: +칩 → 양수 보상
    - 진 경우: -칩 → 음수 보상
    - 무승부: 0칩 → 0 보상
    """
    chip_change = stack_after - stack_before
    bb_change = chip_change / self.big_blind
    
    # ⭐ 정규화: 100BB 기준 (강한 학습 신호)
    # 스택 분포: 5-250BB
    # 일반적 손익: ±100BB
    # 정규화: bb_change / 100
    reward = bb_change / 100.0
    
    # 보상 범위: [-2.5, +2.5]
    # +250BB → +2.5 (극단적 승리)
    # +100BB → +1.0 (큰 승리) ✨
    # +50BB → +0.5 (일반적 승리)
    # -50BB → -0.5 (일반적 손실)
    # -100BB → -1.0 (큰 손실)
    # -250BB → -2.5 (극단적 손실)
    
    return float(reward)
    
# ⚠️ 왜 100으로 나누는가?
# 
# 250으로 나눌 때:
#   +100BB → +0.4 (약한 신호)
#   +250BB → +1.0
#   범위: [-1.0, +1.0] (깔끔하지만 약함)
#
# 100으로 나눌 때:
#   +100BB → +1.0 (강한 신호!) ✨
#   +250BB → +2.5
#   범위: [-2.5, +2.5]
#   
# 장점:
#   1. 더 강한 학습 신호 (100BB = 1.0)
#   2. PPO가 [-2.5, +2.5] 충분히 처리
#   3. 직관적 (100BB = 1.0 보상)
#   4. 일반적 승리가 더 명확

# Multi-Agent step()에서 사용:
def step(self, action_dict):
    # ... 액션 처리 ...
    
    reward_dict = {}
    
    if self.game.is_hand_over:
        # ⭐ Zero-Sum 보장: P0 보상만 계산, P1은 음수 사용
        stack_before_p0 = self.hand_start_stacks[0]
        stack_after_p0 = self.chips[0]
        
        # P0 보상 계산
        chip_change = stack_after_p0 - stack_before_p0
        bb_change = chip_change / self.big_blind
        p0_reward = bb_change / 100.0  # 강한 학습 신호
        
        # ⭐ 완벽한 Zero-Sum 보장
        reward_dict = {
            "player_0": float(p0_reward),
            "player_1": float(-p0_reward)  # 정확히 음수!
        }
        
        # Zero-Sum 검증 (디버깅용)
        # assert abs(p0_reward + (-p0_reward)) < 1e-10, "Zero-Sum violation!"
        
    else:
        # ⭐ 핸드 진행 중: 보상 없음!
        next_player = self.game.get_current_player()
        reward_dict[f"player_{next_player}"] = 0.0
    
    return obs_dict, reward_dict, done_dict, truncated_dict, info_dict
```

**보상 계산 예시**:
```
핸드 시작:
- P0 stack: 1000
- P1 stack: 1000
- BB: 2

액션 시퀀스:
1. P0 raises 10    → reward_P0 = 0.0 (진행 중)
2. P1 calls 10     → reward_P1 = 0.0 (진행 중)
3. (Flop)
4. P0 bets 20      → reward_P0 = 0.0 (진행 중)
5. P1 raises 50    → reward_P1 = 0.0 (진행 중)
6. P0 folds        → reward_P0 = 0.0 (아직!)

핸드 종료:
- P0 stack: 940  (lost 60)
  → reward_P0 = (940-1000)/2/100 = -60/2/100 = -0.30
- P1 stack: 1060 (won 60)
  → reward_P1 = (1060-1000)/2/100 = +60/2/100 = +0.30
  
Zero-sum check: -0.30 + 0.30 = 0.0 ✅
```

**특징**:
- ✅ 칩 EV 최대화 = 포커 최적 전략
- ✅ Zero-sum (P0 + P1 = 0)
- ✅ BB 정규화로 스케일 일관성
- ✅ 클리핑으로 학습 안정성
- ✅ 중간 보상 없음 = 정확한 학습

**정규화 파라미터**:
- `normalization = 100.0` (초기값)
- → ±100BB 변화 = ±1.0 보상
- 필요시 조정: 50.0 (민감) or 200.0 (둔감)


### Phase 2: Transformer (고급)

**구조**:
```
입력 (310) - One-hot 카드 + 게임 상태 + 완전한 액션 히스토리
  → Positional Encoding
  → Transformer Encoder (4 layers, 8 heads)
  → FC(128)
  → FC(7)
```

**장점**:
- 시퀀스 정보 활용 (액션 히스토리!)
- Attention을 통한 의사결정 해석
- 복잡한 패턴 학습
- Preflop → River 스토리 이해
- **미묘한 Bet Sizing Pattern 학습**

**전환 시기**: MLP로 기본 학습 확인 후

---

## 🎓 학습 알고리즘

### PPO (Proximal Policy Optimization)

**선택 이유**:
- 안정적인 학습
- Self-play에 적합
- 연속적인 정책 개선
- RLlib에서 잘 지원됨

**하이퍼파라미터 (초기값)**:
```python
gamma = 0.99              # 할인율
lambda_ = 0.95            # GAE lambda
clip_param = 0.2          # PPO clip 범위
lr = 3e-4                 # 학습률
train_batch_size = 16384  # 배치 크기 (포커 분산 고려, 매우 큰 배치 필수!)
num_sgd_iter = 10         # SGD 반복
entropy_coeff = 0.01      # 탐험 인센티브
```

**포커 특화 고려사항**:

**배치 크기 (16384+)**:
- 포커는 **높은 분산(Variance)** 게임
  - 올바른 플레이 → 질 수 있음 (운 나쁨)
  - 잘못된 플레이 → 이길 수 있음 (운 좋음)
- 작은 배치 (8192 이하):
  - "운 좋게 이긴 나쁜 플레이"를 학습할 위험
  - 노이즈가 많은 그래디언트
  - 불안정한 학습
- **큰 배치 (16384+)**:
  - ✅ 분산이 평균화됨 (Law of Large Numbers)
  - ✅ 진짜 실력이 드러남
  - ✅ 안정적인 그래디언트
  - ✅ 올바른 전략 학습

**권장 배치 크기**:
- 초기 학습: 16384 (안정성 우선)
- 충분한 데이터 후: 32768 (더 안정적)
- 리소스 부족 시 최소: 16384 (이하는 비추천)


```

### 훈련 전략: Curriculum Learning

**Phase 1: Random Agent 부트스트랩 (초기 학습)**

목적: 기본적인 포커 개념 학습 및 무작위 플레이 극복

```python
# 상대: Random Agent (무작위 액션)
policies = {
    "learning_agent": Policy(MLP),      # 학습 중인 에이전트
    "random_opponent": RandomPolicy()   # 고정된 랜덤 에이전트
}

policies_to_train = ["learning_agent"]  # learning_agent만 학습
```

**종료 조건**:
- vs Random Agent 승률 **85%+** 달성
- 예상 시간: 1-2시간 (학습 환경에 따라)

**학습 내용**:
- 기본 베팅 개념 (폴드 vs 콜)
- 명백히 나쁜 액션 회피
- 공격적 플레이의 이점 인식

---

**Phase 2: Self-Play vs Historical Checkpoints (고급 학습)**

목적: 과거 자신과 대결하며 전략 발전 (League Training)

```python
# 상대: 과거 체크포인트 (주기적 업데이트)
policies = {
    "learning_agent": Policy(MLP),           # 현재 학습 중
    "historical_opponent": Policy(MLP)       # 과거 체크포인트
}

policies_to_train = ["learning_agent"]

# 체크포인트 업데이트 주기
# 매 100 iterations마다 historical_opponent 정책 업데이트
```

**업데이트 전략**:
```python
if iteration % 100 == 0:
    # 현재 learning_agent를 historical_opponent로 복사
    save_checkpoint("learning_agent", f"checkpoint_{iteration}")
    load_checkpoint("historical_opponent", f"checkpoint_{iteration}")
```


**종료 조건** (모두 충족 시 Phase 3로 전환):
1. **vs Random Agent**: 95%+ 승률 (100 게임 기준)
2. **vs Call Station**: 80%+ 승률 (100 게임 기준)
   - Call Station: 거의 모든 상황에서 콜만 하는 플레이어 (Fold 5%, Call 85%, Raise 10%)
   - Value betting 능력 검증
3. **vs Historical Checkpoints**: 최근 10개 체크포인트 대비 평균 55%+ 승률
4. **학습 안정성**: 최근 100 iterations 평균 보상의 표준편차 < 0.1
5. **최소 학습 시간**: 10시간 이상
6. **정책 엔트로피**: > 1.0 (액션 다양성 유지)

**학습 내용**:
- 복잡한 베팅 패턴
- 블러핑과 밸류 베팅 균형
- 상대 전략 적응

---

**Phase 3: Self-Play (최종 단계, 선택)**

목적: 두 에이전트 동시 학습으로 GTO 근사

```python
# 양쪽 모두 학습
policies = {
    "player_0": Policy(MLP),
    "player_1": Policy(MLP)
}

policies_to_train = ["player_0", "player_1"]
```

**특징**:
- 대칭적 환경 (공정성)
- 플레이어 간 상호 발전
- GTO 전략에 점진적 수렴

---

### RLlib 구현 예시

```python
# train.py
from ray.rllib.algorithms.ppo import PPOConfig

# Phase 1: vs Random
config_phase1 = (
    PPOConfig()
    .multi_agent(
        policies={
            "learning_agent": (None, obs_space, act_space, {}),
            "random_opponent": (None, obs_space, act_space, {"explore": True}),
        },
        policy_mapping_fn=lambda agent_id: 
            "learning_agent" if agent_id == "player_0" else "random_opponent",
        policies_to_train=["learning_agent"]
    )
)

# Phase 2: vs Historical
config_phase2 = (
    PPOConfig()
    .multi_agent(
        policies={
            "learning_agent": (None, obs_space, act_space, {}),
            "historical_opponent": (None, obs_space, act_space, {}),
        },
        policy_mapping_fn=lambda agent_id:
            "learning_agent" if agent_id == "player_0" else "historical_opponent",
        policies_to_train=["learning_agent"]
    )
    .callbacks(HistoricalCheckpointCallback)  # 주기적 업데이트
)
```

---


## 📊 평가 방법

### 학습 중 메트릭

**TensorBoard 메트릭**:
- `episode_reward_mean` - 평균 에피소드 보상
- `episode_len_mean` - 평균 핸드 수
- `policy_loss` - 정책 손실
- `vf_loss` - 가치 함수 손실
- `entropy` - 정책 엔트로피

**커스텀 메트릭**:
- **bb/100** (필수!) - 100 핸드당 획득 BB
- 평균 팟 크기
- VPIP (자발적 팟 참여율)
- PFR (프리플랍 레이즈율)
- Aggression Factor
- 평균 핸드 길이 (액션 수)

### 벤치마크 에이전트

**1. Random Agent** - 무작위 액션
- 모든 legal action 중 균등 확률로 선택
- 가장 약한 베이스라인

**2. Call Station** - 수동적 콜 중심 플레이
- **정의**: 거의 모든 상황에서 콜만 하는 플레이어
- **행동 분포**: Fold 5%, Call 85%, Raise 10%
- **특징**: 
  - 약한 핸드로도 끝까지 따라감
  - 블러핑에 강함 (폴드 안 함)
  - Value betting에 취약
- **목적**: AI의 value betting 능력 검증

**3. Nit** - 매우 타이트한 플레이 (Phase 3 벤치마크)
- 좋은 핸드만 플레이
- 공격적이지만 예측 가능
- 향후 구현 예정

**4. Historical Checkpoints** - 과거 자신과 대결
- League training의 핵심
- 과적합 방지
- 지속적인 자기 개선 검증

---

**평가 지표: bb/100 (포커 표준)**

**⚠️ 중요**: 승률은 의미 없는 지표!

```python
# ❌ 승률 (쓸모없음!)
90 핸드 승리 (+10 BB)
10 핸드 패배 (-200 BB)
승률: 90% (좋아 보임)
실제: -190 BB (망함!)

# ✅ bb/100 (포커 표준)
bb/100 = (총 획득 BB / 핸드 수) × 100

예시:
1000 핸드, +500 BB
→ bb/100 = (500/1000) × 100 = 50 bb/100
```

**벤치마크 목표 (bb/100)**:

| Phase | 상대 | 목표 bb/100 | 의미 |
|-------|------|-------------|------|
| **Phase 1 종료** | vs Random | **+80 bb/100** | Random 압도 |
| **Phase 2 종료** | vs Random | **+100 bb/100** | Random 완벽 지배 |
| | vs Call Station | **+50 bb/100** | Value betting 능력 |
| **Phase 3 목표** | vs Nit | **+20 bb/100** | 타이트 플레이어 대응 |
| | vs Historical | **+10 bb/100** | 자기 자신 넘어서기 |

**bb/100 기준**:
- **+50 이상**: 매우 강함
- **+20~50**: 강함
- **+5~20**: 괜찮음
- **0~5**: 약간 이김
- **0 미만**: 짐

**측정 방법**:
```python
def evaluate_bb100(agent, opponent, num_hands=1000):
    total_bb = 0
    
    for _ in range(num_hands):
        obs, info = env.reset()
        # 핸드 플레이
        ...
        total_bb += final_bb_change
    
    bb_100 = (total_bb / num_hands) * 100
    return bb_100

# 예시
bb100_vs_random = evaluate_bb100(agent, RandomAgent(), 1000)
print(f"vs Random: {bb100_vs_random:.1f} bb/100")
# 목표: +80 이상
```

---

## 🗓️ 구현 단계

### Phase 0: 환경 구축 (1-2일)

- [ ] Ray/RLlib 설치 및 환경 설정
- [ ] 의존성 정리 (`requirements.txt`)
- [ ] 프로젝트 구조 설계

### Phase 1: Gymnasium 환경 구현 (2-3일)

- [ ] `PokerEnv` 클래스 생성
  - [ ] `__init__()` - 초기화
  - [ ] `reset()` - 에피소드 시작
  - [ ] `step()` - 액션 실행
  - [ ] `_get_observation()` - 관찰 생성
  - [ ] `_get_reward()` - 보상 계산
  - [ ] `_map_action()` - 액션 매핑
- [ ] POKERENGINE 통합
- [ ] 환경 검증
  - [ ] `gymnasium.utils.env_checker` 통과
  - [ ] 수동 플레이 테스트

### Phase 2: RLlib 통합 (1-2일)

- [ ] 환경 등록
- [ ] PPO 설정
- [ ] Multi-Agent 설정
- [ ] 학습 스크립트 작성 (`train.py`)
- [ ] 첫 학습 실행 (5-10분)
- [ ] TensorBoard 확인

### Phase 3: 기본 학습 (3-5일)

- [ ] MLP 모델로 학습
- [ ] 하이퍼파라미터 튜닝
- [ ] 학습 안정성 확인
- [ ] 체크포인트 저장/로드
- [ ] 학습 모니터링 대시보드

### Phase 4: AI 대전 시스템 (2일)

- [ ] `play.py` - AI vs AI 시뮬레이션
- [ ] `play_human.py` - 사람 vs AI
- [ ] 게임 로깅
- [ ] 통계 수집

### Phase 5: 평가 & 개선 (진행중)

- [ ] 벤치마크 에이전트 구현
- [ ] 성능 평가
- [ ] 전략 분석
- [ ] 문제점 식별 및 해결

### Phase 6: 고급 기능 (선택)

- [ ] Transformer 모델 전환
- [ ] ICM 보상 실험
- [ ] 블라인드 레벨업 (토너먼트)
- [ ] 멀티플레이어 (3-9인)
- [ ] 커리큘럼 학습
- [ ] Opponent Modeling

---

## 📁 프로젝트 구조

```
glacial-supernova/
├── POKERENGINE/              # 커스텀 포커 엔진
│   ├── poker_engine/
│   ├── admin.py
│   └── test_poker_engine.py
├── poker_rl/                 # 새 AI 학습 프로젝트
│   ├── __init__.py
│   ├── env.py                # Gymnasium 환경
│   ├── config.py             # 설정 및 하이퍼파라미터
│   ├── train.py              # 학습 스크립트
│   ├── play.py               # AI 대전
│   ├── play_human.py         # 사람 vs AI
│   ├── models/               # 모델 아키텍처
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   └── transformer.py
│   ├── agents/               # 벤치마크 에이전트
│   │   ├── __init__.py
│   │   ├── random_agent.py
│   │   └── call_station.py
│   └── utils/                # 유틸리티
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── experiments/              # 실험 결과
│   ├── checkpoints/
│   └── logs/
├── requirements.txt          # 의존성
├── README.md                 # 프로젝트 설명
└── IMPLEMENTATION_PLAN.md    # 이 문서
```

---

## 🔬 실험 계획

### Experiment 1: Baseline

**목표**: MLP + PPO로 기본 학습 가능성 확인

**설정**:
- 모델: MLP (256-256-128)
- 학습 시간: 2-4시간
- 목표: vs Random 80%+ 승률

### Experiment 2: Hyperparameter Tuning

**목표**: 최적 하이퍼파라미터 탐색

**변수**:
- Learning rate: [1e-4, 3e-4, 1e-3]
- Batch size: [2048, 4096, 8192]
- Entropy coeff: [0.01, 0.02, 0.05]

### Experiment 3: Model Comparison

**목표**: MLP vs Transformer 성능 비교

**설정**:
- 동일한 학습 시간
- 동일한 하이퍼파라미터
- 승률 및 학습 속도 비교

---

## ⚠️ 예상 문제 및 해결책

### 문제 1: 학습 불안정

**증상**: 보상이 발산하거나 수렴하지 않음

**해결책**:
- Learning rate 감소
- Batch size 증가
- Reward clipping
- 정규화 강화

### 문제 2: 과적합

**증상**: 학습 에이전트끼리만 이기고 새로운 상대는 못 이김

**해결책**:
- Population-based training
- League training
- 정기적인 정책 리셋

### 문제 3: 탐험 부족

**증상**: 특정 전략에만 수렴 (예: 항상 폴드)

**해결책**:
- Entropy coefficient 증가
- Curiosity-driven exploration
- 보상 함수 튜닝

---

## 📚 참고 자료

### 논문
- "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games" (DeepStack)
- "Superhuman AI for multiplayer poker" (Pluribus)
- "Mastering the Game of No-Limit Texas Hold'em Poker through Self-Play" (Slumbot)

### 코드
- RLlib 공식 문서: https://docs.ray.io/en/latest/rllib/
- Gymnasium 문서: https://gymnasium.farama.org/
- POKERENGINE: 우리 커스텀 엔진

### 도구
- TensorBoard: 학습 모니터링
- Ray Dashboard: 분산 학습 모니터링

---

## ✅ 체크리스트 & 마일스톤

### Milestone 1: Environment Ready (1주)
- [x] POKERENGINE 완성 및 테스트
- [ ] Gymnasium 환경 구현
- [ ] 환경 검증 완료

### Milestone 2: First Training (2주)
- [ ] RLlib 통합
- [ ] 첫 학습 실행 성공
- [ ] TensorBoard 확인
- [ ] vs Random 50%+ 승률

### Milestone 3: Baseline Agent (1개월)
- [ ] vs Random 90%+ 승률
- [ ] 기본 전략 학습 확인
- [ ] 체크포인트 저장/로드

### Milestone 4: Competitive Agent (2개월)
- [ ] vs Call Station 80%+ 승률
- [ ] 복잡한 전략 학습
- [ ] 사람과 대결 시스템

### Milestone 5: Advanced Features (3개월+)
- [ ] Transformer 모델
- [ ] 멀티플레이어
- [ ] GTO 근사 검증

---

## 🎯 성공 기준

**Phase 1 성공**:
- AI가 무작위 플레이어를 90% 이상 이김
- 기본적인 포커 개념 이해 (폴드, 벳, 레이즈)

**Phase 2 성공**:
- AI가 Call Station을 80% 이상 이김
- 벨류 베팅과 블러핑 구사

**최종 성공**:
- AI가 숙련된 아마추어 플레이어와 대등하게 경쟁
- GTO 전략에 근접하는 플레이
- 다양한 상황에서 적응적 전략

---

**작성일**: 2025-11-30
**작성자**: AI Assistant
**버전**: 1.0
