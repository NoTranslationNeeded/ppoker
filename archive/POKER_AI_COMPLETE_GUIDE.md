# Texas Hold'em AI - 완전한 구현 가이드

**본 문서는 모든 설계 결정사항과 구현 가이드를 통합한 완전판입니다.**

목차:
1. [프로젝트 개요](#프로젝트-개요)
2. [환경 설계](#환경-설계)  
3. [모델 아키텍처](#모델-아키텍처)
4. [학습 전략](#학습-전략)
5. [구현 가이드](#구현-가이드)

---

## 프로젝트 개요

**최종 목표**: 커스텀 POKERENGINE을 사용하여 헤즈업 노리밋 텍사스 홀덤 AI를 강화학습으로 훈련

**핵심 목표**:
- GTO(Game Theory Optimal) 전략에 근접하는 AI 개발
- Self-play를 통한 자가 학습
- 다양한 상황에서 적응 가능한 전략 학습
- bb/100 기준 +50 이상 달성

**기술 스택**:
- Python 3.10+
- Ray/RLlib 2.x (Multi-Agent)
- PyTorch
- Gymnasium
- POKERENGINE (커스텀 TDA 준수)

---

## 환경 설계

### MultiAgentEnv 구조

**왜 MultiAgentEnv?**
- Self-Play 지원 (처음부터)
- RLlib 분산 학습 활용
- Phase 1~3 모두 동일 구조

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class PokerMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        # BB 고정 (수학적으로 의미 있는 것은 BB 단위뿐)
        self.small_blind = 50.0
        self.big_blind = 100.0
        
        # 스택 분포 (중요도 기반)
        self.stack_distribution = {
            'standard': (80, 120, 0.40),   # 100BB, 40%
            'middle': (20, 50, 0.30),      # 35BB, 30%
            'short': (5, 20, 0.20),        # 12BB, 20%
            'deep': (150, 250, 0.10)       # 200BB, 10%
        }
```

### 스택 샘플링 전략 ⭐

**핵심**: 매 핸드 리셋 + 중요도 기반 샘플링

❌ **토너먼트 방식 (비효율)**:
```
Start:500BB → 400핸드 딥스택 → 95핸드 미들 → 5핸드 숏스택
문제: 숏스택만 학습!
```

✅ **매 핸드 리셋 (효율적)**:
```
Hand 1: 100BB
Hand 2: 35BB  
Hand 3: 180BB
Hand 4: 12BB
→ 모든 깊이 균등 학습!
```

**구현**:
```python
def _sample_stack_depth(self) -> float:
    categories = ['standard', 'middle', 'short', 'deep']
    probs = [0.40, 0.30, 0.20, 0.10]
    
    category = np.random.choice(categories, p=probs)
    min_bb, max_bb, _ = self.stack_distribution[category]
    stack_bb = np.random.uniform(min_bb, max_bb)
    
    return stack_bb * self.big_blind

def reset(self):
    # 스택 랜덤 샘플링
    self.chips = [
        self._sample_stack_depth(),
        self._sample_stack_depth()
    ]
    
    # 버튼 랜덤 (50:50)
    self.button = np.random.randint(0, 2)
```

**장점**:
1. 효율성: 1핸드 = 1에피소드
2. 균형: 모든 스택 깊이 학습
3. 현실성: 중요한 상황에 집중

### 턴제 게임 특성 ⭐

**중요**: 포커는 동시 액션 게임이 아닌 **턴제 게임**!

❌ **동시 액션 (스타크래프트)**:
```python
step({"p1": a1, "p2": a2})
return {"p1": obs1, "p2": obs2}  # 둘 다
```

✅ **턴제 (포커)**:
```python
step({"player_0": raise})
return {"player_1": obs}  # 다음 턴만!
```

**구현**:
```python
def step(self, action_dict):
    current_player = self.game.get_current_player()
    action = action_dict[f"player_{current_player}"]
    
    self.game.process_action(current_player, action)
    
    if not self.game.is_hand_over:
        # 진행 중: 다음 턴 플레이어만
        next_player = self.game.get_current_player()
        return {
            f"player_{next_player}": obs
- 더 세밀한 베팅 컨트롤

### Min-Raise 처리 ⭐

**문제**:
```
팟: 100, 상대: 50, Min-Raise: 150
AI: Bet 33% (33) → Total 83 < Min-Raise!
```

**해결**: 스마트 보정
```python
if intended < min_raise:
    if intended >= min_raise * 0.5:
        → Min-Raise로 보정 (공격 유지)
    else:
        → Call로 보정 (약한 공격)
```

**구현**:
```python
def _map_action(self, action_idx, player_id):
    # ... Fold, Call, All-in 처리 ...
    
    # Bet/Raise
    pct = [0.33, 0.75, 1.0, 1.5][action_idx - 2]
    intended_bet = pot * pct
    
    if self.game.current_bet > 0:  # Raise
        min_raise_total = self.game.current_bet + min_raise
        intended_total = self.game.current_bet + intended_bet
        
        if intended_total < min_raise_total:
            if intended_bet >= min_raise * 0.5:
                target = min_raise_total  # 보정
            else:
                return Action.call(to_call)
        else:
            target = intended_total
        
        return Action.raise_to(target)
```

**학습 효과**:
- AI가 다음 관찰에서 보정된 금액 확인
- "75% 눌렀는데 150%가 나갔네" 학습
- 점진적으로 Min-Raise를 고려한 선택

### 보상 함수 ⭐

**타입**: Sparse Reward (핸드 종료 시만)

**Zero-Sum 보장**:
```python
def step(self, action_dict):
    if self.game.is_hand_over:
        # P0 보상만 계산
        chip_change = self.chips[0] - self.hand_start_stacks[0]
        bb_change = chip_change / self.big_blind
        p0_reward = bb_change / 100.0  # 강한 학습 신호
        
        # Zero-Sum 보장!
        reward_dict = {
            "player_0": float(p0_reward),
            "player_1": float(-p0_reward)
        }
    else:
        # 진행 중: 보상 없음
        reward_dict = {next_agent: 0.0}
```

**특징**:
- ✅ 완벽한 Zero-Sum (부동소수점 오차 없음)
- ✅ 범위: [-2.5, +2.5] (100BB로 정규화)
- ✅ 강한 학습 신호 (100BB = 1.0)
- ✅ 클리핑 없음 (정보 손실 없음)

**절대 금지**:
```python
# ❌ 중간 보상 절대 금지!
if action == BET:
    reward = -bet_amount  # NO!
```

### Action Masking ⭐

**필수!** 사후 처리 방식은 학습 비효율!

**환경**:
```python
def _get_legal_actions_mask(self) -> np.ndarray:
    legal = self.game.get_legal_actions(current_player)
    mask = np.zeros(7, dtype=np.int8)
    
    if FOLD in legal: mask[0] = 1
    if CHECK/CALL in legal: mask[1] = 1
    if BET/RAISE in legal: mask[2:6] = 1
    if ALL_IN in legal: mask[6] = 1
    
    return mask

def step(self, action_dict):
    info = {'action_mask': self._get_legal_actions_mask()}
    return obs, reward, done, truncated, info
```

**커스텀 모델**:
```python
class MaskedLSTM(TorchModelV2, nn.Module):
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        action_mask = input_dict["obs"]["action_mask"]
        
        # Network forward
        logits = self.network(obs)
        
        # ⭐ Masking
        inf_mask = torch.clamp(
            torch.log(action_mask), 
            FLOAT_MIN, FLOAT_MIN
        )
        masked_logits = logits + inf_mask
        
        return masked_logits, state
```

**step()에서 절대 금지**:
```python
# ❌ 사후 처리 금지!
if not success:
    action = check_or_call
```
```

**왜 FC가 먼저?**
- 310차원 sparse → LSTM 직접 = 느림
- FC가 특징 추출 → LSTM은 시퀀스만

**RLlib 구현**:
```python
config = (
```python
# 핸드 시작
lstm_state_0 = algo.get_initial_state()
lstm_state_1 = algo.get_initial_state()

# 매 턴
action, lstm_state_0, _ = algo.compute_single_action(
    obs, state=lstm_state_0
)
```

---

## 학습 전략

### PPO 하이퍼파라미터

```python
config = (
    PPOConfig()
    .training(
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        lr=3e-4,
        train_batch_size=16384,  # ⭐ 포커 분산 대응 (매우 큰 배치 필수!)
        num_sgd_iter=10,
        entropy_coeff=0.01
    )
)
```

**왜 16384+?**
- 포커는 고분산 게임
- 운이 학습에 간섭
- 큰 배치로 분산 평균화 (Law of Large Numbers)

### Curriculum Learning

**Phase 1**: vs Random (1-2시간)
- 목표: +80 bb/100

**Phase 2**: vs Historical (10+ 시간)
- 목표: vs Random +100, vs Call Station +50 bb/100
- 체크포인트: 매 100 iterations

**Phase 3**: Self-Play
- 양쪽 모두 학습
- GTO 근접

### 평가 지표: bb/100

❌ **승률은 의미 없음**:
```
90핸드 승리 (+10 BB)
10핸드 패배 (-200 BB)
승률: 90% (좋아 보임)
실제: -190 BB (망함!)
```

✅ **bb/100 (포커 표준)**:
```
bb/100 = (총 획득 BB / 핸드 수) × 100

1000 핸드, +500 BB
→ bb/100 = 50 bb/100
```

**기준**:
- +50 이상: 매우 강함
- +20~50: 강함
- +5~20: 괜찮음

### 벤치마크 목표 (bb/100) ⭐

| Phase | 상대 | 목표 bb/100 | 의미 |
|-------|------|-------------|------|
| **Phase 1 종료** | vs Random | **+80 bb/100** | Random 압도 |
| **Phase 2 종료** | vs Random | **+100 bb/100** | Random 완벽 지배 |
| | vs Call Station | **+50 bb/100** | Value betting 능력 |
| **Phase 3 목표** | vs Nit | **+20 bb/100** | 타이트 플레이어 대응 |
| | vs Historical | **+10 bb/100** | 자기 자신 넘어서기 |

### 벤치마크 에이전트 정의

1. **Random Agent**: 모든 Legal Action 중 균등 확률 선택
2. **Call Station**: Fold 5%, Call 85%, Raise 10% (Value betting 검증용)
3. **Nit**: 상위 15% 핸드만 플레이 (Phase 3용)
4. **Historical**: 과거의 나 (League Training)

### Phase별 종료 조건 (상세)

**Phase 1 (vs Random)**:
- vs Random 승률 85%+ 또는 +80 bb/100
- 예상 시간: 1-2시간

**Phase 2 (vs Historical)**:
- vs Random +100 bb/100
- vs Call Station +50 bb/100
- 학습 안정성: 최근 100 iter 보상 표준편차 < 0.1
- 최소 학습 시간: 10시간 이상

---

## 구현 가이드

### 디렉토리 구조

```
glacial-supernova/
├── poker_rl/
│   ├── env.py              # PokerMultiAgentEnv
│   ├── models/
│   │   └── masked_lstm.py  # MaskedLSTM
│   ├── agents/
│   │   ├── random.py
│   │   └── call_station.py
│   └── utils/
│       └── evaluator.py    # bb/100 계산
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── play_vs_ai.py
├── configs/
│   ├── phase1.yaml
│   ├── phase2.yaml
│   └── phase3.yaml
└── POKERENGINE/
```

### 체크리스트

**환경**:
- [ ] MultiAgentEnv 상속
- [ ] 매 핸드 스택 리셋 (샘플링)
- [ ] 버튼 랜덤화 (50:50)
- [ ] 턴제 반환 (다음 플레이어만)
- [ ] Min-Raise 스마트 보정
- [ ] Zero-Sum 보상 보장
- [ ] Legal Actions Mask 제공

**모델**:
- [ ] FC → LSTM 순서
- [ ] Action Masking 적용
- [ ] State 관리 (Inference)

**학습**:
- [ ] train_batch_size >= 8192
- [ ] bb/100 평가
- [ ] 체크포인트 저장

### 핵심 원칙 요약

1. **매 핸드 리셋** (토너먼트 X)
2. **BB=100 고정** (절대값 무의미)
3. **MultiAgentEnv** (Self-Play)
4. **턴제 반환** (다음 플레이어만)
5. **Zero-Sum 보상** (P1 = -P0)
6. **Sparse Reward** (핸드 종료 시만)
7. **Action Masking** (사후 처리 X)
8. **FC → LSTM** (Feature → Temporal)
9. **bb/100 평가** (승률 X)
10. **큰 배치** (분산 대응)

---

## 참고 자료

- RLlib Multi-Agent: https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical
- Action Masking: https://github.com/ray-project/ray/blob/master/rllib/examples/action_masking.py
- LSTM Models: https://docs.ray.io/en/latest/rllib/rllib-models.html#built-in-models

---

**본 가이드는 모든 설계 결정과 근거를 포함한 완전판입니다.**
**구현 시 이 문서를 단일 참조로 사용하십시오.**
