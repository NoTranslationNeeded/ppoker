# Observation Vector 개선 Implementation Plan

## 📋 Executive Summary

**목적**: 학습 효율을 **3-10배** 향상시키기 위한 관찰 벡터 개선

**근거**: [observation_vector_review.md](file:///c:/Users/99san/.gemini/antigravity/playground/glacial-supernova/observation_vector_review.md)에서 발견된 5가지 치명적 문제점 해결

**예상 효과**:
- 현재 학습 속도: 10M+ 스텝 예상
- 개선 후: **2-3M 스텝** (3-5배 향상)
- Suit symmetry만으로도 4배 개선

---

## 🎯 개선 사항 요약

| # | 개선 | 추가 차원 | 우선순위 | 예상 효과 |
|---|------|----------|---------|----------|
| 1 | 액션 히스토리 & 컨텍스트 | +26 | 🔥🔥🔥 | 5-10배 |
| 2 | Suit Canonicalization | 0 | 🔥🔥🔥 | 4배 |
| 3 | 로그 스케일 정규화 | 0 | 🔥🔥 | 1.5-2배 |
| **총계** | | **+26** | | **복합 3-10배** |

**최종 차원**: 150 + 26 = **176차원**

---

## 📦 Phase 1: 핵심 개선 구현

### 개선 1: 액션 히스토리 & 컨텍스트 (+27차원)

#### 1.1 Street History Features (16차원)

**파일**: `poker_rl/utils/obs_builder.py`

**핵심 원칙**: **Canonical suits는 카드 인코딩에만 사용, Equity calculation은 Original Card 객체 사용!**

**추가 함수**:
```python
def _get_street_context_features(game, player_id, action_history):
    """
    각 스트릿별 요약 통계: 4 streets × 4 features = 16차원
    """
    features = np.zeros(16, dtype=np.float32)
    streets = ['preflop', 'flop', 'turn', 'river']
    
    for i, street in enumerate(streets):
        base_idx = i * 4
        actions = action_history.get(street, [])
        
        # [0] Number of raises (0-10, normalized)
        raises = sum(1 for (action_idx, _, _) in actions 
                    if action_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # All-In 포함!
        features[base_idx + 0] = min(raises, 10) / 10.0
        
        # [1] Aggressor (0=none, 0.5=me, 1.0=opponent)
        last_aggressor = None
        for (action_idx, pid, _) in actions:
            if action_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:  # All-In 포함!
                last_aggressor = pid
        
        if last_aggressor is None:
            features[base_idx + 1] = 0.0
        elif last_aggressor == player_id:
            features[base_idx + 1] = 0.5
        else:
            features[base_idx + 1] = 1.0
        
        # [2] My total investment this street (ACCURATE with actual amounts)
        # env.py stores actual bet_amount in tuple: (action_idx, player_id, bet_ratio, bet_amount)
        my_actions_amounts = [amount for (_, pid, _, amount) in actions if pid == player_id]
        total_invested_absolute = sum(my_actions_amounts)
        features[base_idx + 2] = normalize_chips_log(total_invested_absolute / game.big_blind)
        
        # [3] Was 3-bet or higher? (binary)
        features[base_idx + 3] = 1.0 if raises >= 2 else 0.0
    
    return features
```

**통합**:
```python
def get_observation(game, player_id, action_history, start_stacks=None):
    """start_stacks: List of starting chips for each player"""
    obs_vec = np.zeros(176, dtype=np.float32)
    
    # CRITICAL: Original Card 객체 보존
    hole_cards_original = game.players[player_id].hand  # Card 객체
    board_original = game.community_cards              # Card 객체
    
    # 1. Canonicalize suits (인코딩용)
    canonical = canonicalize_suits(hole_cards_original, board_original)
    
    # 2. Encode cards (0-118) - Canonical 사용
    for i, card in enumerate(canonical[:7]):
        obs_vec[i*17:(i+1)*17] = _encode_card_onehot(card)
    
    # 3. Game state (119-134) - 로그 스케일 적용
    # ...
    
    # 4. Expert features (135-142) - ORIGINAL Card 객체 사용!
    advanced_features = get_8_features(
        hole_cards_original,  # Card 객체 (canonicalized 아님!)
        board_original,       # Card 객체 (canonicalized 아님!)
        game.street.value
    )
    obs_vec[135:143] = advanced_features
    
    # 5. Padding (143-149) - 현재는 0으로 유지
    # Note: 향후 확장성을 위해 유지, hand_index_pos=138 hard-coded 때문
    obs_vec[143:150] = 0.0
    
    # 6. Street history (150-165)
    obs_vec[150:166] = _get_street_context_features(game, player_id, action_history)
    
    # 7. Current street context (166-171)
    obs_vec[166:172] = _get_current_street_context(game, player_id, action_history)
    
    # 8. Investment info (172-173)
    obs_vec[172:174] = _get_investment_features(game, player_id, start_stacks)
    
    # 9. Position info (174-175)
    obs_vec[174:176] = _get_position_features(game, player_id)
    
    return {
        "observations": obs_vec,
        "action_mask": _get_legal_actions_mask(game, player_id)
    }
```

> [!WARNING]
> **Canonical vs Original 분리 필수**: 
> - `canonicalize_suits()` → `(rank_str, suit_int)` 튜플 반환
> - `get_8_features()` → `Card` 객체 필요
> - 카드 인코딩은 canonical 사용, equity 계산은 original 사용!

> [!NOTE]
> **Dead Space (143-149)**: 
> - 7차원이 비어있지만 의도적으로 유지
> - `masked_lstm.py`의 `hand_index_pos=138`이 hard-coded됨
> - 기존 섹션 확장 시 유연성 제공
> - Phase 2에서 compact화 고려 가능

#### 1.2 Current Street Context (6차원)

**추가 위치**: `obs_vec[166:172]`

```python
def _get_current_street_context(game, player_id, action_history):
    """현재 스트릿 액션 패턴: 6차원"""
    features = np.zeros(6, dtype=np.float32)
    
    current_street = game.street.value
    actions = action_history.get(current_street, [])
    
    # [0] Actions count this street
    features[0] = min(len(actions), 10) / 10.0
    
    # [1] I raised this street
    features[1] = 1.0 if any(pid == player_id and action_idx in [2,3,4,5,6,7,8,9,10,11,12,13]  # All-In 포함!
                             for (action_idx, pid, _) in actions) else 0.0
    
    # [2] Opponent raised this street
    features[2] = 1.0 if any(pid != player_id and action_idx in [2,3,4,5,6,7,8,9,10,11,12,13]  # All-In 포함!
                             for (action_idx, pid, _) in actions) else 0.0
    
    # [3] Passive-to-Aggressive transition (개선된 check-raise 감지)
    # Check-Raise 정확한 감지는 복잡하므로, "수동→공격" 패턴으로 단순화
    passive_to_aggressive = False
    my_was_passive = False
    for (action_idx, pid, _) in actions:
        if pid == player_id:
            if action_idx in [0, 1]:  # Fold/Check/Call - 수동적
                my_was_passive = True
            elif action_idx in [2,3,4,5,6,7,8,9,10,11,12,13] and my_was_passive:  # 공격적
                passive_to_aggressive = True
                break
    features[3] = 1.0 if passive_to_aggressive else 0.0
    # Note: 진짜 Check-Raise (Check→Opponent Bet→Raise)보다 넓은 개념
    # 하지만 "태도 전환" 패턴 포착에는 유용
    
    # [4] Donk-bet happened (OOP bets into preflop aggressor)
    # Simplified: bet when not last aggressor
    features[4] = 0.0  # Implement if needed
    
    # [5] Last action was aggressive
    if actions:
        last_action = actions[-1][0]
        features[5] = 1.0 if last_action in [2,3,4,5,6,7,8,9,10,11,12,13] else 0.0
    
    return features
```

#### 1.3 Investment Info (2차원)

**위치**: `obs_vec[172:174]`

```python
def _get_investment_features(game, player_id, start_stacks):
    """투자 정보: 2차원"""
    features = np.zeros(2, dtype=np.float32)
    
    my_total_invested = game.players[player_id].bet_this_hand
    starting_stack = start_stacks[player_id] if start_stacks else 10000.0  # Fallback
    
    # [0] Total investment (log scale)
    features[0] = normalize_chips_log(my_total_invested / game.big_blind)
    
    # [1] Investment ratio (0-1)
    investment_ratio = my_total_invested / starting_stack if starting_stack > 0 else 0.0
    features[1] = min(investment_ratio, 1.0)
    
    return features
```

**통합**: `get_observation()` 시그니처 변경 필요
```python
def get_observation(game, player_id, action_history, start_stacks=None):
    ...
    obs_vec[172:174] = _get_investment_features(game, player_id, start_stacks)
```

#### 1.4 Position Info (2차원)

**위치**: `obs_vec[174:176]`

```python
def _get_position_features(game, player_id):
    """포지션 관련 명시적 정보: 2차원"""
    features = np.zeros(2, dtype=np.float32)
    
    is_button = (game.button_position == player_id)
    is_preflop = (game.street.value == 'preflop')
    
    # [0] Position Value (0.0=OOP, 1.0=IP)
    if is_preflop:
        features[0] = 0.0 if is_button else 1.0  # Non-button is IP preflop
    else:
        features[0] = 1.0 if is_button else 0.0  # Button is IP postflop
    
    # [1] Permanent Position Advantage (postflop only)
    if is_preflop:
        features[1] = 0.5  # Neutral
    else:
        features[1] = 1.0 if is_button else 0.0
    
    return features
```

> [!NOTE]
> **Acting First 피처 제거**: 원래 계획에 있었으나, `get_observation()`은 항상 현재 플레이어를 위해 호출되므로 항상 1.0이 되어 무용지물. [0] Position Value가 IP/OOP를 이미 표현하므로 중복.
```

---

### 개선 2: Suit Canonicalization (0 추가 차원)

**파일**: `poker_rl/utils/obs_builder.py`

**핵심 함수**:
```python
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def canonicalize_suits(hole_cards, board):
    """
    무늬를 첫 등장 순서로 정규화
    
    Returns:
        List of (rank_str, canonical_suit_int) tuples
    """
    suit_map = {}
    next_suit_id = 0
    canonical = []
    
    for card in (hole_cards + board):
        # 새 무늬 등장 시 ID 할당
        if card.suit not in suit_map:
            suit_map[card.suit] = next_suit_id
            next_suit_id += 1
        
        canonical.append((card.rank, suit_map[card.suit]))
    
    return canonical
```

**수정된 인코딩**:
```python
def _encode_card_onehot(canonical_card):
    """
    (rank_str, canonical_suit_int) → 17-dim one-hot
    """
    rank, suit = canonical_card
    encoding = np.zeros(17, dtype=np.float32)
    
    rank_idx = RANKS.index(rank)
    encoding[rank_idx] = 1.0
    encoding[13 + suit] = 1.0  # suit는 이미 0-3
    
    return encoding
```

**통합**:
```python
def get_observation(game, player_id, action_history):
    # 1. Canonicalize
    hole = game.players[player_id].hand
    board = game.community_cards
    canonical = canonicalize_suits(hole, board)
    
    # 2. Encode cards (0-118)
    obs_vec = np.zeros(177, dtype=np.float32)
    for i, card in enumerate(canonical[:7]):
        obs_vec[i*17:(i+1)*17] = _encode_card_onehot(card)
    
    # 3. ... rest of features ...
```

---

### 개선 3: 로그 스케일 정규화 (0 추가 차원)

**파일**: `poker_rl/utils/obs_builder.py`

**정규화 함수**:
```python
def normalize_chips_log(chips_in_bb):
    """로그 스케일 정규화"""
    return np.log1p(chips_in_bb) / np.log1p(500.0)
```

**적용**:
```python
bb = game.big_blind

# Chip-related features (LOG scale)
obs_vec[119] = normalize_chips_log(player.chips / bb)
obs_vec[120] = normalize_chips_log(opponent.chips / bb)
obs_vec[121] = normalize_chips_log(pot / bb)
obs_vec[122] = normalize_chips_log(game.current_bet / bb)
obs_vec[123] = normalize_chips_log(player.bet_this_round / bb)
obs_vec[124] = normalize_chips_log(to_call / bb)
# ... 
obs_vec[131] = normalize_chips_log(game.min_raise / bb)
obs_vec[132] = normalize_chips_log(opponent.bet_this_round / bb)
obs_vec[133] = normalize_chips_log(opponent.bet_this_hand / bb)

# 변경하지 않는 것들 (상대 비율)
obs_vec[127] = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0  # Pot Odds
obs_vec[128] = np.clip((player.chips / pot) / 10.0, 0, 1.0) if pot > 0 else 1.0  # SPR
obs_vec[130] = len(game.community_cards) / 5.0  # Card count
obs_vec[134] = bb / 100.0  # Blind size
```

---

## 🔄 Phase 2: 파일 변경 사항

### 변경 파일 목록

#### [MODIFY] [obs_builder.py](file:///c:/Users/99san/.gemini/antigravity/playground/glacial-supernova/poker_rl/utils/obs_builder.py)

**변경 사항**:
1. ✅ `canonicalize_suits()` 함수 추가
2. ✅ `_encode_card_onehot()` 수정 (canonical 지원)
3. ✅ `normalize_chips_log()` 함수 추가
4. ✅ `_get_street_context_features()` 함수 추가
5. ✅ `_get_current_street_context()` 함수 추가
6. ✅ `_get_position_features()` 함수 추가
7. ✅ `get_observation()` 전체 재구성
   - 차원: 150 → 177
   - 카드 canonicalization 통합
   - 로그 스케일 적용
   - 새 features 추가

#### [MODIFY] [env.py](file:///c:/Users/99san/.gemini/antigravity/playground/glacial-supernova/poker_rl/env.py)

**변경 사항**:
1. ✅ `observation_space` 업데이트
   ```python
   self.observation_space = spaces.Dict({
       "observations": spaces.Box(
           low=0.0,
           high=200.0,
           shape=(176,),  # 150 → 176
           dtype=np.float32
       ),
       ...
   })
   ```

2. ✅ `self.hand_start_stacks` 추적 (이미 존재 확인됨)
   ```python
   def reset(self, ...):
       ...
       self.hand_start_stacks = list(self.chips)  # Line 137
   ```

3. ✅ **action_history 튜플 구조 변경** - 금액 추적 추가
   ```python
   # Line 152-159 - 버그 수정 및 구조 개선
   self.action_history = {
       'preflop': [],
       'flop': [],
       'turn': [],    # 중복 'turn' 제거
       'river': []
   }
   ```

4. ✅ **_record_action() 수정** - 실제 bet amount 저장
   ```python
   def _record_action(self, action_idx: int, player_id: int, bet_amount: float, pot_before: float, street: str):
       """액션 기록 - 비율과 실제 금액 모두 저장"""
       if pot_before > 0:
           bet_ratio = bet_amount / pot_before
       else:
           bet_ratio = 0.0
       bet_ratio = np.clip(bet_ratio, 0.0, 2.5)
       
       if street in self.action_history:
           # 튜플 구조: (action_idx, player_id, bet_ratio, bet_amount)
           # 기존 3개 → 4개 요소로 확장
           self.action_history[street].append(
               (action_idx, player_id, bet_ratio, bet_amount)
           )
   ```

5. ✅ `get_observation()` 호출 시 `start_stacks` 전달
   ```python
   # reset()에서
   obs_dict = {
       "player_0": ObservationBuilder.get_observation(
           self.game, 0, self.action_history, self.hand_start_stacks
       ),
       "player_1": ObservationBuilder.get_observation(
           self.game, 1, self.action_history, self.hand_start_stacks
       )
   }
   
   # step()에서도 동일하게
   obs = ObservationBuilder.get_observation(
       self.game, next_player, self.action_history, self.hand_start_stacks
   )
   ```

#### [MODIFY] [masked_lstm.py](file:///c:/Users/99san/.gemini/antigravity/playground/glacial-supernova/poker_rl/models/masked_lstm.py)

**변경 사항**:
1. ✅ `input_size` 자동 계산 (이미 동적)
   ```python
   # Line 20-23
   if hasattr(obs_space, "original_space"):
       input_size = obs_space.original_space["observations"].shape[0]
   ```
   → 177로 자동 조정됨

2. ✅ `hand_index_pos` 업데이트
   ```python
   self.hand_index_pos = 138  # 변경 없음 (여전히 138 위치)
   ```

3. ✅ `fc_input_size` 재계산
   ```python
   # 176 - 1 (hand_index) + 6 (embedding) = 181
   fc_input_size = input_size - 1 + 6
   ```

---

## ✅ Phase 3: 검증 계획

### 테스트 전략

#### Unit Tests

**파일**: `tests/test_obs_builder.py` (새로 생성)

```python
import pytest
from poker_rl.utils.obs_builder import (
    canonicalize_suits,
    normalize_chips_log,
    _get_street_context_features
)

def test_canonicalize_suits():
    """Suit canonicalization 정확성"""
    from poker_engine import Card
    
    # Test 1: 동일한 무늬 조합은 같은 결과
    hand1 = [Card('H', 'A'), Card('H', 'K')]
    board1 = [Card('H', 'Q'), Card('H', 'J'), Card('C', '2')]
    
    hand2 = [Card('S', 'A'), Card('S', 'K')]
    board2 = [Card('S', 'Q'), Card('S', 'J'), Card('C', '2')]
    
    canonical1 = canonicalize_suits(hand1, board1)
    canonical2 = canonicalize_suits(hand2, board2)
    
    assert canonical1 == canonical2, "Same pattern should canonicalize identically"

def test_log_normalization():
    """로그 정규화 범위 확인"""
    assert 0.0 <= normalize_chips_log(5) <= 1.0
    assert normalize_chips_log(500) == pytest.approx(1.0)
    
    # 해상도 향상 확인
    val_5 = normalize_chips_log(5)
    val_10 = normalize_chips_log(10)
    assert (val_10 - val_5) > 0.05, "Should have good resolution at small values"

def test_observation_shape():
    """최종 observation 차원 확인"""
    # Mock game object
    obs_dict = ObservationBuilder.get_observation(
        mock_game, 0, {}, start_stacks=[10000, 10000]
    )
    assert obs_dict["observations"].shape == (176,)
```

#### Integration Test

```python
def test_full_episode():
    """전체 에피소드 실행"""
    env = PokerMultiAgentEnv()
    obs, info = env.reset()
    
    for _ in range(100):
        # Random actions
        current_player = env.game.get_current_player()
        action_mask = obs[f"player_{current_player}"]["action_mask"]
        legal_actions = np.where(action_mask > 0)[0]
        action = np.random.choice(legal_actions)
        
        obs, rewards, dones, truncated, info = env.step({f"player_{current_player}": action})
        
        # Validate observation shape
        for player in obs:
            assert obs[player]["observations"].shape == (176,)
        
        if dones["__all__"]:
            break
```

---

## 📊 Phase 4: 예상 효과

### 학습 효율 개선

| 개선 | 현재 필요 스텝 | 개선 후 | 향상 배수 |
|------|--------------|---------|---------|
| **Suit Canonicalization** | 4N | N | 4× |
| **Action History** | 5-10N | N | 5-10× |
| **Log Normalization** | 1.5N | N | 1.5× |
| **Position Features** | 2N | N | 2× |
| **복합 효과** | ~10-20N | N | **10-20×** |

### 구체적 예상

- **현재 상태**: 3.5M 스텝에 Q-5o 프리플랍 콜
- **개선 후 1M 스텝**: 
  - ✅ 기본 핸드 선택 정확도 90%+
  - ✅ 포지션 인식 80%+
  - ✅ Aggressor 역할 구분
  
- **개선 후 2-3M 스텝**:
  - ✅ 현재 10M+ 스텝 수준 도달
  - ✅ 체크-레이즈, 3-bet 전략 학습

---

## ⚠️ 주의사항

### Breaking Changes

1. **기존 체크포인트 사용 불가**
   - observation shape 변경 (150 → 176)
   - 처음부터 재학습 필요

2. **get_observation() 시그니처 변경**
   - 이전: `get_observation(game, player_id, action_history)`
   - 이후: `get_observation(game, player_id, action_history, start_stacks=None)`
   - `env.py`의 모든 호출부 업데이트 필요

2. **Equity Calculator 호환성**
   - `canonicalize_suits()`가 equity calculator와 동일한 방식 사용 확인
   - 무늬 정규화 후에도 HS/PPot/NPot 정확히 계산되는지 검증

3. **Action History 구조 변경 (Breaking Change)**
   ```python
   # env.py - 기존
   (action_idx, player_id, bet_ratio)  # 3개 요소
   
   # env.py - 변경 후
   (action_idx, player_id, bet_ratio, bet_amount)  # 4개 요소
   ```
   - `_record_action()` 호출부 확인 필요
   - 다른 곳에서 `action_history` unpacking 하는지 검증

4. **Action History 버그 수정 필수**
   ```python
   # env.py Line 157 - 버그 수정
   self.action_history = {
       'preflop': [],
       'flop': [],
       'turn': [],      # 중복 제거
       'river': []
   }
   ```

---

## 📅 구현 일정 (예상)

| Phase | 작업 | 소요 시간 |
|-------|------|----------|
| 1 | obs_builder.py 구현 | 4-6 시간 |
| 2 | env.py, masked_lstm.py 수정 | 1-2 시간 |
| 3 | 테스트 작성 및 검증 | 2-3 시간 |
| 4 | 학습 시작 및 모니터링 | 1 시간 |
| **총계** | | **8-12 시간** |

---

## 🚀 다음 단계

1. ✅ Implementation Plan 검토 및 승인
2. ⏭️ [task.md](file:///C:/Users/99san/.gemini/antigravity/brain/3dcd237c-db21-4665-a891-41023a127605/task.md) 체크리스트 따라 구현
3. ⏭️ 각 개선 단계별 테스트
4. ⏭️ 통합 후 학습 재시작
5. ⏭️ 성능 모니터링 및 비교 분석
