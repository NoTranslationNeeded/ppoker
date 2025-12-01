# Min-Raise 처리 가이드

## 🚨 문제: The Min-Raise Trap

### 시나리오

```
팟 사이즈: 100
상대 베팅: 50
최소 레이즈: 150 (50 + 50의 2배)

AI 선택: Bet 33% (100 × 0.33 = 33 추가)
→ Total: 83
→ Min-Raise(150) 미달! ❌
```

### 문제점

```python
# ❌ 강제 Min-Raise(150)?
AI: "33% 눌렀는데 150%가 나갔네?"
→ 미세한 베팅 컨트롤 학습 실패

# ❌ 강제 Call?
AI: "공격하려 했는데 수비적으로?"
→ 의도 왜곡
```

---

## ✅ 해결책: 스마트 보정

### 규칙

```python
계산된 베팅 < Min-Raise일 때:

if 베팅 >= Min-Raise × 0.5:
    → Min-Raise로 보정 (공격 의도 유지)
else:
    → Call로 보정 (약한 공격은 수비로)
```

### 구현

```python
def _map_action(self, action_idx: int, player_id: int) -> Action:
    """
    액션 인덱스를 POKERENGINE Action으로 매핑
    Min-Raise 처리 포함
    """
    player = self.game.players[player_id]
    pot = self.game.get_pot_size()
    to_call = self.game.current_bet - player.bet_this_round
    min_raise = self.game.min_raise
    
    # Fold
    if action_idx == 0:
        return Action.fold()
    
    # Check/Call
    elif action_idx == 1:
        if to_call == 0:
            return Action.check()
        else:
            return Action.call(to_call)
    
    # All-in
    elif action_idx == 6:
        return Action.all_in(player.chips)
    
    # Bet/Raise (2-6): 33%, 50%, 75%, 100%, 150%
    else:
        pcts = np.array([0.33, 0.50, 0.75, 1.0, 1.5])
        pct = pcts[action_idx - 2]
        intended_bet = pot * pct
        
        # === Min-Raise 처리 ===
        
        # 이미 베팅이 있는 경우 (Raise 상황)
        if self.game.current_bet > 0:
            # 최소 레이즈 금액 계산
            min_raise_total = self.game.current_bet + min_raise
            intended_total = self.game.current_bet + intended_bet
            
            # ⭐ Min-Raise 체크
            if intended_total < min_raise_total:
                # 의도한 금액이 Min-Raise의 50% 이상?
                if intended_bet >= min_raise * 0.5:
                    # 공격 의도 유지: Min-Raise로 보정
                    target = min_raise_total
                    actual_action = "MIN_RAISE_CORRECTION"
                else:
                    # 약한 공격: Call로 보정
                    return Action.call(to_call)
            else:
                # Min-Raise 이상: 정상 처리
                target = intended_total
                actual_action = "NORMAL_RAISE"
            
            # All-in 체크
            max_bet = player.chips + player.bet_this_round
            if target >= max_bet:
                return Action.all_in(player.chips)
            else:
                return Action.raise_to(target)
        
        # 베팅이 없는 경우 (Bet 상황)
        else:
            # 최소 베팅 = BB
            intended_bet = max(intended_bet, self.big_blind)
            
            # All-in 체크
            if intended_bet >= player.chips:
                return Action.all_in(player.chips)
            else:
                return Action.bet(intended_bet)
```

---

## 📊 보정 예시

### 예시 1: Min-Raise 보정

```
팟: 100
상대 베팅: 50
Min-Raise: 50 추가 (Total 150)

AI: Bet 75% (100 × 0.75 = 75 추가)
→ Total: 125
→ Min-Raise(150) 미달

체크: 75 >= 50 × 0.5? Yes (75 >= 25)
→ Min-Raise로 보정: 150 ✅

다음 관찰:
- obs[bet_ratio] = 150 / 100 = 1.5
- AI가 보정 사실을 학습함
```

### 예시 2: Call 보정

```
팟: 100
상대 베팅: 50
Min-Raise: 50 추가 (Total 150)

AI: Bet 33% (100 × 0.33 = 33 추가)
→ Total: 83
→ Min-Raise(150) 미달

체크: 33 >= 50 × 0.5? No (33 < 25)
→ Call로 보정 ✅

다음 관찰:
- Last action = Call
- AI가 "약한 공격은 Call"을 학습
```

### 예시 3: 정상 처리

```
팟: 100
상대 베팅: 50
Min-Raise: 50 추가 (Total 150)

AI: Bet 150% (100 × 1.5 = 150 추가)
→ Total: 200
→ Min-Raise(150) 이상 ✅

→ 정상 레이즈: 200
```

---

## 🎯 학습 효과

### AI가 배우는 것

1. **Min-Raise 인식**:
   - "33% 베팅은 상황에 따라 Call이 될 수 있구나"
   - Min-Raise 룰을 암묵적으로 학습

2. **보정 패턴**:
   - 약한 레이즈 의도 → Call
   - 중간 레이즈 의도 → Min-Raise
   - 강한 레이즈 의도 → 의도대로

3. **정확한 피드백**:
   - 다음 관찰에서 실제 베팅 금액 확인
   - "내가 75% 눌렀는데 150%가 나갔네" 학습
   - 점진적으로 Min-Raise를 고려한 선택 학습

---

## ⚠️ 중요 사항

### 액션 히스토리 기록

```python
# ✅ 올바른 방법
def step(self, action_dict):
    pot_before = self.game.get_pot_size()
    
    # 액션 실행
    engine_action = self._map_action(action, player)
    self.game.process_action(player, engine_action)
    
    # ⭐ 실제 베팅된 금액 기록 (보정 후)
    actual_bet = self.game.players[player].bet_this_round - bet_before
    self._record_action(action, player, actual_bet, pot_before)
    
    # AI가 다음 턴에서 보정된 금액을 관찰
```

### Action Masking

```python
# Min-Raise 때문에 Masking 복잡해질까?
# → No! 

# Bet/Raise 자체는 legal
# 단지 금액만 보정될 뿐
# Masking에는 영향 없음 ✅
```

---

## 📝 체크리스트

- [ ] _map_action에 Min-Raise 체크 추가
- [ ] 50% 기준으로 Min-Raise vs Call 분기
- [ ] 실제 베팅 금액을 액션 히스토리에 기록
- [ ] 다음 관찰에서 보정된 금액 확인 가능
- [ ] 테스트: 33% 베팅 → Call 보정 확인
- [ ] 테스트: 75% 베팅 → Min-Raise 보정 확인
