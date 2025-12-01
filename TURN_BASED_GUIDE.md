# 턴제 게임 Multi-Agent 구현 가이드

## 🔄 핵심 원칙: 턴제 게임 특성

포커는 **동시 액션 게임**이 아닌 **턴제 게임**입니다.

### 중요한 차이

```python
# ❌ 동시 액션 게임 (스타크래프트):
step({"p1": action1, "p2": action2})  # 둘 다 동시에
→ return {
    "p1": obs1,  # 둘 다 반환
    "p2": obs2
}

# ✅ 턴제 게임 (포커):
step({"player_0": raise_action})  # P0만 액션
→ return {
    "player_1": obs1  # 다음 턴인 P1만 반환!
}
```

## ✅ 올바른 구현

### step() 반환 규칙

**핵심**: "지금 당장 행동해야 하는 플레이어"의 관찰만 반환!

```python
def step(self, action_dict):
    # action_dict = {"player_0": 3} 또는 {"player_1": 1}
    # 한 명만 액션!
    
    current_player = self.game.get_current_player()
    action = action_dict[f"player_{current_player}"]
    
    # 액션 처리
    self.game.process_action(current_player, action)
    
    # 핸드 진행 중
    if not self.game.is_hand_over:
        next_player = self.game.get_current_player()
        
        # ⭐ 다음 턴 플레이어만 반환!
        return {
            f"player_{next_player}": obs  # 한 명만!
        }, {
            f"player_{next_player}": 0.0
        }, {
            "__all__": False
        }, {}
    
    # 핸드 종료
   else:
        # 두 명 모두 반환 (최종 상태)
        return {
            "player_0": obs0,
            "player_1": obs1
        }, {
            "player_0": reward0,
            "player_1": reward1
        }, {
            "__all__": True
        }, {}
```

## 예시: 실제 플레이 시퀀스

```python
# 핸드 시작
reset()
→ return {"player_0": obs}  # SB가 먼저

# Step 1: P0 raises
step({"player_0": 4})
→ return {"player_1": obs}  # P1 턴

# Step 2: P1 calls
step({"player_1": 1})
→ return {"player_0": obs}  # P0 턴 (Flop)

# Step 3: P0 checks
step({"player_0": 1})
→ return {"player_1": obs}  # P1 턴

# Step 4: P1 bets
step({"player_1": 4})
→ return {"player_0": obs}  # P0 턴

# Step 5: P0 folds
step({"player_0": 0})
→ return {
    "player_0": obs0,  # 핸드 종료: 둘 다
    "player_1": obs1
}
```

## ⚠️ 흔한 실수

```python
# ❌ 잘못된 구현
def step(self, action_dict):
    # 액션 처리
    ...
    
    # 항상 둘 다 반환? NO!
    return {
        "player_0": obs0,
        "player_1": obs1
    }, ...
    # → RLlib이 혼란!

# ✅ 올바른 구현
def step(self, action_dict):
    next_player = self.game.get_current_player()
    
    # 다음 턴 플레이어만!
    return {
        f"player_{next_player}": obs
    }, {
        f"player_{next_player}": 0.0
    }, ...
```

## 📝 RLlib 작동 방식

1. 환경: `{"player_0": obs}` 반환
2. RLlib: "아, player_0이 행동할 차례구나"
3. RLlib: player_0 정책 실행 → action0
4. 환경: `step({"player_0": action0})` 호출
5. 환경: `{"player_1": obs}` 반환
6. RLlib: "이제 player_1 차례"
7. ...반복

## ✅ 체크리스트

- [ ] step()이 현재 턴 플레이어만 반환하는가?
- [ ] reset()이 첫 번째 플레이어만 반환하는가?
- [ ] 핸드 종료 시에만 둘 다 반환하는가?
- [ ] reward도 현재 턴 플레이어만 반환하는가? (진행 중)
- [ ] 핸드 종료 시 둘 다에게 reward 주는가?
