# POKERENGINE - 커스텀 텍사스 홀덤 포커 엔진

강력하고 TDA 규칙을 완벽하게 준수하는 강화학습용 텍사스 홀덤 포커 엔진입니다.

## 🎯 주요 기능

### 1. 완벽한 게임 로직
- **Pre-flop, Flop, Turn, River, Showdown** 전체 단계 구현
- **사이드 팟 (Side Pots)**: 다중 올인 시나리오에서 메인 팟과 여러 사이드 팟을 정확하게 계산 및 분배 (TDA Rule 21)
- **자동 진행 (Auto-advance)**: 모든 플레이어가 올인 상태일 때 쇼다운까지 자동으로 카드 딜링 ("Run it out")

### 2. TDA 2024 규칙 준수 (검증 완료)
- **Rule 34-B**: 헤즈업 블라인드 포지션 (SB = Button) 및 행동 순서
- **Rule 43**: 최소 레이즈 규칙 (50% 룰 포함)
- **Rule 47**: 베팅 재오픈 (Re-opening the bet) - 숏스택 올인이 풀 레이즈 미만일 경우 베팅이 닫힘
- **BB Option**: 프리플랍에서 BB의 체크/레이즈 권한 보장

### 3. 견고성 (Robustness)
- 칩 보존 법칙 검증 (게임 전후 총 칩량 일치)
- 10개 이상의 엣지 케이스 테스트 통과
- `eval7` 라이브러리를 활용한 정확한 핸드 평가

## 📁 디렉토리 구조

```
POKERENGINE/
├── poker_engine/          # 핵심 엔진 패키지
│   ├── __init__.py
│   ├── cards.py          # Card, Deck 클래스
│   ├── evaluator.py      # HandEvaluator (eval7 래퍼)
│   ├── player.py         # Player 상태 및 칩 관리
│   ├── actions.py        # Action 타입 및 유효성 검사 (Rule 47 로직 포함)
│   └── game.py           # PokerGame 메인 로직 (사이드팟, 오토 어드밴스 포함)
└── test_poker_engine.py  # 종합 단위 테스트 (10개 시나리오)
```

## 🚀 사용법

### 게임 초기화 및 실행
```python
from poker_engine import PokerGame, Action

# 게임 생성 (블라인드 설정)
game = PokerGame(small_blind=50, big_blind=100)

# 핸드 시작 (칩 스택 및 버튼 위치 설정)
game.start_hand(player0_chips=1000, player1_chips=1000, button=0)

# 행동 처리
# P0 (SB) Call
game.process_action(0, Action.call(50))

# P1 (BB) Check
game.process_action(1, Action.check())

# 상태 확인
print(f"Street: {game.street}")  # Street.FLOP
print(f"Pot: {game.get_pot_size()}")
```

### 테스트 실행
포함된 `test.ps1` 스크립트를 사용하여 모든 테스트를 실행할 수 있습니다.

```powershell
cd POKERENGINE
.\test.ps1
```

## 📊 테스트 시나리오 (10/10 통과)

1. **Card/Deck**: 카드 생성 및 덱 셔플/딜링
2. **Player**: 칩 베팅, 폴드, 상태 관리
3. **HandEvaluator**: 족보 평가 및 비교
4. **PokerGame Basic**: 블라인드, 베팅, 스트리트 진행
5. **TDA Heads-up**: 헤즈업 포지션 및 순서 (Rule 34-B)
6. **Side Pot Basic**: 불균등 스택 올인 및 팟 분배
7. **Side Pot Showdown**: 쇼다운 시 사이드 팟 승자 결정
8. **Equal Stacks All-in**: 동일 스택 올인 처리
9. **TDA Rule 47**: 숏스택 올인 시 베팅 재오픈 여부 검증 (Full Raise vs Short Raise)
10. **Split Pot**: 무승부 시 팟 분할 로직

---
**Version**: 2.0.0 (Phase 3 Completed)
**Last Updated**: 2025-11-30
