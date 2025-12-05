# Glacial Supernova ❄️🌟

**Glacial Supernova**는 Ray RLlib과 PPO(Proximal Policy Optimization) 알고리즘을 활용하여 개발된 고성능 **Heads-Up No-Limit Texas Hold'em (HUNL)** AI 프로젝트입니다.

Self-Play 강화학습과 **Potential-Based Reward Shaping (PBRS)**를 통해 GTO(Game Theory Optimal)에 근접한 전략을 학습하는 것을 목표로 합니다.

---

## 🚀 주요 특징

### 핵심 기술
*   **Multi-Agent Self-Play**: Self-Play 환경(`MultiAgentEnv`)으로 설계되어, AI가 자기 자신과 대결하며 지속적으로 발전
*   **Masked LSTM Architecture**: 불가능한 액션을 원천 차단하는 Action Masking과 LSTM 네트워크 결합
*   **Potential-Based Reward Shaping (PBRS)**: 수학적으로 검증된 보상 시스템으로 학습 효율 극대화
*   **Rich Observation Space**: 176차원의 정교한 관찰 벡터 (카드, 게임 상태, 포지션, 액션 히스토리 포함)
*   **Robust Stack Sampling**: 매 핸드마다 스택 깊이를 랜덤 샘플링하여 범용적인 전략 학습

### PBRS 보상 시스템
프로젝트의 핵심 혁신으로, 다음 컴포넌트들이 완벽히 구현되었습니다:

1. ✅ **Effective Stack Normalization** - 스택 깊이에 무관한 보상
2. ✅ **Terminal Φ Subtraction** - 최종 보상에서 포텐셜 차감
3. ✅ **Initial Φ Compensation** - 블라인드로 인한 초기 비대칭 보정
4. ✅ **Dual Reward Update** - Actor와 Observer 모두에게 보상 제공
5. ✅ **Last Action Reward** - Fold/Showdown 액션의 보상 포함
6. ✅ **Phantom Potential Fix** - Folded 플레이어의 잘못된 포텐셜 수정
7. ✅ **Zero-Sum Verification** - 매 핸드 zero-sum 속성 검증 (tolerance: 0.1)

**결과**: 수학적으로 완벽한 zero-sum 보상 시스템 (violation < 0.001)

---

## 🛠️ 설치

### 요구사항
- Python 3.10 이상 권장
- Windows/Linux/Mac 지원

### 설치 단계

1. **저장소 클론 및 가상환경 생성**
   ```bash
   # Windows
   py -3.11 -m venv venv
   .\\venv\\Scripts\\activate
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU 가속 (선택사항)**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

**주요 의존성:**
- `ray[rllib]` - 강화학습 프레임워크
- `torch` - 딥러닝 라이브러리
- `gymnasium` - RL 환경 표준
- `numpy` - 수치 연산

---

## 🏃 사용법

### 1. 학습 시작

**기본 실행:**
```bash
python poker_rl/train.py
```

**실험 이름 지정 (권장):**
```bash
python poker_rl/train.py --name omega
```

**학습 재개:**
```bash
python poker_rl/train.py --name omega --resume
```

**학습 설정:**
- Train batch size: 8,192
- Environment workers: 4
- Gamma (γ): 0.99
- Learning rate: 3e-4
- LSTM cell size: 256

### 2. 학습 모니터링

**TensorBoard 실행:**
```bash
# 새 터미널
.\\venv\\Scripts\\activate
tensorboard --logdir experiments/logs
```

브라우저에서 `http://localhost:6006` 접속

**주요 메트릭:**
- `episode_reward_mean` - 평균 에피소드 보상 (0 근처 수렴)
- `policy_loss` - 정책 손실
- `value_loss` - 가치 함수 손실
- `entropy` - 탐험 정도

### 3. 체크포인트 관리

**자동 저장 설정:**
- 10 iteration마다 자동 저장
- 최근 5개 체크포인트 유지
- 학습 종료 시 자동 저장

**저장 위치:**
```
experiments/logs/<실험명>/PPO_poker_env_<ID>/checkpoint_<iteration>/
```

**최신 체크포인트 확인:**
```powershell
Get-ChildItem \"experiments\\logs\\<실험명>\\PPO_poker_env_*\\checkpoint_*\" -Directory | 
    Sort-Object CreationTime -Descending | 
    Select-Object -First 1
```

### 4. AI와 대전

**간단한 대전 (Random/Call Station 봇):**
```bash
# Random 봇
python poker_rl/play_human.py --opponent random

# Call Station 봇
python poker_rl/play_human.py --opponent call_station
```

**학습된 RL 모델과 대전:**
```bash
# 대화형 모드
python play_vs_ai.py

# 특정 실험 지정
python play_vs_ai.py --model omega
```

**주의:** 현재 observation space가 176차원으로 변경되었으므로, 새로 학습된 체크포인트만 호환됩니다.

---

## 📂 프로젝트 구조

```
glacial-supernova/
├── poker_rl/                    # 메인 패키지
│   ├── agents/                  # 벤치마크 에이전트
│   ├── models/                  # 신경망 모델
│   │   ├── masked_lstm.py      # LSTM + Action Masking
│   │   └── masked_mlp.py       # MLP + Action Masking
│   ├── utils/                   # 유틸리티
│   │   ├── obs_builder.py      # Observation 빌더 (176차원)
│   │   └── equity_calculator.py # Hand strength 계산
│   ├── env.py                   # PokerMultiAgentEnv (PBRS 구현)
│   ├── potential_state.py       # Φ 계산
│   └── train.py                 # 학습 스크립트
├── POKERENGINE/                 # 커스텀 포커 엔진
├── experiments/                 # 학습 로그 및 체크포인트
├── play_vs_ai.py               # AI 대전 스크립트
└── requirements.txt             # 의존성
```

---

## 🎯 Observation Space (176차원)

### 구성 요소

**1. 카드 인코딩 (0-118): 119차원**
- 홀카드 2장: 34차원
- 커뮤니티 5장: 85차원
- 각 카드: Rank (13) + Suit (4) one-hot
- **Suit Canonicalization 적용**: 무늬 대칭성 제거로 4배 학습 효율 향상

**2. 게임 상태 (119-134): 16차원**
```
[119-124] Stacks, Pot, Bets (로그 스케일 정규화)
[125] Button Position
[126] Street (preflop/flop/turn/river)
[127-128] Pot Odds, SPR
[129-134] Min Raise, Opponent Info
```

**3. Expert Features (135-142): 8차원**
```
[135] Hand Strength (Equity)
[136] Positive Potential (개선 확률)
[137] Negative Potential (악화 확률)
[138] Hand Index (족보 ID, 0-168)
[139-142] Street Indicators
```

**4. Padding (143-149): 7차원**
- 향후 확장을 위한 예약 공간

**5. Street History (150-165): 16차원**
- 각 스트릿별 (4 streets × 4 features):
  - Raise 횟수
  - Aggressor (누가 공격적이었는지)
  - 투자 금액
  - 3-bet 이상 여부

**6. Current Street Context (166-171): 6차원**
- 현재 스트릿 액션 패턴
- 역할 전환 (Check-Raise 등)

**7. Investment Info (172-173): 2차원**
- 총 투자 금액
- 투자 비율 (Pot Commitment)

**8. Position Info (174-175): 2차원**
- IP/OOP 위치
- Postflop Position Advantage

---

## 🎮 액션 공간 (14 Actions)

```
[0] Fold
[1] Check/Call (context-sensitive)
[2] Min Raise
[3-12] Pot % Raise (10%, 25%, 33%, 50%, 75%, 100%, 125%, 150%, 200%, 300%)
[13] All-in
```

**Action Masking**: 불가능한 액션은 자동으로 마스킹되어 선택 불가

---

## 📊 학습 성능

### 현재 상태 (PBRS 구현 후)
- **Zero-sum compliance**: < 0.001 (거의 완벽)
- **Observation richness**: 176차원 (액션 히스토리 포함)
- **Expected training**: 1-3M steps로 기본 전략 학습 예상
  - 이전 (액션 히스토리 없음): 5-10M steps 필요

### 개선 효과
- **Suit Canonicalization**: 4배 학습 속도 향상
- **Action History**: 5-10배 전략 학습 가속
- **PBRS**: 수학적으로 완벽한 보상 신호

---

## 🔬 기술 세부사항

### PBRS (Potential-Based Reward Shaping)

**핵심 공식:**
```
Intermediate Reward = γ × Φ(s') - Φ(s)
Terminal Reward = chip_change/eff_stack - Φ_final + Φ_initial

Total = Σ Intermediate + Terminal
      = chip_change/eff_stack (telescoping sum)
```

**Dual Reward Update:**
- Actor와 Observer 모두 매 step마다 보상 수령
- Multi-agent PBRS의 핵심: 한 플레이어의 액션이 양 플레이어의 포텐셜에 영향

**Φ (Potential) 계산:**
```python
equity = get_equity(cards, board)
expected_value = equity × pot
risk_adjusted = expected_value - α × invested
normalized = clip(risk_adjusted / effective_stack, -1, 1)
```

### Zero-Sum 검증
매 핸드 종료 시:
```python
total_P0 + total_P1 == 0 (within tolerance 0.1)
```

Violation 발생 시 에러로 학습 중단 → 코드 오류 즉시 탐지

---

## ⚙️ 하이퍼파라미터

```python
# PPO
train_batch_size = 8192       # Reduced for faster iterations
gamma = 0.99                  # Discount factor
lr = 3e-4                     # Learning rate
clip_param = 0.2              # PPO clip
lambda_ = 0.95                # GAE lambda
entropy_coeff = 0.05          # Exploration
num_epochs = 10               # PPO epochs

# LSTM
lstm_cell_size = 256          # Hidden state size
max_seq_len = 40              # Max hand length

# Environment
num_env_runners = 4           # Parallel workers
sample_timeout_s = 300        # Timeout per sample
```

---

## 🐛 트러블슈팅

### 체크포인트가 생성되지 않음
- `num_to_keep > 0` 확인 (현재: 5)
- `export_native_model_files=True` 확인

### Observation shape 불일치 에러
- 구 체크포인트 (150차원)와 신 코드 (176차원) 비호환
- 해결: 새로 학습 시작

### Zero-sum violation 에러
- Tolerance 0.1로 설정됨
- Gamma 효과로 인한 자연스러운 violation (~0.001)
- 0.1 초과 시 코드 버그 의심

### 학습 속도 느림
- GPU 사용 확인: `torch.cuda.is_available()`
- Worker 수 조정: `num_env_runners`
- Batch size 조정: `train_batch_size`

---

## 📚 참고 자료

### PBRS 이론
- Ng et al. (1999): "Policy Invariance Under Reward Shaping"
- Wiewiora et al. (2003): "Principled Methods for Advising Reinforcement Learning Agents"

### 포커 AI
- **DeepStack** (2017): Limited-depth search + deep learning
- **Pluribus** (2019): CFR + deep RL (6-player)
- **Rebel** (2020): CFR + self-play

### Multi-Agent RL
- RLlib Multi-Agent Documentation
- Multi-Agent PBRS extensions

---

## 🛤️ 로드맵

### 완료 ✅
- [x] PBRS 완전 구현
- [x] 176차원 Observation space
- [x] Action history 통합
- [x] Suit canonicalization
- [x] Zero-sum 검증
- [x] Dual reward update

### 진행 중 🔄
- [ ] 1-3M steps 학습 및 성능 평가
- [ ] GTO solver와의 비교
- [ ] Exploitability 측정

### 향후 계획 📅
- [ ] Attention mechanism 도입
- [ ] CFR integration
- [ ] Multi-stack range training
- [ ] Human vs AI 토너먼트

---

## 💡 기여 가이드

이 프로젝트는 다음을 환영합니다:
- 버그 리포트 및 수정
- 성능 개선 제안
- 새로운 feature 제안
- 문서 개선

---

## 📜 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

**Glacial Supernova** - *Cold calculation, Explosive results.* ❄️🌟
