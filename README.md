# Glacial Supernova ❄️🌟

**Glacial Supernova**는 Ray RLlib과 PPO(Proximal Policy Optimization) 알고리즘을 활용하여 개발된 고성능 **Heads-Up No-Limit Texas Hold'em (HUNL)** AI 프로젝트입니다.

이 프로젝트는 Self-Play 강화학습을 통해 GTO(Game Theory Optimal)에 근접한 전략을 학습하는 것을 목표로 하며, 커스텀 포커 엔진과 정교한 환경 설계를 기반으로 합니다.

## 🚀 주요 특징 (Key Features)

*   **Multi-Agent Self-Play**: 처음부터 Self-Play 환경(`MultiAgentEnv`)으로 설계되어, AI가 자기 자신과 대결하며 지속적으로 발전합니다.
*   **Masked LSTM Architecture**: 불가능한 액션을 원천 차단하는 Action Masking과 게임의 흐름을 기억하는 LSTM 네트워크를 결합하여 유효하고 전략적인 판단을 내립니다.
*   **Robust Stack Sampling**: 매 핸드마다 스택 깊이(Deep, Standard, Middle, Short)를 랜덤하게 샘플링하여, 특정 상황에 편향되지 않는 범용적인 전략을 학습합니다.
*   **Zero-Sum Reward System**: 포커의 제로섬 특성을 완벽하게 반영하여, 한 플레이어의 이득이 정확히 다른 플레이어의 손실이 되도록 설계되었습니다.
*   **Turn-Based Logic**: 동시 액션이 아닌 실제 포커와 동일한 턴제(Turn-based) 방식으로 환경이 구현되었습니다.

## 🛠️ 설치 (Installation)

이 프로젝트는 Python 3.10 이상을 권장합니다.

1.  저장소를 클론합니다.
2.  가상환경을 생성하고 활성화합니다 (권장).

    ```bash
    # Windows
    py -3.11 -m venv venv
    .\venv\Scripts\activate
    ```

3.  필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

> **Note (GPU 사용자)**: GPU 가속을 사용하려면 PyTorch를 CUDA 버전으로 재설치해야 할 수 있습니다:
> ```bash
> pip uninstall torch torchvision torchaudio
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

**주요 의존성:**
*   `ray[rllib]`: 강화학습 프레임워크
*   `torch`: 딥러닝 라이브러리
*   `gymnasium`: 강화학습 환경 표준
*   `numpy`: 수치 연산

## 🏃‍♂️ 실행 방법 (Usage)

### 학습 시작 (Training)

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 학습을 시작합니다.

```bash
py poker_rl/train.py
```

학습 로그와 체크포인트는 `experiments/logs` 디렉토리에 저장됩니다.

### 학습 모니터링 (Monitoring)

TensorBoard를 사용하여 실시간으로 학습 진행 상황(보상, 승률 등)을 확인할 수 있습니다.

```bash
tensorboard --logdir experiments/logs
```

## 📂 프로젝트 구조 (Project Structure)

```
glacial-supernova/
├── poker_rl/               # 메인 패키지
│   ├── agents/             # (Optional) 벤치마크 에이전트
│   ├── models/             # 신경망 모델 (MaskedLSTM 등)
│   ├── env.py              # PokerMultiAgentEnv 환경 정의
│   └── train.py            # 학습 스크립트
├── POKERENGINE/            # 커스텀 포커 게임 엔진
├── experiments/            # 학습 로그 및 체크포인트 저장소
├── POKER_AI_COMPLETE_GUIDE.md # 📖 상세 구현 가이드 (필독)
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명
```

## 📖 상세 문서 (Documentation)

프로젝트의 설계 철학, 보상 함수, 관찰 공간(Observation Space) 등 자세한 내용은 [POKER_AI_COMPLETE_GUIDE.md](POKER_AI_COMPLETE_GUIDE.md) 파일을 참고하십시오.
---
**Glacial Supernova** - *Cold calculation, Explosive results.*
