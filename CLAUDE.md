# SpaceX 논문 지원 프로젝트

## 목표
최종 목표는 **논문 작성**이며, 이 저장소의 코드는 논문에 필요한 시뮬레이션/실험/데이터 분석을 지원하기 위한 것이다.

## 현재 연구: Spatial Prediction 불확실성 정량화 벤치마크

전체 연구 설계는 [`docs/experiment_context.md`](docs/experiment_context.md)에 있다 — **코드를 작성하기 전에 반드시 전체를 읽는다.** 진행 상황은 [`docs/progress.md`](docs/progress.md)에서 추적한다.

핵심 요약:
- **주장 1**: RBF + bootstrap은 불확실성을 체계적으로 과소평가한다 (bootstrap은 샘플셋 변동성만 포착, 공간 전체의 불확실성은 못 잡음).
- **주장 2**: GPR의 자동 하이퍼파라미터 선택(marginal likelihood / CV-MSE)은 정확도에는 최적이지만 calibration은 보장하지 않는다 — 특히 nugget을 신호로 잘못 학습해 불확실성을 과소평가할 위험.
- **기준선(정답)**: simple kriging(추정), SGS(시뮬레이션) — 이 둘의 불확실성 모델은 추정된 공간통계량에서 직접 유도되므로 "정답" 취급.
- **범위 밖**: accuracy와 uncertainty를 함께 최적화하는 objective function은 이번 연구에서 구현하지 않는다 (future work).
- **현재 약점 & 이번 실험의 목적**: 기존 결과가 단일 ground truth 모델 기반이라 cherry-picking 반론이 가능함. 새 실험은 ground truth의 속성(range, nugget, extrapolation 정도, anisotropy)을 **한 번에 하나씩(one-factor-at-a-time)** 바꿔가며 각 방법이 "언제, 왜" 깨지는지 규명하는 것이 목표. "여러 모델을 확인해봤다"가 아니라 "지배 변수를 하나씩 통제해 실패 조건을 특정했다"는 프레이밍이 핵심.
- **비교 대상**: simple kriging / SGS / RBF+bootstrap / GPR, 4개 방법 모두 동일한 샘플 위치·동일한 replicate 수로 비교.
- **평가**: accuracy(MSE/RMSE/MAE)와 uncertainty quality(coverage, interval width, proper scoring rule)를 **분리해서** 보고 — 둘이 괴리될 수 있음을 보이는 게 논문의 핵심 논지이므로 절대 하나로 뭉뚱그리지 않는다.
- **구현 시 필수 요구사항**: 시드 고정·기록, 케이스별 다중 ground truth realization, 방법 간 동일 샘플 위치, 결과는 tidy long-format 테이블(case, axis, axis level, method, metric, value)로 저장해 재실행 없이 그림 재생성 가능하게.

이 요약은 빠른 참조용이며, 세부 파라미터·근거·기대 결과 패턴은 항상 `docs/experiment_context.md` 원문을 확인한다.

## 코딩 컨벤션
- **패키지**: kriging/SGS는 [geostatspy](https://github.com/GeostatsGuy/GeostatsPy)를 사용한다 (RBF 비교군은 `scipy`, GPR은 `scikit-learn`).
- **스타일 기준**: [`docs/geostatspy_conventions.md`](docs/geostatspy_conventions.md) — Michael Pyrcz의 *GeostatsPy Demos Book*(https://geostatsguy.github.io/GeostatsPyDemos_Book/intro.html)에서 정리한 컨벤션. geostatspy 관련 코드를 쓰거나 검수할 때 항상 이 문서를 기준으로 한다. 새로운 패턴이 필요하면 임의로 만들지 말고 먼저 원본 book에서 해당 챕터를 확인한다.
- 의존성은 `requirements.txt`에 고정한다 (`.venv` 기준, Python 3.8).

## 팀 구조
이 프로젝트는 4개 역할로 나뉜다.

| 역할 | 형태 | 하는 일 |
|---|---|---|
| 조율자 | 메인 세션(이 CLAUDE.md를 읽는 나) | 사용자와 논문 방향·실험 설계 논의, 아래 서브에이전트 오케스트레이션, 진행상황 종합 보고 |
| `coder` | 서브에이전트 (`.claude/agents/coder.md`) | 시뮬레이션/분석 코드 작성, 저장 컨벤션 준수 |
| `reviewer` | 서브에이전트 (`.claude/agents/reviewer.md`) | 코드 검수. **애매한 판단은 절대 스스로 내리지 않고 사용자에게 보고** |
| `data-curator` | 서브에이전트 (`.claude/agents/data-curator.md`) | 누적된 결과 감사·집계·논문용 가공. 이것도 애매한 판단은 사용자에게 보고 |

기본 흐름: 사용자와 방향 논의 → `coder` 구현 → `reviewer` 검수 (문제 있으면 `coder`가 수정, 판단 필요 사항은 사용자에게 보고 후 확정) → 시뮬레이션 실행 → 결과 누적 → 필요 시 `data-curator`로 정리·가공.

## 디렉토리 구조
```
SPACEX/
├── CLAUDE.md
├── .claude/agents/          # coder, reviewer, data-curator 정의
├── src/                     # 시뮬레이션/분석 코드
├── results/
│   ├── raw/                 # 원본 시뮬레이션 출력 (git 추적 안 함, 용량 큰 경우가 많음)
│   │   └── <experiment>/<timestamp>/manifest.json
│   ├── processed/           # data-curator가 가공한 논문용 데이터 (git 추적)
│   └── figures/             # 논문용 그림 (git 추적)
├── docs/                    # 논문 초안, 실험 노트
└── tests/
```

## 결과 저장 컨벤션
모든 시뮬레이션 실행은 `results/raw/<experiment_name>/<timestamp>/`에 저장하고, 다음 필드를 포함한 `manifest.json`을 반드시 남긴다: `experiment`, `timestamp`, `git_commit`, `params`, `seed`, `code_entrypoint`, `output_files`. 자세한 내용은 `.claude/agents/coder.md` 참고.

## 원칙
- `reviewer`와 `data-curator`는 확신이 없는 사안을 임의로 결론짓지 않는다. "명확한 문제/사실"과 "판단이 필요한 사항"을 분리해서 보고하고, 후자는 항상 사용자 확인을 거친다.
- `results/raw/`는 원본이므로 어떤 에이전트도 사후에 수정하지 않는다. 가공은 항상 `results/processed/`에 새로 생성한다.
