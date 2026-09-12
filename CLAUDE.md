# SpaceX 논문 지원 프로젝트

## 목표
최종 목표는 **논문 작성**이며, 이 저장소의 코드는 논문에 필요한 시뮬레이션/실험/데이터 분석을 지원하기 위한 것이다.

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
