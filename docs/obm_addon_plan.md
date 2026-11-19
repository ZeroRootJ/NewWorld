# OBM 검증 실험 계획 (초안, 2026-11-19)

목적: 후속 연구가 아니라 논문 마지막에 붙는 짧은 검증입니다. "본문의 주장이 SGS로 만든 ground truth에서만 성립하는 게 아니라, 독립적으로 생성한 channelized ground truth(ALLUVSIM 기반, ResMill `ChannelLayer`)에서도 성립한다"를 보이는 것이 전부입니다. 그림 1장, 본문 한 문단 분량입니다.

## 1. 이미 있는 것 (브랜치 `worktree-obm-ground-truth`, master에 미병합)
- `obm_benchmark/`: 생성 스크립트, QC, config. 별도 `.venv-obm`(Python 3.11, ResMill commit be0d923 고정)을 씁니다.
- ground truth 30개(seed 0–29)가 두 종류 있습니다. QC는 둘 다 30/30 통과입니다.
  - `obm_ground_truth`(기본)
  - `obm_ground_truth_mud_sd`(이수 facies에도 porosity 변동을 줌)
- 격자는 본 실험과 같습니다(50×50, 20 m). slice는 k = 8로 고정했고, slice마다 mean 15, std 3으로 affine 변환했습니다.
- 기본판은 셀의 56 %가 같은 값입니다. mud_sd판은 0.05 %입니다.
- 이 브랜치는 master보다 64 커밋 뒤처져 있습니다. OBM 관련 커밋(26be478, ad7be1f, 62557d2, d037b8f)만 master로 가져와야 합니다.

## 2. 검증할 주장 (본문과 같은 지표, 같은 방식)
| # | 본문 근거 | OBM에서 확인할 것 |
|---|---|---|
| V1 | F1 | MSE는 4개 방법이 비슷하고, UMG는 RBF+bootstrap만 크게 낮다 |
| V2 | F3, P3 | 밀도가 낮아질 때 GP-MLE의 UMG가 variogram 기반 방법(SGS)보다 더 떨어지고, 하위 꼬리(P10)가 더 벌어진다 |

결론 2(F7 grid search)는 OBM에서 반복하지 않습니다. 비용이 크고, 데이터 생성 방식과 무관한 주장이기 때문입니다.

## 3. 설계
- **ground truth**: 30개 중 seed 0–9의 10개를 테스트로 씁니다(본문 축과 같은 수).
- **샘플 위치**: 본문 density 축(E4)과 같은 sampling seed 1001–1010을 써서, 같은 밀도·같은 샘플 위치를 그대로 재사용합니다. 4개 방법 모두 같은 위치를 씁니다.
- **밀도**: 20, 10, 5, 2, 1 % 다섯 단계입니다. 본문 F3, P3와 x축이 같아 그림 코드를 그대로 씁니다.
- **방법**: simple kriging, SGS, RBF+bootstrap, GP-MLE. 설정은 본문과 같고 N = 50입니다. facies는 어떤 방법에도 주지 않습니다.
- **kriging/SGS의 variogram — 사용자가 직접 그림 (2026-11-19 사용자 결정)**:
  - OBM에는 정답 variogram이 없으므로, 사용자가 실험 variogram을 보고 직접 모델을 그립니다(본문 range·nugget을 사람이 정한다는 주장과 같은 방식).
  - 기존 Variogram Review 화면을 재사용합니다. OBM에는 정답 곡선이 없으므로 화면에 표시할 것도 없습니다.
  - 무엇을 보고 몇 번 그릴지(D2)는 아직 정하지 않았습니다.
  - GP-MLE는 지금처럼 샘플만으로 하이퍼파라미터를 맞춥니다.
- **normal-score 기준 분포**: 본문 파이프라인과 같은 원칙을 씁니다. 구현할 때 본문 방식을 확인합니다.
- **지표**: MSE, UMG, coverage90, variance_mean. 평가 함수는 본문과 같습니다. CRPS, interval width는 2026-10-05 결정대로 계산하지 않습니다.
- **규모**:
  - 10 truth × 10 샘플셋 × 5 밀도 = 500셀, 실행은 500 × 4 방법 = 2,000회입니다.
  - E4와 같은 규모라 20 workers로 약 7시간이 걸리고, 밤새 백그라운드로 돌리면 됩니다.

## 4. 결과를 보기 전에 고정할 판정 기준
결과를 확인하기 전에 이 문서에 확정합니다. 결과는 기준 충족 여부와 상관없이 모두 보고합니다.
- **V1**: 모든 밀도에서 RBF+bootstrap의 평균 UMG가 나머지 세 방법보다 낮은가.
- **V2-a**: UMG 평균 감소폭(20 % → 1 %)이 GP-MLE가 SGS보다 큰가.
- **V2-b**: 1 %에서 셀별 UMG의 P10이 GP-MLE < SGS인가.
- **위험**: channelized, non-Gaussian truth에서는 사용자가 그린 variogram을 쓴 SGS 자체가 잘 calibrate되지 않을 수 있습니다. 그 경우도 결과로 보고하며, 그림이나 설정을 바꿔 맞추지 않습니다.

## 5. 그림 (1장, O1)
- A: 테스트 truth 2개의 map과 1 % 샘플 위치(facies 경계 윤곽 포함)
- B: MSE vs 밀도
- C: UMG vs 밀도
- D: 셀별 UMG 분포 vs 밀도(P3 형식)

layout과 색은 본문 F3, P3와 같게 합니다. 아티팩트에는 마지막 섹션 "검증: ALLUVSIM 기반 ground truth"로 넣습니다.

## 6. 순서 (coder → reviewer)
1. OBM 커밋 4개를 master로 가져오고, main checkout에서 `.venv-obm`으로 truth를 재생성합니다. 결정론적이라 1분이면 되고, 기존 run과 sha256이 같은지 확인합니다.
2. D2 범위에 맞춰 OBM용 Variogram Review 화면을 만들고, 사용자가 variogram을 그립니다. 고른 값은 기존처럼 CSV로 내보냅니다.
3. runner를 만듭니다. multi_truth_axes runner를 그대로 쓰고 truth 로더와 variogram 출처만 바꿉니다. 그다음 smoke test(1셀 × 4 방법)를 합니다.
4. 본 실행을 합니다(백그라운드, 약 7시간).
5. 같은 evaluator로 평가하고, O1 그림과 status 파일을 만든 뒤 아티팩트에 넣습니다.

## 7. 사용자 결정이 필요한 것
| # | 항목 | 선택지 | 제안 |
|---|---|---|---|
| D1 | truth 종류 | 기본 / mud_sd | mud_sd. 기본판은 56 %가 같은 값이라 normal-score 변환에서 동률 처리가 결과를 좌우할 수 있습니다 |
| D2 | 사용자가 그릴 variogram의 근거와 횟수 | (a) 셀별 샘플 variogram, rep0만 50개(실행도 그 50셀로 축소) / (b) 셀별 500개 / (c) seed 10–29 전체 map의 analog variogram 1개를 모든 셀에 적용 | 미정. (a)는 밀도당 10셀이라 P10 같은 꼬리 통계가 약해지고, (c)는 한 번만 그리면 되고 본문 oracle과 성격이 같습니다 |
| D3 | variogram 방향성 | 등방(omni) / 이방 | 그리실 때 판단. 본문은 등방이고 F5(이방성)는 보류 중입니다 |
| D4 | 밀도 | 5단계 / 20, 5, 1 % 3단계 | 5단계(약 7시간). 3단계로 줄이면 약 4시간입니다 |
| D5 | 테스트 truth 수 | 10 / 30(brief 최소) | 10(본문 축과 같음) |
| D6 | OBM 코드 가져오기 | 커밋 4개 cherry-pick | 진행 |
