# 실험 진행 상황

전체 연구 설계는 [`experiment_context.md`](./experiment_context.md) 참고. 여기서는 진행 단계만 추적한다.

Deliverable Order (설계 문서 7절 기준) — 각 단계는 이전 단계에서 kriging이 해당 케이스에서 정상적으로 calibrate됨을 확인한 뒤에만 다음으로 넘어간다. 특정 케이스에서 kriging 자체가 calibrate되지 않으면 그건 결과가 아니라 케이스 설정 오류로 간주.

- [ ] 1. Base case 재현 — 기존 single-model 결과를 새 코드베이스에서 재현
- [ ] 2. Variogram range 변화 실험
- [ ] 3. Nugget 변화 실험
- [ ] 4. Extrapolation / data configuration 변화 실험
- [ ] 5. Anisotropy 변화 실험
- [ ] TODO (신규, 미확정): axis 5 — sample count/sparsity 변화 실험. 구체적 단계는 base case 결과를
      보고 결정 예정 ([experiment_context.md](./experiment_context.md) 3절 참고)

## 로그
| 날짜 | 단계 | 상태 | 비고 |
|---|---|---|---|
| 2026-09-11 | - | 연구 handoff 문서 수령, 프로젝트에 반영 | 아직 구현 시작 전 |
| 2026-09-12 | 1 | Base case 파라미터 확정: grid 50x50 (20m cell, 1000x1000m 도메인), porosity mean=15.0/stdev=3.0, isotropic range≈300m, low nugget, 100 samples(10x10 규칙 격자, interior) | axis 5(sample count) TODO 추가 |
| 2026-09-13 | 1 | 샘플링 파라미터 변경: 이후 4개 방법 비교 실험(RBF+bootstrap, GP-MLE)에서는 100개(10x10 규칙 격자) 대신 500개(20%, 랜덤 interior sampling)로 변경됨 — 사용자 결정. `src/experiments/base_case_conditioning.py`의 `N_SAMPLES=500`이 단일 소스 | 위 100개 규칙 격자 기록은 base_case.py 자체(truth 생성)에는 영향 없음 — 참고용으로 유지 |
| 2026-09-13 | 1 | RBF+bootstrap 방법론 확정: (A) bootstrap replicate 수는 파이프라인 검증 단계인 지금은 `N_BOOTSTRAP=10` 유지, 논문용 최종 결과 산출 시 100~1000으로 상향 예정 — 아직 미확정이므로 그때 다시 결정. (B) bootstrap replicate마다 RBF 하이퍼파라미터(ε, smoothing)를 재튜닝하지 않고, 원본 454개 샘플로 한 번 CV-MSE 튜닝한 값을 10개 replicate 전부에 고정 재사용 — "bootstrap은 샘플셋 변동성만 포착한다"는 `experiment_context.md`의 Claim 1 서술과 부합하는 설계로 최종 확정(사용자 결정, 재론 불필요) | RBF+bootstrap/GP-MLE 코드는 이 결정대로 이미 구현되어 있음 (변경 없음) |
