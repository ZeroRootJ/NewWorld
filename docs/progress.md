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
| 2026-09-14 | 1 | `numba` 미설치로 `geostatspy.geostats` import 자체가 실패하던 환경 문제 발견·해결 (`requirements.txt`에는 있었으나 `.venv`에 누락) | `results/raw`가 그동안 비어있던 이유로 추정 (RBF+bootstrap/GP-MLE도 truth 생성 경로에서 간접적으로 geostats.sgsim을 타므로 동일하게 영향받음) |
| 2026-09-14 | 1 | Simple kriging(`src/experiments/kriging.py`), SGS(`src/experiments/sgs.py`) 구현 완료 — base case 4개 방법(kriging/SGS/RBF+bootstrap/GP-MLE) 모두 구현 완료. kriging: NS(normal-score) 공간에서 `ktype=0`으로 `kb2d` 수행, point estimate는 `backtr_value`로 물리 단위 역변환, variance는 NS 단위 유지(비선형 역변환 미실시, 명시적으로 라벨링). SGS: 조건 데이터 좌표가 grid node와 정확히 겹쳐 `sgsim` 내부에서 특이행렬 에러가 나는 문제를 발견 → `sgsim` 호출용 DataFrame 사본에만 국소 jitter(1e-2m, `SGS_JITTER_SEED`) 적용 + post-hoc honor reinforcement(사후 원본 값 강제 재할당)로 해결, 사용자 승인됨(재론 불필요). `N_REALIZATIONS`는 `rbf_bootstrap.N_BOOTSTRAP`(=10) 재사용 | reviewer 검수에서 kriging.py의 `tmin/tmax`(kb2d 조건 데이터 필터) 버그(엉뚱한 상수 재사용으로 샘플 1개 침묵 드롭) 발견 → 즉시 수정·재검증 완료(454/454 전부 조건화 확인, 향후 재발 시 `RuntimeError`로 감지되도록 가드 추가) |
| 2026-09-14 | 1 | 샘플링 비율 재변경: 문헌조사(`docs/references.md`) 결과를 반영해 500개(20%) → **125개(5%, 2500-cell 중)**로 축소, 사용자 결정. `N_SAMPLES=125`가 단일 소스. 4개 방법(kriging/SGS/RBF+bootstrap/GP-MLE) 전부 재실행 완료(dedup 후 실사용 121개, 4개 방법 일치 확인) | 이전 500-샘플 run들은 `results/raw/`에 원본으로 보존, 수정 안 함 |
| 2026-09-14 | 1 | 평가 모듈 신규 구현: `src/evaluation.py`(MSE, UMG — 참고 코드베이스 `calc_UMG.py`/`dashboard.py` 로직 이식, kriging은 quantile back-transform 방식으로 NS↔물리 단위 불일치 없이 정확히 처리) + `src/experiments/evaluate_base_case.py`(4개 방법 tidy table `results/processed/base_case/metrics.csv` 생성, 조건샘플 cell은 4개 방법 공통으로 평가에서 제외) | **1차 결과(reviewer 검수 전, 잠정)**: MSE는 4개 방법이 3.31~3.73으로 비슷. UMG는 kriging 0.949, SGS 0.904, GP-MLE 0.961, RBF+bootstrap 0.528 — RBF+bootstrap의 심각한 과소평가(Claim 1과 부합)는 뚜렷하나, base case(낮은 nugget)에서는 GP-MLE가 kriging보다 UMG가 오히려 더 높게 나옴(Claim 2가 기대하는 패턴은 nugget이 커지는 axis 2/3에서 더 뚜렷해질 것으로 예상 — 해석은 reviewer 검수 및 사용자 확인 후 논의) |
| 2026-09-14 | 1 | reviewer 검수: 평가 모듈에 기능적 버그 없음 확인(참고 코드 수식과 한 글자 단위 대조, 마스크 정합성, 재현성 전부 확인). 판단 필요 사항으로 kriging UMG의 tail-clipping 비대칭 발견 — `p→1` 근처에서 kriging의 quantile back-transform 결과가 물리단위 고정 경계(`BACKTR_ZMIN/ZMAX`)로 수렴해, kriging variance가 큰 cell일수록 조기에 인위적으로 유리해지는 구조적 효과. 사용자 결정: `BACKTR_ZMIN/ZMAX`를 `POR_MEAN±4·STDEV`→`±10·STDEV`(=[-15,45])로 확장. 재실행 결과 kriging MSE·point estimate는 불변(bit-for-bit), UMG는 0.9491→0.9498로 미세 변화(예상대로 base case에선 영향 작음) — `p=1` 지점 자체는 어떤 폭이든 이 방법론의 수학적 특성상 항상 경계값에 수렴하므로 해소 대상 아님(코드 주석에 명시) | nugget/range 축(2, 3번)에서 kriging variance가 커지는 케이스가 늘어나면 이 효과가 더 커질 수 있으니 그때 다시 관찰 필요 |
| 2026-09-14 | 1 | `BACKTR_ZMIN/ZMAX`를 다시 `±10·STDEV`([-15,45]) → `±4·STDEV`([3,27])로 원복(사용자 결정). 사유: 같은 날 신규 추가한 kriging variance 물리단위 Monte Carlo 근사(`kriging_var_map_physical_mc.npy`)에서, 넓은 경계(zmax=45)가 정규점수 변환표 범위 밖 셀의 선형 tail 외삽을 45%까지 뻗게 만들어 해당 셀 물리단위 variance가 61 %²까지 치솟는(다른 곳은 대부분 5 이하) 역효과가 발견됨. `±4σ` 원복 시 위 UMG tail-clipping 문제가 재발할 수 있음을 감수. 새 run `results/raw/kriging/20260914T184654Z`: kriging point-estimate map(`kriging_mean_map_physical.npy`)은 이전 run(`20260914T144604Z`, ±10σ)과 bit-for-bit 동일(base case 셀들이 변환표 interior에 있어 tail bound 영향 없음) → MSE 불변(3.312265). UMG는 0.9498→0.9491로 미세 변화(±10σ→±4σ 원복 시에도 base case에서는 영향 미미, 예상과 부합). Monte Carlo 물리단위 variance 최댓값은 61.17 %² → 9.82 %²로 대폭 감소, `results/processed/base_case/make_truth_predictions_figures.py`의 shared variance 스케일(SGS/RBF/GP-MLE 기준, ~18.10)을 더는 초과하지 않아 kriging variance 패널이 더 이상 clip되지 않음(패널 주석을 clip 여부에 따라 조건부로 표시하도록 수정) | `metrics.csv`/`source_runs.json`/`accuracy_plot_curves.csv`/`accuracy_plot_comparison.png`/`crossplot_comparison.png`/5개 `qc_truth_predictions_*.png` 전부 재생성 완료. 커밋은 오케스트레이터가 검수 후 처리 |
