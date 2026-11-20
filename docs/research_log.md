# 연구 노트 (일일 로그)

날짜별로 그날 진행한 작업을 워크플로우 단위로 묶어 짧게 기록한다(질의 하나하나를 1:1로 나열하지
않음 — 관련된 질의/작업을 하나의 흐름으로 묶어서 "무엇을 하려고 했는지"가 보이게). 상세 결정/근거는
`progress.md`에 있고, 여기는 타임라인용. 시각은 기록 시점 기준(실제 질의 시각과 다소 오차 가능).
2026-10-16부터 기록.

## 2026-10-16

**Conditional-expectation 패널 재설계** — Mining Geostatistics 책에서 conditional expectation
위치·binning 관행·회귀선 기울기 해석을 조사(책은 binning 대신 회귀직선 방식, 인터넷은
calibration=equal-count bin 우세/kriging 실무=회귀선)한 뒤, 그 근거로 2x3 비교 보드를
quantile-binning → scatter+OLS 회귀선으로 교체.

**GT vs Z* 분포 비교로 확장** — 같은 6패널을 "분포 모양 자체가 얼마나 가까운가"를 보는
histogram overlay로 재구성(진단용) → 가장 가까운 방법을 정량화하려고 Wasserstein distance,
이어서 KS statistic을 추가하고 패널 간 비교가 되도록 y축 스케일 통일 → 뒷받침 선행연구(distribution
reproduction 지표) 조사해 references.md에 반영 → 결과를 Sample Density Axis 아티팩트에 게시.

**연구노트 세팅** — 이 로그 신규 생성 → 길이/형식 피드백 두 차례 받아 반복 수정(한 줄 요약 →
워크플로우 단위 묶음, 지금 이 형식).

## 2026-11-16

**논문 뼈대 확정 + 스토리보드** — 결론 1-a/1-b/2 논리 정리(`paper_outline.md`) → 기존 데이터로 F1–F7 제작(없는 실험은 placeholder) → reviewer 검수·수정 → 아티팩트 게시.

## 2026-11-17

**F7 response surface** — 옛 SNMS grid 코드 검토 → 현재 case용 runner 작성·검수(결정 8건) → multi_truth_axes 종료 후 smoke 게이트 거쳐 자동 실행(490/490, crosscheck PASS) → F7 실제 데이터로 교체·아티팩트 게시.

## 2026-11-18

**스토리라인 초안** — 연구기록·스토리보드 결과로 논리 ①–⑥ 구성 → 그림 설명을 → 연결 bullet로 재작성, 긴장 지점 T1–T4, placeholder P1/P3/P4/P5/P7 제안 → 아티팩트 게시.

## 2026-11-19

**oracle 전용 + 진단 그림** — variogram 기반 방법은 oracle로만 비교하기로 결정(검증은 OBM 후속) → P1/P3/P4/P7/P7b 계산·검수 → 스토리라인 문구 갱신·아티팩트 게시.

**스토리 재구성** — 사용자: 셀링포인트는 "정확도 기반 모델(최신 ML 포함)은 UQ 미내장 → 사후 UQ는 맞추기 어렵다", RBF가 최단순 대표 → 결론 1(핵심) = F2·F7·P7·P7b, 결론 2 = GP 희소 조건; 아티팩트 V29.

**P7b 제거·결론 2 논조** — 사용자: grid search는 "MSE/UMG 최적이 다르다"까지만(P7b·trade-off 크기 문장 삭제), 결론 2 = 희소 데이터에서 GP 파라미터 흔들림 → realization 생성(variogram·분포 재현)에 치명적 → placeholder P8(implied variogram)·P9(GP realization 재현) 추가; 아티팩트 V30.

**F1 오차막대 제거** — 사용자 요청 → 6개 패널 모두 평균만 표시, status·아티팩트 문구 갱신; V31.

**trade-off 결정 정정** — 사용자: 11-19 결정은 "trade-off 정량화용 별도 실험 불필요"였고 기존 값(RBF UMG 최적 MSE 19.52 등) 언급은 허용 → F7을 "UMG로 최적화해도 답이 아님"(RBF MSE 3.9→19.5) 근거로 쓰는 흐름 제안.

**스토리보드 정리** — 사용자: F2B 오차막대 제거, 회색 DATA STATUS 문구를 그림 밖(아티팩트 "그림 읽는 법")으로, F7 별 범례를 MSE/UMG 면 최적점으로, P1 삭제, 그림 번호 Fig. 1–14(전제 1, 결론 1 = 2–4, 결론 2 = 5–14) → 글자 크기 확대는 보류(사용자); 아티팩트 V32.

**P7(Fig. 4) 삭제** — 사용자 → 스토리보드에서 제외, F7 → Fig. 4(F4) 연결 bullet 이전, T2·③ 문구에서 P7 근거 제거, 번호 Fig. 1–13; V33.

**F6 삭제·F3b 부록** — 사용자 → 파라미터 수 vs 데이터량(F6) 제외, 밀도별 map·샘플 위치(F3b)는 부록 Fig. A1; 본문 Fig. 1–11; V34.

**GP-MLE truth 50개/레벨** — 사용자: Fig. 4 점 개수 레벨당 50개 → GP-MLE만 seed 7110–7149 추가(range 8, nugget 6레벨, 700 적합, 157 s), 기존 10개 재현 ≤1.2e-15, `*_n50.csv`; 실험 대시보드 신설.

**Fig. 4 갱신 + truth variogram** — 사용자: F4를 n50(레벨당 50개)·nugget sill 비율(0–1)로 → V35; realization 변동 확인용 truth 700개 실험 variogram(전체 map·샘플) 계산, 실험 대시보드에 GP range 색으로 표시.

**부록 Fig. A2** — 사용자: truth realization 실험 variogram을 부록에 → A2(레벨별 50개, GP range 비율 색), V36.

**GP noise 하한 1e-10 (1 %)** — 사용자: 하한(1e-5)이 GP-MLE 성능을 끌어올렸는지 → E4 1 % 100셀 GP만 재적합, UMG 평균 0.8895→0.8896, 하한 셀 16개 0.828→0.828; 7108 rep8만 다른 국소해; 대시보드 표시, 스토리보드 그대로.

**결정: noise 하한 이슈 종료** — 사용자: 1e-10 재적합에서 거의 변동 없음 → 하한 관련 이슈는 더 다루지 않음.
