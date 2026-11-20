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
