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
