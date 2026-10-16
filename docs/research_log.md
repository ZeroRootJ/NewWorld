# 연구 노트 (일일 로그)

날짜별로 그날 다룬 내용을 한 줄씩 짧게 기록한다. 상세 결정/근거는 `progress.md`에 있고, 여기는
타임라인용. 시각은 기록 시점 기준(실제 질의 시각과 다소 오차 가능). 2026-10-16부터 기록.

## 2026-10-16

- Mining Geostatistics 책에서 conditional expectation 관련 위치 질의 → p.466–468, p.574–581 보고
- 그래프 binning이 range 기준인지 개수 기준인지 질의 → 책은 binning 대신 회귀직선(D1/D2) 방식임을 확인
- 인터넷 일반 관행 질의 → calibration은 equal-count bin 우세, kriging 실무는 산점도+회귀선이 일반적
- 회귀선 기울기 b의 의미(크고 작을 때) 질의 → conditional non-bias(b=1) 기준 설명
- conditional-expectation 2x3 보드를 binning→scatter+OLS로 교체 요청 → 새 스크립트 작성, reviewer 통과, 커밋
- 같은 6패널을 GT vs Z* 히스토그램으로 교체 요청(진단용) → 새 보드 작성, 커밋
- 분포 재현 정량화 선행연구 요청 → literature-scout 조사, `references.md`에 8건 추가, 커밋
- Wasserstein distance란? 질의 → 개념 설명
- 히스토그램 보드에 W1 distance 추가 요청 → 패널에 표시, CSV 컬럼 추가, 커밋
- KS statistic 추가 + y축 통일 요청 → 반영, 커밋
- 해당 그림을 Sample Density Axis 아티팩트에 추가 요청 → 반영
- 연구노트 작성 요청 → 이 파일 생성
- 연구노트가 너무 길다는 피드백 → 항목당 한 줄로 축약
