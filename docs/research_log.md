# 연구 노트 (일일 로그)

날짜별로 "그날 어떤 질의가 있었고 무엇을 진행했는지"를 기록하는 세션 단위 활동 일지다.
`progress.md`가 실험 단계·결과·결정 사항을 추적하는 문서라면, 이 노트는 그보다 가벼운
"언제 무엇을 다뤘는지" 타임라인이다 — 내용이 겹칠 수 있지만 목적이 다르므로 둘 다 유지한다.

**주의**: 세션 내 개별 사용자 메시지의 정확한 전송 시각은 Claude가 알 수 없다. 아래 시각은
해당 항목을 로그에 적은 시점의 실제 시각이며, 실제 질의 시각과 다소(최대 수십 분) 차이가 날 수
있다. **2026-10-16부터 기록 시작**(사용자 지시, 그 이전은 소급하지 않음).

## 2026-10-16

- (기록 시각 14:37 CDT, 세션 진행 중 — 아래는 오늘 세션에서 있었던 질의를 시간 순으로 정리)
  1. *Mining Geostatistics* PDF에서 conditional expectation 관련 내용이 어디 나오는지 질의
     → PDF 텍스트 검색으로 p.466–468(Fig. VI.5, VI.A.3절)과 p.574–581(Fig. VIII.5, Table
     VIII.I, VIII.A.2절) 위치를 찾아 보고.
  2. 실제로 그래프를 그릴 때 점을 찍는 기준이 range(등간격)인지 개수(등개수)인지 질의
     → 책 전체 재검색 결과 그런 class-binning 방식 자체가 없고, 케이스 스터디(Fig. VI.8–VI.10,
     p.473–476)는 산점도에 직접 선형회귀선(D1/D2)을 적합시키는 방식임을 확인해 보고.
  3. (책 말고) 인터넷에서는 일반적으로 어떻게 하는지 질의
     → 웹 검색: calibration/ECE 쪽은 equal-width vs equal-count(quantile) binning 논의가
     활발하고 equal-count가 bias가 작다는 쪽(Roelofs et al. 2022)이 우세; kriging
     cross-validation 실무(ArcGIS 등)는 binning보다 산점도+회귀선 방식이 일반적임을 정리해 보고.
  4. 회귀선 기울기 b가 크고 작을 때 각각 무엇을 나타내는지 질의
     → b=1을 conditional non-bias 기준으로, b<1/b>1을 regression attenuation 관점에서
     설명(고추정값·저추정값 구간에서의 과대/과소추정 방향), 책 사례(kriging vs. polygons of
     influence의 D1 기울기 1.003 vs 0.699)로 근거 제시.
  5. (사용자가 2x3 "A–F" conditional-expectation 비교 보드 이미지 제시) 패널을 quantile binning이
     아니라 raw scatter + OLS 회귀선(기울기 b)으로 업데이트해달라는 요청, 레이아웃은 제공 이미지와
     동일하게 유지
     → 관련 코드 조사 중: `results/processed/sample_replicate_axis/make_length_case_study_figures.py`의
     `compute_conditional_expectation`/`plot_conditional_expectation`(현재 20-equal-count-bin
     곡선 + 전체 셀 기준 OLS slope를 이미 따로 계산 중)과 이를 재사용하는
     `make_fixed_range_case_study_figures.py`를 확인함. 사용자가 보여준 정확한 2×3 보드(A–F,
     MACB 수치 포함) 이미지의 출처 스크립트는 저장소 내에서 아직 특정 못함(노트북
     `method_walkthrough.ipynb`의 "MACB" 문자열 매치는 이미지 base64 데이터 안 우연한 일치로
     확인되어 제외) — 계속 조사 후 반영 예정.
  6. 연구 노트(일일 로그) 작성 요청 → 이 파일(`docs/research_log.md`) 신규 생성, 오늘부터 기록
     시작.
