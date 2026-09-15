# 참고문헌 (Literature Scout 조사 누적본)

이 파일은 `literature-scout` 서브에이전트가 조사한 선행연구를 누적하는 단일 파일이다. 새 항목을 추가하기 전에 항상 전체를 읽어 중복 여부를 확인할 것.

---

### [Applied Geostatistics (Walker Lake dataset)](https://r-spatial.github.io/gstat/reference/walker.html)
- **저자/연도**: Isaaks, E.H. and Srivastava, R.M., 1989, *Applied Geostatistics*, Oxford University Press. (Walker Lake 데이터셋은 이 교과서에서 유래했고, R의 `gstat` 패키지 문서(위 링크)를 2차 출처로 확인함.)
- **한줄 요약**: 지구통계학 교육에서 가장 널리 쓰이는 고전 벤치마크 데이터셋. Nevada Walker Lake 지역 DEM 기반으로 만든 **260×300 = 78,000 cell exhaustive(전수) 그리드**와, 그 중 **470개 지점**을 뽑은 sample 데이터셋으로 구성됨 (470/78,000 ≈ 0.6%).
- **관련 축**: 샘플링 비율/데이터 밀도 (기준선 방법론: kriging/SGS의 전통적 벤치마킹 관례)
- **관련성 판단**: 명확히 관련 있음 — "exhaustive 그리드를 만들고 그 중 일부를 샘플로 뽑아 나머지를 예측"이라는 우리 프로젝트의 실험 설계와 구조적으로 동일한, 지구통계학 분야에서 가장 오래되고 표준적인 벤치마크 사례. 다만 470/78,000(약 0.6%)이라는 비율 자체가 "관례적으로 적절한 비율"인지, 아니면 단지 "이 특정 교육용 데이터셋의 우연한 비율"인지는 판단이 필요함 (교육 목적 데이터셋이라 실무 well spacing과 반드시 대응되지 않을 수 있음).
- **추가한 날짜**: 2026-09-14

### [Geological Field Restoration through the Lens of Image Inpainting](https://arxiv.org/abs/2506.04869)
- **저자/연도**: Trifonov, V., Oseledets, I., Muravleva, E., 2025 (arXiv:2506.04869)
- **한줄 요약**: 저류층 물성(포로시티) 필드 복원을 이미지 인페인팅(텐서 completion, low-rank + smoothness)으로 접근하고, 업계 표준 벤치마크인 **SPE10 Model 2** (Cartesian grid 60×220×85, 약 112만 cell)에서 ordinary kriging과 비교. **100 / 300 / 500 / 700개의 시추공(well, 수직 trajectory 형태로 다층을 관통)**을 샘플로 사용했으며, 이는 (areal, 즉 60×220 평면 격자 기준) active cell의 **0.8% / 2.3% / 3.8% / 5.3%**에 해당. 각 well 개수마다 50회 랜덤 위치 반복 실험.
- **관련 축**: 샘플링 비율/데이터 밀도 (석유공학 분야, kriging vs ML/RBF 계열 비교)
- **관련성 판단**: 명확히 관련 있음 — 석유공학 표준 벤치마크(SPE10)에서 실제로 "몇 개의 well을 몇 %로 샘플링했는가"를 정량적으로 보고한 사례이며, kriging과 비-kriging(텐서completion, RBF/ML 계열에 가까움) 방법을 같은 조건에서 비교했다는 점에서 우리 프로젝트의 실험 설계(4개 방법·동일 샘플 위치 비교)와 직접 비교 가능. 다만 이 논문의 0.8~5.3% 범위가 "일반적인 관례"인지 "이 논문이 의도적으로 희박한 샘플링을 강조하기 위해 택한 범위"인지는 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Rapid Approximation Prediction for Kriging](https://arxiv.org/abs/2605.29284)
- **저자/연도**: Li, Z., Fasshauer, G., Nychka, D. (arXiv:2605.29284, 2026년 버전 확인)
- **한줄 요약**: 대규모 공간 데이터에 대한 kriging 예측을 가속화하는 근사 기법 제안. 벤치마크에서 예측 격자 크기를 **60×60 ~ 500×500**까지 다양하게 사용했고, 몬테카를로 실험에서는 관측점 수를 **n = 200, 500, 1500**으로 설정. 북미 강수량 실제 응용에서는 **n = 1368개 관측점**을 **350×350 = 122,500 격자**에 대해 예측(약 1.1%에 해당하는 밀도이나, 관측점은 격자 위가 아니라 불규칙 위치에 분포).
- **관련 축**: 샘플링 비율/데이터 밀도 (kriging 계열, 통계학 논문)
- **관련성 판단**: 명확히 관련 있음 — kriging 계열 방법론 논문에서 격자 크기 대비 관측점 수 비율(약 1%)을 명시적으로 사용한 사례. 다만 이 논문은 "예측 정확도"가 아니라 "계산 가속"이 목적이므로, 이 비율이 "정확한 불확실성 정량화"를 위한 관례적 비율인지, 단순히 "계산 속도 벤치마크에 흔히 쓰이는 희박한 비율"인지는 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Practical and Rigorous Uncertainty Bounds for Gaussian Process Regression](https://arxiv.org/abs/2105.02796)
- **저자/연도**: Fiedler, C., Scherer, C.W., Trimpe, S., 2021 (arXiv:2105.02796)
- **한줄 요약**: GPR의 불확실성 구간(신뢰구간)에 대한 실용적이고 엄밀한 이론적 상한을 제시. 수치 실험에서 **1000개의 균등 격자 평가점**(1차원, D=[-1,1])을 만들고, 매 학습 인스턴스마다 **50개 입력점을 균등 랜덤 샘플링**(= 평가점 대비 5%)하여 GPR을 학습, ground truth가 불확실성 집합에 포함되는지를 1만 회 반복 검증(함수 50개 사용).
- **관련 축**: 샘플링 비율/데이터 밀도, Claim 2 (GPR calibration) 관련 배경
- **관련성 판단**: 명확히 관련 있음 — GPR 불확실성(calibration) 검증 논문에서 "격자 대비 5% 샘플링"이라는 구체적 비율을 사용한 사례. 다만 이 실험은 **1차원**이며 우리 프로젝트의 2차원 공간(50×50 그리드)과 차원이 다르므로, 이 비율을 2D 공간 문제에 그대로 유추 적용할 수 있는지는 판단 필요.
- **추가한 날짜**: 2026-09-14

### [A review of comparative studies of spatial interpolation methods in environmental sciences: Performance and impact factors](https://www.sciencedirect.com/science/article/abs/pii/S1574954110001147)
- **저자/연도**: Li, J. and Heap, A.D., 2014, *Environmental Modelling & Software* (초기 관련 버전은 2011년 *Ecological Informatics*에도 게재)
- **한줄 요약**: 53편의 비교 연구, 72개 방법/하위방법을 리뷰한 메타분석. 32개 방법을 80개 변수 응용사례에 적용해 표본 밀도(sampling density)가 예측 정확도에 미치는 영향을 정량화했으며, "표본 밀도의 영향은 (다른 요인 대비) 상대적으로 미미하다(marginal)"는 결론을 보고. IDW, ordinary kriging, ordinary co-kriging이 가장 흔히 쓰인 방법으로 집계됨.
- **관련 축**: 샘플링 비율/데이터 밀도 (일반 환경과학, kriging 포함 다수 방법론 메타분석)
- **관련성 판단**: 판단 필요(사용자 확인) — 이 리뷰는 "표본 밀도가 정확도에 미치는 영향이 크지 않다"는 결론을 내리는데, 이는 우리 프로젝트가 강조하려는 "샘플링 조건(및 ground truth 속성)에 따라 각 방법이 언제·왜 깨지는지가 핵심 논지"라는 프레이밍과 표면적으로 배치되는 것처럼 보일 수 있음. 그러나 (a) 이 리뷰는 accuracy만 다루고 uncertainty calibration은 다루지 않으며, (b) "marginal"이라는 결론이 실제 각 개별 연구의 큰 편차를 평균낸 결과일 수 있어 우리 실험의 "one-factor-at-a-time" 설계와 직접 충돌하는지는 원문 표를 더 봐야 판단 가능. 이 리뷰가 우리 논문의 반론 대응용으로 유용한지, 혹은 배경 설명용으로만 쓸지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Minimization of measuring points for the electric field exposure map generation in indoor environments by means of Kriging interpolation and selective sampling](https://www.sciencedirect.com/science/article/pii/S0013935122009045)
- **저자/연도**: Martínez-González, A. et al., 2022, *Environmental Research* (PubMed ID: 35636463)
- **한줄 요약**: 실내 전자기장 노출 지도 작성 시 kriging + 선택적 샘플링(ELSP 전략)으로 측정점을 줄이는 방법 연구. 최댓값/중간값 구역을 우선 샘플링하면 전체 측정점의 **70~80%까지 생략해도** 지도 품질이 유사하게 유지되며, **90% 이상 생략** 시 성능이 저하됨을 보고 (즉, 약 10~30%의 점만 남겨도 kriging 성능이 유지되는 구간이 존재).
- **관련 축**: 샘플링 비율/데이터 밀도 (kriging, 환경측정 분야 — 지구통계 응용이지만 석유공학은 아님)
- **관련성 판단**: 판단 필요(사용자 확인) — 이 연구는 응용 도메인(실내 전자기장)과 그리드 성격이 우리 프로젝트(공간 porosity 필드)와 다르고, "생략 가능한 비율"이 곧 "적절한 샘플링 비율"과 동일한 개념인지도 해석의 여지가 있음. "10~30%의 점으로도 kriging이 잘 작동한다"는 결과를 우리의 20% 샘플링 비율에 대한 지지 근거로 볼 수 있는지, 혹은 도메인 차이 때문에 직접 비교가 부적절한지는 사용자가 판단해야 함.
- **추가한 날짜**: 2026-09-14

### [Estimating Resources in Unconventional Assets: Spatial Bootstrapping with N-effective](https://www.sciencedirect.com/science/article/abs/pii/S0920410522000663)
- **저자/연도**: Farrell, R., Pyrcz, M.J., Bickel, E., 2022, *Journal of Petroleum Science and Engineering*, vol. 212
- **한줄 요약**: 전통적인 spatial bootstrap(SPEE Monograph 3 방식)이 well 간 공간적 독립성을 가정하기 때문에, 공간 상관이 있는 저류층에 적용하면 불확실성을 과소평가(overconfident)하게 됨을 지적. 이를 보완하기 위해 전체 well 개수 대신 "유효 독립 well 개수"(n-effective)를 이용한 spatial bootstrap 방법을 제안 (GeostatsPy 기반 오픈소스 구현).
- **관련 축**: Claim 1 (RBF+bootstrap이 불확실성을 과소평가함) — 샘플링 비율 자체보다는 bootstrap의 공간적 한계에 대한 직접적 선행연구
- **관련성 판단**: 명확히 관련 있음 — "bootstrap이 공간 상관을 무시하면 불확실성을 과소평가한다"는 주장이 우리 프로젝트의 Claim 1과 사실상 동일한 메커니즘을 다루며, 저자(Pyrcz)가 이 프로젝트의 방법론적 기준(geostatspy)과 동일 계보. 다만 이 논문은 석유 매장량(resource) 추정 맥락(well 단위 집계)이고 우리 프로젝트는 공간 필드 전체의 point-wise 불확실성이므로, "동일 메커니즘"으로 봐도 되는지 세부 적용 맥락 차이는 판단 필요. (이 항목은 샘플링 비율 자체에 대한 구체적 수치는 제공하지 않음.)
- **추가한 날짜**: 2026-09-14

### [Density of soil observations in digital soil mapping: A study in the Mayenne region, France](https://www.sciencedirect.com/science/article/abs/pii/S2352009421000031)
- **저자/연도**: Loiseau, T., Arrouays, D., Richer-de-Forges, A.C., Lagacherie, P., Ducommun, C. et al., 2021, *Geoderma Regional*, vol. 24, e00358
- **한줄 요약**: 프랑스 Mayenne 지역에서 토양 입도분포(particle-size distribution) 예측을 위해 ordinary kriging(OK)과 quantile random forest(QRF) 두 방법에 대한 표본 밀도(sampling density)의 영향을 비교. (구체적인 지점당 면적/밀도 수치는 원문 접근 제한으로 확인하지 못함 — 초록 수준 요약만 확보.)
- **관련 축**: 샘플링 비율/데이터 밀도 (kriging vs ML(RF) 비교, 토양과학 — 석유공학 외부 도메인)
- **관련성 판단**: 판단 필요(사용자 확인) — kriging과 ML 계열(QRF)을 같은 샘플 밀도 조건에서 비교했다는 설계는 우리 프로젝트와 유사하나, (a) 정확한 밀도 수치를 확인하지 못했고 (b) 도메인이 토양과학이라 석유공학/공간 porosity 문제에 얼마나 일반화되는지는 추가 확인이 필요. 원문 전체 접근이 가능하면 구체적 수치를 보강할 수 있음.
- **추가한 날짜**: 2026-09-14

### [Digital soil mapping sample density thresholds (여러 출처 종합)](https://www.sciencedirect.com/science/article/pii/S2352009421000031)
- **저자/연도**: 여러 digital soil mapping 문헌 종합 (검색 스니펫 기반 — 개별 원문 미확인)
- **한줄 요약**: 토양 물리 속성 예측에는 약 **3 point/ha**, 화학 속성에는 **1 point/7.2 ha** 미만 밀도가 필요하다는 보고, 그리고 어떤 연구에서는 **약 1 sample/2 km²** 이상에서 성능이 정체(threshold)된다는 보고가 검색 스니펫 수준에서 확인됨.
- **관련 축**: 샘플링 비율/데이터 밀도 (면적당 포인트 밀도 — "격자 대비 %" 프레이밍과는 단위가 다름)
- **관련성 판단**: 판단 필요(사용자 확인) — 이 수치들은 "격자 셀 대비 %" 단위가 아니라 "단위 면적당 포인트 수" 단위이므로, 우리 프로젝트의 20%(그리드 셀 비율) 프레이밍과 직접 비교하려면 단위 환산(및 원 논문 원문 확인)이 필요함. 또한 검색 엔진 요약(스니펫) 수준의 정보로, 원문 확인 전까지는 정확한 출처·수치를 확정하기 어려움 — 사용자가 필요하다고 판단하면 개별 원문 확인을 추가로 진행할 수 있음.
- **추가한 날짜**: 2026-09-14

---

## CRPS / proper scoring rule × geostatistical uncertainty validation (2026-09-14 조사)

> 조사 범위: (A) CRPS 원전·정본 서지정보 검증, (B) Deutsch goodness statistic 서지정보 검증,
> (C) CRPS와 geostatistics의 uncertainty-model-goodness(accuracy plot / goodness) 프레임워크를
> **함께** 쓴 선행연구 탐색. 아래 항목 중 "CRPS 사용 여부 / goodness·accuracy plot 사용 여부"는
> 가능한 경우 Semantic Scholar의 해당 논문 reference list를 직접 조회해 확인했으며, 확인 방법을
> 각 항목에 명시했다.

### A. CRPS 원전 / 정본 문헌

### [Scoring Rules for Continuous Probability Distributions](https://doi.org/10.1287/mnsc.22.10.1087)
- **저자/연도**: Matheson, J.E. and Winkler, R.L., 1976, *Management Science*, 22(10), 1087–1096. DOI: 10.1287/mnsc.22.10.1087
- **한줄 요약**: 연속형 확률예측에 대한 proper scoring rule을 일반적으로 다룬 논문으로, CRPS의 원전으로 널리 인용됨 (CRPS 구현 패키지들도 이 논문을 1차 출처로 표기).
- **관련 축/주장**: 평가지표(calibration/CRPS) — 원전 서지정보
- **관련성 판단**: 명확한 사실 — 서지정보(저널/권/호/페이지/DOI)가 복수 독립 출처(INFORMS, RePEc/IDEAS, Semantic Scholar, ACM DL)에서 일치 확인됨. 다만 "CRPS라는 이름과 현대적 정의가 이 논문에서 처음 명시되었는가"는 원문을 직접 확인하지 못했으므로, 원전으로 단정할지 "널리 원전으로 인용된다"로 쓸지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [A Scoring System for Probability Forecasts of Ranked Categories](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml)
- **저자/연도**: Epstein, E.S., 1969, *Journal of Applied Meteorology*, 8(6), 985–987.
- **한줄 요약**: 순서형(ranked) 범주 확률예측에 대한 점수체계(RPS, Ranked Probability Score)를 제시 — CRPS의 이산형 선행 개념.
- **관련 축/주장**: 평가지표(calibration/CRPS) — 역사적 배경
- **관련성 판단**: 명확한 사실 — AMS 공식 페이지에서 권/호/연도 확인, 페이지(985–987)는 2차 출처 기준. **DOI는 확인 실패** (AMS URL slug로 유추는 가능하나 추측 인용을 피하기 위해 기재하지 않음). 이 논문을 인용할지 여부(역사적 맥락만 필요한지)는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems](https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml)
- **저자/연도**: Hersbach, H., 2000, *Weather and Forecasting*, 15(5), 559–570. DOI: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
- **한줄 요약**: 앙상블 예보 시스템에 대한 CRPS의 실용적 계산식과, CRPS를 reliability 성분과 resolution/uncertainty 성분으로 분해하는 방법을 제시 (Brier score 분해의 연속형 대응).
- **관련 축/주장**: 평가지표(calibration/CRPS) — SGS·RBF bootstrap처럼 **앙상블**로 표현되는 방법의 CRPS 계산 근거
- **관련성 판단**: 명확한 사실 — 서지정보 AMS 공식 페이지에서 확인. **판단 필요(사용자 확인)**: 현재 `src/evaluation.py`는 앙상블에 대해 Hersbach식 추정량이 아니라 **분위수/pinball 적분** 경로를 쓰고 있으므로, 이 문헌을 "우리가 쓴 추정식의 출처"로 인용하면 부정확할 수 있음. "앙상블 CRPS의 표준 참조" 또는 "reliability 분해의 출처"로 인용할지, 아예 인용하지 않을지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437)
- **저자/연도**: Gneiting, T. and Raftery, A.E., 2007, *Journal of the American Statistical Association*, 102(477), 359–378. DOI: 10.1198/016214506000001437
- **한줄 요약**: strictly proper scoring rule의 정본 리뷰 논문. CRPS가 strictly proper임을 포함해 주요 scoring rule들의 성질을 체계적으로 정리.
- **관련 축/주장**: 평가지표(calibration/CRPS) — "CRPS는 proper scoring rule이라 구간만 넓혀서는 점수를 개선할 수 없다"는 논거의 표준 출처
- **관련성 판단**: 명확한 사실 — 서지정보가 Taylor & Francis 공식 페이지, UW Statistics, Semantic Scholar에서 일치. **원문 PDF 본문 검증은 실패**(스캔/압축 PDF로 텍스트 추출 불가)했으므로, "CRPS = 2∫pinball 표현이 이 논문에 있다"고는 **쓰지 말 것** (아래 Laio & Tamea 항목 참고).
- **추가한 날짜**: 2026-09-14

### [Probabilistic Forecasts, Calibration and Sharpness](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- **저자/연도**: Gneiting, T., Balabdaoui, F. and Raftery, A.E., 2007, *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 69(2), 243–268. DOI: 10.1111/j.1467-9868.2007.00587.x
- **한줄 요약**: 확률예측 평가의 "maximizing sharpness subject to calibration" 패러다임을 정식화 — calibration(구간이 맞는가)과 sharpness(구간이 좁은가)를 **분리해서** 보되 calibration을 제약으로 두는 프레임.
- **관련 축/주장**: 평가지표(calibration) — 본 프로젝트가 coverage/UMG와 interval width를 분리 보고하는 설계의 이론적 근거 후보
- **관련성 판단**: 명확한 사실 — 서지정보 Wiley/OUP/JSTOR에서 일치 확인. **판단 필요(사용자 확인)**: 이 프레임(calibration 제약 하 sharpness 최대화)은 Deutsch의 accuracy(goodness) + precision 프레임과 개념적으로 매우 유사하지만, 두 문헌 계열은 서로를 인용하지 않는 것으로 보임 — 이 유사성을 논문에서 "독립적으로 재발견된 같은 원리"로 서술할지, 단순 배경 인용으로 둘지는 해석의 문제라 사용자 확인 필요.
- **추가한 날짜**: 2026-09-14

### [Verification tools for probabilistic forecasts of continuous hydrological variables](https://hess.copernicus.org/articles/11/1267/2007/)
- **저자/연도**: Laio, F. and Tamea, S., 2007, *Hydrology and Earth System Sciences*, 11(4), 1267–1277.
- **한줄 요약**: 연속형 수문 변수의 확률예측 검증 도구를 정리한 논문. 복수의 2차 문헌이 **CRPS의 분위수(quantile score / pinball loss) 적분 표현**의 출처로 이 논문을 지목함 ("The representation is due to Laio and Tamea (2007)").
- **관련 축/주장**: 평가지표(CRPS) — `src/evaluation.py`가 실제로 쓰는 `CRPS = 2∫₀¹ pinball_τ dτ` 계산식의 출처 후보
- **관련성 판단**: **판단 필요(사용자 확인)** — (a) 서지정보(저널/권/페이지/연도)는 HESS 공식 페이지에서 확인했으나, (b) "pinball 적분 표현이 이 논문에서 처음 제시되었다"는 것은 **2차 문헌들의 진술을 근거로 한 것이고 원문 본문을 직접 확인하지 못했다**. 같은 표현을 Gneiting & Ranjan(2011), Friederichs & Thorarinsdottir(2012)와 함께 병렬 인용하는 관행도 관찰됨. 우리 CRPS 구현식의 출처로 이 논문을 단독 인용할지, 복수 병렬 인용할지, 아니면 원문 확인 후 확정할지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Comparing Density Forecasts Using Threshold- and Quantile-Weighted Scoring Rules](https://doi.org/10.1198/jbes.2010.08110)
- **저자/연도**: Gneiting, T. and Ranjan, R., 2011, *Journal of Business & Economic Statistics*, 29(3), 411–422. DOI: 10.1198/jbes.2010.08110
- **한줄 요약**: CRPS의 threshold-가중(Brier 적분) 및 quantile-가중(quantile score 적분) 표현을 다루며, 관심 영역(꼬리/중심)에 가중을 주면서도 propriety를 유지하는 가중 CRPS를 제안.
- **관련 축/주장**: 평가지표(CRPS) — pinball 적분 표현의 병렬 출처
- **관련성 판단**: 명확한 사실(서지정보) — Taylor & Francis / RePEc / JSTOR에서 일치 확인. **판단 필요**: 우리 프로젝트는 가중 CRPS를 쓰지 않으므로, 이 논문을 "적분 표현의 근거"로만 얕게 인용할지 아예 뺄지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### B. Deutsch goodness statistic / accuracy plot 계열

### [Direct Assessment of Local Accuracy and Precision](https://ccg-server.engineering.ualberta.ca/CCG%20Publications/Other/CVD%20Papers/01-Peer%20Reviewed/1997-1996/asses-local-accur-prec96.pdf)
- **저자/연도**: Deutsch, C.V., 1997, in Baafi, E.Y. and Schofield, N.A. (eds), *Geostatistics Wollongong '96*, Vol. 1, Kluwer Academic, Dordrecht. **페이지: 확인 실패 — 2차 출처가 102–113과 115–125 두 가지로 엇갈림.**
- **한줄 요약**: 국소 불확실성 모델의 accuracy(accuracy plot: 대칭 확률구간 p에 실제 값이 들어간 비율 vs p)와 그 요약 통계량인 goodness statistic G, 그리고 precision을 제시한 것으로 널리 인용되는 문헌. 본 프로젝트의 `calc_umg` / `accuracy_plot_fraction_in`이 따르는 프레임.
- **관련 축/주장**: 평가지표(calibration) — 현재 코드의 UMG 지표 원전
- **관련성 판단**: **판단 필요(사용자 확인) + 일부 확인 실패**.
  - 확인된 사실: 이 문헌이 존재하고, 편집자(Baafi & Schofield)·출판사(Kluwer, Dordrecht)·연도(1997)·수록서(Geostatistics Wollongong '96, Vol. 1)는 복수 출처에서 일치. CCG(U. Alberta) 서버에 원문 PDF가 공개되어 있음(위 링크).
  - **확인 실패 1 — 페이지 범위**: 102–113과 115–125가 병존. 논문 제출 전 원본 목차로 반드시 확정 필요.
  - **확인 실패 2 — 사용자 질문의 핵심**: "accuracy plot + goodness + precision 3종이 원래 함께 제시되었는지", 특히 **"goodness 단독으로는 구간을 넓히기만 해도 점수가 오르므로 precision을 함께 봐야 한다"는 취지가 원 문헌에 명시되어 있는지**는 **검증하지 못했다**. 원문 PDF가 스캔 이미지 기반이라 본문 텍스트 추출에 실패했고, 이 환경에는 PDF 렌더링 도구(poppler)가 없어 페이지 이미지로도 읽지 못함. 2차 문헌 수준에서는 "Deutsch(1997)의 goodness statistic은 국소 불확실성 모델의 **accuracy와 precision을** 평가하는 데 쓰인다"(Goovaerts 2009 계열 서술)와 "accurate한 모델들 중에서는 구간이 가장 좁은 것이 가장 precise하다"는 취지의 서술이 반복적으로 관찰되므로 **정황상 3종 세트가 함께 제시되었을 가능성이 높으나, 원문 인용은 사용자가 PDF를 직접 열어 확인한 뒤 확정할 것을 권함.**
  - 참고: `docs/geostatspy_conventions.md`의 출처인 Pyrcz의 GeostatsPy Demos Book 내 model checking 챕터를 확인했으나, accuracy plot을 "적용할 수 있다"고 언급만 할 뿐 goodness/precision의 수식이나 Deutsch(1997) 인용은 없었음.
- **추가한 날짜**: 2026-09-14

### [Geostatistical modelling of uncertainty in soil science](https://doi.org/10.1016/S0016-7061(01)00067-2)
- **저자/연도**: Goovaerts, P., 2001, *Geoderma*, 103(1–2), 3–26. DOI: 10.1016/S0016-7061(01)00067-2
- **한줄 요약**: 토양과학에서 국소·공간 불확실성의 지구통계학적 모델링을 정리한 리뷰. 2차 문헌들은 **accuracy plot을 digital soil mapping 분야에 도입한 문헌**으로 이 논문을 지목하며(별칭: reliability plot), 이후 DSM 계열 논문들이 accuracy plot을 인용할 때 Deutsch(1997) 대신 이 논문을 인용하는 경우가 많음.
- **관련 축/주장**: 평가지표(calibration) — accuracy plot 프레임워크의 대표 후속/전파 문헌
- **관련성 판단**: 명확한 사실(서지정보) — 권/페이지/DOI가 복수 출처에서 일치. **판단 필요**: 우리 논문에서 accuracy plot/UMG의 출처로 Deutsch(1997)와 Goovaerts(2001) 중 무엇을 (또는 둘 다) 인용할지는 관행이 분야마다 갈리므로 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [AUTO-IK: A 2D indicator kriging program for the automated non-parametric modeling of local uncertainty in earth sciences](https://pubmed.ncbi.nlm.nih.gov/20161335/)
- **저자/연도**: Goovaerts, P., 2009, *Computers & Geosciences*, 35(6), 1255–1270.
- **한줄 요약**: indicator kriging 기반 국소 불확실성 모델링 자동화 프로그램. **goodness statistic(Deutsch, 1997)을 국소 불확실성 모델의 accuracy와 precision 평가에 사용**했다고 2차 출처에서 확인됨.
- **관련 축/주장**: 평가지표(calibration) — goodness statistic이 실제 연구에서 쓰인 대표 선례
- **관련성 판단**: **판단 필요(사용자 확인)** — 서지정보(권/페이지)는 2차 출처 기반이며 PubMed 레코드로 논문 존재는 확인. 다만 본문을 직접 확인하지 못해 goodness 수식의 정확한 형태는 미검증. CRPS는 사용하지 않음(비결합 사례로 분류).
- **추가한 날짜**: 2026-09-14

### C. CRPS × geostatistical uncertainty validation 결합 여부 (핵심 조사)

### [Comparison of various uncertainty modelling approaches based on geostatistics and machine learning algorithms](https://doi.org/10.1016/j.geoderma.2018.09.008)
- **저자/연도**: Szatmári, G. and Pásztor, L., 2019, *Geoderma*, 337, 1329–1340. DOI: 10.1016/j.geoderma.2018.09.008
- **한줄 요약**: 헝가리 토양유기탄소 저장량(SOCS)을 대상으로 **universal kriging(UK), sequential Gaussian simulation(SGS), random forest + kriging(RFK), quantile regression forest(QRF)** 네 방법의 **불확실성 모델**을 비교. 평가는 정확도(ME, RMSE)와 **accuracy plot + G statistic(goodness)** 을 분리해 수행했고, SGS와 QRF가 더 나은 불확실성 모델을 준다고 보고.
- **관련 축/주장**: 평가지표(calibration) + 기준선 방법론(kriging·SGS) — 본 프로젝트와 **실험 구조가 가장 유사한 선행연구**(지구통계 vs ML, 정확도와 불확실성 분리 보고)
- **관련성 판단**: **명확한 사실(결합 여부에 한해)** — Semantic Scholar의 이 논문 reference list를 직접 조회한 결과, **Deutsch(1997) "Direct assessment of local accuracy and precision"이 인용되어 있고, Gneiting·Hersbach·Matheson & Winkler 등 CRPS/scoring rule 계열 문헌은 하나도 인용되지 않음**. 즉 이 논문은 **goodness/accuracy plot만 쓰고 CRPS는 쓰지 않는다**.
  **판단 필요(사용자 확인)**: 이 논문이 우리 연구의 (a) 가장 가까운 선행연구로서 차별점을 설명할 대상인지, (b) 단순 배경 인용인지, 그리고 (c) "동일한 비교 구도를 이미 다뤘다"는 점이 우리 novelty에 부담이 되는지 여부는 사용자 판단 필요. (차이점 후보: 본 연구는 합성 ground truth로 전역 coverage를 계산하고 one-factor-at-a-time으로 실패 조건을 특정하며, RBF+bootstrap과 GPR을 포함한다.)
- **추가한 날짜**: 2026-09-14

### [Validation of uncertainty predictions in digital soil mapping](https://doi.org/10.1016/j.geoderma.2023.116585)
- **저자/연도**: Schmidinger, J. and Heuvelink, G.B.M., 2023, *Geoderma*, 437, 116585. DOI: 10.1016/j.geoderma.2023.116585
- **한줄 요약**: DSM의 확률예측 검증을 체계화한 논문. 기존에 쓰이던 **PICP / reliability(= accuracy) plot**이 구간 경계의 한쪽 치우침을 못 잡는다는 한계를 지적하고, **quantile coverage probability(QCP), PIT 히스토그램, interval score(IS), CRPS**를 보완 지표로 제안. 사례연구에서 **kriging with external drift(KED)** 를 포함한 5개 확률예측 모델을 동일 지표로 비교.
- **관련 축/주장**: 평가지표(calibration + CRPS) — **coverage 계열 지표와 CRPS를 한 프레임에서 같이 쓴, 현재까지 찾은 가장 근접한 선례**
- **관련성 판단**: **명확한 사실(결합 여부에 한해)** — Semantic Scholar의 reference list 직접 조회 결과: **Goovaerts(2001), Gneiting & Raftery(2007), Gneiting·Balabdaoui·Raftery(2007), Hersbach(2000)가 모두 인용되어 있고, Deutsch(1997)와 Matheson & Winkler(1976)는 인용되어 있지 않음.** 즉 이 논문은 **accuracy plot을 Goovaerts(2001) 계보로 인용하면서 CRPS를 함께 사용하지만, Deutsch의 goodness statistic G 자체는 쓰지 않는 것으로 보임**(본문 미확인 — reference list 기준 추론).
  **판단 필요(사용자 확인)**: (a) "coverage 계열 + CRPS 결합"이라는 넓은 의미에서는 이 논문이 명백한 선례이므로, 우리 기여를 "CRPS와 coverage를 함께 쓴 것"으로 프레이밍하면 **선례에 밀린다**. (b) 반면 "Deutsch의 **goodness statistic G**라는 특정 요약 통계량 + CRPS" 조합에 한정하면 이 논문은 선례가 아니다. 어느 범위로 기여를 주장할지는 전적으로 사용자 판단 사항이며, 본 에이전트는 결론을 내리지 않음. (c) 본문 직접 확인(ScienceDirect 403으로 실패)이 필요하며, 논문 제출 전 원문 확인 권장.
- **추가한 날짜**: 2026-09-14

### [A Framework for the Cross-Validation of Categorical Geostatistical Simulations](https://doi.org/10.1029/2020EA001152)
- **저자/연도**: Juda, P., Renard, P. and Straubhaar, J., 2020, *Earth and Space Science*, 7, e2020EA001152. DOI: 10.1029/2020EA001152
- **한줄 요약**: 범주형(지질 facies) 지구통계 시뮬레이션 모델을 **K-fold 교차검증 + proper scoring rule**(quadratic/zero-one/balanced linear score)로 체계적으로 검증·순위화하는 프레임워크 제안. 오픈소스 구현(`randlab/geocv`) 제공.
- **관련 축/주장**: 평가지표(proper scoring rule) + 기준선 방법론(지구통계 시뮬레이션)
- **관련성 판단**: **판단 필요(사용자 확인)** — "proper scoring rule을 지구통계 시뮬레이션 검증에 도입"했다는 점에서 우리 CRPS 도입의 선례적 성격이 있으나, (a) 대상이 **범주형**이라 연속형 CRPS가 아니라 Brier/quadratic score 계열이고, (b) accuracy plot/goodness는 쓰지 않음. 따라서 "결합 사례"로 볼지 "인접 사례"로 볼지는 해석의 문제. 본문 미확인(서지정보와 요지는 AGU/저자 PDF 링크로 확인).
- **추가한 날짜**: 2026-09-14

### [BIS-4D: mapping soil properties and their uncertainties at 25 m resolution in the Netherlands](https://doi.org/10.5194/essd-16-2941-2024)
- **저자/연도**: Helfenstein, A. et al., 2024, *Earth System Science Data*, 16, 2941–… (Copernicus, open access)
- **한줄 요약**: 네덜란드 전역 토양물성 지도 + 불확실성. 불확실성 검증은 **PICP90과 PI90(구간폭)** 만 사용하며, 원문 확인 결과 **CRPS도 accuracy plot도 Deutsch의 goodness도 사용하지 않음** (PICP 출처로 Papadopoulos et al., 2001을 인용).
- **관련 축/주장**: 평가지표(calibration) — **부정 사례(negative evidence)**
- **관련성 판단**: 명확한 사실 — 본문(HTML) 직접 확인. **판단 필요**: 이런 "coverage만 쓰고 proper scoring rule은 안 쓰는" 대규모 응용 논문을 "현행 관행의 한계"를 보이는 근거로 인용할지, 아니면 분야가 달라(토양) 인용하지 않을지는 사용자 판단 필요.
- **추가한 날짜**: 2026-09-14

### [Using Monte Carlo conformal prediction to evaluate the uncertainty of deep-learning soil spectral models](https://doi.org/10.5194/soil-11-553-2025)
- **저자/연도**: Huang, Padarian, Minasny, McBratney, 2025, *SOIL*, 11, 553–563 (Copernicus, open access)
- **한줄 요약**: 불확실성 평가에 **PICP + MPIW(평균 구간폭)** 만 사용(출처: Shrestha & Solomatine, 2006). CRPS는 사용하지 않으며, 본문에서 "Schmidinger & Heuvelink(2023)가 PICP의 한쪽 치우침 문제를 지적했으므로 향후 QCP·PIT 등이 필요하다"고 **한계로 인정**.
- **관련 축/주장**: 평가지표(calibration) — **부정 사례 + coverage 지표의 한계에 대한 최근 인식 사례**
- **관련성 판단**: **판단 필요(사용자 확인)** — "coverage 계열 지표만으로는 부족하다"는 최근 분야 인식을 보여주는 인용거리가 될 수 있으나, 도메인(토양 분광 + 딥러닝)이 우리와 멀어 인용 적합성은 사용자 판단 필요. 본문(HTML) 직접 확인함.
- **추가한 날짜**: 2026-09-14
