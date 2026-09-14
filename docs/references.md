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
