# 논문 뼈대 (사용자 확정, 2026-11-16)

Conventional geostatistics(simple kriging, SGS)와 ML 모델(RBF+bootstrap, GP)을 accuracy와
uncertainty quantification 양쪽에서 비교한다.

## 결론 1 — 모든 것을 data-driven하게 판단하는 것은 위험하다
- **1-a**: 불확실성이 모델에 내재되지 않은 방법(RBF, accuracy 기반 ML 대다수)은 bootstrap으로
  사후 평가해도 불확실성을 제대로 평가하지 못한다.
- **1-b**: 불확실성이 내재된 모델(GP)은 데이터가 풍부하면 잘 작동하지만 데이터가 scarce해지면
  fail한다.
  - 이방성 reservoir: conventional은 사람이 variogram을 수정해 대응할 수 있지만 GP는 그렇지 못하다.
    더 다양한 형상의 variogram을 optimize하도록 변형해도 그만큼 더 많은 데이터가 필요해
    지구통계 관점에서는 적절하지 않다. (GP를 실제로 변형해 맞추는 것을 보일 필요는 없음 —
    파라미터 수와 필요한 데이터 양의 관계를 density 축과 겹쳐 보여주는 방식.)
- **시사점**: 사람이 range, nugget 같은 geoparameter를 적절히 설정해주는 것이 중요하다.

## 결론 2 — grid search로도 동시 최적화는 어렵다
Accuracy만 보면 simulation(grid search)으로 최적 파라미터를 찾을 수 있지만, accuracy와 uncertainty
goodness를 동시에 고려하면 두 response surface의 최적점이 서로 다른 곳에 있어 최적화가 어렵다.

## Figure 매핑 (스토리보드)
그림: `results/figures/paper_storyboard/`, 스크립트:
`results/processed/paper_storyboard/make_storyboard_figures.py`,
아티팩트: https://claude.ai/artifact/PcTKZW3cQKQ2AhxEGEpHhw

| 결론 | 그림 | 내용 | 상태 |
|---|---|---|---|
| 1-a | F1 | MSE·UMG, 3개 축(range/nugget/density), 4개 방법 | 실제 데이터 |
| 1-a | F2 | Base case calibration 곡선, 예측 분산 vs density | 실제 데이터 |
| 1-b | F3 | 밀도별 GP vs kriging, GP fitted range | 실제 데이터 |
| 1-b | F4 | GP fitted range·nugget vs truth (10 realization) | 실제 데이터 |
| 1-b | F5 | 이방성 range·variogram(A, B), 4개 방법 비교(C) | C는 placeholder |
| 1-b | F6 | 파라미터 수 vs 필요 데이터량 | B는 placeholder |
| 2 | F7 | Accuracy·UMG response surface | 전부 placeholder |

## 그림을 채우는 데 필요한 실험
- range/nugget 축 4개 방법을 여러 ground-truth realization으로 재실행 (현재 레벨당 1개)
- N_BOOTSTRAP / SGS N_REALIZATIONS를 최종 설정으로 재실행 (현재 둘 다 10)
- density 축 ground-truth realization 확장 (현재 1개, 샘플 배치 replicate만 10)
- 이방성 축 본편 (F5-C); F5-A(300/600 m)와 F5-B(900/300 m)의 파라미터 세트 통일
- 파라미터 수 vs 데이터량 (F6-B)
- (range × nugget) 하이퍼파라미터 grid에서 MSE·UMG 동시 평가 (F7)

Extrapolation 축(`experiment_context.md` 3절 axis 3)은 이 뼈대에 포함되지 않았다.
