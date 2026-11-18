# -*- coding: utf-8 -*-
"""Build the Paper Storyboard artifact HTML (embeds the storyboard PNGs)."""
import base64
import sys
from pathlib import Path

FIG = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def img(stem, alt):
    b = base64.b64encode((FIG / (stem + ".png")).read_bytes()).decode()
    return '<div class="plate"><img alt="%s" src="data:image/png;base64,%s"></div>' % (alt, b)


REAL, PART, NONE = "real", "part", "none"
CHIP = {REAL: "실제 데이터", PART: "일부 가상", NONE: "전부 가상"}

FIGS = {
    "F1": dict(
        stem="F1_accuracy_vs_calibration_three_axes", st=REAL,
        title="정확도와 calibration, 3개 축",
        body=[
            "위 행은 MSE, 아래 행은 UMG입니다. 열은 왼쪽부터 ground-truth range, nugget, sample density 축입니다.",
            "range와 nugget 축은 레벨당 ground-truth realization 10개(샘플셋은 같음, n = 121), density 축은 ground truth 10개 × 샘플셋 10개 = 100셀입니다. 점은 평균, 막대는 레벨 내 모든 셀의 ±1 표준편차입니다. SGS와 RBF+bootstrap의 ensemble 크기는 N = 50입니다.",
        ]),
    "F2": dict(
        stem="F2_base_case_calibration_and_variance", st=REAL,
        title="Base case calibration 곡선과 예측 분산",
        body=[
            "A는 base case(ground truth 1개, 샘플셋 1개, N = 50)의 accuracy plot(명목 확률 구간 대 실제 포함 비율)입니다. B는 방법별 평균 예측 분산을 sample density에 대해 그린 것입니다(레벨당 100셀의 평균 ± 1 표준편차).",
            "분산은 모두 porosity %² 단위입니다. kriging 분산은 normal-score 분산을 Monte Carlo로 역변환한 값입니다.",
        ]),
    "F3": dict(
        stem="F3-2_density_rep0_picked_variogram", st=REAL,
        title="밀도별 oracle kriging/SGS와 GP, GP가 맞춘 range",
        body=[
            "한 행 그림입니다. A는 MSE, B는 UMG, C는 range입니다. 점은 레벨 평균이고 error bar는 없습니다.",
            "A, B의 kriging과 SGS는 oracle(정답, range 300 m) variogram으로 돌린 결과이고, GP-MLE와 같은 밀도당 100셀(ground truth 10개 × 샘플셋 10개)의 평균입니다(SGS N = 50). 직접 고른 variogram과 자동 적합 variogram으로 돌린 kriging/SGS, 그리고 90% coverage는 그리지 않았고, 값은 status 파일에 있습니다.",
            "C는 GP-MLE practical range(상관 0.05 기준)를 셀마다 점 하나로 찍은 것입니다(레벨당 100개, 로그 축). 점선은 oracle(정답 variogram)의 range로, 300 m로 고정입니다.",
        ]),
    "F3b": dict(
        stem="F3-2b_density_truth_and_samples", st=REAL,
        title="밀도별 실제 map과 샘플 위치, realization 1개",
        body=[
            "F3에서 쓴 ground truth 중 하나(truth seed 101, range 300 m, nugget 0.05)의 porosity map입니다. 다섯 패널 모두 같은 map이고, 흰 점은 밀도마다 rep0 샘플셋(sample seed 1001)이 뽑은 위치입니다. 제목의 n은 실제 샘플 위치 수입니다.",
            "map과 샘플은 실험 코드의 생성 함수로 다시 만들었고, 샘플 위치가 각 셀의 kriging run에 저장된 samples.csv와 같은지 확인했습니다. 색 범위는 map의 최솟값~최댓값입니다.",
        ]),
    "F3c": dict(
        stem="F3c_density_gp_range_extremes", st=REAL,
        title="극단값: 밀도 2 %에서 GP range가 가장 짧은 셀과 가장 긴 셀",
        body=[
            "밀도 2 %의 100셀 중 GP-MLE practical range가 가장 짧은 셀(위)과 가장 긴 셀(아래)입니다. 열은 왼쪽부터 정답 porosity map과 샘플 위치(n = 49, 50), 같은 셀의 GP-MLE 예측 map(posterior mean, 제목에 MSE와 UMG), 같은 셀(같은 realization, 같은 샘플링)의 SGS realization 1개(50개 중 0번, oracle variogram, range 300 m), 이 세 map의 실험 variogram, 히스토그램입니다.",
            "가운데 열은 posterior 평균(추정값)이고 오른쪽 map 열은 시뮬레이션 1개라서 서로 종류가 다른 map입니다. SGS의 MSE, UMG는 realization 50개 ensemble 기준 값이고 그림에는 적지 않았습니다(아래 목록과 status 파일).",
            "variogram은 각 map의 전체 50×50 grid에서 무작위 셀 쌍 200,000개(모든 map에 같은 쌍), 25 m 구간으로 계산했고, 기존 variogram 재현 그림의 계산 함수를 그대로 썼습니다. 점선은 정답 모델(spherical, range 300 m, nugget 0.45, 구조 sill 8.55, porosity %² 단위)입니다. "
            "y축은 0~11.7로 고정했고, 위쪽 행 SGS 곡선의 6개 구간이 이 위로 잘렸습니다. 히스토그램은 두 행의 map 6개에 같은 구간 30개(같은 x축)를 쓴 밀도 히스토그램입니다.",
            "MSE와 UMG는 F3과 같은 평가 함수로 다시 계산해 저장된 값과 같은지 확인했습니다(샘플이 없는 셀에서 평가). SGS 평균·분산 map도 realization에서 다시 계산해 저장된 map과 같은지 확인했습니다. map 여섯 개는 같은 색 범위(두 정답 map의 최솟값~최댓값)를 쓰고, 이 범위를 벗어난 SGS realization 셀(위 63개, 아래 10개, 2,500개 중)은 끝 색으로 그려집니다. 두 셀은 정답 realization과 샘플셋이 모두 다릅니다.",
        ]),
    "F3d": dict(
        stem="F3d_density_gp_range_extremes_1pct", st=REAL,
        title="극단값: 밀도 1 %에서 GP range가 가장 짧은 셀과 가장 긴 셀",
        body=[
            "F3c와 같은 그림을 밀도 1 %(100셀)에 대해 그린 것입니다. 가장 짧은 range 셀(위)과 가장 긴 range 셀(아래)이고, 샘플 위치는 두 셀 모두 n = 25입니다. 열 구성, variogram 계산, 히스토그램 구간(두 행 공통 30개), 색 범위 규칙, 검증 방법은 F3c와 같습니다.",
            "variogram y축은 0~11.7로 고정했고, 아래쪽 행 SGS 곡선의 19개 구간이 이 위로 잘렸습니다. 색 범위(두 정답 map의 최솟값~최댓값)를 벗어난 SGS realization 셀은 위 33개, 아래 51개(2,500개 중)입니다.",
        ]),
    "F4": dict(
        stem="F4_gp_fitted_vs_truth_hyperparameters", st=REAL,
        title="GP가 맞춘 range, nugget과 실제 값",
        body=[
            "A는 ground-truth range 대 GP-MLE가 맞춘 practical range, B는 ground-truth nugget 대 GP-MLE가 맞춘 noise variance입니다. 점선은 y = x입니다.",
            "레벨당 ground-truth realization 10개이고, 샘플 위치는 모두 같습니다(n = 121). optimizer 경계에 걸린 적합은 A 80개 중 0개, B 60개 중 0개입니다.",
        ]),
    "F5": dict(
        stem="F5_anisotropy", st=PART,
        title="이방성: 실제 range와 variogram 모델링",
        body=[
            "A는 방위각별 실제 practical range(해석값)입니다. B는 major, minor 방향의 실험 variogram과 모델 variogram입니다(ground truth 10개 × 샘플 추출 2회).",
            "A와 B는 서로 다른 파라미터 세트입니다. A의 라벨은 저장된 그대로 300/300, 300/150, 300/600 m이고, B는 900/300 m입니다.",
            "C는 가상 그림입니다. 이방성 축에서 4개 방법을 비교한 실험이 아직 없습니다.",
        ]),
    "F6": dict(
        stem="F6_hyperparameter_stability_vs_density", st=PART,
        title="파라미터 수와 필요한 데이터 양",
        body=[
            "A는 등방성 GP-MLE가 맞춘 practical range의 변동계수(표준편차 ÷ 평균, replicate 10개)입니다. 밀도 20, 10, 5, 2, 1 %에서 0.068, 0.114, 0.111, 0.466, 0.590입니다.",
            "B는 가상 그림입니다. 파라미터가 더 많은 커널을 맞춘 실험이 없습니다.",
        ]),
    "F7": dict(
        stem="F7_response_surfaces_accuracy_vs_uncertainty", st=REAL,
        title="정확도와 UMG의 반응면",
        body=[
            "위 행은 MSE, 아래 행은 UMG이고, 열은 왼쪽부터 simple kriging, SGS, RBF+bootstrap, GP입니다. 색은 grid 노드마다 ground truth 10개(base case: range 300 m, nugget 0.05)에 대한 평균입니다. SGS와 RBF+bootstrap의 ensemble 크기는 N = 50입니다.",
            "하이퍼파라미터는 grid 값으로 고정했고 자동 튜닝(CV, marginal likelihood)은 하지 않았습니다. 색은 grid 노드 값 사이를 bilinear 보간한 등고선(14단계)이고, 노드 사이 값은 계산값이 아니라 보간값입니다. 로그 축은 로그 공간에서 보간했고, 바깥 노드 밖으로는 그리지 않았습니다. 동그라미는 실제 base case(300 m, nugget 0.05)입니다. 마름모는 RBF와 GP에 찍은 대략적인 대응점으로, 의미가 정확히 같지는 않고 비교용입니다. GP는 noise = nugget = 0.05, RBF는 λ = nugget/(1 − nugget) ≈ 0.0526(noise 대 signal 비)로 두었습니다. 최적점은 grid 노드 중에서 골랐습니다(MSE 최소, UMG 최대). 색 범위는 패널마다 다르고, 두 면 모두 밝을수록 좋은 쪽입니다. RBF와 GP의 MSE는 로그 색 척도입니다.",
            "grid 크기는 kriging 8×9, SGS 5×5, RBF 8×13, GP 8×11입니다. y축은 kriging과 SGS가 nugget(sill 대비 비율, 선형), RBF가 smoothing λ(로그), GP가 noise variance(정규화 단위의 sill 대비 비율, 로그)입니다. GP는 signal = 1 − noise로 두었습니다(unit sill).",
            "x축 range는 방법마다 같은 규칙으로 맞춘 등가값입니다. kriging과 SGS는 spherical variogram의 range 그대로입니다. RBF는 ε = √(−ln 0.05)/range, GP는 length scale = range/√(−2 ln 0.05)로, Gaussian 커널이 range 거리에서 0.05로 떨어지도록 변환했습니다. 두 covariance 함수가 같다는 뜻은 아닙니다.",
            "채운 별은 그 면의 최적점, 빈 별은 다른 면의 최적점입니다. kriging과 SGS 패널의 원은 실제 base case 값(300 m, 0.05)입니다.",
        ]),
}

PROP = "prop"
CHIP[PROP] = "제안(placeholder)"

# Storyline draft 2026-11-18: one chain from F1 to F7; each figure's last bullet hands off to the next.
STORY = [
    ("①", "정확도로는 방법이 구분되지 않는다. 불확실성은 따로 평가해야 한다.", "F1"),
    ("②", "불확실성이 모델 밖에 있는 RBF+bootstrap은 구간이 체계적으로 좁고, 하이퍼파라미터를 바꿔도 정확도를 버려야만 고쳐진다.", "F2 · F7 RBF 열"),
    ("③", "불확실성이 내재된 GP는 데이터가 충분하면 잘 calibrate되지만, 그 하이퍼파라미터는 range·nugget이 아니다. 지질 지식으로 검증할 손잡이가 없다.", "F4"),
    ("④", "데이터가 희소해지면 그 하이퍼파라미터가 흩어지고, 정확도는 conventional과 같이 나빠지는데 calibration만 GP에서 따로 무너진다. 평균보다 꼬리에서 먼저.", "F3 · F6 · F3c/d · P3 · P4"),
    ("⑤", "range·nugget을 사람이 variogram으로 정하면 흩어짐이 줄어든다. (현재 근거가 가장 약한 고리)", "P5"),
    ("⑥", "그렇다고 grid search로 고르는 것도 답이 아니다. 정확도 최적점과 UQ 최적점이 다르고, 실제 geoparameter는 그 사이에 있다.", "F7 · P7"),
]
STORY_END = ("결론: geoparameter는 데이터 적합 objective(CV-MSE, marginal likelihood, grid search)가 아니라 "
             "공간통계 해석으로 정해야 하고, 정확도와 불확실성은 분리해 보고해야 한다.")

TENSION = [
    ("T1", "F1 range 축에서 oracle kriging/SGS UMG가 100 → 800 m에서 0.946 → 0.794, 0.954 → 0.808로 내려가고 GP-MLE는 0.953을 유지합니다. "
           "'kriging/SGS = 정답 기준선'이라는 전제와 부딪히므로 본문에서 이유를 설명해야 합니다.", "P1"),
    ("T2", "밀도 20 %에서 GP-MLE UMG 0.977, oracle kriging 0.933, 자동 적합 variogram kriging 0.985, 직접 고른 variogram kriging 0.983입니다. "
           "데이터에서 정한 variogram이 oracle보다 높으며 T1과 같은 원인일 수 있습니다.", "P1 · P5"),
    ("T3", "직접 고른 variogram은 Variogram Review 화면에 정답 variogram 곡선이 보이는 상태에서 골랐습니다. ⑤의 근거로 쓰려면 blind pick이 필요합니다.", "P5"),
    ("T4", "F7에서 최적 노드와 2위 노드의 차이(MSE 0.0004~0.034, UMG 0.0006~0.010)가 ground truth 간 표준편차(MSE 1.28~1.33, UMG 0.025~0.029)보다 훨씬 작습니다. "
           "최적점 위치가 truth마다 일관된지 보여야 합니다.", "P7"),
]

CLAIMS = {
    "F1": [
        "MSE: 세 축 모두 4개 방법이 거의 겹칩니다(예: range 300 m에서 kriging 3.89, SGS 3.93, GP-MLE 3.86, RBF+bootstrap 4.17, 레벨 내 표준편차 1.2~1.3). 정확도만으로는 방법을 고를 근거가 없습니다.",
        "UMG: 방법이 뚜렷하게 갈립니다. RBF+bootstrap은 0.37~0.55, 나머지 세 방법은 0.79 이상입니다. 정확도가 같아도 불확실성 품질은 전혀 다를 수 있으므로 둘을 분리해 보고합니다.",
        "range, nugget, 밀도를 하나씩 바꿔도 이 구도는 유지됩니다. ground truth 하나에서만 나온 결과가 아니라는 근거입니다.",
        "주의(T1): range 축에서 oracle kriging/SGS의 UMG가 긴 range에서 내려갑니다(0.946 → 0.794). 정답 기준선이 왜 떨어지는지 P1으로 설명할 자리가 필요합니다.",
        "→ F2: RBF+bootstrap의 UMG는 왜 낮은가.",
    ],
    "F2": [
        "A: base case에서 RBF+bootstrap만 1:1 선 아래로 크게 내려갑니다. 명목 95 % 구간에 참값이 47 %만 들어옵니다(UMG 0.454, 나머지 0.950~0.971).",
        "B: 원인은 예측 분산의 크기입니다. RBF+bootstrap 0.4~1.5, kriging 2.2~6.9(porosity %²). bootstrap은 샘플셋이 바뀔 때의 변동만 잡고, 샘플이 없는 곳의 공간 불확실성은 잡지 못합니다.",
        "밀도가 낮아져도 이 차이는 좁혀지지 않습니다(F1 밀도 축 UMG 0.48~0.52).",
        "튜닝으로 고칠 수 있는가: F7의 RBF 열에서 UMG를 0.962까지 올리는 노드가 있지만 그때 MSE는 19.52로, MSE 최적값 3.91의 5배입니다. 사후 bootstrap은 정확도를 포기하지 않고는 calibrate되지 않습니다 → 결론 1-a.",
        "→ F4: 그렇다면 불확실성이 모델에 내재된 GP는? F1에서 GP-MLE는 오히려 oracle kriging 이상으로 calibrate됩니다.",
    ],
    "F4": [
        "출발점: F1에서 GP-MLE는 range 축 전체에서 UMG 0.95를 유지합니다(n = 121, 밀도 약 5 %).",
        "A: 그런데 GP가 맞춘 range는 실제 range를 따라가지 않습니다. 실제 100 → 800 m일 때 평균 130 → 337 m입니다.",
        "B: GP가 맞춘 noise는 실제 nugget보다 평균 1.0~1.5 크고, nugget이 0일 때도 1.47입니다.",
        "즉 GP는 '짧은 range + 큰 noise'라는 실제와 다른 모델로도 calibrate됩니다. 이 하이퍼파라미터는 marginal likelihood를 최대화한 값이지 range, nugget이 아니므로, 지질 지식으로 맞는지 검증하거나 고칠 수 없습니다.",
        "ground truth 10개에서 같은 방향이므로 realization 하나의 우연이 아닙니다.",
        "→ F3: 이 '다른 모델'은 데이터가 충분할 때는 안정적입니다. 데이터가 줄면?",
    ],
    "F3": [
        "기준선 주의: kriging/SGS는 oracle variogram(range 300 m)이므로 '파라미터를 맞게 정했을 때'의 기준입니다. 데이터에서 정한 variogram과의 비교는 P5입니다.",
        "A: MSE는 밀도 20 → 1 %에서 세 방법이 함께 오릅니다(kriging 2.48 → 6.56, SGS 2.52 → 6.59, GP-MLE 2.63 → 6.74). 정확도로는 GP도 희소 데이터를 conventional과 똑같이 견딥니다.",
        "B: UMG는 다르게 움직입니다. SGS 0.932 → 0.926으로 거의 그대로, GP-MLE 0.977 → 0.890. 5 %까지는 GP-MLE가 앞서고 2 %에서 교차(kriging 0.926, GP-MLE 0.927, SGS 0.938), 1 %에서 GP-MLE가 가장 낮습니다. 데이터 손실에 정확도는 같이 나빠지는데 calibration만 GP에서 따로 무너집니다. F1의 '정확도와 UQ는 분리' 전제가 밀도 축에서 다시 나타납니다.",
        "C: 같은 구간에서 GP range가 흩어집니다. 범위 148~248 m(20 %) → 57~740 m(1 %), 300 m를 넘는 셀은 0개 → 38개(100셀 중).",
        "셀 간 UMG 표준편차도 GP-MLE 0.017 → 0.10, SGS 0.054 → 0.068입니다. 평균보다 꼬리가 먼저 벌어집니다 → P3.",
        "→ F6: range의 흩어짐을 밀도별로 정량화.",
    ],
    "F3b": [
        "밀도 20, 10, 5, 2, 1 %에서 샘플 위치 수는 442, 244, 120, 48, 25개입니다. 2 %부터 샘플 간격이 실제 range(300 m)에 가까워집니다.",
    ],
    "F6": [
        "A: 등방성 GP-MLE range의 변동계수는 20, 10, 5, 2, 1 %에서 0.068, 0.114, 0.111, 0.466, 0.590입니다. 5 → 2 %에서 급증하며, F3 B에서 GP UMG가 떨어지기 시작하는 지점과 같습니다.",
        "파라미터가 1개(length scale)인 가장 단순한 커널도 2 % 아래에서는 데이터가 모자랍니다. 이방성(major, minor, 방위각)을 표현하려 파라미터를 늘리면 더 많은 데이터가 필요하다는 논의로 이어집니다(F5는 보류, 논의로만).",
        "A는 아직 이전 density 실험(replicate 10개) 데이터입니다. F3 C와 같은 E4 100셀로 교체하면 두 그림의 숫자가 일치합니다(TODO).",
        "B(placeholder, 선택): 파라미터가 더 많은 커널(ARD 이방성 GP)의 같은 지표를 겹쳐 보일 자리입니다. 신규 실행이 필요하며, 없어도 A와 논의로 논리는 이어집니다.",
        "→ F3c, F3d: 흩어진 파라미터가 실제로 어떤 map을 만드는가.",
    ],
    "F3c": [
        "밀도 2 %에서 GP range가 가장 짧은 셀과 가장 긴 셀입니다. 두 가지 실패 모양이 나옵니다.",
        "(a) 짧은 range(92 m) + noise가 하한값(1e-5): 샘플 주변만 점 모양으로 바뀌고 나머지는 평균값. map 분산 1.26(정답 9.00).",
        "(b) 긴 range(728 m) + 큰 noise(0.37): 위가 낮고 아래가 높은 완만한 경사면. map 분산 3.91.",
        "같은 셀의 SGS(oracle)는 정답과 비슷한 variogram과 히스토그램을 재현합니다(287.5 m 구간 semivariance: 정답 8.87/8.09, SGS 10.01/7.63, GP 평균 1.27/1.47).",
        "UMG: 긴 range 셀은 GP-MLE 0.856, SGS 0.923으로 GP가 낮고, 짧은 range 셀은 GP-MLE 0.878, SGS 0.831로 SGS도 낮습니다. 짧은 range 셀은 그 realization·샘플셋 자체가 어려운 셀일 수 있습니다.",
    ],
    "F3d": [
        "1 %에서도 같은 두 모양이 나옵니다. 짧은 range 57 m(noise 1e-5, map 분산 0.19), 긴 range 740 m(noise 0.33, map 분산 3.91).",
        "UMG: 긴 range 셀 GP-MLE 0.787, SGS 0.963 / 짧은 range 셀 GP-MLE 0.823, SGS 0.804.",
        "→ P4: 이 두 셀이 예외인지, '파라미터가 실제에서 멀수록 UMG가 떨어진다'는 일반 경향인지 확인해야 F3 C → B를 잇는 고리가 생깁니다.",
    ],
    "F7": [
        "질문: 1-b가 '데이터 적합으로 정한 파라미터는 희소할 때 불안정하다'라면, ground truth를 알고 전 영역에서 평가할 수 있는 이상적 조건에서 grid search로 직접 고르면 되는가?",
        "4개 방법 모두 MSE 최적 노드와 UMG 최적 노드가 다릅니다. 예: kriging MSE 최적 (600 m, 0.05)의 UMG 0.808, UMG 최적 (106 m, 0)의 MSE 5.562.",
        "방향이 일정합니다. MSE 최적은 긴 range 쪽(kriging, SGS 600 m), UMG 최적은 짧은 range 쪽(kriging, SGS, GP 모두 106 m)입니다.",
        "실제 값(300 m, 0.05)은 그 사이의 절충점입니다. kriging MSE 3.887(최적 3.861), UMG 0.928(최적 0.965). 한 objective로 튜닝하면 실제 geoparameter에서 멀어지며, ⑤의 '사람이 정한 geoparameter' 쪽을 지지합니다.",
        "RBF+bootstrap은 UMG 0.962를 얻으려면 MSE 19.52를 감수해야 합니다(1-a 재확인).",
        "주의(T4): 최적 노드와 2위 노드의 차이는 ground truth 간 표준편차보다 훨씬 작습니다. 면이 평평하므로 최적점 위치가 truth마다 일관된지 P7로 보여야 합니다.",
    ],
}

PH = {
    "P1": dict(
        title="정답 기준선 점검: range별 realization 통계와 모델의 차이",
        why="T1, T2 해소. 영역(1 km) 대비 range가 길수록 realization 하나의 실제 분산·variogram이 모델(sill 9.0)과 달라져 oracle variogram이 그 realization에는 '정답'이 아니게 되는지 확인합니다.",
        data="range 축 80셀의 ground truth map 전체 분산과 실험 variogram을 모델과 비교(range별). 선택: range 축에서 자동 적합 variogram으로 kriging/SGS 재실행(fit은 이미 641셀 모두 있음).",
        cost="기존 데이터로 계산 / 선택 항목은 신규 실행(80셀, 소규모)"),
    "P3": dict(
        title="평균이 아닌 꼬리: 셀별 UMG 분포",
        why="④의 '꼬리에서 먼저 무너진다'를 직접 보입니다. 실무에서 위험한 것은 평균이 아니라 최악의 셀입니다.",
        data="밀도별 100셀 UMG의 상자그림 또는 하위 10 % 분위수, GP-MLE vs oracle SGS(가능하면 picked SGS 10셀도).",
        cost="기존 데이터로 계산"),
    "P4": dict(
        title="파라미터 오차와 calibration 손실의 연결",
        why="F3 C(range 흩어짐)와 F3 B(UMG 하락)를 인과적으로 잇는 고리입니다. 지금은 두 패널이 같은 밀도에서 함께 움직인다는 것만 있습니다.",
        data="2 %, 1 % 200셀의 산점도: x = |ln(GP range / 300 m)|(또는 noise), y = 같은 셀의 UMG(GP-MLE) − UMG(oracle SGS). 셀 난이도를 SGS로 빼서 통제.",
        cost="기존 데이터로 계산"),
    "P5": dict(
        title="사람이 정한 variogram vs GP-MLE, 같은 셀에서",
        why="⑤(시사점)의 직접 근거. 현재 rep0 10셀 값: 1 %에서 picked range 239~384 m(GP-MLE 57~740 m, 자동 적합 75~1000 m)로 흩어짐은 작지만, UMG는 picked kriging 0.853, picked SGS 0.876, 자동 적합 SGS 0.902 vs GP-MLE 0.890(100셀)으로 셀 집합이 달라 비교가 안 되고, 우위도 아직 보이지 않습니다.",
        data="(i) 같은 rep0 10셀의 GP-MLE 값 추출, (ii) 정답 곡선을 가린 blind pick 50셀 + kriging/SGS 재실행(T3), (iii) 여유가 되면 샘플셋 10개 전체로 확장.",
        cost="(i) 기존 데이터 / (ii) 사용자 picking + 소규모 신규 실행 / (iii) 신규 실행"),
    "P7": dict(
        title="최적점의 일관성과 trade-off 경계",
        why="T4 해소. 평균 면의 최적점 차이가 작으므로, truth별로도 MSE 최적은 긴 range, UMG 최적은 짧은 range에 있는지 보여야 결론 2가 노이즈가 아님이 됩니다.",
        data="truth 10개 각각의 MSE·UMG 최적 노드 위치(F7 패널 위 점 10개씩), 방법별 MSE–UMG Pareto 경계(실제 값 노드 표시).",
        cost="기존 raw grid 데이터로 계산"),
}

SECTIONS = [
    ("전제", "정확도와 불확실성 품질은 따로 움직인다", "논문 전체의 평가 틀입니다. 이후 모든 그림은 MSE와 UMG를 따로 봅니다.", ["F1", "P1"]),
    ("결론 1-a", "불확실성이 모델 밖에 있으면 bootstrap으로도, 튜닝으로도 고쳐지지 않는다", "②. F7의 RBF 열을 미리 끌어와 '튜닝으로도 안 된다'까지 닫습니다.", ["F2"]),
    ("결론 1-b", "불확실성이 내재된 GP는 데이터가 희소해지면 calibration부터 무너진다", "③ → ④. 파라미터가 지질 통계량이 아님(F4) → 희소할수록 흩어짐(F3, F6) → 흩어진 파라미터의 결과(F3c, F3d) → 꼬리와 인과 고리(P3, P4).", ["F4", "F3", "F3b", "F6", "F3c", "F3d", "P3", "P4"]),
    ("시사점", "range·nugget은 사람이 공간통계로 정해야 한다", "⑤. 지금 논리에서 근거가 가장 약한 고리입니다. 이방성(F5)은 보류 상태이므로 F6와 함께 논의로만 다룹니다.", ["P5"]),
    ("결론 2", "grid search로 골라도 정확도와 UQ의 최적점은 다르다", "⑥. 데이터가 충분한 이상적 조건에서도 하나의 objective로는 실제 geoparameter에 도달하지 못합니다.", ["F7", "P7"]),
]

FLOW = [
    ("전제", "F1", "MSE와 UMG, 3개 축, 4개 방법", REAL),
    ("전제", "P1", "range별 realization 통계 vs 모델 (기준선 점검)", PROP),
    ("1-a", "F2", "Calibration 곡선, 예측 분산 (+ F7 RBF 열)", REAL),
    ("1-b", "F4", "GP가 맞춘 range, nugget 대 실제 값", REAL),
    ("1-b", "F3", "밀도별 oracle kriging/SGS와 GP-MLE MSE, UMG, GP range", REAL),
    ("1-b", "F3b", "밀도별 실제 map과 샘플 위치", REAL),
    ("1-b", "F6", "GP range 변동계수 vs 밀도 (B는 선택 placeholder)", PART),
    ("1-b", "F3c", "극단값 2 %: GP range 최소·최대 셀", REAL),
    ("1-b", "F3d", "극단값 1 %: 같은 구성", REAL),
    ("1-b", "P3", "셀별 UMG 분포(꼬리)", PROP),
    ("1-b", "P4", "파라미터 오차 vs UMG 손실 산점도", PROP),
    ("시사점", "P5", "picked/자동 적합 variogram vs GP-MLE, 같은 셀", PROP),
    ("2", "F7", "정확도와 UMG 반응면, 4개 방법", REAL),
    ("2", "P7", "truth별 최적점, Pareto 경계", PROP),
]

TODO = [
    ("P1, P3, P4, P7", "기존 데이터에서 계산만 하면 되는 placeholder", "신규 시뮬레이션 없음"),
    ("P5 (i)", "rep0 10셀의 GP-MLE 값 추출", "기존 데이터"),
    ("P5 (ii)", "정답 곡선을 가린 blind variogram pick + kriging/SGS 재실행", "사용자 picking 필요, 소규모 신규 실행"),
    ("F6 A", "E4(100셀) 데이터로 교체", "E4 데이터 있음, 미반영"),
    ("F1, F2, F3", "SGS realization, bootstrap 수를 N = 100, 150으로", "현재 N = 50 (배치 1)"),
    ("선택", "P1 range 축 자동 적합 variogram 재실행, F6-B ARD 커널, P5 (iii) 샘플셋 확장", "신규 실행, 논리상 필수 아님"),
]

CSS = """
<title>논문 스토리보드</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; a flow table up top, then one collapsible band per conclusion, figures full width */
:root{
  --bg:#f6f7f9; --surface:#ffffff; --fg:#1b2330; --muted:#5b6676; --line:#d9dee6;
  --accent:#1f5f8b; --real:#1f7a4d; --real-bg:#e3f3ea; --part:#9a6200; --part-bg:#fbefd6;
  --none:#6a5d7a; --none-bg:#ece8f1; --plate:#ffffff;
  --sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#12161c; --surface:#1a2029; --fg:#e4e9f0; --muted:#9aa6b6; --line:#2e3744;
  --accent:#7db8e0; --real:#6fcf9b; --real-bg:#173326; --part:#e8b860; --part-bg:#3a2c10;
  --none:#b9a9d1; --none-bg:#2b2536; --plate:#ffffff; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#12161c; --surface:#1a2029; --fg:#e4e9f0; --muted:#9aa6b6; --line:#2e3744;
  --accent:#7db8e0; --real:#6fcf9b; --real-bg:#173326; --part:#e8b860; --part-bg:#3a2c10;
  --none:#b9a9d1; --none-bg:#2b2536; --plate:#ffffff; color-scheme:dark }
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.65}
.wrap{max-width:1040px;margin:0 auto;padding-block:28px 56px;
  padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px));
  display:flex;flex-direction:column;gap:22px}
h1{font-size:26px;font-weight:600;margin:0;text-wrap:balance}
.lede{color:var(--muted);margin:6px 0 0;max-width:68ch}
h2{font-size:13px;font-weight:600;letter-spacing:.06em;color:var(--muted);margin:0 0 8px}
.tablebox{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:500;color:var(--muted);font-size:12.5px}
tr:last-child td{border-bottom:0}
td.k{font-family:var(--mono);font-size:13px;white-space:nowrap}
td.k a{color:var(--accent);text-decoration:none} td.k a:hover{text-decoration:underline}
.chip{display:inline-block;font-size:12px;font-weight:500;padding:1px 8px;border-radius:999px;white-space:nowrap}
.chip.real{color:var(--real);background:var(--real-bg)}
.chip.part{color:var(--part);background:var(--part-bg)}
.chip.none{color:var(--none);background:var(--none-bg)}
details.sec{border-top:2px solid var(--fg);padding-top:10px}
details.sec>summary{cursor:pointer;list-style:none;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::before{content:"+";font-family:var(--mono);color:var(--muted);width:1ch}
details.sec[open]>summary::before{content:"\\2212"}
details.sec>summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.tag{font-family:var(--mono);font-size:13px;color:var(--accent);font-weight:500}
.claim{font-size:18px;font-weight:600;text-wrap:balance}
.figs{display:flex;flex-direction:column;gap:30px;margin-top:18px}
.fig{display:flex;flex-direction:column;gap:10px;min-width:0}
.fig h3{margin:0;font-size:16px;font-weight:600;display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
.fig h3 .n{font-family:var(--mono);font-weight:500;color:var(--muted);font-size:14px}
.plate{background:var(--plate);border:1px solid var(--line);border-radius:4px;padding:6px}
.plate img{display:block;width:100%;height:auto}
.fig p{margin:6px 0 0;max-width:72ch}
.claims{margin:0;padding-left:1.2em;max-width:76ch;display:flex;flex-direction:column;gap:6px}
.how{color:var(--muted);font-size:14px}
.how>summary{cursor:pointer;font-size:13px}
.how>summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.file{font-family:var(--mono);font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.chip.prop{color:var(--none);background:var(--none-bg)}
.story{margin:12px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:8px;max-width:80ch}
.story li{display:grid;grid-template-columns:2ch 1fr;gap:2px 10px}
.story .sn{color:var(--accent);font-weight:600}
.story .sf{grid-column:2;font-family:var(--mono);font-size:12px;color:var(--muted)}
.storyend{margin:14px 0 0;font-weight:600;max-width:80ch}
.lead{margin:8px 0 0;color:var(--muted);max-width:76ch}
.claims li.next{color:var(--accent);font-weight:500;list-style:none;margin-left:-1.2em}
.fig.ph{border:1.5px dashed var(--line);border-radius:6px;padding:14px 16px;background:var(--surface)}
.phd{margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 14px;max-width:80ch}
.phd dt{font-size:12.5px;color:var(--muted);white-space:nowrap;padding-top:2px}
.phd dd{margin:0}
@media (max-width:560px){.phd{grid-template-columns:1fr}.phd dt{padding-top:6px}}
</style>
"""


def chip(st):
    return '<span class="chip %s">%s</span>' % (st, CHIP[st])


def main():
    n_real = sum(1 for r in FLOW if r[3] != PROP)
    n_ph = sum(1 for r in FLOW if r[3] == PROP)
    h = [CSS, '<div class="wrap">']
    h.append('<header><h1>논문 스토리보드</h1>'
             '<p class="lede">그림 %d개와 placeholder %d개를 하나의 논리로 엮은 초안입니다(2026-11-18). 각 그림의 마지막 → 줄이 다음 그림으로 넘어가는 연결이고, '
             '점선 상자는 논리를 채우기 위해 제안하는 데이터입니다. F5 이방성은 보류입니다.</p></header>' % (n_real, n_ph))
    h.append('<details class="sec" open><summary><span class="tag">논리</span><span class="claim">한 줄씩 이어지는 주장</span></summary>'
             '<ol class="story">')
    for n, s, figs in STORY:
        h.append('<li><span class="sn">%s</span><span>%s</span><span class="sf">%s</span></li>' % (n, s, figs))
    h.append('</ol><p class="storyend">%s</p>' % STORY_END)
    h.append('<h2 style="margin-top:18px">판단이 필요한 긴장 지점</h2><div class="tablebox"><table><thead><tr><th>#</th><th>내용</th><th>해소</th></tr></thead><tbody>')
    for t, s, p in TENSION:
        h.append('<tr><td class="k">%s</td><td>%s</td><td class="k">%s</td></tr>' % (t, s, p))
    h.append('</tbody></table></div></details>')
    h.append('<section><h2>흐름</h2><div class="tablebox"><table><thead><tr><th>단계</th><th>그림</th><th>내용</th><th>상태</th></tr></thead><tbody>')
    for c, f, d, st in FLOW:
        h.append('<tr><td class="k">%s</td><td class="k"><a href="#%s">%s</a></td><td>%s</td><td>%s</td></tr>' % (c, f.lower(), f, d, chip(st)))
    h.append('</tbody></table></div></section>')
    for tag, claim, lead, keys in SECTIONS:
        h.append('<details class="sec" open><summary><span class="tag">%s</span><span class="claim">%s</span></summary>'
                 '<p class="lead">%s</p><div class="figs">' % (tag, claim, lead))
        for k in keys:
            if k in PH:
                p = PH[k]
                h.append('<article class="fig ph" id="%s"><h3><span class="n">%s</span>%s %s</h3>' % (k.lower(), k, p["title"], chip(PROP)))
                h.append('<dl class="phd"><dt>논리상 역할</dt><dd>%s</dd><dt>제안 데이터</dt><dd>%s</dd><dt>비용</dt><dd>%s</dd></dl></article>'
                         % (p["why"], p["data"], p["cost"]))
                continue
            f = FIGS[k]
            h.append('<article class="fig" id="%s"><h3><span class="n">%s</span>%s %s</h3>' % (k.lower(), k, f["title"], chip(f["st"])))
            h.append(img(f["stem"], k + " " + f["title"]))
            h.append('<ul class="claims">' + "".join('<li%s>%s</li>' % (' class="next"' if c.startswith("→") else "", c) for c in CLAIMS[k]) + '</ul>')
            h.append('<details class="how"><summary>그림 읽는 법</summary>' + "".join('<p>%s</p>' % p for p in f["body"]) + '</details>')
            h.append('<div class="file">results/figures/paper_storyboard/%s.pdf</div></article>' % f["stem"])
        h.append('</div></details>')
    h.append('<details class="sec" open><summary><span class="tag">TODO</span><span class="claim">placeholder를 채우는 데 필요한 작업</span></summary>'
             '<div class="figs"><div class="tablebox"><table><thead><tr><th>대상</th><th>필요한 작업</th><th>비용</th></tr></thead><tbody>')
    for f, w, cur in TODO:
        h.append('<tr><td class="k">%s</td><td>%s</td><td>%s</td></tr>' % (f, w, cur))
    h.append('</tbody></table></div></div></details>')
    h.append('</div>')
    OUT.write_text("\n".join(h), encoding="utf-8")
    print(OUT, OUT.stat().st_size)


main()
