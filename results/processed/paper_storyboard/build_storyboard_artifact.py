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

CLAIMS = {
    "F1": [
        "정확도(MSE)는 4개 방법이 세 축 모두에서 비슷합니다. 정확도만 보면 방법 간 차이가 드러나지 않습니다.",
        "UMG는 RBF+bootstrap만 크게 낮습니다. range 축 0.37~0.55, nugget 축 0.37~0.55, density 축 0.48~0.52이고, 나머지 세 방법은 0.79 이상입니다.",
        "range 축에서 kriging과 SGS의 UMG는 range 100 m → 800 m에서 0.95 → 0.79, 0.95 → 0.81로 내려가고, GP-MLE는 0.95로 일정합니다.",
        "이 차이는 ground truth의 range, nugget, 샘플 밀도를 바꿔도 유지됩니다. 특정 조건 하나에서만 나온 결과가 아니라는 근거입니다.",
        "정확도와 uncertainty 품질은 따로 봐야 한다는 논문의 전제를 한 장으로 보여줍니다.",
    ],
    "F2": [
        "A: RBF+bootstrap의 곡선이 1:1 선 아래로 크게 내려갑니다. 명목 95 % 구간에 참값이 실제로는 47 %만 들어옵니다. 구간이 너무 좁다는 뜻입니다.",
        "B: 그 원인이 예측 분산의 크기에 있음을 보여줍니다. RBF+bootstrap의 평균 분산은 0.4~1.5로, kriging의 2.2~6.9보다 훨씬 작습니다.",
        "bootstrap은 샘플셋이 바뀔 때 생기는 변동만 잡고, 샘플이 없는 곳의 공간적 불확실성은 잡지 못한다는 결론 1-a의 근거입니다.",
    ],
    "F3": [
        "A: MSE는 밀도 20 %에서 1 %로 갈 때 kriging 2.48 → 6.56, SGS 2.52 → 6.59, GP-MLE 2.63 → 6.74입니다(모두 100셀).",
        "B: UMG는 kriging 0.933 → 0.899, SGS 0.932 → 0.926, GP-MLE 0.977 → 0.890입니다. 20, 10, 5 %에서는 GP-MLE가 kriging/SGS보다 높고, 2 %에서는 kriging 0.926, GP-MLE 0.927, SGS 0.938, 1 %에서는 GP-MLE가 가장 낮습니다.",
        "C: GP-MLE practical range(100셀)의 중앙값은 20 %에서 181 m, 1 %에서 270 m이고, 범위는 20 %에서 148~248 m, 1 %에서 57~740 m입니다. 300 m를 넘는 셀은 100개 중 20 % 0개, 10 % 4개, 5 % 16개, 2 % 26개, 1 % 38개입니다. oracle range는 300 m 고정입니다.",
    ],
    "F3b": [
        "밀도 20, 10, 5, 2, 1 %에서 샘플 위치 수는 442, 244, 120, 48, 25개입니다.",
    ],
    "F3c": [
        "가장 짧은 range: truth 7108, rep2(n = 49), range 92 m, MSE 8.21, UMG 0.878. 적합된 noise variance(정규화)는 1e-5로 하한값입니다. 예측 map은 샘플 위치 주변에서 둥근 점 모양으로 값이 바뀌고 그 사이는 평균값에 가깝습니다.",
        "가장 긴 range: truth 101, rep7(n = 50), range 728 m, MSE 5.66, UMG 0.856. 적합된 noise variance(정규화)는 0.37입니다. 예측 map은 위쪽이 낮고 아래쪽이 높은 완만한 경사면입니다.",
        "같은 셀의 SGS(oracle variogram, realization 50개 ensemble 기준): 가장 짧은 range 셀에서 MSE 6.69, UMG 0.831, 가장 긴 range 셀에서 MSE 4.87, UMG 0.923입니다.",
        "전체 2,500셀 분산(porosity %²): 가장 짧은 range 셀에서 정답 9.00, GP-MLE 평균 1.26, SGS realization 0번 10.62, 가장 긴 range 셀에서 정답 9.00, GP-MLE 평균 3.91, SGS realization 0번 8.62입니다.",
        "287.5 m 구간(300 m에 가장 가까운 구간)의 semivariance: 가장 짧은 range 셀에서 정답 8.87, GP-MLE 평균 1.27, SGS 10.01, 가장 긴 range 셀에서 정답 8.09, GP-MLE 평균 1.47, SGS 7.63입니다. 정답 모델 값은 8.98입니다.",
        "비교: 밀도 2 %의 100셀 평균은 GP-MLE MSE 5.34, UMG 0.927, SGS MSE 5.20, UMG 0.938입니다.",
    ],
    "F3d": [
        "가장 짧은 range: truth 7102, rep0(n = 25), range 57 m, MSE 9.37, UMG 0.823. 적합된 noise variance(정규화)는 1e-5입니다.",
        "가장 긴 range: truth 101, rep8(n = 25), range 740 m, MSE 7.51, UMG 0.787. 적합된 noise variance(정규화)는 0.33입니다.",
        "같은 셀의 SGS(oracle variogram, realization 50개 ensemble 기준): 가장 짧은 range 셀에서 MSE 8.54, UMG 0.804, 가장 긴 range 셀에서 MSE 7.96, UMG 0.963입니다.",
        "전체 2,500셀 분산(porosity %²): 가장 짧은 range 셀에서 정답 9.00, GP-MLE 평균 0.19, SGS realization 0번 7.53, 가장 긴 range 셀에서 정답 9.00, GP-MLE 평균 3.91, SGS realization 0번 13.26입니다.",
        "287.5 m 구간의 semivariance: 가장 짧은 range 셀에서 정답 9.35, GP-MLE 평균 0.22, SGS 7.26, 가장 긴 range 셀에서 정답 8.09, GP-MLE 평균 1.42, SGS 11.60입니다. 정답 모델 값은 8.98입니다.",
    ],
    "F4": [
        "A: ground truth의 range가 길어져도 GP가 맞춘 range는 따라가지 못합니다. 실제 값이 100 m에서 800 m로 늘 때 평균은 130 m에서 337 m로만 늡니다.",
        "B: GP가 맞춘 noise variance는 모든 nugget 레벨에서 실제 nugget보다 평균 1.0~1.5만큼 큽니다. nugget이 0인 경우에도 평균 1.47을 noise로 배정합니다.",
        "marginal likelihood로 자동 선택한 하이퍼파라미터가 실제 공간 통계량과 체계적으로 어긋난다는 근거입니다. 데이터에서 자동으로 고른 값을 그대로 믿기 어렵다는 결론 1-b의 시사점과 연결됩니다.",
        "ground-truth realization 10개에서 반복해도 같은 방향이므로, realization 하나의 우연이 아니라는 점도 함께 보여줍니다.",
    ],
    "F5": [
        "A: 이방성 reservoir에서는 방향에 따라 실제 range가 달라집니다. 등방성 모델은 이 구조를 표현할 수 없습니다.",
        "B: 사람은 실험 variogram을 방향별로 보고 major와 minor range를 따로 모델링할 수 있습니다. conventional 방법이 이방성에 대응하는 방식입니다.",
        "C (예정): 이방성이 강해질 때 4개 방법의 MSE와 UMG가 어떻게 달라지는지 보일 자리입니다. 실험이 없어 아직 주장할 수 있는 내용이 없습니다.",
    ],
    "F6": [
        "A: 파라미터가 1개(length scale)뿐인 등방성 GP도 밀도가 낮아지면 추정이 흔들립니다. 변동계수가 20 %에서 0.07, 1 %에서 0.59입니다.",
        "가장 단순한 커널도 이 정도 데이터가 필요하므로, 이방성을 표현하려고 파라미터를 늘리면 더 많은 데이터가 필요하다는 논의의 출발점입니다.",
        "B (예정): 파라미터가 더 많은 커널을 같은 축에 겹쳐 보일 자리입니다. 데이터가 없어 아직 주장할 수 있는 내용이 없습니다.",
    ],
    "F7": [
        "Simple kriging: MSE 최적점은 (600 m, nugget 0.05)에서 3.861, UMG 최적점은 (106 m, nugget 0)에서 0.965입니다. MSE 최적점의 UMG는 0.808, UMG 최적점의 MSE는 5.562입니다. 실제 값(300 m, 0.05)에서는 MSE 3.887, UMG 0.928입니다.",
        "SGS: MSE 최적점은 (600 m, 0.05)에서 3.892, UMG 최적점은 (106 m, 0)에서 0.963입니다. MSE 최적점의 UMG는 0.816, UMG 최적점의 MSE는 5.779입니다. 실제 값(300 m, 0.05)에서는 MSE 3.926, UMG 0.933입니다.",
        "RBF+bootstrap: MSE 최적점은 (300 m, λ 0.316)에서 3.912, UMG 최적점은 (212 m, λ 3.2×10⁻⁴)에서 0.962입니다. MSE 최적점의 UMG는 0.489, UMG 최적점의 MSE는 19.52입니다.",
        "GP: MSE 최적점은 (300 m, noise 0.169)에서 3.897, UMG 최적점은 (106 m, noise 0.169)에서 0.958입니다. MSE 최적점의 UMG는 0.883, UMG 최적점의 MSE는 5.096입니다.",
        "4개 방법 모두 MSE 최적점과 UMG 최적점이 서로 다른 grid 노드입니다. 최적 노드와 2위 노드의 평균 차이는 MSE 0.0004~0.034, UMG 0.0006~0.010이고, ground truth 10개 사이의 표준편차는 MSE 1.28~1.33, UMG 0.025~0.029입니다.",
    ],
}

SECTIONS = [
    ("결론 1-a", "불확실성이 모델에 없는 방법은 bootstrap으로도 평가하지 못한다", ["F1", "F2"]),
    ("결론 1-b", "불확실성이 내재된 GP는 데이터가 희소해지면 흔들린다", ["F3", "F3b", "F3c", "F3d", "F4", "F6"]),
    ("결론 2", "정확도와 uncertainty goodness의 최적점은 서로 다르다", ["F7"]),
]

FLOW = [
    ("1-a", "F1", "MSE와 UMG, 3개 축, 4개 방법", REAL),
    ("1-a", "F2", "Calibration 곡선, 예측 분산", REAL),
    ("1-b", "F3", "밀도별 oracle kriging/SGS와 GP-MLE(100셀) MSE, UMG, GP가 맞춘 range", REAL),
    ("1-b", "F3b", "밀도별 실제 map과 샘플 위치(realization 1개)", REAL),
    ("1-b", "F3c", "극단값: 밀도 2 %에서 GP range 최소·최대 셀의 map(정답, GP-MLE 평균, SGS realization 1개), variogram, 히스토그램, MSE, UMG", REAL),
    ("1-b", "F3d", "극단값: 밀도 1 %에서 같은 구성(F3c와 동일 패널)", REAL),
    ("1-b", "F4", "GP가 맞춘 range, nugget 대 실제 값", REAL),
    ("1-b", "F6", "파라미터 수와 필요한 데이터 양", PART),
    ("2", "F7", "정확도와 UMG 반응면, 고정 하이퍼파라미터 grid, 4개 방법", REAL),
]

TODO = [
    ("F1, F2, F3", "SGS realization, bootstrap 수를 50개 배치로 추가(N = 100, 150)", "현재 N = 50 (배치 1)"),
    ("F6", "density 축을 ground truth 10개 데이터(E4)로 교체", "E4 데이터 있음, F6 미반영"),
    ("F6-B", "파라미터가 더 많은 커널의 적합 안정성", "실험 없음"),
    ("F7", "range × 두 번째 하이퍼파라미터 grid에서 MSE와 UMG 동시 평가", "완료: ground truth 10개, N = 50, grid 노드 2,890개 (results/raw/hyperparam_grid_surface/20261117T071222580635Z), 그림 반영"),
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
</style>
"""


def chip(st):
    return '<span class="chip %s">%s</span>' % (st, CHIP[st])


def main():
    h = [CSS, '<div class="wrap">']
    h.append('<header><h1>논문 스토리보드</h1>'
             '<p class="lede">지금 있는 데이터로 그린 그림 %d개입니다(F5 이방성은 보류). 실험이 없는 패널은 회색 바탕의 가상 그림이고, '
             '데이터 한계는 각 그림 아래 DATA STATUS 줄에 적혀 있습니다.</p></header>' % len(FLOW))
    h.append('<section><h2>흐름</h2><div class="tablebox"><table><thead><tr><th>결론</th><th>그림</th><th>내용</th><th>상태</th></tr></thead><tbody>')
    for c, f, d, st in FLOW:
        h.append('<tr><td class="k">%s</td><td class="k"><a href="#%s">%s</a></td><td>%s</td><td>%s</td></tr>' % (c, f.lower(), f, d, chip(st)))
    h.append('</tbody></table></div></section>')
    for tag, claim, keys in SECTIONS:
        h.append('<details class="sec" open><summary><span class="tag">%s</span><span class="claim">%s</span></summary><div class="figs">' % (tag, claim))
        for k in keys:
            f = FIGS[k]
            h.append('<article class="fig" id="%s"><h3><span class="n">%s</span>%s %s</h3>' % (k.lower(), k, f["title"], chip(f["st"])))
            h.append(img(f["stem"], k + " " + f["title"]))
            h.append('<ul class="claims">' + "".join('<li>%s</li>' % c for c in CLAIMS[k]) + '</ul>')
            h.append('<details class="how"><summary>그림 읽는 법</summary>' + "".join('<p>%s</p>' % p for p in f["body"]) + '</details>')
            h.append('<div class="file">results/figures/paper_storyboard/%s.pdf</div></article>' % f["stem"])
        h.append('</div></details>')
    h.append('<details class="sec" open><summary><span class="tag">TODO</span><span class="claim">그림을 채우는 데 필요한 실험</span></summary>'
             '<div class="figs"><div class="tablebox"><table><thead><tr><th>그림</th><th>필요한 작업</th><th>현재</th></tr></thead><tbody>')
    for f, w, cur in TODO:
        h.append('<tr><td class="k">%s</td><td>%s</td><td>%s</td></tr>' % (f, w, cur))
    h.append('</tbody></table></div></div></details>')
    h.append('</div>')
    OUT.write_text("\n".join(h), encoding="utf-8")
    print(OUT, OUT.stat().st_size)


main()
