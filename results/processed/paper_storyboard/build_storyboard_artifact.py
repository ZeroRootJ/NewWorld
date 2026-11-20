# -*- coding: utf-8 -*-
"""Build the Paper Storyboard artifact HTML (embeds the storyboard PNGs)."""
import base64
import re
import sys
from pathlib import Path

FIG = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def img(stem, alt):
    b = base64.b64encode((FIG / (stem + ".png")).read_bytes()).decode()
    return '<div class="plate"><img alt="%s" src="data:image/png;base64,%s"></div>' % (alt, b)


# Figure numbers in reading order (user request 2026-11-20): premise = Fig. 1,
# conclusion 1 = Fig. 2-3, conclusion 2 = Fig. 4-13 (P1, P7 dropped 2026-11-20). Internal keys / file stems unchanged.
ORDER = ["F1", "F2", "F7", "F4", "F3", "P3", "F3c", "F3d", "P8", "P9"]
NUM = {k: i + 1 for i, k in enumerate(ORDER)}
# appendix (user request 2026-11-20: F3b moved to the appendix, F6 dropped)
APPENDIX = ["F3b", "A2"]
LABEL = {k: "Fig. %d" % n for k, n in NUM.items()}
LABEL.update({k: "Fig. A%d" % (i + 1) for i, k in enumerate(APPENDIX)})
_KEY_RE = re.compile(r"(?<![A-Za-z0-9_/-])(F3[bcd]|F[1-7]|P[0-9]b?)(?![A-Za-z0-9_-])")


def fnum(text):
    """Replace internal figure keys with the paper's figure numbers."""
    return _KEY_RE.sub(lambda m: LABEL[m.group(1)] if m.group(1) in LABEL else m.group(1), text)


def status_note(stem):
    """DATA STATUS note that used to be drawn under the figure (moved out 2026-11-20)."""
    lines = (FIG / (stem + ".status.txt")).read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("DATA STATUS note"):
            return lines[i + 1].strip().replace("DATA STATUS: ", "", 1)
    return ""


REAL, PART, NONE = "real", "part", "none"
CHIP = {REAL: "실제 데이터", PART: "일부 가상", NONE: "전부 가상"}

FIGS = {
    "F1": dict(
        stem="F1_accuracy_vs_calibration_three_axes", st=REAL,
        title="정확도와 calibration, 3개 축",
        body=[
            "위 행은 MSE, 아래 행은 UMG입니다. 열은 왼쪽부터 ground-truth range, nugget, sample density 축입니다.",
            "range와 nugget 축은 레벨당 ground-truth realization 10개(샘플셋은 같음, n = 121), density 축은 ground truth 10개 × 샘플셋 10개 = 100셀입니다. 점은 레벨 내 모든 셀의 평균이고, 오차막대는 그리지 않았습니다(2026-11-20 사용자 요청). SGS와 RBF+bootstrap의 ensemble 크기는 N = 50입니다.",
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
            "A, B의 kriging과 SGS는 oracle(정답, range 300 m) variogram으로 돌린 결과이고, GP-MLE와 같은 밀도당 100셀(ground truth 10개 × 샘플셋 10개)의 평균입니다(SGS N = 50). 90% coverage는 그리지 않았고 값은 status 파일에 있습니다. variogram 기반 방법은 oracle variogram으로만 비교합니다(2026-11-19 사용자 결정).",
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
            "A는 ground-truth range 대 GP-MLE가 맞춘 practical range, B는 ground-truth nugget 대 GP-MLE가 맞춘 noise로, 둘 다 sill 대비 비율(0~1)입니다(GP는 noise / (signal + noise)). 점선은 y = x입니다.",
            "레벨당 ground-truth realization 50개이고(2026-11-20, 10개에서 확장), 샘플 위치는 모두 같습니다(n = 121). optimizer 경계에 걸린 적합은 A 400개 중 2개(range 100 m, noise 하한), B 300개 중 0개입니다.",
        ]),
    "A2": dict(
        stem="A2_truth_realization_variograms", st=REAL,
        title="ground truth realization의 실험 variogram",
        body=[
            "Fig. 4와 같은 truth(레벨당 50개)의 전체 map(50 × 50셀, 모든 셀 쌍, 20 m bin) 실험 variogram을 모델 sill(9)로 나눈 값입니다. A–H는 range 축(nugget 0.05), I–N은 nugget 축(range 300 m)입니다.",
            "선 색은 같은 truth의 샘플로 GP-MLE가 맞춘 practical range를 실제 range로 나눈 비율(로그, 0.25–4)이고, 검은 점선은 truth 생성에 쓴 spherical 모델입니다.",
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
            "모든 패널에서 채운 별은 MSE 면의 최적점(MSE 최소), 빈 별은 UMG 면의 최적점(UMG 최대)입니다. kriging과 SGS 패널의 원은 실제 base case 값(300 m, 0.05)입니다.",
        ]),
    "P3": dict(
        stem="P3_density_umg_distribution", st=REAL,
        title="셀별 UMG 분포와 sample density",
        body=[
            "density 축(E4), 밀도당 100셀(ground truth 10개 × 샘플셋 10개). kriging과 SGS는 oracle variogram입니다. 상자 = 25~75 백분위, 선 = 중앙값, 수염 = 상자에서 1.5 × IQR 안의 가장 먼 값, 원 = 그 밖의 셀, 마름모 = 평균입니다.",
        ]),
    "P4": dict(
        stem="P4_gp_param_error_vs_umg_loss", st=REAL,
        title="GP 하이퍼파라미터와 같은 셀의 UMG 차이",
        body=[
            "density 축(E4) 500셀. 점 하나가 같은 ground truth·같은 샘플셋의 GP-MLE와 SGS(oracle)이고, y = UMG(GP-MLE) − UMG(SGS)입니다. A의 x는 |ln(GP practical range / 300 m)|, B의 x는 GP가 맞춘 noise variance(샘플 분산 대비 비율, 로그)입니다. B의 점선은 실제 nugget 비율 0.05(모델 sill 대비)로 참고용입니다. 10⁻⁵는 optimizer의 noise 하한값입니다.",
        ]),
}

PROP = "prop"
CHIP[PROP] = "제안(placeholder)"

# Storyline draft 2026-11-18: one chain from F1 to F7; each figure's last bullet hands off to the next.
STORY = [
    ("⓪", "대부분의 정확도 기반 모델(RBF 보간부터 최신 ML 회귀 모델까지)은 불확실성을 내장하지 않는다. 필요하면 bootstrap 같은 사후 방법을 덧붙인다. RBF+bootstrap은 그 가장 단순한 대표다.", "문제 제기"),
    ("①", "정확도로는 방법이 구분되지 않는다. 불확실성은 따로 평가해야 한다.", "F1"),
    ("②", "사후로 붙인 불확실성(RBF+bootstrap)은 구간이 체계적으로 좁고, 밀도를 바꿔도 좁혀지지 않는다.", "F2"),
    ("③", "ground truth를 아는 grid search에서도 MSE로 고른 하이퍼파라미터와 UMG로 고른 하이퍼파라미터가 다르다. 4개 방법 모두 그렇다.", "F7"),
    ("④", "불확실성이 내장된 GP-MLE는 데이터가 충분하면 잘 calibrate된다. 실제와 다른 range·noise로도 conventional 이상의 UMG를 낸다.", "F1 · F4 · F3"),
    ("⑤", "그러나 현장에서 흔한 희소 데이터에서는 marginal likelihood로 최적화한 파라미터가 흔들린다. range가 흩어지고, calibration은 평균보다 하위 꼬리에서 먼저 나빠진다.", "F3 · P3"),
    ("⑥", "realization을 만들 때는 이 파라미터가 그대로 variogram·분포 모델이 된다. 흔들린 파라미터는 variogram과 분포 재현에 치명적일 수 있다.", "F3c · F3d · P8 · P9"),
    ("⑦", "range·nugget을 올바르게 정한 conventional 방법(oracle variogram)은 같은 희소 조건에서 calibration을 유지한다. 데이터에서 variogram을 정하는 경우는 논문 끝 OBM 검증에서 다룬다(보류 중).", "F3 · P3 · OBM(보류)"),
]
STORY_END = ("결론: 정확도 기반 모델에 사후로 붙인 불확실성은 calibration을 맞추기 어렵고, 정확도로 고른 파라미터와 UQ로 고른 파라미터는 다르다. "
             "불확실성이 내장된 GP-MLE도 현장에서 흔한 희소 데이터에서는 최적화된 파라미터가 흔들려 realization 생성에 치명적일 수 있다. "
             "정확도와 불확실성은 분리해 보고해야 한다.")

TENSION = [
    ("T1", "F1 range 축에서 oracle kriging/SGS UMG가 100 → 800 m에서 0.946 → 0.794, 0.954 → 0.808로 내려가고 GP-MLE는 0.953을 유지합니다. "
           "'kriging/SGS = 정답 기준선'이라는 전제와 부딪힙니다(진단 그림 P1은 2026-11-20 사용자 결정으로 삭제).", "—"),
    ("T2", "F7에서 최적 노드와 2위 노드의 차이(MSE 0.0004~0.034, UMG 0.0006~0.010)가 ground truth 간 표준편차(MSE 1.28~1.33, UMG 0.025~0.029)보다 훨씬 작습니다. "
           "'MSE 최적은 긴 range, UMG 최적은 짧은 range'라는 방향 문장은 F7에서 뺐습니다(truth별 최적점 그림 P7은 2026-11-20 사용자 결정으로 삭제).", "—"),
]

CLAIMS = {
    "F1": [
        "MSE: 세 축 모두 4개 방법이 거의 겹칩니다(예: range 300 m에서 kriging 3.89, SGS 3.93, GP-MLE 3.86, RBF+bootstrap 4.17, 레벨 내 표준편차 1.2~1.3). 정확도만으로는 방법을 고를 근거가 없습니다.",
        "UMG: 방법이 뚜렷하게 갈립니다. RBF+bootstrap은 0.37~0.55, 나머지 세 방법은 0.79 이상입니다. 정확도가 같아도 불확실성 품질은 전혀 다를 수 있으므로 둘을 분리해 보고합니다.",
        "range, nugget, 밀도를 하나씩 바꿔도 이 구도는 유지됩니다. ground truth 하나에서만 나온 결과가 아니라는 근거입니다.",
        "주의(T1): range 축에서 oracle kriging/SGS의 UMG가 긴 range에서 내려갑니다(0.946 → 0.794).",
        "→ F2: RBF+bootstrap의 UMG는 왜 낮은가.",
    ],
    "F2": [
        "A: base case에서 RBF+bootstrap만 1:1 선 아래로 크게 내려갑니다. 명목 95 % 구간에 참값이 47 %만 들어옵니다(UMG 0.454, 나머지 0.950~0.971).",
        "B: 원인은 예측 분산의 크기입니다. RBF+bootstrap 0.4~1.5, kriging 2.2~6.9(porosity %²). bootstrap은 샘플셋이 바뀔 때의 변동만 잡고, 샘플이 없는 곳의 공간 불확실성은 잡지 못합니다.",
        "밀도가 낮아져도 이 차이는 좁혀지지 않습니다(F1 밀도 축 UMG 0.48~0.52).",
        "→ F7: 하이퍼파라미터를 grid search로 고르면 정확도 최적과 UQ 최적이 같은가.",
    ],
    "F4": [
        "출발점: F1에서 GP-MLE는 range 축 전체에서 UMG 0.95를 유지합니다(n = 121, 밀도 약 5 %). oracle kriging 이상으로 calibrate됩니다.",
        "A: 그런데 GP가 맞춘 range는 실제 range를 따라가지 않습니다. 실제 100 → 800 m일 때 평균 107 → 388 m입니다.",
        "B: GP가 맞춘 noise 비율은 실제 nugget보다 레벨 평균 0.12~0.17 크고, nugget이 0일 때도 0.167입니다.",
        "즉 GP는 '짧은 range + 큰 noise'라는 실제와 다른 모델로도 calibrate됩니다. 이 하이퍼파라미터는 marginal likelihood를 최대화한 값이지 range, nugget이 아니므로, 지질 지식으로 맞는지 검증하거나 고칠 수 없습니다.",
        "ground truth 50개에서 같은 방향이므로 realization 하나의 우연이 아닙니다.",
        "→ F3: 이 '다른 모델'은 데이터가 충분할 때는 안정적입니다. 현장에서 흔한 희소 데이터에서는?",
    ],
    "F3": [
        "기준선: kriging/SGS는 oracle variogram(range 300 m), 즉 'range·nugget을 맞게 정했을 때'의 기준입니다. 데이터에서 variogram을 정하는 경우는 논문 끝 OBM 검증에서 다룹니다(2026-11-19 사용자 결정, 현재 보류).",
        "A: MSE는 밀도 20 → 1 %에서 세 방법이 함께 오릅니다(kriging 2.48 → 6.56, SGS 2.52 → 6.59, GP-MLE 2.63 → 6.74). 정확도로는 GP도 희소 데이터를 conventional과 똑같이 견딥니다.",
        "B: UMG는 다르게 움직입니다. SGS 0.932 → 0.926으로 거의 그대로, GP-MLE 0.977 → 0.890. 5 %까지는 GP-MLE가 앞서고 2 %에서 교차(kriging 0.926, GP-MLE 0.927, SGS 0.938), 1 %에서 GP-MLE가 가장 낮습니다. 데이터 손실에 정확도는 같이 나빠지는데 calibration만 GP에서 따로 무너집니다. F1의 '정확도와 UQ는 분리' 전제가 밀도 축에서 다시 나타납니다.",
        "C: 같은 구간에서 GP range가 흩어집니다. 범위 148~248 m(20 %) → 57~740 m(1 %), 300 m를 넘는 셀은 0개 → 38개(100셀 중).",
        "셀 간 UMG 표준편차도 GP-MLE 0.017 → 0.10, SGS 0.054 → 0.068입니다. 평균보다 꼬리가 먼저 벌어집니다.",
        "→ P3: 파라미터가 흔들리는 밀도에서 calibration은 어디서부터 나빠지는가.",
    ],
    "A2": [
        "먼 lag(560–700 m) 평균 γ/sill은 range 100 m에서 1.006(표준편차 0.018, 0.97–1.04), 300 m에서 1.070(0.071, 0.95–1.26), 800 m에서 1.187(0.090, 0.98–1.38)입니다.",
        "nugget 축에서는 nugget 0에서 1.072(0.074, 0.95–1.27), 0.5에서 1.025(0.029, 0.97–1.11)입니다.",
    ],
    "F3b": [
        "밀도 20, 10, 5, 2, 1 %에서 샘플 위치 수는 442, 244, 120, 48, 25개입니다. 2 %부터 샘플 간격이 실제 range(300 m)에 가까워집니다.",
    ],
    "F6": [
        "A: 등방성 GP-MLE range의 변동계수는 20, 10, 5, 2, 1 %에서 0.068, 0.114, 0.111, 0.466, 0.590입니다. 5 → 2 %에서 급증하며, F3 B에서 GP UMG가 떨어지기 시작하는 지점과 같습니다.",
        "파라미터가 1개(length scale)인 가장 단순한 커널도 2 % 아래에서는 데이터가 모자랍니다. 이방성(major, minor, 방위각)을 표현하려 파라미터를 늘리면 더 많은 데이터가 필요하다는 논의로 이어집니다(F5는 보류, 논의로만).",
        "A는 아직 이전 density 실험(replicate 10개) 데이터입니다. F3 C와 같은 E4 100셀로 교체하면 두 그림의 숫자가 일치합니다(TODO).",
        "B(placeholder, 선택): 파라미터가 더 많은 커널(ARD 이방성 GP)의 같은 지표를 겹쳐 보일 자리입니다. 신규 실행이 필요하며, 없어도 A와 논의로 논리는 이어집니다.",
        "→ P3: 파라미터가 흔들리는 밀도에서 calibration은 어디서부터 나빠지는가.",
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
        "F3c, F3d의 GP 열은 posterior mean(추정값)이고 SGS 열은 realization이라 종류가 다른 map입니다. GP 파라미터로 만든 realization의 variogram·분포 재현은 아직 직접 보여주지 않습니다.",
        "→ P8, P9: 흔들린 GP 파라미터가 뜻하는 variogram 모델(P8), 그 파라미터로 뽑은 realization의 재현(P9).",
    ],
    "F7": [
        "질문: ground truth를 알고 전 영역에서 평가할 수 있는 이상적 조건에서 하이퍼파라미터를 grid search로 고르면, MSE로 고른 노드와 UMG로 고른 노드가 같은가?",
        "4개 방법 모두 MSE 최적 노드와 UMG 최적 노드가 다릅니다. 예: kriging MSE 최적 (600 m, 0.05)의 UMG 0.808, UMG 최적 (106 m, 0)의 MSE 5.562.",
                "10개 truth 평균 면에서 MSE 최적은 kriging, SGS 600 m, UMG 최적은 kriging, SGS, GP 106 m입니다. 실제 값(300 m, 0.05)은 kriging MSE 3.887, UMG 0.928입니다.",
        "주의(T2): 최적 노드와 2위 노드의 차이는 ground truth 간 표준편차보다 훨씬 작습니다.",
        "→ F4: 실제로는 ground truth로 grid search를 할 수 없습니다. 불확실성이 내장된 GP-MLE가 데이터만으로(marginal likelihood) 정한 파라미터는 어떤가.",
    ],
    "P3": [
        "중앙값은 밀도 20 → 1 %에서 GP-MLE 0.982 → 0.915, oracle kriging 0.957 → 0.930, oracle SGS 0.956 → 0.946입니다.",
        "하위 꼬리가 먼저 갈립니다. 1 %에서 10번째 백분위수는 GP-MLE 0.774, kriging 0.777, SGS 0.842이고, 최솟값은 GP-MLE 0.519, kriging 0.614, SGS 0.685입니다.",
        "셀 간 표준편차는 GP-MLE 0.017 → 0.101, SGS는 0.048~0.068 사이입니다.",
        "→ F3c, F3d: 흔들린 파라미터는 실제로 어떤 map을 만드는가. realization 생성에서는 이 파라미터가 그대로 variogram·분포 모델이 됩니다.",
    ],
    "P4": [
        "A: range 오차와 UMG 차이의 순위상관은 희소할수록 약해지고 부호가 바뀝니다. Spearman ρ 20 % +0.457, 10 % +0.366, 5 % +0.115, 2 % −0.126, 1 % −0.046입니다. range가 실제에서 멀다고 GP가 SGS보다 덜 calibrate되지는 않습니다.",
        "B: noise와의 순위상관은 모든 밀도에서 양수입니다(+0.774, +0.639, +0.609, +0.347, +0.313). noise를 크게 잡은 셀일수록 GP가 SGS보다 UMG가 높습니다.",
        "noise가 하한값(10⁻⁵)에 걸린 셀은 2 % 6개, 1 % 16개입니다. 이 22셀 중 17셀에서 GP-MLE UMG가 SGS보다 낮고(평균 차이 −0.083), 같은 밀도의 나머지 셀은 평균 −0.016입니다. 1 %에서 GP-MLE UMG 평균은 하한 셀 0.828, 나머지 0.901입니다.",
        "GP가 SGS보다 낮은 셀 수는 20 % 16개 → 1 % 61개입니다(100개 중).",
        "⑤의 고리는 'range가 흩어진다'(F3 C)보다 'noise가 0으로 붕괴한다'(F3c, F3d의 짧은 range 셀) 쪽이 데이터와 맞습니다. range의 흩어짐은 불안정성 지표로 남기고, calibration 손실과의 연결은 noise로 잡습니다.",
        "→ F3c, F3d: 흔들린 파라미터는 실제로 어떤 map을 만드는가. realization 생성에서는 이 파라미터가 그대로 variogram·분포 모델이 됩니다.",
    ],
}

PH = {
    "P8": dict(
        title="GP가 맞춘 파라미터가 뜻하는 variogram 모델, 밀도별",
        why="realization을 만들 때 쓰일 variogram 모델 자체가 밀도별로 얼마나 흔들리는지 보입니다. range 숫자 하나(F3 C)가 아니라 range·noise·sill을 함께 담은 곡선이라 ⑥의 직접 근거가 됩니다.",
        data="E4 100셀의 GP-MLE 하이퍼파라미터(ConstantKernel 분산, RBF length scale, WhiteKernel noise)로 implied variogram γ(h) = noise + c(1 − exp(−h²/2l²))를 그려 밀도별 패널(20, 5, 2, 1 %)에 겹치고, 정답 모델(spherical 300 m, nugget 0.45)을 점선으로 둡니다. noise 하한 셀은 색으로 구분합니다.",
        cost="기존 데이터만, 신규 실행 없음(data-curator → reviewer, 수 분).",
    ),
    "P9": dict(
        title="GP 파라미터로 뽑은 realization의 variogram·히스토그램 재현",
        why="F3c, F3d는 GP posterior mean과 SGS realization을 비교해서 종류가 다른 map입니다. GP posterior에서 뽑은 realization과 SGS realization을 같은 셀에서 비교해야 '흔들린 파라미터가 realization 생성에 치명적'을 직접 보입니다.",
        data="밀도 2 %, 1 %의 셀(최소 F3c, F3d의 극단 셀 4개, 가능하면 200셀 전부)에서 GP posterior realization 50개(sklearn sample_y, 변환은 본문 GP 파이프라인과 같게) → 셀별 variogram 재현 오차와 분포 재현(분산 비, 히스토그램)을 같은 셀의 SGS 50개와 비교.",
        cost="신규 계산(소규모). 셀당 2,500×2,500 공분산이라 200셀도 수십 분 수준으로 추정(coder → reviewer).",
    ),
}

SECTIONS = [
    ("문제 제기", "대부분의 정확도 기반 모델은 불확실성을 내장하지 않는다", "⓪. RBF 보간부터 최신 ML 회귀 모델까지 많은 공간 예측 모델은 정확도를 목표로 학습하고 예측 분산을 내놓지 않습니다. 불확실성이 필요하면 bootstrap 같은 사후 방법을 덧붙입니다. 이 논문은 그 가장 단순한 대표로 RBF+bootstrap을 쓰고, 불확실성이 내장된 kriging·SGS(oracle variogram)와 GP-MLE를 비교 대상으로 둡니다. 근거 문헌은 TODO입니다.", []),
    ("전제", "정확도와 불확실성 품질은 따로 움직인다", "①. 논문 전체의 평가 틀입니다. 이후 모든 그림은 MSE와 UMG를 따로 봅니다.", ["F1"]),
    ("결론 1 (핵심)", "정확도 기반 모델에 사후로 붙인 불확실성은 맞지 않고, 정확도로 고른 파라미터는 UQ 최적과 다르다", "② → ③. 원인(예측 분산이 작음, F2) → ground truth를 아는 grid search에서도 MSE 최적과 UMG 최적 노드가 다름(F7).", ["F2", "F7"]),
    ("결론 2", "GP-MLE는 잘 calibrate되지만, 현장에서 흔한 희소 데이터에서는 최적화된 파라미터가 흔들리고 이는 realization 생성에 치명적일 수 있다", "④ → ⑤ → ⑥. 데이터가 충분하면 잘 맞음(F4, F3) → 희소할수록 파라미터가 흔들림(F3) → calibration 하위 꼬리(P3) → 흔들린 파라미터로 만든 map(F3c, F3d) → realization 생성의 variogram·분포 재현(P8, P9 placeholder).", ["F4", "F3", "P3", "F3c", "F3d", "P8", "P9"]),
    ("시사점", "range·nugget은 데이터 적합 objective가 아니라 공간통계로 정해야 한다", "⑦. 이 논문에서 kriging/SGS는 oracle variogram, 즉 range·nugget을 올바르게 정했을 때의 기준입니다(F3, P3에서 희소 조건에서도 UMG 유지). 데이터에서 variogram을 정하는 경우는 논문 끝 OBM 검증으로 다룹니다(보류 중). 이방성(F5)은 보류 상태이므로 논의로만 다룹니다.", []),
    ("부록", "밀도별 실제 map과 샘플 위치", "본문 결론 2의 밀도 축 설정(Fig. A1)과 Fig. 4 truth들의 실험 variogram(Fig. A2)입니다(2026-11-20 사용자 결정).", ["F3b", "A2"]),
]

FLOW = [
    ("전제", "F1", "MSE와 UMG, 3개 축, 4개 방법", REAL),
    ("1", "F2", "Calibration 곡선, 예측 분산", REAL),
    ("1", "F7", "정확도와 UMG 반응면, 4개 방법", REAL),
    ("2", "F4", "GP가 맞춘 range, nugget 대 실제 값", REAL),
    ("2", "F3", "밀도별 oracle kriging/SGS와 GP-MLE MSE, UMG, GP range", REAL),
    ("2", "P3", "셀별 UMG 분포(꼬리)", REAL),
    ("2", "F3c", "극단값 2 %: GP range 최소·최대 셀", REAL),
    ("2", "F3d", "극단값 1 %: 같은 구성", REAL),
    ("2", "P8", "GP 파라미터가 뜻하는 variogram 모델, 밀도별 (기존 데이터)", PROP),
    ("2", "P9", "GP realization의 variogram·히스토그램 재현 (소규모 신규 계산)", PROP),
    ("부록", "F3b", "밀도별 실제 map과 샘플 위치", REAL),
    ("부록", "A2", "Fig. 4 truth realization의 실험 variogram", REAL),
]

TODO = [
    ("P8", "GP implied variogram 곡선, 밀도별", "기존 데이터, 수 분"),
    ("P9", "GP posterior realization 재현 vs SGS", "소규모 신규 계산"),
    ("F1, F2, F3", "SGS realization, bootstrap 수를 N = 100, 150으로", "현재 N = 50 (배치 1)"),
    ("문제 제기", "최신 ML 공간 예측 모델의 UQ 내장 여부·사후 UQ 관행 문헌(literature-scout)", "문헌 조사만"),
    ("OBM 검증", "OBM ground truth에서 사용자가 그린 variogram으로 2 % 검증(docs/obm_addon_plan.md)", "보류 중, 약 1~2시간 실행"),
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
             '<p class="lede">그림 %d개를 하나의 논리로 엮은 초안입니다(2026-11-20 갱신). 각 그림의 마지막 → 줄이 다음 그림으로 넘어가는 연결입니다. '
             '중심 주장은 ‘정확도 기반 모델에 사후로 붙인 불확실성은 맞추기 어렵다’이고, RBF+bootstrap이 그 가장 단순한 대표입니다(2026-11-19 사용자 결정). kriging/SGS는 oracle variogram으로만 비교합니다. F5 이방성은 보류입니다.</p></header>' % n_real)
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
        h.append('<tr><td class="k">%s</td><td class="k"><a href="#%s">%s</a></td><td>%s</td><td>%s</td></tr>' % (c, f.lower(), LABEL[f], d, chip(st)))
    h.append('</tbody></table></div></section>')
    for tag, claim, lead, keys in SECTIONS:
        h.append('<details class="sec" open><summary><span class="tag">%s</span><span class="claim">%s</span></summary>'
                 '<p class="lead">%s</p><div class="figs">' % (tag, claim, lead))
        for k in keys:
            if k in PH:
                p = PH[k]
                h.append('<article class="fig ph" id="%s"><h3><span class="n">%s</span>%s %s</h3>' % (k.lower(), LABEL[k], p["title"], chip(PROP)))
                h.append('<dl class="phd"><dt>논리상 역할</dt><dd>%s</dd><dt>제안 데이터</dt><dd>%s</dd><dt>비용</dt><dd>%s</dd></dl></article>'
                         % (p["why"], p["data"], p["cost"]))
                continue
            f = FIGS[k]
            h.append('<article class="fig" id="%s"><h3><span class="n">%s</span>%s %s</h3>' % (k.lower(), LABEL[k], f["title"], chip(f["st"])))
            h.append(img(f["stem"], "%s %s" % (LABEL[k], f["title"])))
            h.append('<ul class="claims">' + "".join('<li%s>%s</li>' % (' class="next"' if c.startswith("→") else "", c) for c in CLAIMS[k]) + '</ul>')
            h.append('<details class="how"><summary>그림 읽는 법</summary>' + "".join('<p>%s</p>' % p for p in f["body"]) + '<p>데이터: %s</p>' % status_note(f["stem"]) + '</details>')
            h.append('<div class="file">results/figures/paper_storyboard/%s.pdf</div></article>' % f["stem"])
        h.append('</div></details>')
    h.append('<details class="sec" open><summary><span class="tag">TODO</span><span class="claim">placeholder를 채우는 데 필요한 작업</span></summary>'
             '<div class="figs"><div class="tablebox"><table><thead><tr><th>대상</th><th>필요한 작업</th><th>비용</th></tr></thead><tbody>')
    for f, w, cur in TODO:
        h.append('<tr><td class="k">%s</td><td>%s</td><td>%s</td></tr>' % (f, w, cur))
    h.append('</tbody></table></div></div></details>')
    h.append('</div>')
    html = "\n".join(h)
    # renumber figure keys in the prose only (not inside base64 image data)
    parts = re.split(r'(src="data:image/png;base64,[^"]*")', html)
    html = "".join(x if x.startswith('src="data:') else fnum(x) for x in parts)
    OUT.write_text(html, encoding="utf-8")
    print(OUT, OUT.stat().st_size)


main()
