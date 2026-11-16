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
            "range와 nugget 축은 레벨당 ground-truth realization 1개, 샘플셋 1개입니다. density 축은 샘플 배치 replicate 10개의 평균과 ±1 표준편차입니다.",
        ]),
    "F2": dict(
        stem="F2_base_case_calibration_and_variance", st=REAL,
        title="Base case calibration 곡선과 예측 분산",
        body=[
            "A는 base case의 accuracy plot(명목 확률 구간 대 실제 포함 비율)입니다. B는 방법별 평균 예측 분산을 sample density에 대해 그린 것입니다.",
            "분산은 모두 porosity %² 단위입니다. kriging 분산은 normal-score 분산을 Monte Carlo로 역변환한 값입니다.",
        ]),
    "F3": dict(
        stem="F3_gp_vs_sample_density", st=REAL,
        title="밀도에 따른 GP와 kriging",
        body=[
            "A는 MSE, B는 UMG, C는 90% 구간 coverage입니다. kriging과 GP-MLE를 진하게, 나머지 두 방법을 옅게 그렸습니다. C의 점선은 공칭값 0.90입니다.",
            "90% coverage는 각 방법이 내놓은 90% 예측 구간 안에 truth가 실제로 들어간 셀의 비율입니다. UMG를 계산하는 accuracy plot에서 p = 0.90인 한 점이고, UMG와 같은 계산 경로를 씁니다(kriging은 normal-score 구간의 back-transform, 나머지는 평균 ± 표준편차의 정규분포).",
            "D는 GP-MLE가 맞춘 practical range를 replicate마다 점 하나로 찍은 것입니다. 점선은 ground-truth range 300 m입니다. 최소 144.6 m, 최대 739.9 m입니다.",
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
            "A는 등방성 GP-MLE가 맞춘 practical range의 변동계수(표준편차 ÷ 평균, replicate 10개)입니다. 밀도 17.8, 9.5, 4.8, 2.0, 1.0 %에서 0.068, 0.114, 0.111, 0.466, 0.590입니다.",
            "B는 가상 그림입니다. 파라미터가 더 많은 커널을 맞춘 실험이 없습니다.",
        ]),
    "F7": dict(
        stem="F7_response_surfaces_accuracy_vs_uncertainty", st=NONE,
        title="정확도와 UMG의 반응면",
        body=[
            "두 패널 모두 가상 그림입니다. 등고선 모양과 두 최적점의 위치는 임의로 정한 것이며 어떤 결과에서도 나오지 않았습니다.",
            "들어갈 그림의 형식만 보여줍니다. x축은 range, y축은 nugget 하이퍼파라미터이고, 채운 별은 해당 면의 최적점, 빈 별은 다른 면의 최적점입니다.",
        ]),
}

CLAIMS = {
    "F1": [
        "정확도(MSE)는 4개 방법이 세 축 모두에서 비슷합니다. 정확도만 보면 방법 간 차이가 드러나지 않습니다.",
        "UMG는 RBF+bootstrap만 크게 낮습니다. range 축 0.29~0.47, nugget 축 0.30~0.47, density 축 0.46~0.53이고, 나머지 세 방법은 0.83 이상입니다.",
        "이 차이는 ground truth의 range, nugget, 샘플 밀도를 바꿔도 유지됩니다. 특정 조건 하나에서만 나온 결과가 아니라는 근거입니다.",
        "정확도와 uncertainty 품질은 따로 봐야 한다는 논문의 전제를 한 장으로 보여줍니다.",
    ],
    "F2": [
        "A: RBF+bootstrap의 곡선이 1:1 선 아래로 크게 내려갑니다. 명목 95 % 구간에 참값이 실제로는 48 %만 들어옵니다. 구간이 너무 좁다는 뜻입니다.",
        "B: 그 원인이 예측 분산의 크기에 있음을 보여줍니다. RBF+bootstrap의 평균 분산은 0.3~1.5로, kriging의 2.3~7.3보다 훨씬 작습니다.",
        "bootstrap은 샘플셋이 바뀔 때 생기는 변동만 잡고, 샘플이 없는 곳의 공간적 불확실성은 잡지 못한다는 결론 1-a의 근거입니다.",
    ],
    "F3": [
        "A, B: 평균만 보면 GP-MLE는 모든 밀도에서 kriging과 MSE, UMG가 비슷합니다. 현재 데이터에서는 저밀도에서도 평균 UMG 차이가 작습니다(1 %에서 약 0.90 대 0.91).",
        "C: 90% coverage는 밀도가 17.8 %에서 1 %로 줄 때 GP-MLE가 0.90에서 0.85로, kriging이 0.92에서 0.86으로 떨어집니다. 모든 밀도에서 GP-MLE가 kriging보다 낮지만, 둘의 차이는 밀도가 줄어도 커지지 않습니다. 1 %에서 replicate 간 표준편차는 0.07~0.08입니다.",
        "D: GP가 맞춘 range는 밀도가 낮아지면 replicate마다 크게 달라집니다. 17.8 %에서는 168~208 m에 모이고, 1 %에서는 145~740 m로 퍼집니다.",
        "따라서 이 그림이 뒷받침하는 것은 '저밀도에서 GP의 하이퍼파라미터 추정이 샘플 배치에 따라 불안정해진다'입니다. 'GP의 uncertainty가 fail한다'까지는 현재 평균 UMG로는 보이지 않습니다.",
        "밀도가 충분한 구간(4.8 % 이상)에서도 GP가 맞춘 range는 실제 값 300 m보다 짧은 쪽에 모여 있습니다.",
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
        "A: 파라미터가 1개(length scale)뿐인 등방성 GP도 밀도가 낮아지면 추정이 흔들립니다. 변동계수가 17.8 %에서 0.07, 1 %에서 0.59입니다.",
        "가장 단순한 커널도 이 정도 데이터가 필요하므로, 이방성을 표현하려고 파라미터를 늘리면 더 많은 데이터가 필요하다는 논의의 출발점입니다.",
        "B (예정): 파라미터가 더 많은 커널을 같은 축에 겹쳐 보일 자리입니다. 데이터가 없어 아직 주장할 수 있는 내용이 없습니다.",
    ],
    "F7": [
        "(예정) 정확도(MSE)가 가장 좋은 하이퍼파라미터 조합과 UMG가 가장 좋은 조합이 서로 다른 위치에 있음을 보일 자리입니다.",
        "(예정) 그렇다면 grid search로 하나의 최적점을 찾는 방식으로는 둘을 동시에 만족시키기 어렵다는 결론 2의 근거가 됩니다.",
        "실험이 없어 현재는 그림 형식만 있습니다. 두 최적점이 실제로 다른 곳에 있는지는 아직 확인되지 않았습니다.",
    ],
}

SECTIONS = [
    ("결론 1-a", "불확실성이 모델에 없는 방법은 bootstrap으로도 평가하지 못한다", ["F1", "F2"]),
    ("결론 1-b", "불확실성이 내재된 GP는 데이터가 희소해지면 흔들린다", ["F3", "F4", "F6"]),
    ("결론 2", "정확도와 uncertainty goodness의 최적점은 서로 다르다", ["F7"]),
]

FLOW = [
    ("1-a", "F1", "MSE와 UMG, 3개 축, 4개 방법", REAL),
    ("1-a", "F2", "Calibration 곡선, 예측 분산", REAL),
    ("1-b", "F3", "밀도별 GP와 kriging(MSE, UMG, 90% coverage), GP가 맞춘 range", REAL),
    ("1-b", "F4", "GP가 맞춘 range, nugget 대 실제 값", REAL),
    ("1-b", "F6", "파라미터 수와 필요한 데이터 양", PART),
    ("2", "F7", "정확도와 UMG 반응면", NONE),
]

TODO = [
    ("F1, F2, F3", "range, nugget 축을 여러 ground-truth realization으로 재실행", "현재 레벨당 1개"),
    ("F1, F2, F3", "bootstrap 수와 SGS realization 수를 최종 설정으로 재실행", "현재 둘 다 10"),
    ("F3, F6", "density 축을 여러 ground-truth realization으로 확장", "현재 1개, 샘플 배치만 10회"),
    ("F6-B", "파라미터가 더 많은 커널의 적합 안정성", "실험 없음"),
    ("F7", "range × nugget grid에서 MSE와 UMG 동시 평가", "실험 없음"),
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
