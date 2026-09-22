# GeostatsPy 코딩 컨벤션

이 프로젝트에서 kriging / SGS 코드를 작성할 때는 **항상 이 문서의 컨벤션을 기준으로 한다.**
출처: Michael Pyrcz, *GeostatsPy Demos Book*
(https://geostatsguy.github.io/GeostatsPyDemos_Book/intro.html). 새 패턴이 필요할 때는
임의로 만들지 말고 먼저 이 책에서 해당하는 챕터를 확인한다.

## 1. 기본 import 블록
모든 geostatspy 관련 스크립트/노트북 상단은 이 형태를 따른다:

```python
import geostatspy.GSLIB as GSLIB          # 래퍼/시각화/변환 유틸
import geostatspy.geostats as geostats    # GSLIB 수치 알고리즘 (gam, gamv, kb2d, sgsim, nscore, vmodel ...)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

모듈 역할 분리를 그대로 따른다: `GSLIB.*` = 시각화·데이터 변환(`locmap_st`, `pixelplt_st`,
`locpix_st`, `make_variogram`, `nscore`, `affine`), `geostats.*` = 핵심 수치 루틴
(`gamv`, `kb2d`, `sgsim`, `nscore`, `vmodel`).

> **예외 1건**: `geostats.nscore`는 이 프로젝트에서 **직접 호출하지 않는다.** 0.0.79에 off-by-one
> 버그가 있어 `src/nscore.py` 래퍼를 통해서만 쓴다 — 3.1절 참고.

## 2. 재현성: 시드
- 책 전체에서 `random_state`/`seed`로 고정된 시드 상수를 재사용한다 (책의 예시는 `73073`).
- 이 프로젝트에서는 **[coder.md](../.claude/agents/coder.md)의 저장 컨벤션에 따라 시드를 하드코딩하지 말고
  `manifest.json`에 기록되는 실험 파라미터의 일부로 관리**한다. 다만 "시드는 스크립트 상단에 이름 붙은
  상수로 한 번만 정의하고 모든 랜덤 연산에 그 상수를 재사용한다"는 패턴 자체는 그대로 따른다
  (예: `SEED = 42` 하나를 정의해 pandas sampling과 `sgsim(seed=...)`에 동일하게 전달).

## 3. Variogram 컨벤션
- Gaussian simulation에 쓸 데이터는 variography 전에 **normal-score 변환**을 명시적으로 먼저 한다.
  책의 원문 패턴은 `geostats.nscore`를 직접 부르지만, **이 프로젝트에서는 그 호출을 금지하고
  로컬 래퍼 `src/nscore.py`를 쓴다** (아래 3.1 참고):
  ```python
  # 책 원문(사용 금지):  df['NVar'], tv, tns = geostats.nscore(df, 'Var')
  from src.nscore import nscore as nscore_corrected

  ns, vr, vrg, nscore_corrections = nscore_corrected(df, 'Var')
  df['NVar'] = ns
  ```
  반환값 4번째(`nscore_corrections`)는 JSON 직렬화 가능한 감사 기록이므로 그대로
  `manifest.json`의 `params`에 넣는다 (`kriging.py`가 그렇게 한다).
  (단, `sgsim`은 `itrans=1`이면 내부적으로 자체 변환을 수행하므로 **`sgsim` 호출 전에 수동으로
  다시 변환하지 않는다** — 중복 변환 금지, 6절 gotcha 참고)

### 3.1 `geostats.nscore` 직접 호출 금지 — 이 프로젝트의 로컬 수정
**이 항목은 Demos Book 원문 규약과 의도적으로 다르다.** 책은 `geostats.nscore(df, 'Var')`를
그대로 쓰지만, 설치본 **geostatspy 0.0.79**의 `nscore`에는 1-based → 0-based 포팅 off-by-one이
있어 **모든 데이터셋의 최솟값 데이텀 하나**에 잘못된 정규점수를 준다.

- 메커니즘: 변환 루프가 `j = dlocate(...)` 뒤에 `j = min(max(1, j), nd-1)`로 clamp한다. 원본
  GSLIB Fortran은 1-based라 이 식이 "첫 번째 구간으로 clamp"를 뜻하지만, Python 포팅은 0-based
  numpy 배열을 인덱싱하므로 **두 번째 구간으로** clamp된다. 최솟값 데이텀은 `dlocate`가
  `j=0`을 옳게 돌려주는데도 `j=1`로 밀려, 자기가 속한 `vr[0]→vr[1]` 구간 대신
  `vr[1]→vr[2]` 구간을 따라 **역방향 외삽**된다. 2·3번째로 작은 값이 가까울수록 그 구간의
  기울기가 폭발해 정규점수가 변환표 범위 밖으로 튄다.
- 발견 시점/근거: 2026-09-22, nugget axis(nug=0.3, 조건화 샘플 121개) run에서 `NPor`의 최솟값이
  **-52.145**로 나왔는데 같은 run의 변환표 최솟값 `vrg[0]`은 **-2.6411**이었다. 같은 레벨에서
  simple kriging MSE 9.4425 → 5.4079로 바뀌었다 (정량화: `results/processed/nscore_bug_impact.csv`).
- 수정 방식: `src/nscore.py`가 변환 루프를 **같은 라이브러리의 `sgsim`이 자기 내부 변환에서 쓰는
  올바른 0-based clamp**(`j = min(max(0, j), nd-2)`, `geostats.py` L3807)로 다시 돌린다. 즉 이
  프로젝트가 임의 규약을 만든 게 아니라, `nscore`를 같은 라이브러리의 `sgsim`과 일치시킨 것이다.
  변환표(`vr`/`vrg`) 자체는 다른 코드 경로에서 나오며 정상이므로 손대지 않는다.
- 래퍼는 결과를 검증한다: 비유한값(NaN/inf), 변환표 범위 `[vrg[0], vrg[-1]]` 이탈, 순서 비보존이
  있으면 `RuntimeError`로 중단한다 (알려지지 않은 실패 모드를 조용히 통과시키지 않기 위함).
  `ismooth=True`(참조분포로 변환표를 만드는 모드)는 표 바깥 값 처리 정책이 미정이라
  `NotImplementedError`를 던진다.
- 라이브러리는 **직접 패치하지 않는다** — `.venv` 안의 수정은 git에 남지 않아 `manifest.json`의
  `git_commit`이 실제 실행 코드와 어긋나고, 환경 재구축 시 조용히 사라진다.
- 회귀 테스트: `tests/test_nscore_offbyone.py`. 그 중 하나는 **라이브러리 원본을 직접 호출해
  버그가 여전히 존재함**을 확인하므로, upstream이 고쳐지면 이 테스트가 실패해서 알려준다.
  `requirements.txt`에 버전이 고정돼 있고, 설치 버전이 달라지면 `src/nscore.py`가
  `UserWarning`을 낸다.
- 워크스루: `notebooks/method_walkthrough.ipynb` §1.1이 래퍼 호출과 버그의 차이를 셀 출력으로 보여준다.
- 실험적 variogram은 `geostats.gamv`로 계산:
  ```python
  lag, gamma, npair = geostats.gamv(df, "X", "Y", "NVar", tmin, tmax,
      lag_dist, lag_tol, nlag, azi, atol, bandh, isill)
  ```
  minor 방향은 별도 파라미터가 아니라 `azi + 90.0`으로 계산하는 관행을 따른다.
- Variogram 모델은 `GSLIB.make_variogram(...)`으로 만든 **dict 하나**를 kriging과 SGS 양쪽에
  그대로 재사용한다 (변형해서 다시 만들지 않는다):
  ```python
  vario = GSLIB.make_variogram(nug=nug, nst=1, it1=it1, cc1=cc1,
      azi1=azi1, hmaj1=hmaj1, hmin1=hmin1)
  ```
  파라미터 키 이름: `nug`(nugget), `nst`(구조 개수), `itN`(1=spherical, 2=exponential, 3=Gaussian),
  `ccN`(sill 기여분), `aziN`(방위각), `hmajN`/`hminN`(주/부축 range).
- **반드시 sill까지 모델링한다** (`nug` + 모든 `ccN`의 합 = sill, 보통 normal-score 데이터는 1.0) —
  그렇지 않으면 시뮬레이션 realization들이 전역 분포를 재현하지 못한다.
- 기하학적 anisotropy는 major/minor range가 서로 다른 타원으로 표현되며 `aziN`이 major축 방향이다
  (이번 연구의 axis 4, anisotropy 실험과 직결).

## 4. Kriging 컨벤션
- 그리드 정의는 SGS와 동일한 변수명을 공유한다 (5절 참고).
- Simple/ordinary kriging은 `geostats.kb2d`, 반환값은 `(estimate_map, variance_map)` 튜플:
  ```python
  kmap, vmap = geostats.kb2d(df, 'X', 'Y', 'Var', vmin, vmax, nx, xmn, xsiz,
      ny, ymn, ysiz, nxdis, nydis, ndmin, ndmax, radius, ktype, skmean, vario)
  ```
- `ktype` (0=simple kriging, 1=ordinary kriging)는 `kb2d`와 `sgsim`에서 **동일한 이름/의미**로 쓰이는
  공유 플래그다 — 이 프로젝트는 simple kriging(`ktype=0`)이 기준선이므로 항상 명시적으로 `ktype=0`을 쓴다.
- Simple kriging의 평균/분산은 데이터로부터 명시적으로 계산해서 전달한다:
  ```python
  skmean = np.average(df['Var'].values)
  sill = np.var(df['Var'].values)
  ```
- **동작 특성 문서화**: simple kriging은 데이터에서 멀어질수록 전역 평균으로 revert하고 분산은 sill에
  근접한다 (이번 연구 axis 1, short range 실험에서 기대하는 "정답" 거동). ordinary kriging은 로컬 평균으로
  revert한다는 점과 구분해서 주석에 남긴다.

## 5. SGS 컨벤션
- 그리드 변수명은 kriging과 통일: `nx, ny`(셀 개수), `xmn, ymn`(그리드 원점 = **첫 셀의 중심**, GSLIB 관례),
  `xsiz, ysiz`(셀 크기). bounding box `xmin/xmax/ymin/ymax`와 `xmn/ymn`을 혼동하지 않는다
  (`xmin=0.0`이어도 `xsiz=10`이면 `xmn=5`).
- `geostats.sgsim` 호출은 keyword argument를 명시적으로 채워서 쓴다 (magic number로 나열하지 않는다):
  ```python
  sim = geostats.sgsim(df, 'X', 'Y', 'Var', wcol=-1, scol=-1,
      tmin=-999, tmax=999, itrans=1, ismooth=0, dftrans=0, tcol=0, twtcol=0,
      zmin=zmin, zmax=zmax, ltail=1, ltpar=0.0, utail=1, utpar=0.3,
      nsim=nreal, nx=nx, xmn=xmn, xsiz=xsiz, ny=ny, ymn=ymn, ysiz=ysiz,
      seed=SEED, ndmin=0, ndmax=20, nodmax=20, mults=1, nmult=3,
      noct=-1, ktype=0, colocorr=0.0, sec_map=0, vario=vario)
  ```
- **여러 realization이 있어야 유효한 불확실성 모델**이다 (`nsim=nreal`, `nreal`은 이 연구에서 축별로
  통일된 replicate 수를 써야 함 — [experiment_context.md](experiment_context.md) 4절 참고).
  결과는 `sim[0]`, `sim[1]`, ... 형태로 realization별로 인덱싱한다.
- **QC 체크리스트** (매 실험 케이스마다 확인, [reviewer.md](../.claude/agents/reviewer.md)의 재현성
  체크리스트에 포함): 각 realization이 (1) 조건 데이터를 정확히 지나는지(honor), (2) 원본 히스토그램을
  재현하는지, (3) variogram 구조를 보존하는지.
- 단일 realization의 요약 통계는 목표 통계량과 어긋날 수 있다는 ergodic fluctuation을 항상 염두에 둔다
  — 이 연구가 "단일 ground truth의 cherry-picking" 문제를 다루는 것과 같은 맥락이므로 특히 유의.

## 6. 그리드/데이터 컨벤션 및 gotcha
- DataFrame 변수명은 `df`, 좌표 컬럼은 문자열 리터럴 `'X'`, `'Y'`로 전달 (별도 `xcol`/`ycol` 변수로
  감싸지 않는 게 책의 관례지만, 이 프로젝트에서는 여러 축(axis) 실험을 스크립트화해야 하므로
  `xcol='X'`처럼 이름 붙은 상수로 한 번 선언해 재사용하는 걸 권장 — coder.md의 "named constants" 원칙과 일치).
- **중복 정규분포 변환 금지**: variography용으로 `nscore`를 수동 호출했더라도, `sgsim(itrans=1)`은
  내부적으로 다시 정규분포 변환을 수행한다. 두 변환을 혼동해서 이중 적용하지 않는다.
- Facies/indicator는 `0/1` 이진 코딩 관례를 따른다 (이 연구에서 indicator 변수가 필요할 경우).
- 플롯은 가능하면 geostatspy의 고수준 래퍼(`GSLIB.locmap_st`, `GSLIB.pixelplt_st`, `GSLIB.locpix_st`)를
  우선 사용하고, `vmin/vmax`, 축 라벨, colorbar 라벨을 항상 명시한다. 여러 realization/방법 비교는
  `plt.subplot(nrows, ncols, i)` 그리드 + 마지막에 `plt.subplots_adjust(...)` → `plt.savefig(..., dpi=600,
  bbox_inches="tight")` 패턴을 따른다 (이 연구의 "axis당 accuracy/calibration 동시 표시" 요구사항과 부합).
- **Black-box 금지 원칙** (저자가 명시): variogram/kriging/SGS 파라미터를 이름 붙은 상수로 문서화하고,
  기본값을 그대로 쓰지 않는다. 이는 이 프로젝트의 `coder.md`에 있는 "base case 파라미터는 명시적 상수로"
  원칙과 정확히 일치하므로 강제 사항으로 취급한다.

## 7. 이 프로젝트에 적용 시 우선순위
위 컨벤션 중 이 연구와 직접 관련된 것부터 우선 적용한다:
1. Variogram dict를 kriging(`kb2d`)과 SGS(`sgsim`) 사이에 공유 — 4개 방법 비교 시 "동일 조건" 요구사항과 직결.
2. `ktype=0`(simple kriging) 명시 — RBF/GPR과 대조되는 기준선이므로 절대 기본값에 의존하지 않는다.
3. SGS의 realization 개수(`nreal`)를 RBF bootstrap replicate 개수와 맞춰 "동일 replicate 수" 요구사항 충족.
4. 시드는 named constant로 한 번 정의 후 재사용하되, 실행마다 `manifest.json`에 기록 (책의 고정 시드
   관행과 이 프로젝트의 재현성 요구사항을 결합한 것).
