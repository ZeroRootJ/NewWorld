# Project Context: Benchmarking Uncertainty Quantification in Spatial Prediction

> Handoff document for Claude Code. This describes the research claim, the current
> weakness in the study, and the experimental design intended to address it.
> Read this fully before writing any code.

---

## 1. Research Claim

The paper benchmarks **machine learning spatial predictors against conventional
geostatistical methods**, with the focus on *uncertainty quality*, not accuracy alone.

Two specific claims:

1. **RBF + bootstrap underestimates uncertainty.**
   The Radial Basis Function interpolator is deterministic and has no native
   uncertainty metric. Bootstrapping is bolted on to produce a distribution of
   outcomes, but bootstrap resampling only captures variability arising from the
   *sample set*, not the full spatial uncertainty at unsampled locations. The
   resulting uncertainty model is expected to be systematically too narrow.

2. **Automatic hyperparameter selection in GPR is risky.**
   Gaussian Process Regression does have an intrinsic probabilistic output
   (posterior mean + variance). However, hyperparameters chosen automatically —
   by marginal likelihood maximization or cross-validated MSE minimization — are
   optimized for *accuracy*. There is no guarantee that an accuracy-optimal
   configuration yields a calibrated uncertainty model.

Reference baselines: **simple kriging** (estimation) and **sequential Gaussian
simulation / SGS** (simulation). These are the "correct answer" for uncertainty,
since their uncertainty model follows directly from the inferred spatial statistics.

### Explicitly out of scope
An objective function that jointly balances accuracy and uncertainty quality is
**deferred to future work**. Do not implement or optimize one. This study is a
*diagnostic benchmark* only — it establishes that the problem exists and under
what conditions.

---

## 2. The Weakness Being Addressed

All current results come from a **single ground truth model**. A reviewer can
reasonably object that the findings are cherry-picked — that this one realization
happens to favor the conclusion.

**Goal of the new experiments:** demonstrate that the two claims above hold
across a systematically varied set of ground truth models, and identify the
conditions under which each method fails or holds up.

Framing matters here. The result should not read as *"we checked several models
and it still worked."* It should read as *"we varied the controlling properties
one at a time and characterized where each method breaks down."* That is a
stronger paper and it neutralizes the cherry-picking objection directly.

---

## 3. Experimental Design

### Structure
**One base case, one axis varied at a time.** Do not vary multiple axes
simultaneously in the primary results. Full factorial combinations may be added
later as a supplement, but the headline results must be one-factor-at-a-time so
each effect is attributable.

The ground truth models are synthetic and fully controllable — they are generated
by unconditional simulation, so the true field is known everywhere and uncertainty
model calibration can be measured exactly.

### Base case
A single reference model from which each variation departs. Keep every property
fixed except the one under test. Base case parameters (isotropic, moderate range,
zero-to-low nugget, interior-sampled data configuration) should be stated
explicitly in the code as named constants so the variations are obvious diffs.

### Axes, in priority order

**1. Variogram range (spatial continuity)**
- Short range, on the order of the data spacing — most unsampled locations are
  effectively uncorrelated with the data. Kriging correctly reverts toward the
  global mean with variance approaching the sill. The question is whether GPR and
  RBF-bootstrap do the same, or collapse their uncertainty inappropriately.
- Long range — a regime that should favor the ML methods. If the claims survive
  here too, they are much stronger.

**2. Nugget effect (noise / short-scale variability)**
- Zero nugget vs. substantial nugget (e.g. 20–30% of sill).
- This axis targets Claim 2 most directly. The RBF smoothing parameter (λ) and
  the GPR noise term are both nominally meant to absorb nugget-scale variance.
  The question is whether automatic tuning actually assigns it there, or instead
  fits the nugget as signal — which compresses the predicted uncertainty and
  produces overconfident models.

**3. Extrapolation / data configuration**
- Same number of samples (same sparsity) but arranged so a large fraction of the
  prediction domain falls outside the data convex hull.
- Kriging degrades gracefully toward the mean with increasing variance. RBF and
  GPR are expected to be more sensitive here, both in estimate and in uncertainty.
- Keep sample count identical to the base case so the effect is attributable to
  configuration, not sample density.

**4. Anisotropy**
- Isotropic base case vs. strong geometric anisotropy (ratio ≥ 3:1) in a
  non-axis-aligned direction.
- RBF with an isotropic kernel is structurally unable to represent this. GPR with
  ARD can in principle learn it, but with sparse data may learn the wrong
  direction — a failure mode worth showing explicitly.

**5. Sample count / sparsity (TODO — added 2026-09-12, not yet specced)**
- 별도 축으로: 샘플 **개수**를 base case(100) 대비 줄여가며(예: 100 → 50 → 25) sparsity 자체가
  4개 방법의 calibration에 미치는 영향을 측정한다. Sample *위치/구성*(convex hull 안/밖)을 바꾸는
  axis 3과는 분리한다 — axis 3은 여전히 샘플 개수를 base case와 동일하게 고정해서 "configuration
  자체의 효과"만 보는 통제된 실험으로 유지한다 (밑에 원래 문구 그대로 둠).
- 구체적인 sample count 단계, margin, 배치 규칙은 base case 결과를 본 뒤 정한다 — 지금은 TODO.

---

## 4. Methods to Compare

Four predictors, evaluated on every ground truth model:

| Method | Type | Uncertainty source |
|---|---|---|
| Simple kriging | Geostatistical, estimation | Kriging variance |
| Sequential Gaussian simulation | Geostatistical, simulation | Ensemble of realizations |
| RBF + bootstrap | ML, deterministic + resampling | Spread across bootstrap replicates |
| GPR | ML, Bayesian | Posterior predictive variance |

Notes:
- SGS requires **multiple realizations** evaluated jointly. A single realization
  is not a valid uncertainty model. Same for the bootstrap ensemble — use a
  consistent, adequate replicate count across all methods and cases.
- RBF has two hyperparameters of interest: the shape parameter (ε) and the
  smoothing parameter (λ). GPR has kernel length scale(s), variance, and noise.
  Tune these by the automatic procedure the paper is critiquing (CV-MSE for RBF,
  marginal likelihood or CV-MSE for GPR) — the point is to show what that
  procedure does to uncertainty, so do not hand-tune them.
- Geostatistical baselines should use the **true** variogram of the ground truth
  model, or a variogram inferred from the samples. Decide once and apply
  consistently; if inferred, note that inference error is part of the comparison.

---

## 5. Evaluation

Accuracy and uncertainty quality are reported **separately**. The entire argument
depends on showing they can diverge.

- **Accuracy:** MSE / RMSE, MAE at withheld locations.
- **Uncertainty quality:** calibration of the predicted distributions against the
  known truth — e.g. coverage of prediction intervals across nominal levels
  (an accuracy plot / goodness measure), interval width, and a proper scoring
  rule. The key diagnostic is whether observed coverage falls below nominal
  (overconfidence / underestimated uncertainty), which is the predicted failure
  mode for both RBF-bootstrap and auto-tuned GPR.

Because the ground truth is synthetic and fully known, coverage can be computed
over the entire domain, not just at a handful of withheld points. Use that.

**Expected result pattern to test (not to assume):** ML methods competitive or
better on MSE, while showing materially worse calibration than kriging/SGS — and
the gap widening along the axes above.

---

## 6. Practical Requirements for the Implementation

- Random seeds controlled and recorded; results must be reproducible.
- Multiple ground truth realizations per configuration where feasible, so
  reported differences are not a single-realization artifact — this is the exact
  criticism the work is answering, so it applies to the new experiments too.
- Identical sample locations across methods within a given case. Method
  comparison must not be confounded by different conditioning data.
- Results stored in a tidy long-format table (case, axis, axis level, method,
  metric, value) so figures can be regenerated without rerunning experiments.
- One figure per axis showing accuracy and calibration side by side, so the
  divergence between the two is visible in a single panel.

---

## 7. Deliverable Order

1. Base case, reproducing the existing single-model result within the new codebase.
2. Variogram range variation.
3. Nugget variation.
4. Extrapolation / configuration variation.
5. Anisotropy variation.

Confirm each stage reproduces sensible geostatistical behavior before moving on —
if kriging itself is not calibrated on a given case, the case is misconfigured,
not informative.
