"""Project-level replacement for ``geostatspy.geostats.nscore``'s transform
loop, repairing a 1-based-to-0-based porting off-by-one in the upstream
library.

WHY THIS MODULE EXISTS (do not delete it, and do not "simplify" it away)
=======================================================================
``geostatspy.geostats.nscore`` (installed version 0.0.79, see
GEOSTATSPY_VERSION_VERIFIED below) assigns the WRONG normal score to the
SMALLEST datum of every data set it transforms. The bug is in the final
transform loop::

    j = dlocate(vr, 1, nd, vrr)
    j = min(max(1, j), (nd - 1))        # <-- 1-based clamp on a 0-based array
    ns[i] = dpowint(vr[j], vr[j + 1], wt_ns[j], wt_ns[j + 1], vrr, pwr)

The original GSLIB Fortran is 1-based, so ``max(1, j)`` there means "clamp to
the FIRST table segment". This Python port indexes 0-based numpy arrays, so
the same ``max(1, j)`` clamps to the SECOND segment instead. ``dlocate``
correctly returns ``j = 0`` for the minimum datum (``vrr < vr[1]``), the
clamp bumps it to ``j = 1``, and the datum is then obtained by extrapolating
BACKWARDS along the slope of the ``vr[1] -> vr[2]`` segment instead of being
read off the ``vr[0] -> vr[1]`` segment it actually lies on. The correct
answer for a strictly-smallest datum is exactly ``vrg[0]`` (i.e. ``wt_ns[0]``),
because that datum IS the table's first knot.

Whenever the 2nd and 3rd smallest values happen to be nearly tied, that
borrowed segment has a near-zero run, its slope explodes, and the minimum
datum gets a wildly out-of-range normal score.

THE FIX: THE SAME LIBRARY'S OWN 0-BASED CLAMP
---------------------------------------------
This module does not invent a correction rule. ``geostatspy.geostats.sgsim``
performs the identical normal-score transform internally and there the port
uses the CORRECT 0-based clamp (``geostats.py`` L3807)::

    j = dlocate(vrtr, 1, nd, vrr)
    j = min(max(0, j), (nd - 2))
    vrg = dpowint(vrtr[j], vrtr[j + 1], vrgtr[j], vrgtr[j + 1], vrr, 1.0)

``min(max(0, j), nd - 2)`` is the exact 0-based translation of what the GSLIB
Fortran expresses 1-based, and it is what the same library already does in
another function -- so adopting it here is not a project-local convention,
it is making ``nscore`` agree with ``sgsim``. :func:`_nscore_transform` below
re-runs ``nscore``'s transform loop verbatim except for that one clamp.

Rewriting the whole loop (rather than patching only data at/below ``vr[0]``,
which an earlier version of this module did) also removes an ambiguity when
the minimum value is TIED. With ties ``vr[0] == vr[1]``, so evaluating the
first segment hits ``dpowint``'s ``xhigh - xlow < EPSLON`` branch and returns
``(vrg[0] + vrg[1]) / 2`` -- neither a table knot nor the tied group's mean.
The correct clamp leaves ``dlocate``'s answer alone for tied minima and so
has no such special case. (Measured on the toy set ``[5, 5, 6, 7, 9, 12]``:
library ``-0.6745``, the old value-based patch ``-1.0287``, this module
``-0.6745``.) None of the project's data sets contain ties -- they are
continuous simulated values -- so this changes no existing number; see
``tests/test_nscore_offbyone.py``.

REPRODUCED ARITHMETIC (measured 2026-09-22, not quoted from a report)
---------------------------------------------------------------------
From ``results/raw/kriging/20260922T155131942671Z/`` (nugget axis, nug=0.3;
121 conditioning samples)::

    vr[0] = 7.89485789   vr[1] = 9.57722464   vr[2] = 9.58411332
    vr[2] - vr[1] = 0.00688868                     <-- near-tie
    vrg[1] = -2.24460677  vrg[2] = -2.04028133

    geostats.nscore's value for the minimum datum
      = vrg[1] + (vr[0] - vr[1]) / (vr[2] - vr[1]) * (vrg[2] - vrg[1])
      = -52.14538652095075
    correct value (this module's)
      = vrg[0] = -2.641070048882167

-52.145 is exactly the ``NPor`` minimum stored in that run's ``samples.csv``,
and -2.6411 is exactly ``vrg[0]`` of the same run's
``nscore_transform_table.csv`` -- i.e. the transform TABLE is fine, only the
transformed value assigned to the minimum datum is wrong. The maximum datum
is NOT affected: ``dlocate`` returns ``nd - 2`` there, which both the buggy
and the correct clamp leave alone, and interpolating at the right end of the
last usable segment returns ``vrg[-1]`` correctly.

Downstream impact measured on the nugget axis at nug=0.3: simple-kriging MSE
9.4425 -> 5.4079, UMG 0.8361 -> 0.9244, evaluated cells clipped at
BACKTR_ZMIN 60 -> 0.

WHY THE LIBRARY IS NOT PATCHED DIRECTLY
---------------------------------------
Editing ``.venv/lib/site-packages/geostatspy/geostats.py`` would fix the
symptom but the edit lives outside git: it would not appear in any commit, it
would be silently lost on any ``pip install -r requirements.txt`` /
environment rebuild, and every ``manifest.json``'s ``git_commit`` would then
describe code that does not match what actually ran. Keeping the correction
inside the repository makes it reviewable, versioned, testable, and
reproducible on a clean checkout. (Upstreaming the fix to GeostatsPy is a
separate, out-of-band action and does not change anything here.)

WHAT THIS WRAPPER DOES
----------------------
1. Calls ``geostats.nscore`` unchanged, and keeps its transform TABLE
   (``vr``, ``wt_ns``) and its raw ``ns`` verbatim. The table is computed by
   a different code path from the buggy loop and is correct.
2. Recomputes ``ns`` with :func:`_nscore_transform`, i.e. the library's own
   loop with ``sgsim``'s correct 0-based clamp. The EPSILON dither the
   library adds to each value is reproduced bit-for-bit from a private
   ``np.random.RandomState(73073)`` (the library seeds the GLOBAL numpy RNG
   with 73073 and draws ``len(df)`` values from it; using a private
   RandomState reproduces the same numbers without perturbing the global
   stream the library already advanced, so this wrapper leaves the process
   RNG in exactly the state a bare ``geostats.nscore`` call would).
3. VERIFIES the result and raises ``RuntimeError`` on anything that is wrong
   -- any non-finite normal score, any score outside ``[vrg[0], vrg[-1]]``,
   or any violation of the transform's order-preserving property. It
   deliberately does NOT silently "repair" unknown failure modes: an unknown
   mode must surface as a loud failure, in the same spirit as the existing
   ``n_samples_used_by_kb2d`` and ``KB2D_TMIN/TMAX`` guards in
   ``src/experiments/kriging.py``.

   Note that ``sgsim`` additionally CLAMPS its transformed value into
   ``[vrgtr[0], vrgtr[nd-1]]`` (``geostats.py`` L3808-3809). That clamp is
   deliberately NOT copied here: with the data themselves defining the table
   (``ismooth=False``, the only mode this module supports), every value lies
   inside ``[vr[0], vr[-1]]``, so a linear interpolation cannot leave the
   table range. Anything that did leave it would be an unknown failure and
   must be loud, not silently clamped. Guard 2 below is, for the same
   reason, currently unreachable on this code path (a monotone table
   interpolated monotonically cannot invert order); it is kept as a
   cheap invariant check against future changes, not because it is known to
   fire.
4. Returns the corrections it applied so the caller can log them and record
   them in its ``manifest.json``.

Usage (this is the ONLY nscore entry point any script in this repo may use --
calling ``geostats.nscore`` directly is the bug)::

    from src.nscore import nscore

    ns, vr, vrg, nscore_corrections = nscore(samples_df, VCOL)

VERSION SENSITIVITY
-------------------
This module exists to work around a bug in ONE specific release of
GeostatsPy. ``requirements.txt`` pins that release. If the installed version
ever differs from GEOSTATSPY_VERSION_VERIFIED a ``UserWarning`` is emitted
(not an exception -- the wrapper may well still be correct): a human must
then re-read the upstream ``nscore`` transform loop and decide whether this
work-around is still needed, still sufficient, or now double-corrects an
upstream fix. ``tests/test_nscore_offbyone.py`` contains a test that
exercises the raw library and fails loudly if upstream is fixed.
"""

import warnings
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import pandas as pd

import geostatspy
import geostatspy.geostats as geostats

# The version this off-by-one was read out of the installed source and
# reproduced arithmetically against (see module docstring). Recorded in the
# returned corrections dict / manifests so a future environment bump is
# visible in the results rather than silent.
GEOSTATSPY_VERSION_VERIFIED = "0.0.79"

# Absolute tolerance for the "every normal score lies inside the transform
# table's [vrg[0], vrg[-1]] range" guard. The table values are O(1) (|vrg| <=
# ~3 for realistic sample counts), so 1e-9 is far tighter than any genuine
# failure mode and far looser than float noise in a linear interpolation.
RANGE_TOL = 1e-9

# Absolute tolerance for the order-preserving guard (a normal-score transform
# is a monotone map of the data values, so sorting by value must give
# non-decreasing normal scores).
MONOTONE_TOL = 1e-9

# GSLIB hard-codes dpowint's interpolation power to 1.0 inside nscore
# (``pwr = 1.0  # interpolation power, hard coded to 1.0 in GSLIB``), i.e.
# plain linear interpolation. Reused here so the repaired values are computed
# by exactly the routine the library would have used had the index been right.
_NSCORE_PWR = 1.0

# The library's own two transform-loop constants, copied verbatim from
# geostatspy.geostats.nscore so the reimplemented loop is identical to it
# apart from the clamp: ``np.random.seed(73073)`` and
# ``vrr = val[i] + np.random.rand() * EPSILON`` with ``EPSILON = 1.0e-20``.
_NSCORE_DITHER_SEED = 73073
_NSCORE_EPSILON = 1.0e-20


def _warn_on_version_drift() -> None:
    installed = getattr(geostatspy, "__version__", None)
    if installed is not None and str(installed) != GEOSTATSPY_VERSION_VERIFIED:
        warnings.warn(
            "src.nscore works around a transform-loop off-by-one that was read "
            f"out of geostatspy {GEOSTATSPY_VERSION_VERIFIED}, but geostatspy "
            f"{installed} is installed. Re-read geostats.nscore's final loop "
            "and confirm this work-around is still needed and still correct "
            "(if upstream fixed the clamp, this module would now be "
            "redundant, not wrong -- but that must be checked by a human, and "
            "requirements.txt should be updated deliberately). See "
            "src/nscore.py's docstring.",
            UserWarning,
            stacklevel=3,
        )


class NscoreResult(NamedTuple):
    """Return value of :func:`nscore`.

    Unpacks as ``ns, vr, vrg, corrections`` -- the first three are exactly
    what ``geostats.nscore`` returns (with ``ns`` recomputed using the
    correct clamp), the fourth is this wrapper's audit record, intended to be
    dropped straight into a ``manifest.json``.
    """

    ns: np.ndarray
    vr: np.ndarray
    vrg: np.ndarray
    corrections: Dict[str, Any]


def _nscore_transform(values: np.ndarray, vr: np.ndarray, vrg: np.ndarray) -> np.ndarray:
    """``geostats.nscore``'s transform loop with ``sgsim``'s correct clamp.

    Line-for-line the upstream loop (same ``dlocate``, same ``dpowint``, same
    EPSILON dither drawn from the same seed) except that

        j = min(max(1, j), nd - 1)      # upstream nscore, 1-based clamp
    becomes
        j = min(max(0, j), nd - 2)      # upstream sgsim, 0-based clamp

    The dither comes from a private ``RandomState`` rather than the global
    numpy RNG so this function does not advance a stream the caller (or the
    ``geostats.nscore`` call that produced ``vr``/``vrg``) also uses.
    """
    nd = int(vr.size)
    n = int(values.size)
    dither = np.random.RandomState(_NSCORE_DITHER_SEED).rand(n) * _NSCORE_EPSILON
    ns = np.zeros(n, dtype=float)
    for i in range(n):
        vrr = values[i] + dither[i]
        j = geostats.dlocate(vr, 1, nd, vrr)
        j = min(max(0, j), nd - 2)
        ns[i] = geostats.dpowint(vr[j], vr[j + 1], vrg[j], vrg[j + 1], vrr, _NSCORE_PWR)
    return ns


def _correction_record(
    n_data: int,
    indices,
    values,
    raw,
    fixed,
) -> Dict[str, Any]:
    indices = list(map(int, indices))
    return {
        "geostatspy_version": GEOSTATSPY_VERSION_VERIFIED,
        "geostatspy_version_installed": getattr(geostatspy, "__version__", None),
        "bug": (
            "geostats.nscore's `j = min(max(1, j), nd-1)` is a 1-based GSLIB "
            "clamp applied to 0-based numpy arrays, so the minimum datum is "
            "extrapolated backwards off the vr[1]->vr[2] segment instead of "
            "being read off vr[0]->vr[1]. src/nscore.py re-runs the transform "
            "loop with the same library's own correct 0-based clamp from "
            "sgsim, `min(max(0, j), nd-2)`. See src/nscore.py's docstring."
        ),
        "n_data": int(n_data),
        "n_corrected": len(indices),
        "corrected_indices": indices,
        "corrected_data_values": [float(v) for v in values],
        "raw_normal_scores": [float(v) for v in raw],
        "corrected_normal_scores": [float(v) for v in fixed],
        "max_abs_correction": (
            float(np.max(np.abs(np.asarray(fixed) - np.asarray(raw)))) if indices else 0.0
        ),
    }


def nscore(
    df: pd.DataFrame,
    vcol: str,
    wcol: Optional[str] = None,
    ismooth: bool = False,
    dfsmooth: Optional[pd.DataFrame] = None,
    smcol: int = 0,
    smwcol: int = 0,
    verbose: bool = True,
) -> NscoreResult:
    """Normal-score transform of ``df[vcol]`` with the upstream transform-loop
    off-by-one repaired and every other outcome verified.

    Parameters other than ``verbose`` mirror
    ``geostatspy.geostats.nscore``; see that function for their meaning.
    ``verbose`` only controls whether applied corrections are printed.

    ``ismooth=True`` (build the transform table from a separate reference
    distribution ``dfsmooth`` instead of from the data) is NOT supported and
    raises ``NotImplementedError``. The parameters are kept in the signature
    only so the call is a drop-in for the library's. Reason: with a reference
    table, data can fall OUTSIDE ``[vr[0], vr[-1]]``, and what should happen
    then is an unmade decision. ``sgsim`` clamps such values to the table's
    end knots; this module instead treats an out-of-table normal score as an
    unknown failure and raises (Guard 1). Those two policies contradict each
    other, so rather than pick one silently -- and rather than let the guard
    turn a legitimate reference-distribution workflow into a crash -- the
    unsupported mode is rejected up front. Nothing in this repo uses it.

    Returns
    -------
    NscoreResult(ns, vr, vrg, corrections)
        ``ns``   : repaired normal scores, one per row of ``df``
        ``vr``   : sorted data values of the transform table (unchanged)
        ``vrg``  : the table's normal scores (unchanged; this is what the
                   library calls ``wt_ns`` on return)
        ``corrections`` : JSON-serializable audit dict -- always present, with
                   ``n_corrected == 0`` when the correct clamp happened to
                   agree with the library everywhere.

    Raises
    ------
    NotImplementedError
        If ``ismooth`` is true (see above).
    RuntimeError
        If the transform table is degenerate or non-monotonic, or if any
        recomputed normal score is non-finite, falls outside the transform
        table's ``[vrg[0], vrg[-1]]`` range, or breaks order preservation.
        These indicate failure modes this module does not know about, and are
        deliberately fatal rather than silently passed through.
    """
    if ismooth:
        raise NotImplementedError(
            "src.nscore.nscore does not support ismooth=True (transform table "
            "from a reference distribution): the policy for data lying "
            "outside the reference table's range is undecided -- sgsim clamps "
            "to the end knots, this module's Guard 1 treats out-of-table "
            "values as an unknown failure and raises. Decide that policy "
            "before enabling this mode. See src/nscore.py's docstring."
        )
    _warn_on_version_drift()

    ns_raw, vr, vrg = geostats.nscore(df, vcol, wcol=wcol)
    ns_raw = np.asarray(ns_raw, dtype=float)
    vr = np.asarray(vr, dtype=float)
    vrg = np.asarray(vrg, dtype=float)
    values = np.asarray(df[vcol].values, dtype=float)

    if vr.size < 2:
        raise RuntimeError(
            f"normal-score transform table has only {vr.size} point(s); a "
            "transform needs at least 2 to define a segment."
        )
    if not np.all(np.isfinite(vr)) or not np.all(np.isfinite(vrg)):
        raise RuntimeError(
            "geostats.nscore returned a transform table containing non-finite "
            "values (NaN/inf in vr or vrg) -- check the input data for "
            "missing values before transforming."
        )
    if np.any(np.diff(vr) < 0) or np.any(np.diff(vrg) < 0):
        raise RuntimeError(
            "geostats.nscore returned a non-monotonic transform table "
            "(vr or vrg is not sorted ascending) -- this module's repair "
            "assumes the GSLIB table ordering and cannot be trusted here."
        )

    # --- Redo the transform loop with sgsim's correct 0-based clamp -------
    ns = _nscore_transform(values, vr, vrg)

    differs = np.flatnonzero(ns != ns_raw)
    corrections = _correction_record(
        len(values), differs, values[differs], ns_raw[differs], ns[differs]
    )
    if verbose and differs.size:
        for i in differs:
            print(
                f"  src.nscore: repaired geostats.nscore off-by-one at row {int(i)} "
                f"({vcol}={values[i]!r}): normal score {ns_raw[i]!r} -> {ns[i]!r}"
            )

    # --- Guard 0: no NaN/inf ---------------------------------------------
    # Must come FIRST: NaN compares False against every bound, so Guards 1
    # and 2 below would let non-finite scores through silently.
    bad = np.flatnonzero(~np.isfinite(ns))
    if bad.size:
        k = int(bad[0])
        raise RuntimeError(
            f"{bad.size} normal score(s) are not finite (NaN/inf). First: row "
            f"{k} ({vcol}={values[k]!r}) -> {ns[k]!r}. This is an UNKNOWN "
            "failure mode of the normal-score transform; refusing to continue "
            "with corrupted normal scores. See src/nscore.py's docstring."
        )

    # --- Guard 1: every normal score must lie inside the table's range ----
    lo, hi = float(vrg[0]), float(vrg[-1])
    out_of_range = np.flatnonzero((ns < lo - RANGE_TOL) | (ns > hi + RANGE_TOL))
    if out_of_range.size:
        worst = out_of_range[np.argmax(np.abs(ns[out_of_range]))]
        raise RuntimeError(
            f"{out_of_range.size} normal score(s) fall outside the transform "
            f"table range [{lo!r}, {hi!r}] even after applying the correct "
            f"0-based clamp. Worst: row {int(worst)} "
            f"({vcol}={values[worst]!r}) -> {ns[worst]!r}. This is an UNKNOWN "
            "failure mode of geostats.nscore; refusing to continue with "
            "corrupted normal scores. See src/nscore.py's docstring."
        )

    # --- Guard 2: the transform must be order-preserving ------------------
    order = np.argsort(values, kind="stable")
    ns_sorted = ns[order]
    drops = np.flatnonzero(np.diff(ns_sorted) < -MONOTONE_TOL)
    if drops.size:
        k = int(drops[0])
        raise RuntimeError(
            "the normal-score transform is not order-preserving: sorting by "
            f"{vcol} gives normal score {ns_sorted[k]!r} at "
            f"{vcol}={values[order[k]]!r} followed by {ns_sorted[k + 1]!r} at "
            f"{vcol}={values[order[k + 1]]!r} ({drops.size} such inversion(s)). "
            "This is an UNKNOWN failure mode of geostats.nscore; refusing to "
            "continue. See src/nscore.py's docstring."
        )

    return NscoreResult(ns, vr, vrg, corrections)
