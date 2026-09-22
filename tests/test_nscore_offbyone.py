"""Tests for ``src/nscore.py`` -- the project-level repair of
``geostatspy.geostats.nscore``'s 1-based -> 0-based off-by-one.

These tests are deliberately NOT tautological: they do not re-assert the
wrapper's own formula. Each one pins the wrapper against something
independent of it:

1. ``test_raw_geostats_nscore_blows_up_on_near_tied_second_and_third``
   exercises the UPSTREAM library directly and shows it really does produce a
   wildly out-of-transform-table normal score for the minimum datum when the
   2nd and 3rd smallest values are nearly tied. If GeostatsPy is ever fixed
   upstream this test fails loudly, which is exactly when the wrapper's
   repair should be revisited.
2. ``test_wrapper_matches_independent_reference_on_every_datum`` is the main
   pin: a reference normal-score transform written from scratch in this file
   (vectorised ``np.searchsorted`` + explicit linear interpolation, sharing
   no code with ``src/nscore.py`` beyond the library's transform TABLE) must
   agree with the wrapper on EVERY datum of several data sets, including one
   with ties. This is what fixes the whole transform, not just its minimum.
3. ``test_wrapper_returns_table_minimum_for_the_smallest_datum`` checks the
   repaired value against ``vrg[0]``, which comes out of the library's own
   transform TABLE (a different code path from the transform loop that has
   the bug) -- not against a re-derivation of the wrapper's arithmetic.
4. ``test_wrapper_leaves_every_other_datum_bit_identical`` is a regression pin
   on the SCOPE of the repair for untied data: with no ties, only the minimum
   datum may change. (Since the wrapper now recomputes the whole loop rather
   than patching one datum, this is an empirical claim about the correct
   clamp, not a property of the implementation.)
5. ``test_tied_minimum_agrees_with_library_and_is_a_table_knot`` pins the tie
   behaviour that motivated rewriting the loop instead of special-casing
   ``values <= vr[0]``.
6. ``test_wrapper_raises_on_*`` inject failure modes the wrapper does NOT
   know how to repair (NaN, out-of-range, non-monotone) and require a
   ``RuntimeError`` -- the "don't let a second unknown bug hide" requirement.
7. ``test_ismooth_is_rejected`` pins that the unsupported reference-
   distribution mode fails fast with an explanation instead of dying inside a
   guard.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_nscore_offbyone.py -q
or  .venv/Scripts/python.exe -m tests.test_nscore_offbyone
"""

import numpy as np
import pandas as pd

import geostatspy.geostats as geostats

import src.nscore as nscore_module
from src.nscore import nscore

VCOL = "Por"

# Pathological data set: the 2nd and 3rd smallest values are nearly tied, so
# the segment the buggy clamp borrows (vr[1] -> vr[2]) has a ~1e-6 run. The
# minimum datum sits ~10 units below vr[1], so back-extrapolating along that
# segment's slope overshoots by a factor of ~1e7. This is a synthetic,
# exaggerated version of what was measured in
# results/raw/kriging/20260922T155131942671Z (real gap 0.00689, real overshoot
# to -52.15 against a table minimum of -2.6411).
PATHOLOGICAL_VALUES = [0.0, 10.0, 10.0 + 1e-6, 20.0, 30.0, 40.0, 55.0, 70.0]

# Well-separated data: no near-tie anywhere, so the borrowed segment has a
# sane slope. The minimum datum is STILL mis-assigned by the library (the
# off-by-one is unconditional), just not dramatically -- which is precisely
# why the wrapper corrects unconditionally rather than only when the value
# looks extreme.
ORDINARY_VALUES = [1.0, 3.0, 7.0, 11.0, 14.0, 19.0, 26.0, 31.0, 38.0, 44.0]

# Exact ties, including at the minimum. This is the case that made the
# earlier "patch every datum with value <= vr[0]" approach worse than the
# library: with vr[0] == vr[1] the first segment is degenerate and dpowint
# falls into its (ylow + yhigh) / 2 branch. The correct 0-based clamp has no
# such special case.
TIED_VALUES = [5.0, 5.0, 6.0, 7.0, 9.0, 12.0]

# A larger, unstructured set so the independent-reference comparison is not
# only exercised on hand-picked shapes.
RANDOM_VALUES = list(np.random.RandomState(20260922).normal(12.0, 3.0, 137))


def _df(values):
    return pd.DataFrame({VCOL: np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# Independent reference implementation (J-5).
#
# Written from the GSLIB definition + the 0-based clamp that geostatspy's own
# sgsim uses (geostats.py L3807, `j = min(max(0, j), (nd - 2))`), but sharing
# no code with src/nscore.py: no geostats.dlocate, no geostats.dpowint, no
# import of the wrapper's helpers. Vectorised with np.searchsorted so even the
# index arithmetic is expressed differently.
#
# dlocate(vr, 1, nd, x) is bisect_right(vr[1 : nd-1], x), i.e. the number of
# entries of vr[1..nd-2] that are <= x. np.searchsorted(vr[1:nd-1], x, 'right')
# is the same quantity.
# ---------------------------------------------------------------------------
_EPSILON = 1.0e-20
_DITHER_SEED = 73073


def reference_nscore(values, vr, vrg):
    values = np.asarray(values, dtype=float)
    vr = np.asarray(vr, dtype=float)
    vrg = np.asarray(vrg, dtype=float)
    nd = vr.size
    n = values.size

    # The library dithers each value by rand() * 1e-20 off a 73073-seeded
    # global RNG; reproduce the same draws so the comparison is exact.
    vrr = values + np.random.RandomState(_DITHER_SEED).rand(n) * _EPSILON

    j = np.searchsorted(vr[1: nd - 1], vrr, side="right")
    j = np.clip(j, 0, nd - 2)

    xlow, xhigh = vr[j], vr[j + 1]
    ylow, yhigh = vrg[j], vrg[j + 1]
    run = xhigh - xlow
    out = np.where(
        run < _EPSILON,
        (ylow + yhigh) / 2.0,                      # dpowint's degenerate branch
        ylow + (yhigh - ylow) * ((vrr - xlow) / np.where(run < _EPSILON, 1.0, run)),
    )
    return out


def test_wrapper_matches_independent_reference_on_every_datum():
    """The whole transform -- not just the minimum -- must match a reference
    implementation that shares no code with src/nscore.py."""
    for name, vals in [
        ("pathological", PATHOLOGICAL_VALUES),
        ("ordinary", ORDINARY_VALUES),
        ("tied", TIED_VALUES),
        ("random", RANDOM_VALUES),
    ]:
        df = _df(vals)
        ns, vr, vrg, _ = nscore(df, VCOL, verbose=False)
        expected = reference_nscore(df[VCOL].values, vr, vrg)
        assert ns.shape == expected.shape
        assert np.allclose(ns, expected, rtol=0.0, atol=1e-12), (
            f"[{name}] wrapper disagrees with the independent reference at "
            f"rows {np.flatnonzero(~np.isclose(ns, expected, rtol=0.0, atol=1e-12))}"
        )


def test_raw_geostats_nscore_blows_up_on_near_tied_second_and_third():
    """The upstream library, called directly, must be shown to be broken --
    otherwise the wrapper is guarding against nothing."""
    df = _df(PATHOLOGICAL_VALUES)
    ns_raw, vr, vrg = geostats.nscore(df, VCOL)

    i_min = int(np.argmin(df[VCOL].values))
    assert np.isclose(vr[0], min(PATHOLOGICAL_VALUES))
    # The library's own transform table says the smallest datum's normal
    # score is vrg[0] (a finite, O(1) number).
    assert -4.0 < vrg[0] < 0.0, f"unexpected table minimum vrg[0]={vrg[0]}"
    # ... yet the value it actually assigns is orders of magnitude outside it.
    assert ns_raw[i_min] < -1e5, (
        "expected geostats.nscore to back-extrapolate the minimum datum into "
        f"absurdity, but it returned {ns_raw[i_min]} (has the upstream "
        "off-by-one been fixed? then revisit src/nscore.py)"
    )
    # Confirm the mechanism, not just the symptom: the bad value is exactly
    # the vr[1] -> vr[2] segment evaluated at vr[0].
    expected_bad = vrg[1] + (vr[0] - vr[1]) / (vr[2] - vr[1]) * (vrg[2] - vrg[1])
    assert np.isclose(ns_raw[i_min], expected_bad, rtol=1e-9), (
        f"library gave {ns_raw[i_min]}, wrong-segment extrapolation predicts "
        f"{expected_bad} -- the failure mechanism is not the documented one"
    )
    # And the maximum datum is fine (the clamp does not bite at the top).
    assert np.isclose(ns_raw[int(np.argmax(df[VCOL].values))], vrg[-1])


def test_wrapper_returns_table_minimum_for_the_smallest_datum():
    df = _df(PATHOLOGICAL_VALUES)
    ns, vr, vrg, corrections = nscore(df, VCOL, verbose=False)

    i_min = int(np.argmin(df[VCOL].values))
    assert np.isclose(ns[i_min], vrg[0], rtol=0.0, atol=1e-12), (
        f"repaired normal score {ns[i_min]} != transform table minimum {vrg[0]}"
    )
    assert ns.min() >= vrg[0] - 1e-12 and ns.max() <= vrg[-1] + 1e-12
    assert corrections["n_corrected"] == 1, corrections
    assert corrections["corrected_indices"] == [i_min]
    assert corrections["max_abs_correction"] > 1e5, corrections


def test_wrapper_leaves_every_other_datum_bit_identical():
    """Regression pin on the SCOPE of the repair for untied data: only the
    minimum datum may change."""
    for vals in (PATHOLOGICAL_VALUES, ORDINARY_VALUES, RANDOM_VALUES):
        df = _df(vals)
        ns_raw, _, _ = geostats.nscore(df, VCOL)
        ns, _, _, _ = nscore(df, VCOL, verbose=False)

        i_min = int(np.argmin(df[VCOL].values))
        others = [i for i in range(len(df)) if i != i_min]
        assert np.array_equal(ns[others], np.asarray(ns_raw)[others]), (
            "the wrapper changed a datum other than the minimum"
        )


def test_tied_minimum_agrees_with_library_and_is_a_table_knot():
    """With an exactly tied minimum the correct 0-based clamp reproduces the
    library (the buggy clamp does not bite when dlocate already returns 1),
    and the result is a table knot rather than the midpoint of a degenerate
    segment that a `values <= vr[0]` patch would have produced."""
    df = _df(TIED_VALUES)
    ns_raw, vr, vrg = geostats.nscore(df, VCOL)
    ns, vr_w, vrg_w, corrections = nscore(df, VCOL, verbose=False)

    assert vr[0] == vr[1], "TIED_VALUES no longer produces a tied minimum"
    assert np.array_equal(ns, np.asarray(ns_raw)), (
        "with a tied minimum the correct clamp must agree with the library "
        f"everywhere: wrapper {ns}, library {np.asarray(ns_raw)}"
    )
    assert corrections["n_corrected"] == 0, corrections
    # Both tied data land on the table knot vrg[1], NOT on the degenerate
    # midpoint (vrg[0] + vrg[1]) / 2 that the earlier value-based patch gave.
    tied = np.flatnonzero(df[VCOL].values == vr[0])
    midpoint = (vrg[0] + vrg[1]) / 2.0
    for i in tied:
        assert np.isclose(ns[i], vrg[1], rtol=0.0, atol=1e-12), (ns[i], vrg[1])
        assert not np.isclose(ns[i], midpoint, rtol=0.0, atol=1e-12)
    assert np.array_equal(vr, vr_w) and np.array_equal(vrg, vrg_w)


def test_ordinary_data_needs_no_repair_and_matches_library():
    """With well-separated data the library's minimum-datum value is still
    wrong (the off-by-one is unconditional), but only mildly -- the wrapper
    must still correct it, and must not touch anything else."""
    df = _df(ORDINARY_VALUES)
    ns_raw, vr, vrg = geostats.nscore(df, VCOL)
    ns, _, _, corrections = nscore(df, VCOL, verbose=False)

    i_min = int(np.argmin(df[VCOL].values))
    assert np.isclose(ns[i_min], vrg[0])
    # The library's value here is in-range but still not vrg[0] -- proof the
    # bug is not detectable by a range check alone, which is why the wrapper
    # corrects by construction rather than only when a value looks extreme.
    assert vrg[0] - 4.0 < ns_raw[i_min] < vrg[-1]
    assert not np.isclose(ns_raw[i_min], vrg[0]), (
        f"library happened to return the correct value {ns_raw[i_min]} for "
        "this data set; pick a different ORDINARY_VALUES to keep this test "
        "meaningful"
    )
    assert corrections["n_corrected"] == 1
    others = [i for i in range(len(df)) if i != i_min]
    assert np.array_equal(ns[others], np.asarray(ns_raw)[others])


class _StubTransform:
    """Context manager that replaces ``src.nscore._nscore_transform`` with a
    function returning a deliberately corrupted ``ns``, so the wrapper's
    UNKNOWN-failure guards can be exercised without needing a second real
    library bug. The guards are separate logic from the transform loop, so
    stubbing the loop is the right seam to inject at."""

    def __init__(self, corrupt):
        self.corrupt = corrupt
        self._original = None

    def __enter__(self):
        self._original = nscore_module._nscore_transform
        original = self._original
        corrupt = self.corrupt

        def stub(values, vr, vrg):
            return corrupt(np.asarray(original(values, vr, vrg), dtype=float), vr, vrg, values)

        nscore_module._nscore_transform = stub
        return self

    def __exit__(self, *exc):
        nscore_module._nscore_transform = self._original
        return False


def _expect_runtime_error(df, fragment):
    try:
        nscore(df, VCOL, verbose=False)
    except RuntimeError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(
            f"wrapper accepted a corrupted transform instead of raising "
            f"(expected a RuntimeError mentioning {fragment!r})"
        )


def test_wrapper_raises_on_unknown_out_of_range_failure():
    """An out-of-range normal score at a datum that is NOT the minimum is a
    failure mode this module does not know how to repair -- it must be fatal,
    not silently passed through."""
    def corrupt(ns, vr, vrg, values):
        ns[int(np.argmax(values))] = float(vrg[-1]) + 17.0
        return ns

    with _StubTransform(corrupt):
        _expect_runtime_error(_df(ORDINARY_VALUES), "outside the transform table range")


def test_wrapper_raises_on_non_monotone_failure():
    """A normal-score transform is order-preserving by definition. An
    in-range but order-violating result is another unknown failure mode and
    must also be fatal."""
    def corrupt(ns, vr, vrg, values):
        order = np.argsort(values, kind="stable")
        # Swap the normal scores of the 2nd and 3rd smallest data: both
        # values stay inside [vrg[0], vrg[-1]], so only the monotonicity
        # guard can catch this.
        a, b = int(order[1]), int(order[2])
        ns[a], ns[b] = ns[b], ns[a]
        return ns

    with _StubTransform(corrupt):
        _expect_runtime_error(_df(ORDINARY_VALUES), "not order-preserving")


def test_wrapper_raises_on_nan_normal_score():
    """NaN compares False against every bound, so the range and monotonicity
    guards cannot see it -- there must be an explicit finiteness guard."""
    def corrupt(ns, vr, vrg, values):
        ns[2] = np.nan
        return ns

    with _StubTransform(corrupt):
        _expect_runtime_error(_df(ORDINARY_VALUES), "not finite")


def test_wrapper_raises_on_inf_normal_score():
    def corrupt(ns, vr, vrg, values):
        ns[1] = -np.inf
        return ns

    with _StubTransform(corrupt):
        _expect_runtime_error(_df(ORDINARY_VALUES), "not finite")


def test_nan_in_input_data_is_rejected():
    """A NaN in the input propagates into the transform table; it must be
    caught there rather than silently producing NaN normal scores."""
    values = list(ORDINARY_VALUES)
    values[3] = np.nan
    _expect_runtime_error(_df(values), "non-finite")


def test_ismooth_is_rejected():
    """The reference-distribution mode is exposed for signature compatibility
    only; it must fail fast with an explanation."""
    df = _df(ORDINARY_VALUES)
    try:
        nscore(df, VCOL, ismooth=True, dfsmooth=df, smcol=VCOL, verbose=False)
    except NotImplementedError as exc:
        assert "ismooth" in str(exc)
    else:
        raise AssertionError("ismooth=True was accepted instead of raising")


def test_stub_is_restored_after_guard_tests():
    """The guard tests monkeypatch a module-level function; make sure nothing
    leaks into the rest of the suite."""
    df = _df(ORDINARY_VALUES)
    ns, _, vrg, corrections = nscore(df, VCOL, verbose=False)
    assert corrections["n_corrected"] == 1
    assert np.isclose(ns[int(np.argmin(df[VCOL].values))], vrg[0])


def main():
    tests = [
        test_wrapper_matches_independent_reference_on_every_datum,
        test_raw_geostats_nscore_blows_up_on_near_tied_second_and_third,
        test_wrapper_returns_table_minimum_for_the_smallest_datum,
        test_wrapper_leaves_every_other_datum_bit_identical,
        test_tied_minimum_agrees_with_library_and_is_a_table_knot,
        test_ordinary_data_needs_no_repair_and_matches_library,
        test_wrapper_raises_on_unknown_out_of_range_failure,
        test_wrapper_raises_on_non_monotone_failure,
        test_wrapper_raises_on_nan_normal_score,
        test_wrapper_raises_on_inf_normal_score,
        test_nan_in_input_data_is_rejected,
        test_ismooth_is_rejected,
        test_stub_is_restored_after_guard_tests,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
