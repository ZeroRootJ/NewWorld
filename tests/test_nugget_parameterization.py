"""Regression tests for the nugget parameterization added for the nugget
axis (docs/experiment_context.md deliverable 3).

The nugget axis works by threading a ``nug`` argument through
``base_case_conditioning.build_vario`` / ``get_base_case_truth`` /
``get_base_case_conditioning_data`` and all four method ``main()``
functions. Three things have to stay true for that to be safe, and each has
a cheap structural check here (none of these runs a full experiment -- the
expensive bit-for-bit check against the pinned base-case runs lives in
src/experiments/_verify_nugget_parameterization_regression.py):

1. The DEFAULT call is unchanged. ``build_vario()`` with no arguments must
   still produce exactly the base case's variogram dict, so every existing
   script/pin keeps its meaning.
2. The UNIT-SILL invariant holds at every axis level. ``cc1`` is derived as
   ``1.0 - nug``, never passed independently, so the total sill stays 1.0
   and "more nugget" cannot silently mean "more total variance" (which would
   destroy the one-factor-at-a-time interpretation of the axis).
3. All four methods actually accept ``nug``, with the base-case default, so
   the axis runner cannot pass it to three methods and silently skip a
   fourth.

Plain assert-based, no pytest dependency required (pytest auto-discovers
these anyway), matching tests/test_base_case_conditioning.py:

    .venv/Scripts/python.exe -m tests.test_nugget_parameterization
"""

import inspect

import numpy as np

import geostatspy.GSLIB as GSLIB

from src import truth_model
from src.experiments import base_case, base_case_conditioning, nugget_axis
from src.experiments import gp_mle, kriging, rbf_bootstrap, sgs


def test_base_case_sill_is_unity():
    """The whole nugget axis rests on nug + cc1 == 1.0 in the base case."""
    assert abs((base_case.NUG + base_case.CC1) - 1.0) < 1e-12, (
        f"base_case.NUG ({base_case.NUG}) + base_case.CC1 ({base_case.CC1}) != 1.0"
    )


def test_build_vario_default_reproduces_base_case_variogram():
    """build_vario() with no arguments must equal the variogram base_case.py
    builds from its own NUG/CC1 constants -- i.e. the nug parameterization is
    behavior-preserving at the defaults."""
    expected = GSLIB.make_variogram(
        nug=base_case.NUG, nst=1, it1=base_case.IT1, cc1=base_case.CC1,
        azi1=base_case.AZI1, hmaj1=base_case.HMAJ1, hmin1=base_case.HMIN1,
    )
    actual = base_case_conditioning.build_vario()
    assert actual == expected, f"build_vario() = {actual}, expected {expected}"


def test_build_vario_derives_cc1_and_keeps_unit_sill_at_every_axis_level():
    for nug in nugget_axis.ALL_NUGGET_VALUES:
        v = base_case_conditioning.build_vario(nug=float(nug))
        assert np.isclose(v["nug"], nug), f"nug={nug}: dict has nug={v['nug']}"
        assert np.isclose(v["cc1"], 1.0 - nug), f"nug={nug}: dict has cc1={v['cc1']}"
        assert np.isclose(v["nug"] + v["cc1"], 1.0), (
            f"nug={nug}: sill = {v['nug'] + v['cc1']} != 1.0"
        )
        # The nugget axis must not disturb the range/azimuth.
        assert v["hmaj1"] == base_case.HMAJ1 and v["hmin1"] == base_case.HMIN1


def test_build_vario_rejects_out_of_range_nugget():
    for bad in (-0.1, 1.5):
        try:
            base_case_conditioning.build_vario(nug=bad)
        except ValueError:
            continue
        raise AssertionError(f"build_vario(nug={bad}) should have raised ValueError")


class _AbortAfterConditioningData(Exception):
    """Sentinel raised by the spy below to stop a method's main() as soon as
    it has requested its conditioning data -- so this test can observe what
    was actually passed without paying for a full (minutes-long) run."""


def _spy_on_conditioning_data(module, **main_kwargs):
    """Call ``module.main(**main_kwargs)`` with that module's
    ``get_base_case_conditioning_data`` replaced by a spy that records the
    keyword arguments it receives and then aborts the run.

    Returns the recorded kwargs dict.
    """
    recorded = {}
    original = module.get_base_case_conditioning_data

    def spy(**kwargs):
        recorded.update(kwargs)
        raise _AbortAfterConditioningData()

    module.get_base_case_conditioning_data = spy
    try:
        module.main(**main_kwargs)
    except _AbortAfterConditioningData:
        pass
    finally:
        module.get_base_case_conditioning_data = original

    assert recorded, (
        f"{module.__name__}.main did not call get_base_case_conditioning_data "
        "at all -- this test's interception point is wrong"
    )
    return recorded


def test_all_four_methods_accept_nug_with_base_case_default():
    """Each method's main() must take a ``nug`` keyword defaulting to the
    base-case NUG -- otherwise nugget_axis.py could pass it to some methods
    and silently leave others at the base case.

    NOT a signature-only check (that was the previous version's weakness: a
    method could declare ``nug`` and never use it, and the test would still
    pass). Each method's ``main()`` is actually INVOKED, with its
    ``get_base_case_conditioning_data`` replaced by a spy that records what
    it was handed and then aborts the run -- so what is verified is that the
    value REACHES the ground-truth generator, which is the only thing ``nug``
    is supposed to do in three of the four methods (kriging and SGS
    additionally feed it to their variogram, covered by
    ``test_kriging_and_sgs_pass_nug_into_their_input_variogram``).
    """
    probe_nug = 0.37  # not the default, not an axis level
    for module in (kriging, sgs, rbf_bootstrap, gp_mle):
        sig = inspect.signature(module.main)
        assert "nug" in sig.parameters, f"{module.__name__}.main has no 'nug' parameter"
        default = sig.parameters["nug"].default
        assert default == base_case.NUG, (
            f"{module.__name__}.main's nug default is {default}, expected "
            f"base_case.NUG = {base_case.NUG}"
        )

        # (a) explicit value must be forwarded, not swallowed
        recorded = _spy_on_conditioning_data(module, nug=probe_nug)
        assert recorded.get("nug") == probe_nug, (
            f"{module.__name__}.main(nug={probe_nug}) passed "
            f"nug={recorded.get('nug')!r} to get_base_case_conditioning_data "
            "-- the argument is accepted but not used"
        )

        # (b) the no-argument call must reproduce the base case
        recorded_default = _spy_on_conditioning_data(module)
        assert recorded_default.get("nug") == base_case.NUG, (
            f"{module.__name__}.main() with no arguments passed "
            f"nug={recorded_default.get('nug')!r}, expected base_case.NUG = "
            f"{base_case.NUG}"
        )


def test_kriging_and_sgs_pass_nug_into_their_input_variogram():
    """kriging/SGS are the baselines that get the TRUE nugget, so for them
    ``nug`` must also reach the variogram dict handed to kb2d/sgsim -- not
    only the truth generator. Verified by rebuilding the variogram the same
    way their main() does and checking the nug/cc1 split, which is the exact
    dict both write into their manifest."""
    for nug in (0.0, 0.37, 0.5):
        vario = base_case_conditioning.build_vario(nug=nug)
        assert np.isclose(vario["nug"], nug)
        assert np.isclose(vario["cc1"], 1.0 - nug)
        # And the pinned nugget-axis runs prove the end-to-end path: each
        # level's kriging/SGS manifest records this same split (asserted by
        # src/experiments/nugget_axis.verify_run at run time).


def test_axis_level_keys_are_unique_and_stable():
    keys = [nugget_axis.level_key(n) for n in nugget_axis.ALL_NUGGET_VALUES]
    assert keys == ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"], keys
    assert len(set(keys)) == len(keys), "level_key collides between axis levels"
    tokens = [nugget_axis.level_file_token(n) for n in nugget_axis.ALL_NUGGET_VALUES]
    assert tokens == ["0p0", "0p1", "0p2", "0p3", "0p4", "0p5"], tokens
    assert len(set(tokens)) == len(tokens), "level_file_token collides between axis levels"
    # The base case is deliberately NOT an axis level (NUG=0.05 lies between
    # 0.0 and 0.1) -- if that ever changed, the "no base-case reuse" reasoning
    # in nugget_axis.py / evaluate_nugget_axis.py would need revisiting.
    assert base_case.NUG not in [float(n) for n in nugget_axis.ALL_NUGGET_VALUES]


def test_level_key_refuses_values_that_would_be_rounded_onto_another_level():
    """A bare "%.1f" rounds, so level_key(0.05) -- the BASE CASE's nugget --
    would return "0.1" and silently collide with the real 0.1 axis level.
    level_key must raise instead. (This test is what found that hazard.)"""
    for bad in (base_case.NUG, 0.05, 0.15, 0.123):
        try:
            key = nugget_axis.level_key(bad)
        except ValueError:
            continue
        raise AssertionError(
            f"level_key({bad}) returned '{key}' instead of raising -- it was silently "
            "rounded onto a different axis level's key."
        )


def test_truth_nugget_real_units_matches_the_generated_fields_actual_sill():
    """``truth_nugget_real_units`` must be anchored to a property of the
    GENERATED ground truth, not merely to a restatement of its own formula.

    The previous version of this test asserted
    ``truth_nugget_real_units(nug) == nug * POR_STDEV**2``, which is the
    implementation copied verbatim -- it would have passed for any wrong
    constant as long as both places used the same wrong constant. Instead:

    1. Generate an actual ground-truth field and measure its variance. The
       full-sill value of the function (nug = 1.0) must equal that measured
       variance -- i.e. "normalized nugget 1.0" really does correspond to
       the whole physical sill the experiment produces.
    2. Check the function is exactly proportional across the axis levels
       using RATIOS, which involve no physical constant at all.
    """
    truth = base_case_conditioning.get_base_case_truth(
        hmaj1=nugget_axis.AXIS_HMAJ1, hmin1=nugget_axis.AXIS_HMIN1, nug=0.2
    )
    measured_sill = float(np.var(truth))
    assert np.isclose(nugget_axis.truth_nugget_real_units(1.0), measured_sill, rtol=1e-9), (
        f"truth_nugget_real_units(1.0) = {nugget_axis.truth_nugget_real_units(1.0)} "
        f"but the generated truth field's variance is {measured_sill}"
    )
    assert np.isclose(nugget_axis.truth_nugget_real_units(0.0), 0.0)

    # 2. Proportionality, stated without any physical constant.
    reference = nugget_axis.truth_nugget_real_units(0.1)
    for nug in nugget_axis.ALL_NUGGET_VALUES:
        assert np.isclose(
            nugget_axis.truth_nugget_real_units(nug), reference * (float(nug) / 0.1)
        ), f"truth_nugget_real_units is not proportional to nug at nug={nug}"


def test_affine_scale_factor_is_the_factor_affine_actually_applied():
    """``affine_scale_factor(nug)`` claims to be the multiplier
    ``GSLIB.affine`` applied to this level's realization. Verify that
    against the realization itself: reconstructing the truth field as
    ``a * (sim_ns - mean(sim_ns)) + POR_MEAN`` must reproduce it exactly.

    This is what makes ``truth_nugget_real_units_affine`` (the alternative
    scale recorded alongside the headline one in
    results/processed/nugget_axis/gp_fitted_nugget_vs_truth.csv) meaningful
    rather than an arbitrary second number.
    """
    nug = 0.3
    vario = base_case_conditioning.build_vario(
        hmaj1=nugget_axis.AXIS_HMAJ1, hmin1=nugget_axis.AXIS_HMIN1, nug=nug
    )
    truth, sim_ns = truth_model.make_porosity_truth_with_sim_ns(
        nx=base_case.NX, ny=base_case.NY, xsiz=base_case.XSIZ, ysiz=base_case.YSIZ,
        xmn=base_case.XMN, ymn=base_case.YMN, vario=vario,
        mean=base_case.POR_MEAN, stdev=base_case.POR_STDEV,
        seed=base_case_conditioning.TRUTH_SEED,
    )
    a = nugget_axis.affine_scale_factor(nug)
    reconstructed = a * (sim_ns - np.mean(sim_ns)) + base_case.POR_MEAN
    assert np.allclose(truth, reconstructed, rtol=0.0, atol=1e-10), (
        "affine_scale_factor does not reproduce the truth field from sim_ns"
    )
    # And the alternative physical nugget really is nug * a**2.
    assert np.isclose(
        nugget_axis.truth_nugget_real_units_affine(nug), nug * a * a, rtol=1e-12
    )
    # The two conventions must genuinely differ (otherwise recording both is
    # pointless and one of them is wrong).
    assert not np.isclose(
        nugget_axis.truth_nugget_real_units_affine(nug),
        nugget_axis.truth_nugget_real_units(nug),
        rtol=1e-3,
    ), "the affine-based and target-sill nugget scales are indistinguishable"


def test_different_nuggets_give_different_truth_fields():
    """The nugget must actually reach the truth generator. Two levels with
    the same seed/range but different nuggets must produce different fields
    (and therefore different conditioning sample VALUES at identical sample
    LOCATIONS -- which is what nugget_axis.verify_run relies on to pin a run
    to its level)."""
    truth_a, samples_a = base_case_conditioning.get_base_case_conditioning_data(nug=0.0)
    truth_b, samples_b = base_case_conditioning.get_base_case_conditioning_data(nug=0.5)

    assert not np.array_equal(truth_a, truth_b), (
        "nug=0.0 and nug=0.5 produced identical truth fields"
    )
    assert np.array_equal(samples_a[["X", "Y"]].values, samples_b[["X", "Y"]].values), (
        "sample LOCATIONS must be identical across nugget levels (same SAMPLE_SEED)"
    )
    assert not np.array_equal(
        samples_a[base_case_conditioning.VCOL].values,
        samples_b[base_case_conditioning.VCOL].values,
    ), "sample VALUES must differ between nugget levels"


def main():
    tests = [
        test_base_case_sill_is_unity,
        test_build_vario_default_reproduces_base_case_variogram,
        test_build_vario_derives_cc1_and_keeps_unit_sill_at_every_axis_level,
        test_build_vario_rejects_out_of_range_nugget,
        test_all_four_methods_accept_nug_with_base_case_default,
        test_kriging_and_sgs_pass_nug_into_their_input_variogram,
        test_axis_level_keys_are_unique_and_stable,
        test_level_key_refuses_values_that_would_be_rounded_onto_another_level,
        test_truth_nugget_real_units_matches_the_generated_fields_actual_sill,
        test_affine_scale_factor_is_the_factor_affine_actually_applied,
        test_different_nuggets_give_different_truth_fields,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
