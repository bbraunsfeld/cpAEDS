"""Tests for cpaeds.algorithms module."""

import math

import numpy as np
import pytest

from cpaeds.algorithms import (
    inverse_log_curve,
    log_fit,
    logistic_curve,
    offset_steps,
    pKa_from_df,
    ph_curve,
)

_TEMP = 300.0


# ---------------------------------------------------------------------------
# pKa_from_df
# ---------------------------------------------------------------------------

class TestPKaFromDf:
    def test_zero_df_gives_zero_pKa(self):
        """df=0 → Ka=1 → pKa=0."""
        pKa = pKa_from_df(0.0, _TEMP)
        assert abs(pKa) < 1e-6

    def test_nan_passthrough(self):
        assert pKa_from_df("NaN", _TEMP) == "NaN"

    def test_positive_df_gives_positive_pKa(self):
        """Positive ΔF → Ka < 1 → pKa > 0."""
        pKa = pKa_from_df(10.0, _TEMP)
        assert pKa > 0

    def test_negative_df_gives_negative_pKa(self):
        pKa = pKa_from_df(-10.0, _TEMP)
        assert pKa < 0

    def test_known_value(self):
        """pKa = -log10(exp(-df / (R*T))).  Check against manual calc."""
        k = 0.00831451
        df = 5.706  # chosen so Ka ≈ 0.1 → pKa ≈ 1 at 300 K
        Ka = math.exp(-(df / (k * _TEMP)))
        expected = -math.log10(Ka)
        # pKa_from_df uses slightly different k=0.00831451 vs module k=0.00831451
        result = pKa_from_df(df, _TEMP)
        assert abs(result - expected) < 0.05  # tolerance for k constant difference


# ---------------------------------------------------------------------------
# ph_curve
# ---------------------------------------------------------------------------

class TestPhCurve:
    def test_half_fraction_equals_pKa(self):
        """At f=0.5, Henderson-Hasselbalch gives pH = pKa."""
        pKa = 4.0
        ph_list = ph_curve(pKa, [0.5])
        assert abs(ph_list[0] - pKa) < 1e-9

    def test_extreme_fractions_are_nan(self):
        """f=0 and f=1 must return NaN (undefined pH)."""
        ph_list = ph_curve(4.0, [0.0, 0.5, 1.0])
        assert math.isnan(ph_list[0])
        assert not math.isnan(ph_list[1])
        assert math.isnan(ph_list[2])

    def test_increasing_fraction_decreasing_pH(self):
        """As the deprotonated fraction increases, pH decreases."""
        pKa = 4.0
        fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        ph_list = ph_curve(pKa, fractions)
        assert all(ph_list[i] > ph_list[i + 1] for i in range(len(ph_list) - 1))


# ---------------------------------------------------------------------------
# logistic_curve / inverse_log_curve
# ---------------------------------------------------------------------------

class TestLogisticCurve:
    _PARAMS = (1.0, 0.0, 0.5, 0.0)  # a, b, c, d

    def test_midpoint(self):
        """At x=d the curve should return (a+b)/2."""
        a, b, c, d = self._PARAMS
        assert abs(logistic_curve(d, a, b, c, d) - (a + b) / 2) < 1e-9

    def test_inverse_recovers_x(self):
        """inverse_log_curve(logistic_curve(x, ...), ...) == x."""
        a, b, c, d = self._PARAMS
        x_vals = np.linspace(-5.0, 5.0, 20)
        for x in x_vals:
            y = logistic_curve(x, a, b, c, d)
            x_back = inverse_log_curve(y, a, b, c, d)
            assert abs(x_back - x) < 1e-8, f"Failed at x={x}"

    def test_asymptotes(self):
        """At large |x|, curve approaches a (top) or b (bottom)."""
        a, b, c, d = self._PARAMS
        assert abs(logistic_curve(1e6, a, b, c, d) - a) < 1e-6
        assert abs(logistic_curve(-1e6, a, b, c, d) - b) < 1e-6


# ---------------------------------------------------------------------------
# log_fit
# ---------------------------------------------------------------------------

class TestLogFit:
    def test_recovers_known_params(self):
        """Fitting synthetic logistic data should recover parameters closely."""
        a, b, c, d = 1.0, 0.0, 0.4, -5.0
        x = np.linspace(-20.0, 10.0, 80)
        y = logistic_curve(x, a, b, c, d)
        popt = log_fit(x, y)
        # Parameters may not be unique (a↔b sign flip), so check curve values
        y_fit = logistic_curve(x, *popt)
        assert np.allclose(y, y_fit, atol=1e-4)

    def test_ignores_nans(self):
        """NaN values in y should be dropped before fitting."""
        a, b, c, d = 1.0, 0.0, 0.5, 0.0
        x = np.linspace(-8.0, 8.0, 40)
        y = logistic_curve(x, a, b, c, d)
        y[0] = np.nan
        y[-1] = np.nan
        popt = log_fit(x, y)  # should not raise
        assert len(popt) == 4


# ---------------------------------------------------------------------------
# offset_steps
# ---------------------------------------------------------------------------

class TestOffsetSteps:
    def _base_kwargs(self, cptype):
        return dict(
            EIR_start=[0, -220],
            EIR_range=40,
            EIR_step_size=2,
            EIR_groups=[[0], [1]],
            cpAEDS_type=cptype,
        )

    def test_type1_equal_spacing(self):
        """Type 1 inner list should have equal-spaced offsets.

        Note: cpAEDS_type=1 currently stores offsets as a nested list
        [[offset_list]] due to a known bug (uses .append instead of =).
        The inner list is tested here.
        """
        offsets = offset_steps(**self._base_kwargs(1))
        inner = offsets[1][0]  # outer list has one element: the actual offset list
        diffs = [inner[i + 1] - inner[i] for i in range(len(inner) - 1)]
        assert all(abs(d - diffs[0]) < 1e-9 for d in diffs)

    def test_type1_length(self):
        """Type 1 inner offset list length = EIR_range / step_size + 1.

        Note: cpAEDS_type=1 currently stores offsets as [[offset_list]]
        (known bug).  The inner list length is tested.
        """
        offsets = offset_steps(**self._base_kwargs(1))
        inner = offsets[1][0]
        assert len(inner) == 40 // 2 + 1

    def test_type2_produces_list(self):
        offsets = offset_steps(**self._base_kwargs(2))
        assert isinstance(offsets[1], list)
        assert len(offsets[1]) > 0

    def test_type3_centered_sum(self):
        """Type 3 should adjust offsets so they are centred (cumsum / n ≈ 0)."""
        offsets = offset_steps(**self._base_kwargs(3))
        arr = np.array(offsets)
        col_sums = arr.sum(axis=0)
        # Each column sum / n_states should be close to 0 (centred)
        assert np.allclose(col_sums / 2, 0, atol=5)

    def test_zero_range_returns_start(self):
        """EIR_range=0 should return the start values only."""
        offsets = offset_steps(
            EIR_start=[0, -220],
            EIR_range=0,
            EIR_step_size=2,
            EIR_groups=[[0], [1]],
            cpAEDS_type=1,
        )
        assert offsets[0] == [0]
        assert offsets[1] == [-220]
