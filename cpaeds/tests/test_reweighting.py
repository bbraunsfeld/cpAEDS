"""Tests for cpaeds.reweighting module."""

import os
import math
import tempfile

import numpy as np
import pytest

from cpaeds.reweighting import (
    Hr,
    Hr_s,
    ReweightResult,
    get_eminmax,
    get_offsets,
    mixing_by_states,
    read_offset_file,
    reweighting,
    reweighting_constpH,
)

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_data", "reweighting")

_BOLTZMANN = 0.00831441
_TEMP = 300.0
_BETA = 1.0 / (_TEMP * _BOLTZMANN)


# ---------------------------------------------------------------------------
# Hr
# ---------------------------------------------------------------------------

class TestHr:
    def test_avg_scalar(self):
        """Hr with avg=True returns a single float."""
        energy = np.array([-400.0, -410.0, -420.0])
        result = Hr(_BETA, energy, offset=-220.0, avg=True)
        assert isinstance(result, float)

    def test_no_avg_shape(self):
        """Hr with avg=False returns an array of the same shape as energy."""
        energy = np.array([-400.0, -410.0, -420.0])
        result = Hr(_BETA, energy, offset=-220.0, avg=False)
        assert result.shape == energy.shape

    def test_no_avg_positive(self):
        """Per-frame exponential values must be positive."""
        energy = np.array([-400.0, -500.0, -300.0])
        result = Hr(_BETA, energy, offset=0.0, avg=False)
        assert np.all(result > 0)

    def test_avg_consistent_with_logsumexp(self):
        """Manual log-sum-exp should agree with Hr."""
        energy = np.array([-400.0, -410.0])
        offset = -220.0
        expected = -(1.0 / _BETA) * math.log(
            sum(math.exp(-_BETA * (e - offset)) for e in energy)
        )
        assert abs(Hr(_BETA, energy, offset, avg=True) - expected) < 1e-8


# ---------------------------------------------------------------------------
# Hr_s
# ---------------------------------------------------------------------------

class TestHr_s:
    """Test the three branches of the soft-core correction."""

    def _make_energy(self, val: float) -> np.ndarray:
        """Single-frame, single-state energy array (shape 1×1)."""
        return np.array([[val]])

    def test_above_emax_branch(self):
        """When Href >= emax, correction is -(emax-emin)/2."""
        emin, emax = -550.0, -100.0
        # Choose offset such that Hr scalar >> emax
        # Use a very small energy so exp is large → Hr is small (negative)
        # To force Href >= emax we need Href to be > -100
        # Hr = -(1/BETA)*log(sum(exp(-BETA*(e-offset))))
        # With energy=-50, offset=0 → Href ≈ -50 which is > emax=-100
        energy = np.array([[-50.0]])
        result = Hr_s(_BETA, energy, offset=0.0, emin=emin, emax=emax, avg=True)
        assert result.shape == (1,)
        # The corrected value must be smaller than the raw Hr
        raw = Hr(_BETA, energy[0], 0.0, avg=True)
        assert result[0] < raw

    def test_below_emin_branch(self):
        """When Href <= emin, no correction is applied."""
        emin, emax = -100.0, 200.0
        # Force Href well below emin
        energy = np.array([[-500.0]])
        result = Hr_s(_BETA, energy, offset=0.0, emin=emin, emax=emax, avg=True)
        raw = Hr(_BETA, energy[0], 0.0, avg=True)
        assert abs(result[0] - raw) < 1e-8

    def test_output_shape_matches_n_frames(self):
        """avg=True output shape should be (n_frames,)."""
        energy = np.array([[-400.0, -420.0, -380.0], [-300.0, -310.0, -290.0]])
        result = Hr_s(_BETA, energy, offset=0.0, emin=-550.0, emax=-100.0, avg=True)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# reweighting
# ---------------------------------------------------------------------------

class TestReweighting:
    def test_uniform_Q_returns_Q(self):
        """If Q is constant, reweighted result equals that constant."""
        Q = np.ones(100) * 0.5
        H_i = np.random.default_rng(0).normal(-400, 10, 100)
        H_ref = H_i.copy()
        Q_rew, p = reweighting(Q, H_i, H_ref, _BETA)
        assert abs(Q_rew - 0.5) < 1e-8

    def test_weights_sum_to_n_frames(self):
        """Importance weights should sum to n_frames (not 1)."""
        rng = np.random.default_rng(42)
        Q = rng.uniform(0, 1, 50)
        H_i = rng.normal(-400, 20, 50)
        H_ref = rng.normal(-400, 20, 50)
        _, p = reweighting(Q, H_i, H_ref, _BETA)
        assert abs(p.sum() - len(Q)) < 1e-6

    def test_binary_Q_fraction(self):
        """Binary Q (0/1) gives a fraction between 0 and 1."""
        rng = np.random.default_rng(7)
        Q = rng.integers(0, 2, 200).astype(float)
        H_i = rng.normal(-400, 15, 200)
        H_ref = H_i + rng.normal(0, 2, 200)
        Q_rew, _ = reweighting(Q, H_i, H_ref, _BETA)
        assert 0.0 <= Q_rew <= 1.0

    def test_identical_hamiltonians_no_bias(self):
        """When H_i == H_ref, reweighted result equals simple mean of Q."""
        rng = np.random.default_rng(99)
        Q = rng.uniform(0, 1, 80)
        H = rng.normal(-420, 5, 80)
        Q_rew, _ = reweighting(Q, H, H, _BETA)
        assert abs(Q_rew - Q.mean()) < 1e-8


# ---------------------------------------------------------------------------
# read_offset_file / get_offsets
# ---------------------------------------------------------------------------

class TestOffsetIO:
    def test_read_offset_file(self):
        """read_offset_file returns the correct float from a temp file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("# header\n")
            f.write("  0.0 -220.5\n")
            fname = f.name
        try:
            val = read_offset_file(fname)
            assert abs(val - (-220.5)) < 1e-9
        finally:
            os.unlink(fname)

    def test_get_offsets_returns_list(self):
        """get_offsets from the fixture dir returns a 2-element list."""
        offsets = get_offsets(FIXTURE_DIR)
        assert len(offsets) == 2

    def test_get_offsets_values(self):
        """Offset values match what was written to the fixtures."""
        offsets = get_offsets(FIXTURE_DIR)
        assert abs(offsets[0] - (-220.0)) < 1e-9
        assert abs(offsets[1] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# get_eminmax
# ---------------------------------------------------------------------------

class TestGetEminmax:
    def test_values(self):
        emin, emax = get_eminmax(FIXTURE_DIR)
        assert abs(emin - (-550.0)) < 1e-9
        assert abs(emax - (-100.0)) < 1e-9


# ---------------------------------------------------------------------------
# mixing_by_states
# ---------------------------------------------------------------------------

class TestMixingByStates:
    def test_single_group_returns_one_entry(self):
        e0 = np.array([-400.0, -420.0, -380.0])
        e1 = np.array([-300.0, -310.0, -290.0])
        mix_groups, diff_groups = mixing_by_states([e0, e1], [[0, 1]], _BETA)
        assert len(mix_groups) == 1
        assert len(diff_groups) == 1

    def test_single_state_group_skipped(self):
        """Groups with only one state should be skipped."""
        e0 = np.array([-400.0, -420.0])
        e1 = np.array([-300.0, -310.0])
        mix_groups, diff_groups = mixing_by_states([e0, e1], [[0], [1]], _BETA)
        assert len(mix_groups) == 0

    def test_mix_below_min(self):
        """Mixing energy must be <= min of the individual states (logsumexp)."""
        e0 = np.array([-400.0, -420.0])
        e1 = np.array([-300.0, -310.0])
        mix_groups, _ = mixing_by_states([e0, e1], [[0, 1]], _BETA)
        mins = np.minimum(e0, e1)
        assert np.all(mix_groups[0] <= mins + 1e-6)


# ---------------------------------------------------------------------------
# reweighting_constpH  (integration test with synthetic fixtures)
# ---------------------------------------------------------------------------

class TestReweightingConstpH:
    def test_fractions_sum_to_one(self):
        """A_frac + B_frac must equal 1 within floating-point tolerance."""
        result = reweighting_constpH(
            cutoff=-400.0,
            temp=_TEMP,
            stepsize=0.5,
            itime=0.0,
            group=[["1", "2"], ["0"]],
            path=FIXTURE_DIR,
            H1_eds=False,
            H1_aeds=False,
        )
        assert abs(result.A_frac + result.B_frac - 1.0) < 1e-8

    def test_returns_namedtuple(self):
        """Result must be a ReweightResult NamedTuple."""
        result = reweighting_constpH(
            cutoff=-400.0,
            temp=_TEMP,
            stepsize=0.5,
            itime=0.0,
            group=[["1", "2"], ["0"]],
            path=FIXTURE_DIR,
        )
        assert isinstance(result, ReweightResult)

    def test_A_frac_between_0_and_1(self):
        result = reweighting_constpH(
            cutoff=-400.0,
            temp=_TEMP,
            stepsize=0.5,
            itime=0.0,
            group=[["1"], ["0"]],
            path=FIXTURE_DIR,
        )
        assert 0.0 <= result.A_frac <= 1.0

    def test_h1_eds_changes_result(self):
        """Using H1_eds should produce a different result than the default."""
        r_default = reweighting_constpH(
            cutoff=-400.0,
            temp=_TEMP,
            stepsize=0.5,
            itime=0.0,
            group=[["1"], ["0"]],
            path=FIXTURE_DIR,
        )
        r_eds = reweighting_constpH(
            cutoff=-400.0,
            temp=_TEMP,
            stepsize=0.5,
            itime=0.0,
            group=[["1"], ["0"]],
            path=FIXTURE_DIR,
            H1_eds=True,
            art_eir=[-220.0, 0.0],
        )
        # The fractions will differ because mix changes
        assert r_default.A_frac != r_eds.A_frac or r_default.A_rew != r_eds.A_rew

    def test_missing_art_eir_raises(self):
        """H1_eds=True without art_eir should raise ValueError."""
        with pytest.raises(ValueError):
            reweighting_constpH(
                cutoff=-400.0,
                temp=_TEMP,
                stepsize=0.5,
                itime=0.0,
                group=[["1"], ["0"]],
                path=FIXTURE_DIR,
                H1_eds=True,
                art_eir=[],
            )
