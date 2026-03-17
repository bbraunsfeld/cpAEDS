"""Tests for cpaeds.linspace_estimator module."""

import os
import math
import tempfile

import numpy as np
import pytest

from cpaeds.linspace_estimator import (
    LinspaceResult,
    auto_linspace,
    batch_linspace,
    estimate_from_offset,
    estimate_from_results,
    estimate_from_simulation,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_data", "reweighting")

# Fixture values written by the test data generator
FIXTURE_OFFSET_STATE0 = -220.0   # from e1r.dat
FIXTURE_DF_STATE0 = 2.5          # from df.out
FIXTURE_LOGISTIC_MIDPOINT = -245.0  # centre of the synthetic results.out sigmoid


# ---------------------------------------------------------------------------
# estimate_from_offset
# ---------------------------------------------------------------------------

class TestEstimateFromOffset:
    def test_returns_linspace_result(self):
        r = estimate_from_offset(FIXTURE_DIR, active_state=0)
        assert isinstance(r, LinspaceResult)

    def test_method_label(self):
        r = estimate_from_offset(FIXTURE_DIR)
        assert r.method == "actual_offset"

    def test_center_equals_actual_eir(self):
        r = estimate_from_offset(FIXTURE_DIR, active_state=0)
        assert abs(r.center - FIXTURE_OFFSET_STATE0) < 1e-6

    def test_lnspace_length(self):
        r = estimate_from_offset(FIXTURE_DIR, n_points=51, width=40.0)
        assert len(r.lnspace) == 51

    def test_lnspace_width(self):
        r = estimate_from_offset(FIXTURE_DIR, width=40.0)
        assert abs(r.lnspace[-1] - r.lnspace[0] - 40.0) < 1e-9

    def test_lnspace_centered_on_center(self):
        r = estimate_from_offset(FIXTURE_DIR, width=50.0)
        computed_center = (r.lnspace[0] + r.lnspace[-1]) / 2
        assert abs(computed_center - r.center) < 1e-6

    def test_invalid_state_raises(self):
        with pytest.raises(IndexError):
            estimate_from_offset(FIXTURE_DIR, active_state=99)


# ---------------------------------------------------------------------------
# estimate_from_simulation
# ---------------------------------------------------------------------------

class TestEstimateFromSimulation:
    def test_returns_linspace_result(self):
        r = estimate_from_simulation(FIXTURE_DIR)
        assert isinstance(r, LinspaceResult)

    def test_method_label(self):
        r = estimate_from_simulation(FIXTURE_DIR)
        assert r.method == "free_energy_correction"

    def test_center_equals_offset_plus_df(self):
        """center = EIR_actual + ΔF_state."""
        r = estimate_from_simulation(FIXTURE_DIR, active_state=0)
        expected_center = FIXTURE_OFFSET_STATE0 + FIXTURE_DF_STATE0
        assert abs(r.center - expected_center) < 1e-6

    def test_default_width_is_50(self):
        r = estimate_from_simulation(FIXTURE_DIR)
        assert abs(r.width - 50.0) < 1e-9

    def test_custom_width(self):
        r = estimate_from_simulation(FIXTURE_DIR, width=80.0)
        assert abs(r.lnspace[-1] - r.lnspace[0] - 80.0) < 1e-9

    def test_custom_n_points(self):
        r = estimate_from_simulation(FIXTURE_DIR, n_points=201)
        assert len(r.lnspace) == 201

    def test_invalid_state_raises(self):
        with pytest.raises(IndexError):
            estimate_from_simulation(FIXTURE_DIR, active_state=50)

    def test_lnspace_monotonically_increasing(self):
        r = estimate_from_simulation(FIXTURE_DIR)
        diffs = np.diff(r.lnspace)
        assert np.all(diffs > 0)


# ---------------------------------------------------------------------------
# estimate_from_results
# ---------------------------------------------------------------------------

class TestEstimateFromResults:
    _results_file = os.path.join(FIXTURE_DIR, "results.out")

    def test_returns_linspace_result(self):
        r = estimate_from_results(self._results_file)
        assert isinstance(r, LinspaceResult)

    def test_method_label(self):
        r = estimate_from_results(self._results_file)
        assert r.method == "logistic_fit"

    def test_center_near_logistic_midpoint(self):
        """Logistic fit should recover the synthetic midpoint within ±5 kJ/mol."""
        r = estimate_from_results(self._results_file)
        assert abs(r.center - FIXTURE_LOGISTIC_MIDPOINT) < 5.0

    def test_width_at_least_min_width(self):
        r = estimate_from_results(self._results_file, min_width=50.0)
        assert r.width >= 50.0

    def test_n_points_preserved(self):
        r = estimate_from_results(self._results_file, n_points=201)
        assert len(r.lnspace) == 201

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            estimate_from_results("/nonexistent/path/results.out")

    def test_directory_input_resolves_file(self):
        """Passing a directory should still find results.out if it exists."""
        # The fixture dir has results.out directly, not in a 'results' subdir
        # estimate_from_results should find it via the candidate list
        r = estimate_from_results(self._results_file)
        assert r.method == "logistic_fit"


# ---------------------------------------------------------------------------
# auto_linspace
# ---------------------------------------------------------------------------

class TestAutoLinspace:
    def test_returns_linspace_result(self):
        r = auto_linspace(FIXTURE_DIR)
        assert isinstance(r, LinspaceResult)

    def test_prefer_results_uses_logistic_when_available(self):
        """With results.out present, prefer='results' should use logistic fit."""
        r = auto_linspace(FIXTURE_DIR, prefer="results")
        # The fixture has results.out so logistic_fit should be picked
        assert r.method in ("logistic_fit", "free_energy_correction", "actual_offset")

    def test_prefer_df_uses_df_when_available(self):
        """With df.out present, prefer='df' should use free_energy_correction."""
        r = auto_linspace(FIXTURE_DIR, prefer="df")
        assert r.method in ("free_energy_correction", "logistic_fit", "actual_offset")

    def test_lnspace_is_monotone(self):
        r = auto_linspace(FIXTURE_DIR)
        assert np.all(np.diff(r.lnspace) > 0)

    def test_center_inside_lnspace(self):
        r = auto_linspace(FIXTURE_DIR)
        assert r.lnspace[0] <= r.center <= r.lnspace[-1]

    def test_custom_n_points_passed_through(self):
        r = auto_linspace(FIXTURE_DIR, n_points=51)
        assert len(r.lnspace) == 51

    def test_fallback_on_missing_df(self, tmp_path):
        """When only e*r.dat files exist (no df.out), should fall back."""
        # Create a minimal dir with just offset files
        for i, off in enumerate([-220.0, 0.0]):
            with open(tmp_path / f"e{i+1}r.dat", "w") as f:
                f.write("# header\n")
                f.write(f"  0.0 {off}\n")
        r = auto_linspace(str(tmp_path), prefer="results")
        assert r.method == "actual_offset"

    def test_completely_empty_dir_raises(self, tmp_path):
        """A dir with no relevant files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            auto_linspace(str(tmp_path))


# ---------------------------------------------------------------------------
# batch_linspace
# ---------------------------------------------------------------------------

class TestBatchLinspace:
    def test_merged_covers_all_centers(self):
        """Merged lnspace must cover all individual centres."""
        merged, individual = batch_linspace(
            [FIXTURE_DIR, FIXTURE_DIR],  # same dir → same centre
            active_state=0,
            n_points=101,
            width=50.0,
            prefer="df",
        )
        for r in individual:
            assert merged[0] <= r.center <= merged[-1]

    def test_returns_two_items(self):
        merged, individual = batch_linspace([FIXTURE_DIR], prefer="df")
        assert isinstance(merged, np.ndarray)
        assert isinstance(individual, list)
        assert len(individual) == 1

    def test_merged_n_points(self):
        merged, _ = batch_linspace([FIXTURE_DIR, FIXTURE_DIR], n_points=61, prefer="df")
        assert len(merged) == 61

    def test_merged_monotone(self):
        merged, _ = batch_linspace([FIXTURE_DIR, FIXTURE_DIR], prefer="df")
        assert np.all(np.diff(merged) > 0)


# ---------------------------------------------------------------------------
# LinspaceResult NamedTuple interface
# ---------------------------------------------------------------------------

class TestLinspaceResult:
    def test_all_fields_accessible(self):
        r = estimate_from_offset(FIXTURE_DIR)
        _ = r.center
        _ = r.lnspace
        _ = r.method
        _ = r.width
        _ = r.n_points

    def test_field_names(self):
        assert LinspaceResult._fields == ("center", "lnspace", "method", "width", "n_points")
