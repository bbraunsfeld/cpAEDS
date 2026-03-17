"""Tests for cpaeds.aeds_sampling module."""

import os
import tempfile

import numpy as np
import pytest

from cpaeds.aeds_sampling import sampling

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_data", "reweighting")

_BOLTZMANN = 0.00831441
_TEMP = 300.0


# ---------------------------------------------------------------------------
# sampling.read_energy_file (static method)
# ---------------------------------------------------------------------------

class TestReadEnergyFile:
    def test_reads_correct_values(self):
        """Values from the fixture file should match what was written."""
        path = os.path.join(FIXTURE_DIR, "e1.dat")
        vals = sampling.read_energy_file(path)
        expected = np.array([-500.0, -520.0, -480.0, -350.0, -300.0])
        np.testing.assert_array_almost_equal(vals, expected)

    def test_returns_float64_array(self):
        path = os.path.join(FIXTURE_DIR, "eds_vr.dat")
        vals = sampling.read_energy_file(path)
        assert vals.dtype == np.float64

    def test_synthetic_file(self):
        """Read a hand-crafted temp file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("# header\n")
            f.write("  1.0 42.5\n")
            f.write("  2.0 -10.0\n")
            fname = f.name
        try:
            vals = sampling.read_energy_file(fname)
            np.testing.assert_array_almost_equal(vals, [42.5, -10.0])
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# sampling.main (integration test with synthetic fixtures)
# ---------------------------------------------------------------------------

class TestSamplingMain:
    """Integration tests that run sampling.main() against fixture files."""

    def _make_config(self):
        return {
            "simulation": {
                "parameters": {
                    "temp": _TEMP,
                }
            }
        }

    def test_main_returns_five_outputs(self):
        """sampling.main() should return a 5-tuple."""
        import os
        config = self._make_config()
        # offsets and dfs must match the number of end-states (2)
        offsets = [-220.0, 0.0]
        dfs = [2.5, 0.0]

        orig_dir = os.getcwd()
        try:
            os.chdir(FIXTURE_DIR)
            s = sampling(config, offsets, dfs)
            result = s.main()
        finally:
            os.chdir(orig_dir)

        assert len(result) == 5

    def test_fractions_contrib_sum_to_one(self):
        """Boltzmann-weighted fractions must sum to 1."""
        import os
        config = self._make_config()
        offsets = [-220.0, 0.0]
        dfs = [2.5, 0.0]

        orig_dir = os.getcwd()
        try:
            os.chdir(FIXTURE_DIR)
            s = sampling(config, offsets, dfs)
            fractions_contrib, fractions_cutoff, energies, contrib_accum, frame_contribution = s.main()
        finally:
            os.chdir(orig_dir)

        assert abs(sum(fractions_contrib) - 1.0) < 1e-6

    def test_fractions_cutoff_sum_to_one(self):
        """Cutoff-based fractions must also sum to 1."""
        import os
        config = self._make_config()
        offsets = [-220.0, 0.0]
        dfs = [2.5, 0.0]

        orig_dir = os.getcwd()
        try:
            os.chdir(FIXTURE_DIR)
            s = sampling(config, offsets, dfs)
            fractions_contrib, fractions_cutoff, energies, contrib_accum, frame_contribution = s.main()
        finally:
            os.chdir(orig_dir)

        assert abs(sum(fractions_cutoff) - 1.0) < 1e-6

    def test_mismatched_offsets_raises(self):
        """Providing wrong number of offsets should raise ValueError."""
        import os
        config = self._make_config()
        offsets = [-220.0]  # only 1 offset for 2 states
        dfs = [2.5, 0.0]

        orig_dir = os.getcwd()
        try:
            os.chdir(FIXTURE_DIR)
            s = sampling(config, offsets, dfs)
            with pytest.raises(ValueError):
                s.main()
        finally:
            os.chdir(orig_dir)
