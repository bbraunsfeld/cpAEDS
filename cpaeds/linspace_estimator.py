"""Automatic linspace estimation for cpAEDS exponential reweighting.

When running :func:`~cpaeds.reweighting.parallel_reweight`, the user needs to
specify a ``lnspace`` array of artificial EIR offset values to scan.  Choosing
this range by hand is tedious and error-prone: the centre of the scan must
coincide with the **equilibrium offset** — the offset value at which the system
would show 50/50 protonation/deprotonation, i.e. the offset that cancels the
free-energy difference between the two states.

This module provides two methods to estimate that centre automatically:

``estimate_from_simulation``
    Reads the actual simulation EIR offsets (``e*r.dat``) and the free-energy
    differences from dfmult (``df.out``) and computes::

        EIR_eq = EIR_actual + ΔF_state_R

    This works **before** any post-processing visualisation and requires only
    the files produced by ene_ana + dfmult.

``estimate_from_results``
    Reads the ``results.out`` file produced by
    :class:`~cpaeds.postprocessing_parallel.postprocessing_parallel`, fits a
    logistic curve to the fraction-vs-offset data, and returns the midpoint
    (inflection point) as the centre.  This is more accurate but requires a
    completed post-processing run.

``auto_linspace``
    Convenience wrapper that tries ``estimate_from_results`` first (more
    accurate) and falls back to ``estimate_from_simulation``.

All three functions return a ``LinspaceResult`` named tuple containing the
centre, the full ``lnspace`` array, and metadata about which estimation method
was used.

Typical usage::

    from cpaeds.linspace_estimator import auto_linspace

    result = auto_linspace(
        path="/path/to/run/ene_ana",
        active_state=0,       # index of the state to scan
        n_points=101,
        width=50.0,
    )
    print(f"centre = {result.center:.1f} kJ/mol  (method: {result.method})")
    # Pass to parallel_reweight:
    from cpaeds.reweighting import parallel_reweight
    rew_results = parallel_reweight(
        runs=[1, 2, 3],
        lnspace=result.lnspace,
        nthreads=8,
        path_template="/path/to/aeds/ASPD_{run}/ene_ana",
        cutoff=-400.0, temp=300.0, stepsize=0.5, itime=4000.0,
        group=[["1"], ["0"]], H1_eds=True,
    )
"""

from __future__ import annotations

import glob
from typing import NamedTuple

import numpy as np
import pandas as pd

from cpaeds.algorithms import log_fit, logistic_curve
from cpaeds.context_manager import set_directory
from cpaeds.reweighting import get_offsets

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class LinspaceResult(NamedTuple):
    """Output of linspace estimation functions.

    Attributes:
        center: Estimated equilibrium EIR offset (kJ/mol).  The midpoint of
            ``lnspace``.
        lnspace: 1-D array of ``n_points`` EIR values centred on ``center``
            and spanning ``width`` kJ/mol.
        method: Human-readable description of which estimation method was used
            (``'logistic_fit'``, ``'free_energy_correction'``, or
            ``'actual_offset'``).
        width: Total scan width in kJ/mol (= ``lnspace[-1] - lnspace[0]``).
        n_points: Number of points in ``lnspace``.
    """

    center: float
    lnspace: np.ndarray
    method: str
    width: float
    n_points: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_free_energies(df_path: str) -> np.ndarray:
    """Read per-state ΔF values from a ``df.out`` file produced by dfmult.

    Only rows whose index ends with ``_R`` (reference-state comparisons) are
    returned, which gives one ΔF value per end-state.

    Args:
        df_path: Path to the ``df.out`` file.

    Returns:
        1-D array of ΔF values (kJ/mol), one per end-state, in the order
        dfmult writes them.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no ``_R`` rows are found in the file.
    """
    df = pd.read_table(
        df_path,
        names=["DF", "err"],
        index_col=0,
        sep=r"\s+",
        header=0,
    )
    refstates = df.index.str.endswith("_R")
    if not refstates.any():
        raise ValueError(f"No reference-state rows (ending in '_R') found in {df_path}")
    return np.array(df[refstates]["DF"].values, dtype=np.float64)


def _build_lnspace(center: float, width: float, n_points: int) -> np.ndarray:
    """Construct a linearly-spaced array centred on ``center``."""
    half = width / 2.0
    return np.linspace(center - half, center + half, n_points)


# ---------------------------------------------------------------------------
# Method 1: free-energy correction (requires df.out + e*r.dat)
# ---------------------------------------------------------------------------

def estimate_from_simulation(
    path: str,
    active_state: int = 0,
    n_points: int = 101,
    width: float = 50.0,
) -> LinspaceResult:
    """Estimate the optimal linspace from simulation EIR offsets and ΔF.

    Computes the equilibrium offset as::

        EIR_eq = EIR_actual[active_state] + ΔF[active_state]

    where ``EIR_actual`` is read from the ``e*r.dat`` offset files and ``ΔF``
    is read from ``df.out`` (produced by dfmult after ene_ana).

    This method can be called **before** any post-processing visualisation.

    Args:
        path: Directory containing ``e*r.dat`` and ``df.out`` files (the
            ``ene_ana`` subdirectory of a cpAEDS run).
        active_state: Index (0-based) of the end-state whose offset is being
            scanned in the reweighting.  Defaults to ``0`` (the first
            end-state, conventionally the protonated form).
        n_points: Number of EIR values in the returned ``lnspace`` array.
            Defaults to ``101`` (matching the notebook convention).
        width: Total scan width in kJ/mol.  The returned array spans
            ``[center - width/2, center + width/2]``.  Defaults to ``50.0``
            kJ/mol (matching the notebook convention).

    Returns:
        :class:`LinspaceResult` with ``method = 'free_energy_correction'``.

    Raises:
        FileNotFoundError: If offset files or ``df.out`` are missing.
        IndexError: If ``active_state`` exceeds the number of end-states.

    Example::

        result = estimate_from_simulation(
            path="/data/ASPD_trans_aq_1/ASPD_trans_aq_1_5/ene_ana",
            active_state=0,
        )
        print(result.center)   # e.g. -269.8 kJ/mol
        print(result.lnspace)  # array from -294.8 to -244.8 (101 pts)
    """
    offsets = get_offsets(path)
    if active_state >= len(offsets):
        raise IndexError(
            f"active_state={active_state} but only {len(offsets)} end-states found."
        )

    with set_directory(path):
        free_energies = _read_free_energies("df.out")

    if active_state >= len(free_energies):
        raise IndexError(
            f"active_state={active_state} but only {len(free_energies)} ΔF values found."
        )

    center = offsets[active_state] + free_energies[active_state]
    lnspace = _build_lnspace(center, width, n_points)

    return LinspaceResult(
        center=center,
        lnspace=lnspace,
        method="free_energy_correction",
        width=width,
        n_points=n_points,
    )


# ---------------------------------------------------------------------------
# Method 2: logistic fit to results.out
# ---------------------------------------------------------------------------

def estimate_from_results(
    results_path: str,
    state_col: int = -1,
    offset_col: int = -1,
    n_points: int = 101,
    width_factor: float = 1.0,
    min_width: float = 50.0,
) -> LinspaceResult:
    """Estimate the optimal linspace by fitting the logistic curve to post-processing results.

    Reads the ``results.out`` CSV file produced by
    :class:`~cpaeds.postprocessing_parallel.postprocessing_parallel`, fits a
    logistic curve to the fraction-vs-offset relationship, and extracts the
    inflection point (50/50 crossing) as the centre.  The scan width is derived
    from the logistic steepness parameter.

    This method is more accurate than
    :func:`estimate_from_simulation` but requires a completed post-processing
    run.

    Args:
        results_path: Path to the ``results.out`` file (or its parent directory
            — if a directory is given, ``results/results.out`` is appended).
        state_col: Column index (0-based among ``FRACTION*`` columns) of the
            state to use for fitting.  Defaults to ``-1`` (last fraction
            column, conventionally the deprotonated form).
        offset_col: Column index (0-based among ``OFFSET*`` columns) of the
            offset to use as x-axis.  Defaults to ``-1`` (last offset column).
        n_points: Number of EIR values in the returned ``lnspace`` array.
        width_factor: Multiplier applied to the logistic transition width
            ``6 / |c|`` (which spans ~99% of the sigmoid) to set the scan
            range.  Defaults to ``1.0`` (use the natural transition width).
        min_width: Minimum scan width in kJ/mol.  The width will never be
            smaller than this value even if the logistic fit yields a very
            steep curve.  Defaults to ``50.0`` kJ/mol.

    Returns:
        :class:`LinspaceResult` with ``method = 'logistic_fit'``.

    Raises:
        FileNotFoundError: If ``results.out`` cannot be found.
        RuntimeError: If the logistic fit fails to converge.

    Example::

        result = estimate_from_results(
            results_path="/data/ASPD_trans_aq_1/results",
        )
        print(result.center)  # e.g. -221.3 kJ/mol (logistic midpoint)
    """
    import os

    # Accept either a path to results.out or its parent directory
    if os.path.isdir(results_path):
        results_path = os.path.join(results_path, "results", "results.out")
        if not os.path.exists(results_path):
            results_path = os.path.join(
                os.path.dirname(results_path), "results.out"
            )

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.out not found at {results_path}")

    df = pd.read_csv(results_path, index_col=0)

    offsets = df.loc[:, df.columns.str.startswith("OFFSET")].iloc[:, offset_col]
    fractions = df.loc[:, df.columns.str.startswith("FRACTION")].iloc[:, state_col]

    x = np.array(offsets, dtype=np.float64)
    y = np.array(fractions, dtype=np.float64)

    try:
        popt = log_fit(x, y)
    except RuntimeError as e:
        raise RuntimeError(
            f"Logistic fit failed to converge on data from {results_path}. "
            f"Original error: {e}"
        ) from e

    # popt = [a, b, c, d]: d is the midpoint (50/50 crossing)
    center = float(popt[3])

    # Natural transition width: the sigmoid spans ~99% of its range over 6/|c|
    natural_width = 6.0 / abs(float(popt[2]))
    width = max(natural_width * width_factor, min_width)
    lnspace = _build_lnspace(center, width, n_points)

    return LinspaceResult(
        center=center,
        lnspace=lnspace,
        method="logistic_fit",
        width=width,
        n_points=n_points,
    )


# ---------------------------------------------------------------------------
# Method 3: fallback — use actual simulation offset as centre
# ---------------------------------------------------------------------------

def estimate_from_offset(
    path: str,
    active_state: int = 0,
    n_points: int = 101,
    width: float = 50.0,
) -> LinspaceResult:
    """Use the actual simulation EIR offset as the linspace centre.

    This is the simplest fallback: if ``df.out`` is not yet available (e.g.
    dfmult has not been run), centre the scan on the offset the simulation was
    actually run at.

    Args:
        path: Directory containing the ``e*r.dat`` offset files.
        active_state: Index (0-based) of the end-state being scanned.
        n_points: Number of EIR values in ``lnspace``.
        width: Total scan width in kJ/mol.

    Returns:
        :class:`LinspaceResult` with ``method = 'actual_offset'``.
    """
    offsets = get_offsets(path)
    center = offsets[active_state]
    lnspace = _build_lnspace(center, width, n_points)
    return LinspaceResult(
        center=center,
        lnspace=lnspace,
        method="actual_offset",
        width=width,
        n_points=n_points,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def auto_linspace(
    path: str,
    active_state: int = 0,
    n_points: int = 101,
    width: float = 50.0,
    width_factor: float = 1.0,
    min_width: float = 50.0,
    state_col: int = -1,
    offset_col: int = -1,
    prefer: str = "results",
) -> LinspaceResult:
    """Automatically estimate the optimal linspace for reweighting.

    Tries estimation methods in priority order and returns the first that
    succeeds:

    1. **Logistic fit** (``prefer='results'``, default): fits ``results.out``
       if it exists under *path*.  Most accurate — uses real simulation data.
    2. **Free-energy correction** (``prefer='df'``): reads ``df.out`` +
       ``e*r.dat``.  Works after dfmult but before full post-processing.
    3. **Actual offset fallback**: uses the raw simulation EIR offset.

    When ``prefer='df'``, the priority becomes: free-energy correction →
    logistic fit → actual offset.

    Args:
        path: Run directory (``ene_ana`` subdirectory or its parent).  The
            function looks for ``results/results.out``, ``df.out``, and
            ``e*r.dat`` relative to this path.
        active_state: Index (0-based) of the end-state to scan.
        n_points: Number of EIR values in ``lnspace``.  Defaults to ``101``.
        width: Total scan width in kJ/mol for non-logistic methods.
            Defaults to ``50.0`` kJ/mol.
        width_factor: Multiplier for the logistic natural width.  Only used
            when the logistic fit method is selected.  Defaults to ``1.0``.
        min_width: Minimum scan width in kJ/mol.  Defaults to ``50.0``.
        state_col: Fraction column index for the logistic fit method.
        offset_col: Offset column index for the logistic fit method.
        prefer: ``'results'`` (default) tries logistic fit first; ``'df'``
            tries free-energy correction first.

    Returns:
        :class:`LinspaceResult` from the first successful method.

    Example::

        from cpaeds.linspace_estimator import auto_linspace
        from cpaeds.reweighting import parallel_reweight
        import numpy as np

        est = auto_linspace(
            path="/data/ASPD_trans_aq_1_5/ene_ana",
            active_state=0,
        )
        print(f"Scanning {est.n_points} points centred on {est.center:.1f} kJ/mol")
        print(f"Method used: {est.method}")

        results = parallel_reweight(
            runs=[1, 2, 3],
            lnspace=est.lnspace,
            nthreads=8,
            path_template="/data/ASPD_trans_aq_1_{run}/ene_ana",
            cutoff=-400.0, temp=300.0, stepsize=0.5, itime=4000.0,
            group=[["1"], ["0"]], H1_eds=True,
        )
    """
    import os

    def _try_results() -> LinspaceResult | None:
        """Attempt logistic fit from results.out."""
        # Look for results.out relative to path or its parent
        candidates = [
            os.path.join(path, "results", "results.out"),
            os.path.join(os.path.dirname(path), "results", "results.out"),
            os.path.join(path, "results.out"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    return estimate_from_results(
                        candidate,
                        state_col=state_col,
                        offset_col=offset_col,
                        n_points=n_points,
                        width_factor=width_factor,
                        min_width=min_width,
                    )
                except (RuntimeError, ValueError):
                    continue
        return None

    def _try_df() -> LinspaceResult | None:
        """Attempt free-energy correction from df.out + e*r.dat."""
        df_candidates = [
            os.path.join(path, "df.out"),
            path,  # path might already be ene_ana dir
        ]
        for candidate_dir in [path, os.path.dirname(path)]:
            df_file = os.path.join(candidate_dir, "df.out")
            if os.path.exists(df_file):
                try:
                    return estimate_from_simulation(
                        candidate_dir,
                        active_state=active_state,
                        n_points=n_points,
                        width=width,
                    )
                except (FileNotFoundError, IndexError, ValueError):
                    continue
        return None

    def _fallback() -> LinspaceResult:
        """Always succeeds — uses actual simulation offset."""
        for candidate_dir in [path, os.path.dirname(path)]:
            try:
                return estimate_from_offset(
                    candidate_dir,
                    active_state=active_state,
                    n_points=n_points,
                    width=width,
                )
            except (FileNotFoundError, IndexError):
                continue
        raise FileNotFoundError(
            f"Could not find e*r.dat offset files in or near {path}"
        )

    if prefer == "results":
        methods = [_try_results, _try_df, _fallback]
    else:
        methods = [_try_df, _try_results, _fallback]

    for method_fn in methods:
        result = method_fn()
        if result is not None:
            return result

    # _fallback raises rather than returning None, so we never reach here
    return _fallback()  # pragma: no cover


# ---------------------------------------------------------------------------
# Batch helper for multiple runs
# ---------------------------------------------------------------------------

def batch_linspace(
    run_paths: list[str],
    active_state: int = 0,
    n_points: int = 101,
    width: float = 50.0,
    prefer: str = "results",
    **kwargs,
) -> tuple[np.ndarray, list[LinspaceResult]]:
    """Estimate and merge linspaces from multiple run directories.

    Estimates the equilibrium offset independently for each run in
    *run_paths*, then returns a single merged ``lnspace`` array that covers
    all of them plus a list of individual :class:`LinspaceResult` objects.

    The merged array spans from ``min(center) - width/2`` to
    ``max(center) + width/2`` with *n_points* points.

    Args:
        run_paths: List of ene_ana directory paths, one per statistical run.
        active_state: Index of the end-state to scan.
        n_points: Number of points in the merged ``lnspace``.
        width: Individual scan width per run (kJ/mol).
        prefer: Passed to :func:`auto_linspace`.
        **kwargs: Additional keyword arguments forwarded to
            :func:`auto_linspace`.

    Returns:
        Tuple ``(merged_lnspace, individual_results)`` where
        ``merged_lnspace`` is a 1-D array and ``individual_results`` is a
        list of :class:`LinspaceResult` objects.
    """
    individual = [
        auto_linspace(p, active_state=active_state, n_points=n_points,
                      width=width, prefer=prefer, **kwargs)
        for p in run_paths
    ]
    centers = np.array([r.center for r in individual])
    merged_min = centers.min() - width / 2.0
    merged_max = centers.max() + width / 2.0
    merged = np.linspace(merged_min, merged_max, n_points)
    return merged, individual
