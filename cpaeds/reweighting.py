"""Exponential reweighting for constant-pH AEDS simulations.

This module implements the exponential reweighting methodology that allows
reconstruction of protonation fractions at arbitrary EIR offset values from
a single simulation run. This avoids running a full titration series of
independent simulations.

Core formula:
    <Q>_rew = <Q · exp(-β(H_i - H_ref))> / <exp(-β(H_i - H_ref))>

where H_i is the reference Hamiltonian at offset i and H_ref is the
simulation reference energy (eds_vr.dat).

Typical workflow:
    1. Run a cpAEDS simulation with a fixed EIR offset.
    2. Post-process with ``postprocessing_parallel.py`` to produce energy files.
    3. Call :func:`parallel_reweight` to scan over a range of artificial offsets.
    4. Visualise with :class:`cpaeds.reweighting_plots.ReweightPlot`.
"""

from __future__ import annotations

import glob
from math import log
from multiprocessing import Pool
from typing import NamedTuple

import numpy as np
import pandas as pd
from natsort import natsorted

from cpaeds.aeds_sampling import sampling as _Sampling
from cpaeds.context_manager import set_directory

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class ReweightResult(NamedTuple):
    """Container for the output of :func:`reweighting_constpH`.

    Attributes:
        mix: Reference/mixing energy array used for reweighting.  This is
            either the raw ``eds_vmix.dat`` array or a computed Hr / Hr_s
            value depending on the ``H1_eds`` / ``H1_aeds`` flags.
        reference: Reference free-energy array from ``eds_vr.dat``.
        A_rew: Reweighted population estimate for state group A (scalar).
        B_rew: Reweighted population estimate for state group B (scalar).
        A_frac: Fraction of population in group A = A_rew / (A_rew + B_rew).
        B_frac: Fraction of population in group B = B_rew / (A_rew + B_rew).
        p_A: Per-frame importance weights for group A (shape: n_frames).
        p_B: Per-frame importance weights for group B (shape: n_frames).
        emin: Minimum EDS energy value read from ``eds_emin.dat``.
        emax: Maximum EDS energy value read from ``eds_emax.dat``.
    """

    mix: np.ndarray
    reference: np.ndarray
    A_rew: float
    B_rew: float
    A_frac: float
    B_frac: float
    p_A: np.ndarray
    p_B: np.ndarray
    emin: float
    emax: float


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_offset_file(file_name: str) -> float:
    """Read the single offset value stored in a GROMOS ``e*r.dat`` file.

    The file format is a two-column text file where the last value on the
    second line is the offset in kJ/mol.

    Args:
        file_name: Path to the offset file (e.g. ``e1r.dat``).

    Returns:
        Offset value in kJ/mol.

    Raises:
        ValueError: If the file contains no data after the header line.
    """
    with open(file_name) as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.rstrip()
            if line:
                return float(line.split()[-1])
    raise ValueError(f"No data found in offset file: {file_name}")


def get_offsets(path: str = ".") -> list[float]:
    """Collect all per-state EIR offset values from a simulation directory.

    Reads every ``e*r.dat`` file (sorted in natural order) and returns the
    offset values as a list where index *i* corresponds to end-state *i*.

    Args:
        path: Directory containing the ``e*r.dat`` files.  Defaults to the
            current working directory.

    Returns:
        List of offset values in kJ/mol, one per end-state.
    """
    with set_directory(path):
        offset_files = sorted(
            glob.glob("e*[0-9]r.dat"),
            key=lambda x: int(x.split(".")[0][1:-1]),
        )
        return [read_offset_file(f) for f in offset_files]


def get_eminmax(path: str = ".") -> tuple[float, float]:
    """Read the EDS energy window boundaries from a simulation directory.

    Args:
        path: Directory containing ``eds_emin.dat`` and ``eds_emax.dat``.

    Returns:
        Tuple ``(emin, emax)`` in kJ/mol.
    """
    with set_directory(path):
        emin = _Sampling.read_energy_file("eds_emin.dat")
        emax = _Sampling.read_energy_file("eds_emax.dat")
    return float(emin[0]), float(emax[0])


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

_BOLTZMANN = 0.00831441  # kJ / (mol · K)


def Hr(
    BETA: float,
    energy: np.ndarray,
    offset: float,
    avg: bool = True,
) -> float | np.ndarray:
    """Compute the EDS reference energy via exponential averaging.

    When ``avg=True`` returns a single scalar (the exponential average over
    the whole trajectory).  When ``avg=False`` returns the per-frame
    exponential values before averaging, which can be used for reweighting.

    Args:
        BETA: Inverse thermal energy β = 1 / (k_B · T) in mol/kJ.
        energy: 1-D array of end-state energies (n_frames,).
        offset: EIR offset for this end-state in kJ/mol.
        avg: If ``True``, return the log-sum-exp scalar; otherwise return
            the per-frame array ``exp(-β · (energy - offset))``.

    Returns:
        Scalar exponential average (``avg=True``) or array of
        per-frame values (``avg=False``).
    """
    exp_vals = np.exp(-BETA * (energy - offset))
    if avg:
        return -(1.0 / BETA) * log(exp_vals.sum())
    return exp_vals


def Hr_eds(
    BETA: float,
    energy: np.ndarray,
    offset: float | list[float],
    avg: bool = True,
) -> np.ndarray:
    """Compute per-state EDS reference energies for a multi-state system.

    Args:
        BETA: Inverse thermal energy β = 1 / (k_B · T) in mol/kJ.
        energy: 2-D array of end-state energies with shape
            ``(n_states, n_frames)``.
        offset: Scalar offset applied to all states, or list of per-state
            offsets of length ``n_states``.
        avg: If ``True``, return a 1-D array of per-frame exponential
            averages (shape: ``n_frames``).  If ``False``, return the
            full per-state per-frame array (shape: ``(n_states, n_frames)``).

    Returns:
        Exponential average array.
    """
    if avg:
        # Loop over frames; pass full offset vector to Hr so it can broadcast
        # over states (energy shape per frame: (n_states,)).
        Href_eds = np.zeros(energy.shape[1], dtype=np.float64)
        for i, hi in enumerate(energy.T):
            Href_eds[i] = Hr(BETA, hi, offset, avg=True)
        return Href_eds
    else:
        # Loop over states; each state gets its own scalar offset.
        Hi_eds = np.empty_like(energy)
        for i, hi_state in enumerate(energy):
            off = offset[i] if hasattr(offset, "__len__") else offset
            Hi_eds[i] = Hr(BETA, hi_state, off, avg=False)
        return Hi_eds


def Hr_s(
    BETA: float,
    energy: np.ndarray,
    offset: float | list[float],
    emin: float,
    emax: float,
    avg: bool = True,
) -> np.ndarray:
    """Compute soft-core corrected EDS reference energies.

    Applies a quadratic smoothing correction to the exponential average in
    the transition region between *emin* and *emax* to avoid discontinuities
    in the reweighting potential.

    Three branches are applied per frame:
        * ``Href >= emax``: subtract ``(emax - emin) / 2``
        * ``emin < Href < emax``: subtract quadratic correction
          ``(Href - emin)² / (2 · (emax - emin))``
        * ``Href <= emin``: use as-is

    Args:
        BETA: Inverse thermal energy β = 1 / (k_B · T) in mol/kJ.
        energy: 2-D array of end-state energies ``(n_states, n_frames)``.
        offset: Scalar or per-state list of offsets in kJ/mol.
        emin: Lower EDS energy window boundary in kJ/mol.
        emax: Upper EDS energy window boundary in kJ/mol.
        avg: If ``True``, compute and correct the per-frame scalar average;
            otherwise apply correction element-wise to the full array.

    Returns:
        Corrected energy array of shape ``(n_frames,)`` if ``avg=True``,
        or ``(n_states, n_frames)`` if ``avg=False``.
    """
    window = emax - emin

    if avg:
        # Loop over frames; pass full offset vector to Hr so it broadcasts
        # over states for the per-frame log-sum-exp.
        Href_acc = np.empty(energy.shape[1], dtype=np.float64)
        for i, hi in enumerate(energy.T):
            Href = Hr(BETA, hi, offset, avg=True)
            if Href >= emax:
                Href_acc[i] = Href - window / 2.0
            elif Href > emin:
                Href_acc[i] = Href - ((Href - emin) ** 2) / (2.0 * window)
            else:
                Href_acc[i] = Href
        return Href_acc
    else:
        Hi_acc = np.empty_like(energy, dtype=np.float64)
        for i, hi_state in enumerate(energy):
            off = offset[i] if hasattr(offset, "__len__") else offset
            Hi_array = Hr(BETA, hi_state, off, avg=False)
            above = Hi_array >= emax
            mid = (Hi_array > emin) & ~above
            below = ~above & ~mid
            Hi_acc[i][above] = Hi_array[above] - window / 2.0
            Hi_acc[i][mid] = Hi_array[mid] - ((Hi_array[mid] - emin) ** 2) / (2.0 * window)
            Hi_acc[i][below] = Hi_array[below]
        return Hi_acc


def reweighting(
    Q: np.ndarray | list,
    H_i: np.ndarray,
    H_ref: np.ndarray,
    beta: float,
) -> tuple[float, np.ndarray]:
    """Perform exponential reweighting of observable Q.

    Implements the standard free-energy perturbation reweighting formula::

        <Q>_rew = <Q · exp(-β(H_i - H_ref))> / <exp(-β(H_i - H_ref))>

    Args:
        Q: Observable time series to reweight, shape ``(n_frames,)``.
            Can be a binary indicator (0/1) for state populations.
        H_i: Target Hamiltonian time series in kJ/mol, shape ``(n_frames,)``.
        H_ref: Reference Hamiltonian time series in kJ/mol, shape
            ``(n_frames,)``.
        beta: Inverse thermal energy β = 1 / (k_B · T) in mol/kJ.

    Returns:
        Tuple ``(Q_rew, p)`` where:

        * ``Q_rew`` — reweighted estimate of Q (scalar).
        * ``p`` — normalised per-frame importance weights, shape
          ``(n_frames,)``.  Weights sum to ``n_frames`` (not 1).
    """
    Q = np.asarray(Q, dtype=np.float64)
    H_i = np.asarray(H_i, dtype=np.float64)
    H_ref = np.asarray(H_ref, dtype=np.float64)

    H_diff = -beta * (H_i - H_ref)
    expH_diff = np.exp(H_diff)
    H_avg = np.mean(expH_diff)
    Q_rew = np.mean(Q * expH_diff) / H_avg
    p = expH_diff / H_avg
    return float(Q_rew), p


# ---------------------------------------------------------------------------
# Group-mixing (from reweight_by_state.py)
# ---------------------------------------------------------------------------

def mixing_by_states(
    endstates_e: list[np.ndarray],
    groups: list[list[int]],
    BETA: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Compute EDS mixture energies grouped by user-defined state sets.

    For each group of end-states, computes the per-frame log-sum-exp mixing
    energy and the mean difference between the mixing energy and the minimum
    state energy.

    Args:
        endstates_e: List of per-state energy arrays, each shape
            ``(n_frames,)``.  Index *i* corresponds to end-state *i*.
        groups: List of state-index groups.  Only groups with more than one
            member are processed.  Example: ``[[1, 2, 3], [0]]``.
        BETA: Inverse thermal energy β = 1 / (k_B · T) in mol/kJ.

    Returns:
        Tuple ``(E_mix_groups, E_diff_groups)`` where:

        * ``E_mix_groups`` — list of per-frame mixing energy arrays, one per
          multi-member group.
        * ``E_diff_groups`` — list of mean (mixing - minimum) energy
          differences, one scalar per multi-member group.
    """
    enes = np.array(endstates_e)
    E_mix_groups: list[np.ndarray] = []
    E_diff_groups: list[float] = []

    for group in groups:
        if len(group) <= 1:
            continue
        idx = [int(s) for s in group]
        E_states = enes[idx]  # shape: (n_group_states, n_frames)
        E_mix = np.empty(E_states.shape[1], dtype=np.float64)
        E_min = np.empty(E_states.shape[1], dtype=np.float64)
        for i, frame in enumerate(E_states.T):
            E_mix[i] = -(1.0 / BETA) * log(np.exp(-BETA * frame).sum())
            E_min[i] = frame.min()
        E_diff_groups.append(float(np.mean(E_mix - E_min)))
        E_mix_groups.append(E_mix)

    return E_mix_groups, E_diff_groups


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def reweighting_constpH(
    cutoff: float,
    temp: float,
    stepsize: float,
    itime: float,
    group: list[list],
    path: str,
    H1_eds: bool = False,
    H1_aeds: bool = False,
    art_eir: list[float] | None = None,
    art_eminmax: list[float] | None = None,
    gro_rew: bool = False,
) -> ReweightResult:
    """Compute reweighted protonation fractions for a cpAEDS run directory.

    Reads GROMOS energy files from *path*, classifies each frame into one of
    two groups (A = protonated / B = deprotonated) based on the energy
    *cutoff*, and applies exponential reweighting to estimate the population
    of each group at an artificial offset (*art_eir*).

    Args:
        cutoff: Energy cutoff in kJ/mol for A/B state classification.
            Frames where the lowest end-state energy is below this value are
            assigned to group A.  Typical value: ``-400``.
        temp: Simulation temperature in K.
        stepsize: MD time step between energy output frames in ps.
        itime: Initial simulation time in ps (used only when
            ``gro_rew=True``).
        group: State grouping, e.g. ``[['1', '2', '3'], ['0']]``.
            Currently used only for documentation / future multi-group support.
        path: Absolute path to the ``ene_ana`` subdirectory containing the
            energy files (``e*[0-9].dat``, ``eds_vr.dat``, etc.).
        H1_eds: If ``True``, replace ``eds_vmix.dat`` with
            :func:`Hr_eds`-computed reference using *art_eir*.
        H1_aeds: If ``True``, replace ``eds_vmix.dat`` with
            :func:`Hr_s`-computed soft-core reference using *art_eir*.
            Overrides ``H1_eds`` if both are ``True``.
        art_eir: Artificial EIR offsets (one per end-state) used when
            ``H1_eds`` or ``H1_aeds`` is ``True``.  Defaults to ``None``
            (uses the offsets stored in the simulation files).
        art_eminmax: ``[emin, emax]`` override for :func:`Hr_s`.  If
            ``None``, the values are read from ``eds_emin.dat`` /
            ``eds_emax.dat``.
        gro_rew: If ``True``, write state time-series and GROMOS reweighting
            input files and run the external ``reweight`` program.

    Returns:
        :class:`ReweightResult` named tuple with all reweighting outputs.

    Raises:
        FileNotFoundError: If required energy files are missing in *path*.
        ValueError: If *art_eir* is required but not provided.
    """
    if art_eir is None:
        art_eir = []

    BETA = 1.0 / (temp * _BOLTZMANN)

    with set_directory(path):
        reference = _Sampling.read_energy_file("eds_vr.dat")
        mix = _Sampling.read_energy_file("eds_vmix.dat")

        efiles = natsorted(glob.glob("e*[0-9].dat"))
        OFFSETS = get_offsets(".")
        emin_val, emax_val = get_eminmax(".")

        df_free = pd.read_table(
            "df.out",
            names=["DF", "err"],
            index_col=0,
            sep=r"\s+",
            header=0,
        )
        refstates = df_free.index.str.endswith("_R")
        FREE = np.array(df_free[refstates]["DF"].values)

        endstates_e = [_Sampling.read_energy_file(f) for f in efiles]

    n_states = len(efiles)
    enes = np.array(endstates_e)  # (n_states, n_frames)

    # Build per-state exp(-β(H-offset)) and find dominant state per frame
    endstates_totals = np.empty_like(enes)
    for i, hi in enumerate(enes):
        de = (hi - OFFSETS[i]) * BETA * -1.0
        endstates_totals[i] = np.exp(de)

    minstates_exp = np.argmax(endstates_totals, axis=0)

    # Classify frames: A = lowest-energy state is below cutoff
    state_series_A = np.zeros(enes.shape[1], dtype=np.float64)
    state_series_B = np.zeros(enes.shape[1], dtype=np.float64)
    for i, lowest in enumerate(minstates_exp):
        if enes[lowest, i] < cutoff:
            state_series_A[i] = 1.0
        else:
            state_series_B[i] = 1.0

    # Optionally override the mixing/reference energy
    if H1_aeds:
        if not art_eir:
            raise ValueError("art_eir must be provided when H1_aeds=True")
        emin_use = art_eminmax[0] if art_eminmax else emin_val
        emax_use = art_eminmax[1] if art_eminmax else emax_val
        mix = Hr_s(BETA, enes, offset=art_eir, emin=emin_use, emax=emax_use, avg=True)
    elif H1_eds:
        if not art_eir:
            raise ValueError("art_eir must be provided when H1_eds=True")
        mix = Hr_eds(BETA, enes, offset=art_eir, avg=True)

    A_rew, p_A = reweighting(state_series_A, mix, reference, BETA)
    B_rew, p_B = reweighting(state_series_B, mix, reference, BETA)
    A_frac = A_rew / (A_rew + B_rew)
    B_frac = B_rew / (A_rew + B_rew)

    if gro_rew:
        _write_state_series("tser_stateA.dat", state_series_A, itime, stepsize)
        _write_state_series("tser_stateB.dat", state_series_B, itime, stepsize)
        _create_reweight_input(path, temp, "A")
        _create_reweight_input(path, temp, "B")
        _run_gromos_reweight(path)

    return ReweightResult(
        mix=mix,
        reference=reference,
        A_rew=A_rew,
        B_rew=B_rew,
        A_frac=A_frac,
        B_frac=B_frac,
        p_A=p_A,
        p_B=p_B,
        emin=emin_val,
        emax=emax_val,
    )


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def _reweight_worker(args: tuple) -> ReweightResult:
    """Unpacking wrapper for :func:`multiprocessing.Pool.starmap`."""
    eir_value, run_path, kwargs = args
    return reweighting_constpH(path=run_path, art_eir=eir_value, **kwargs)


def parallel_reweight(
    runs: list[int],
    lnspace: np.ndarray,
    nthreads: int,
    path_template: str,
    **kwargs,
) -> list[list[ReweightResult]]:
    """Run :func:`reweighting_constpH` in parallel over an EIR scan range.

    For each run index in *runs*, processes all EIR values in *lnspace* using
    a :class:`multiprocessing.Pool` of *nthreads* workers.

    Args:
        runs: List of integer run indices (e.g. ``[5, 7, 8, 9]``).
        lnspace: 1-D array of artificial EIR offset values to scan over.
            Each value is passed as ``art_eir[0]`` (primary state offset)
            while the remaining states are held at 0.  For multi-state scans,
            pass full offset lists via ``kwargs``.
        nthreads: Number of parallel worker processes.
        path_template: Path template where ``{run}`` is replaced by the run
            index, e.g. ``"/data/ASPD_trans_aq_1/ASPD_trans_aq_1_{run}/ene_ana"``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`reweighting_constpH` (e.g. ``cutoff``, ``temp``,
            ``H1_eds``, ``group``).

    Returns:
        Nested list of shape ``[n_runs][n_eir_values]`` where each element is
        a :class:`ReweightResult`.

    Example::

        results = parallel_reweight(
            runs=[5, 7, 8, 9],
            lnspace=np.linspace(-250, -200, 101),
            nthreads=14,
            path_template="/data/ASP_{run}/ene_ana",
            cutoff=-400,
            temp=300,
            stepsize=0.5,
            itime=4000,
            group=[["1", "2"], ["0"]],
            H1_eds=True,
        )
        # Access A_frac for run index 0, EIR scan point 50:
        results[0][50].A_frac
    """
    out_list: list[list[ReweightResult]] = []
    for run_i in runs:
        run_path = path_template.format(run=run_i)
        tasks = [
            ([float(eir)] + [0.0] * (len(kwargs.get("art_eir", [0])) - 1 or 0), run_path, kwargs)
            for eir in lnspace
        ]
        with Pool(nthreads) as pool:
            results = pool.map(_reweight_worker, tasks)
        out_list.append(results)
    return out_list


# ---------------------------------------------------------------------------
# GROMOS reweighting helpers (gro_rew=True path)
# ---------------------------------------------------------------------------

def _write_state_series(
    outfile: str,
    state: np.ndarray,
    itime: float,
    step: float,
) -> None:
    """Write a binary state time series to a two-column text file."""
    time = itime
    with open(outfile, "w") as out:
        for element in state:
            out.write(f"{round(time, 3)}\t{element}\n")
            time += step


def _create_reweight_input(path: str, temp: float, state: str) -> None:
    """Write GROMOS reweighting argument file for a given state."""
    with open(f"{path}/gromos_reweight_state{state}.arg", "w") as fh:
        fh.write(f"@temp {temp}\n")
        fh.write(f"@x tser_state{state}.dat\n")
        fh.write("@vr eds_vr.dat\n")
        fh.write("@vy eds_vmix.dat")


def _run_gromos_reweight(path: str) -> None:
    """Call the external GROMOS ``reweight`` binary for both states."""
    import subprocess

    for state in ("A", "B"):
        with open(f"{path}/state{state}_rew.dat", "w") as out:
            subprocess.run(
                ["reweight", "@f", f"{path}/gromos_reweight_state{state}.arg"],
                stdout=out,
                check=True,
            )
