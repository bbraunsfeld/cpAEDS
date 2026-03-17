"""Visualisation tools for exponential reweighting results.

This module provides :class:`ReweightPlot`, which takes the nested output of
:func:`cpaeds.reweighting.parallel_reweight` and generates publication-quality
figures for:

* Reweighted state-A population vs. delta-EIR with run-to-run variability bands.
* Reconstructed pH titration curves with error bands.
* pKa estimation from the inflection point of the reweighted titration curve.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from cpaeds.algorithms import ph_curve, log_fit, logistic_curve
from cpaeds.reweighting import ReweightResult


class ReweightPlot:
    """Visualise the output of :func:`~cpaeds.reweighting.parallel_reweight`.

    Args:
        results: Nested list of shape ``[n_runs][n_eir_values]``, as returned
            by :func:`~cpaeds.reweighting.parallel_reweight`.  Each element
            must be a :class:`~cpaeds.reweighting.ReweightResult`.
        lnspace: 1-D array of artificial EIR offset values used during the
            scan.  Must match the second dimension of *results*.
        ref_eir: Reference EIR offset (the value used in the actual
            simulation).  Used to compute ``delta_EIR = lnspace - ref_eir``
            for the x-axis.
    """

    def __init__(
        self,
        results: list[list[ReweightResult]],
        lnspace: np.ndarray,
        ref_eir: float = 0.0,
    ) -> None:
        self.results = results
        self.lnspace = np.asarray(lnspace)
        self.ref_eir = ref_eir
        self.delta_eir = self.lnspace - ref_eir

        # Cache A_frac array: shape (n_runs, n_eir_values)
        self._A_frac = np.array(
            [[r.A_frac for r in run] for run in results]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-EIR-point mean and std over runs."""
        return self._A_frac.mean(axis=0), self._A_frac.std(axis=0)

    # ------------------------------------------------------------------
    # Public plotting methods
    # ------------------------------------------------------------------

    def plot_A_rew(
        self,
        ax: plt.Axes | None = None,
        colors: list[str] | None = None,
        show_runs: bool = True,
        show_mean: bool = True,
        show_band: bool = True,
        plotArgs: dict | None = None,
    ) -> plt.Axes:
        """Plot reweighted state-A fraction vs. delta-EIR.

        Optionally overlays individual run traces and a mean ± std band.

        Args:
            ax: Matplotlib axes to draw on.  A new figure is created and
                saved if ``None``.
            colors: List of colours for ``[individual runs, mean line, band]``.
                Defaults to a built-in palette.
            show_runs: Draw individual per-run traces.
            show_mean: Draw the mean line.
            show_band: Draw a ``fill_between`` band for ± 1 std deviation.
            plotArgs: Additional keyword arguments passed to line plots.

        Returns:
            The axes object.
        """
        plotArgs = plotArgs or {}
        colors = colors or ["#999999", "#E69F00", "#56B4E9"]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(1, 1, dpi=200)

        mean, std = self._mean_std()

        if show_runs:
            for i, run in enumerate(self._A_frac):
                ax.plot(
                    self.delta_eir,
                    run,
                    color=colors[0],
                    alpha=0.4,
                    linewidth=0.8,
                    label=f"run {i + 1}" if i == 0 else None,
                    **plotArgs,
                )

        if show_band:
            ax.fill_between(
                self.delta_eir,
                mean - std,
                mean + std,
                color=colors[2],
                alpha=0.3,
                label="± 1 std",
            )

        if show_mean:
            ax.plot(
                self.delta_eir,
                mean,
                color=colors[1],
                linewidth=1.5,
                label="mean",
                **plotArgs,
            )

        ax.set_xlabel("ΔOffset (kJ/mol)")
        ax.set_ylabel("A_rew (reweighted fraction)")
        ax.legend(fontsize=8)

        if standalone:
            plt.savefig("A_rew_vs_delta_eir.png", bbox_inches="tight")
            plt.close()

        gc.collect()
        return ax

    def plot_titration(
        self,
        pKa: float,
        ax: plt.Axes | None = None,
        colors: list[str] | None = None,
        show_runs: bool = True,
        show_mean: bool = True,
        show_band: bool = True,
        show_pKa_line: bool = True,
        plotArgs: dict | None = None,
    ) -> plt.Axes:
        """Plot reconstructed pH titration curve from reweighted fractions.

        Converts each reweighted ``A_frac`` profile to a pH curve using
        :func:`~cpaeds.algorithms.ph_curve` (Henderson–Hasselbalch) and plots
        it against ``delta_EIR``.

        Args:
            pKa: Experimental or reference pKa used for the Henderson–
                Hasselbalch conversion.
            ax: Matplotlib axes to draw on.  A new figure is created and
                saved if ``None``.
            colors: Colour list for ``[runs, mean, band]``.
            show_runs: Draw individual per-run pH traces.
            show_mean: Draw the mean pH trace.
            show_band: Draw ± 1 std band.
            show_pKa_line: Draw a horizontal reference line at *pKa*.
            plotArgs: Additional kwargs for line plots.

        Returns:
            The axes object.
        """
        plotArgs = plotArgs or {}
        colors = colors or ["#999999", "#E69F00", "#56B4E9"]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(1, 1, dpi=200)

        # Convert A_frac → pH for every run
        pH_array = np.array(
            [ph_curve(pKa, list(run)) for run in self._A_frac],
            dtype=np.float64,
        )

        mean_pH = np.nanmean(pH_array, axis=0)
        std_pH = np.nanstd(pH_array, axis=0)

        if show_runs:
            for i, ph_run in enumerate(pH_array):
                ax.plot(
                    self.delta_eir,
                    ph_run,
                    color=colors[0],
                    alpha=0.4,
                    linewidth=0.8,
                    label=f"run {i + 1}" if i == 0 else None,
                    **plotArgs,
                )

        if show_band:
            ax.fill_between(
                self.delta_eir,
                mean_pH - std_pH,
                mean_pH + std_pH,
                color=colors[2],
                alpha=0.3,
                label="± 1 std",
            )

        if show_mean:
            ax.plot(
                self.delta_eir,
                mean_pH,
                color=colors[1],
                linewidth=1.5,
                label="mean",
                **plotArgs,
            )

        if show_pKa_line:
            ax.axhline(pKa, color="gray", linewidth=0.7, linestyle="--", label=f"pKa = {pKa}")

        ax.set_xlabel("ΔOffset (kJ/mol)")
        ax.set_ylabel("theoretical pH")
        ax.legend(fontsize=8)

        if standalone:
            plt.savefig("titration_curve.png", bbox_inches="tight")
            plt.close()

        gc.collect()
        return ax

    def plot_logfit(
        self,
        ax: plt.Axes | None = None,
        color: str = "#009E73",
        plotArgs: dict | None = None,
    ) -> tuple[plt.Axes, np.ndarray]:
        """Fit the mean A_frac profile to a logistic curve and plot.

        Args:
            ax: Matplotlib axes.  New figure saved to file if ``None``.
            color: Colour for the fitted curve.
            plotArgs: Additional kwargs for the plot call.

        Returns:
            Tuple ``(ax, popt)`` where *popt* contains the four fitted
            logistic parameters ``[a, b, c, d]`` (see
            :func:`~cpaeds.algorithms.logistic_curve`).
        """
        plotArgs = plotArgs or {}
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(1, 1, dpi=200)

        mean, _ = self._mean_std()
        popt = log_fit(self.delta_eir, mean)
        x_dense = np.linspace(self.delta_eir.min(), self.delta_eir.max(), 300)
        ax.plot(x_dense, logistic_curve(x_dense, *popt), color=color, **plotArgs)

        if standalone:
            plt.savefig("logfit.png", bbox_inches="tight")
            plt.close()

        gc.collect()
        return ax, popt
