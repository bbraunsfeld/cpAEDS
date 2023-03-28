"""
This is an example of what a plotting script could look like. It will create a figure with all available plots from this package.
It is intended to be used from the results directory after running postprocessing.
"""

from cpaeds.plots import StdPlot
import matplotlib.pyplot as plt

cls = StdPlot(basepath="./", pKa=4.88)

fig, axes = plt.subplots(5,2, dpi=300, figsize=(12,20))

cls.offset_fraction(ax=axes[0][0])
cls.offset_pH(ax=axes[0][1], state=-1, linfit_subset=[8,15])
cls.offset_pH_fraction(ax=axes[1][0], fit='log')
cls.kde_vmix(ax=axes[1][1], threshold=[0.25, 0.75])
cls.kde_e(axes = [axes[2][0], axes[2][1]], states=[0,-1], threshold=[0.25, 0.75])
cls.kde_ees(ax=axes[3][0], states=[0], threshold=[0.25, 0.75])
cls.kde_ees(ax=axes[3][1], threshold=[0.25, 0.75])
cls.fit_residuals(ax = axes[4][0], fit="log")
cls.fit_residuals(ax= axes[4][1], fit="lin", linfit_subset=[8,15])

plt.tight_layout()
fig.savefig("plots.png")