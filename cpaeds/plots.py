import seaborn as sns
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cpaeds.algorithms import ph_curve, log_fit, logistic_curve, inverse_log_curve
from scipy.stats import linregress
import os

class Plot():
    def __init__(self, basepath: str ="./", pKa: float = 4.81) -> None:
        """
        Takes the files "energies.npy" and "results.out" from a postprocessing-run and allows for plotting of the runs

        Args:
            datafile (_type_): _description_
        """
        # Read in the results and energies files
        self.pka = pKa
        path_energies = f"{basepath}/energies.npy" # Colums must be "vmix, v_r, e1, e2,..., en" for n states. Hierarchy: runs -> points -> values
        path_results = f"{basepath}/results.out"

        energy_arrays = np.load(path_energies)
        e_colnames = ["vmix", "vr"] + [f"e{n}" for n in range(1, len(energy_arrays[0])-1) ]
        self.energies = [pd.DataFrame(arr.T, columns=e_colnames) for arr in energy_arrays]
        self.results = pd.read_csv(path_results, index_col = 0)

        return
    
class StdPlot(Plot):
    def __init__(self, basepath: str ="./", pKa: float = 4.81) -> None:
        super().__init__(basepath, pKa)

        return
    
    def getOffset(self, states= -1) -> pd.DataFrame:
        """
        Returns a dataframe with the offsets of the given states

        Args:
            states (optinal): States to return the offset data from. Defaults to "-1"
        Returns:
            pd.DataFrame: _description_
        """
        df = self.results
        offsets = df.loc[:, df.columns.str.startswith('OFFSET')]
        return offsets.iloc[:, states]

    def getFractions(self, states = slice(None) ) -> pd.DataFrame:
        """
        Returns a dataframe with the fractions of the given states

        Args:
            states (str, optional): States to return the data from. Defaults to ":".

        Returns:
            pd.DataFrame: _description_
        """
        df = self.results
        fractions = df.loc[:, df.columns.str.startswith('FRACTION')]
        return fractions.iloc[:, states]
    
    def offset_fraction(self, ax=None, refstate: int = -1):
        """
        Generates a plot with the offset on the x-axis and the fraction of the states on the y-axis.
        If there is only two states, it will only plot the fraction of the first state (assuming that fraction(1) + fraction(2) = 1)

        Args:
        """
        x = self.getOffset(refstate)
        ys = self.getFractions()
        
        standalone = False
        if ax is None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        if len(ys.columns) > 2:
            for y in ys:
                ax.scatter(x, ys[y], label=y)        
        else:
            ax.scatter(x,ys['FRACTION1'].values, label='FRACTION1')

        ax.legend()
        ax.set_ylabel(f"Fraction of time")
        ax.set_xlabel(f"Offset [kJ/mol]")
        if standalone:
            ax.plot()
            plt.savefig("offset_fraction.png")
        gc.collect()

    def offset_pH(self, state: int = -1, offset_state: int = -1, ax = None, linfit_subset: list = [0,-1]):
        """
        Generates a plot with the offset on the x-axis and the computed pH (from the pKa) on the y-axis.
        If no state is given, then the last state is assumed to be the fully deprotonated state.
        Currently only works for molecules with a single deprotonated state.

        Args:
        state: int, index of state of the fully deprotonated form.
        linfit_subset: list, can be used to subset the datapoints used for the linear fit. Defaults to all datapoints (from 0 to -1)
        """

        x = self.getOffset(offset_state)
        fractions = self.getFractions(states=state)

        pH = ph_curve(self.pka, fractions)

        # Fitting to the data
        x_subset = x[linfit_subset[0]:linfit_subset[1]]
        pH_subset = pH[linfit_subset[0]:linfit_subset[1]]
        self.linfit = linregress(x_subset, pH_subset) 

        # Logfit
        self.logfit = log_fit(x, pH)

        standalone = False
        if ax is None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        plotpoints = np.linspace(x.min(), x.max())
        ax.plot(plotpoints, logistic_curve(plotpoints, *self.logfit))
        ax.plot(x, self.linfit.intercept + x * self.linfit.slope)
        ax.scatter(x, pH)
        ax.set_ylabel("theoretical pH")
        ax.set_xlabel("Offset [kJ/mol]")
        ax.axhline(self.pka, color="gray", linewidth=0.5)

        if standalone:
            ax.plot()
            plt.savefig("offset_pH.png")

        gc.collect() 

    def offset_pH_fraction(self, state: int = -1, offset_state: int = -1, ax = None, linfit_subset: list = [0,-1], fit = 'log'):
        """
        Generates a plot with the offset on the lower x-axis, the computed correspondign pH on the upper x-axis and the fraction of states on the y-axis.

        Args:
            state (int, optional): index of state of the fully deprotonated form. Defaults to -1.
            fit (str, optional): either 'log' for logistic fit or 'lin' for linear fit
        """
        x = self.getOffset(states=offset_state)
        fractions = self.getFractions(states=state)
        
        pH = ph_curve(self.pka, fractions)

        # Fitting to the data
        if fit == 'lin':
            x_subset = x[linfit_subset[0]:linfit_subset[1]]
            pH_subset = pH[linfit_subset[0]:linfit_subset[1]]
            self.linfit = linregress(x_subset, pH_subset) 
            offsetToPH = lambda x: self.linfit.intercept + self.linfit.slope * x
            pHToOffset = lambda x: (x - self.linfit.intercept) / self.linfit.slope
            secondary_label = "pH (linear fit)"
        elif fit == 'log':
            # Logfit
            self.logfit = log_fit(x, pH)
            offsetToPH = lambda x: logistic_curve(x, *self.logfit)
            pHToOffset = lambda x: inverse_log_curve(x, *self.logfit)
            secondary_label = "pH (logistic fit)"
        else:
            raise SyntaxError

        standalone = False
        if ax is None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        ax.scatter(x, fractions)
        ax.set_ylabel("fraction of time")
        ax.set_xlabel("Offset [kJ/mol]")
        secxax = ax.secondary_xaxis('top', functions=(offsetToPH, pHToOffset))
        secxax.set_xlabel(secondary_label)

        if standalone:
            ax.plot()
            plt.savefig("offset_pH_fraction.png")

        gc.collect() 

    def fit_residuals(self, state=-1, offset_state: int = -1, ax = None, linfit_subset: list = [0,-1], fit = 'log'):
        """
        Plots a residual plot for the selected fit

        Args:
            state (int, optional): Index of the reference state. Defaults to -1.
            ax (_type_, optional): matplotlib ax object. Defaults to None.
            linfit_subset (list, optional): Subset data for linear fit. Has no effect if the fit type is set to 'log'. Defaults to [0,-1].
            fit (str, optional): Type of fit, either 'log' or 'lin'. Defaults to 'log'.
        """
        x = self.getOffset(states=offset_state)
        fractions = self.getFractions(states=state)

        pH = ph_curve(self.pka, fractions)

        # Fitting to the data
        if fit == 'lin':
            x_subset = x[linfit_subset[0]:linfit_subset[1]]
            pH_subset = pH[linfit_subset[0]:linfit_subset[1]]
            self.linfit = linregress(x_subset, pH_subset)
            offsetToPH = lambda x: self.linfit.intercept + self.linfit.slope * x
            label = "Residuals linear fit"
        elif fit == 'log':
            # Logfit
            self.logfit = log_fit(x, pH)
            offsetToPH = lambda x: logistic_curve(x, *self.logfit)
            label = "Residuals logisitc fit"
        else:
            raise SyntaxError

        standalone = False
        if ax is None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        residuals = pH - offsetToPH(x)

        ax.scatter(x, residuals)
        ax.set_title(label)
        ax.set_xlabel('Offset [kJ/mol]')
        ax.set_ylabel('Residuals')

        ax.set_ylim(-1.1*max(abs(residuals)), 1.1*max(abs(residuals)))
        ax.axhline(0, color="gray", linewidth=0.5)

        if standalone:
            ax.plot()
            plt.savefig("residuals.png")

        gc.collect() 


    def kde_vmix(self, ax= None, refstate: int = -1, threshold: list = [0.15, 0.85] ):
        df = pd.DataFrame([runs['vmix'] for runs in self.energies])
        df = df.transpose()
        df.columns = [f"run {n}" for n in range(1, len(df.columns)+1)]

        fractions = self.getFractions(refstate)
        run_selection = [i > threshold[0] and i < threshold[1] for i in fractions]


        kdeplot = sns.kdeplot(df.iloc[:,run_selection], fill=False, ax=ax)
        fig = kdeplot.get_figure()

        if ax is None:
            fig.savefig("kde_vmix.png")

        gc.collect()

    def kde_e(self, axes: list = None, which: list = None):
        """
        Generates kde plots for different states.

        Args:
            axes (list, optional): List of axes to plot on. Default generates a new plot for each state.
            which (list, optional): e_1 corresponds to index 0! List of states to plot. Needs to be the same length as axes. Default plots all states.

        Raises:
            ValueError: axes and which need to be the same length.
        """
        if  axes is not None and which is not None:
            # Raise value error if length of which and axes is not the same and neither of them is None
            if not len(axes) == len(which):
                raise ValueError

        if which is None:
            dfs = self.data['energies']
            which = range(0, len(self.data['energies']))
        else:
            dfs = [self.data['energies'][n] for n in which]

        standalone = False
        if axes is None:
            axes = [None for n in dfs]
            standalone = True

        for n, df, ax in zip(which, dfs, axes):
            df = df.set_index(df.columns[0])
            kdeplot = sns.kdeplot(df, ax=ax)
            fig = kdeplot.get_figure()
            kdeplot.set(title=f"e_{n+1}")

            if standalone:
                fig.savefig(f"e_{n+1}.png")

        gc.collect()


    def kde_ees(self, which: list = None, ax = None):
        """
        Generates kernel density estimate plots for the given energies and plots them onto a single plot.

        Args:
            which (list, optional): Which energies to plot - 0 based list. Default plots all energies.
            ax (_type_, optional): Ax object onto which to plot the kde. Default generates a new figure and saved it to a file in the cwd.
        """

        dfs = self.data['energies']

        if which is None:
            which = range(0, len(dfs))

        dfs = [dfs[n].set_index(dfs[n].columns[0]) for n in which]

        standalone = False
        if ax is None:
            standalone = True

        for df in dfs:
            kdeplot = sns.kdeplot(df, ax=ax)
        kdeplot.set(title=f"e_{'_'.join([str(e+1) for e in which])}")

        if standalone:
            fig = kdeplot.get_figure()
            fig.savefig(f"e_{'_'.join([str(e+1) for e in which])}.png")
        
        gc.collect()
