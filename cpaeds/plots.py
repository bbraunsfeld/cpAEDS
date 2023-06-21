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

        self.ph_curve_deprot = lambda pKa, f: ph_curve(pKa, [1-i for i in f])

        return
    
    def getOffset(self, states= -1) -> pd.DataFrame:
        """
        Returns a dataframe with the offsets of the given states

        Args:
            states (optional): States to return the offset data from. Defaults to "-1"
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
    
    def offset_fraction(self, ax=None, refstate: int = -1, plotArgs:dict = {}, colors:list = None):
        """
        Generates a plot with the offset on the x-axis and the fraction of the states on the y-axis.
        If there is only two states, it will only plot the fraction of the first state (assuming that fraction(1) + fraction(2) = 1)

        Args:
            ax (matplotlib subplots ax, optional): Ax to plot on. If left empty, a standalone plot will be created
            refstate (int, optional): The state to get the offset values from
            plotArgs (dictionary, optional): Additional arguments to pass along to plt.plot (e.g. marker, color,...)
            colors (list, optional): A list of colors to be used for the states. If there is only two fractions, the length should be 1, otherwise it should be the same length as the number of fractions.
        """
        x = self.getOffset(refstate)
        ys = self.getFractions()
        
        standalone = False
        if ax is None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        if len(ys.columns) > 2:
            if colors is None:
                for y in ys:
                    ax.scatter(x, ys[y], label=y, **plotArgs)        
            else:
                for c, y in zip(colors, ys):
                    ax.scatter(x, ys[y], label=y, c=c, **plotArgs)        
        else:
            if colors is None:
                ax.scatter(x,ys['FRACTION1'].values, label='FRACTION1', **plotArgs)
            else:
                ax.scatter(x,ys['FRACTION1'].values, label='FRACTION1', c=colors[0], **plotArgs)

        ax.legend()
        ax.set_ylabel(f"Fraction of time")
        ax.set_xlabel(f"Offset [kJ/mol]")
        if standalone:
            ax.plot()
            plt.savefig("offset_fraction.png")
        gc.collect()

    def offset_pH(self, state: int = -1, offset_state: int = -1, ax = None, linfit_subset: list = [0,-1], ref_is_protonated: bool = False, plotArgs: dict = {}, colors: list = None, plotArgsTrend: dict = {}):
        """
        Generates a plot with the offset on the x-axis and the computed pH (from the pKa) on the y-axis.
        If no state is given, then the last state is assumed to be reference state. By default, the reference state is the fully deprotonated one.

        Args:
        state: int, index of state of the reference state for the calculation of the theoretical pH. By default this is the fully deprotonated form. Defaults to -1
        offset_state (int, optional): state to take the offset from. Defaults to -1
        linfit_subset: list, can be used to subset the datapoints used for the linear fit. Defaults to all datapoints (from 0 to -1)
        ref_is_protonated (bool, optional): indicates wether the offset state is the protonated or deprotonated state. Defaults to False (state is deprotonated)
        plotArgs (dictionary, optional): Additional arguments to pass along to the plotting of the points (e.g. marker)
        plotArgsTrend (dictionary, optional): Additional arguments to pass along to the plotting of the trend lines (e.g. marker)
        colors (list, optional): list of colors for points, LogFit, linearFit
        """

        x = self.getOffset(offset_state)
        fractions = self.getFractions(states=state)

        pH = self.ph_curve_deprot(self.pka, (1 - fractions) if ref_is_protonated else fractions )

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
        ax.plot(plotpoints, logistic_curve(plotpoints, *self.logfit), **({} if colors is None else {'c': colors[1]}), **plotArgsTrend)
        ax.plot(x, self.linfit.intercept + x * self.linfit.slope, **({} if colors is None else {'c': colors[2]}), **plotArgsTrend)
        ax.scatter(x, pH, **({} if colors is None else {'c': colors[0]}), **plotArgs)
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
        
        pH = self.ph_curve_deprot(self.pka, fractions)

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

        pH = self.ph_curve_deprot(self.pka, fractions)

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


    def kde_vmix(self, ax= None, refstate: int = -1, threshold: list = [0.15, 0.85], subsample: float = 0.01 ):
        df = pd.DataFrame([runs['vmix'] for runs in self.energies])
        df = df.transpose()
        df.columns = [f"run {n}" for n in range(1, len(df.columns)+1)]

        fractions = self.getFractions(refstate)
        run_selection = [i > threshold[0] and i < threshold[1] for i in fractions]


        kdeplot = sns.kdeplot(df.iloc[:,run_selection].sample(frac=subsample), fill=False, ax=ax)
        fig = kdeplot.get_figure()
        kdeplot.set(title="kde_vmix")

        if ax is None:
            fig.savefig("kde_vmix.png")

        gc.collect()


    def kde_e(self, axes: list = None, states: list = None, threshold: list = [0.15, 0.85], refstate: int = -1, subsample: float=0.01):
        """
        Generates kde plots for different states.

        Args:
            axes (list, optional): List of axes to plot on. Default generates a new plot for each state.
            states (list, optional): e_1 corresponds to index 0! List of states to plot. Needs to be the same length as axes. Default plots all states.

        Raises:
            ValueError: axes and states need to be the same length.
        """
        
        # Get the indices of the interesting runs (those where the fraction of the refstate is between the set threshold values)
        fractions = self.getFractions(refstate)
        run_selection = [i > threshold[0] and i < threshold[1] for i in fractions]

        # dfs is a list containing dataframes of each run with the states in the columns
        dfs = [run.loc[:, run.columns.str.startswith('e')] for run in self.energies]        

        # dfs_runs is a list containing dataframes where each dataframe corresponds to a state and the columns are the run numbers.
        dfs_runs = list()
        for state in range(len(dfs[0].columns)):
            dfs_runs.append(pd.DataFrame([df.iloc[:,state] for df in dfs]).transpose())
            dfs_runs[-1].columns = [f"e{state+1}_run{n}" for n in range(1, len(dfs)+1)]
            dfs_runs[-1].reset_index(drop=True, inplace=True)

        # Input validation and set-up of plotting type (standalone or as axes object)
        if  axes is not None and states is not None:
            # Raise value error if length of states and axes is not the same and neither of them is None
            if not len(axes) == len(states):
                raise ValueError

        if states is None:
            states = range(0, len(dfs_runs))

        standalone = False
        if axes is None:
            axes = [None for n in dfs_runs]
            standalone = True

        # plotting
        for state, ax in zip(states, axes):
            kdeplot = sns.kdeplot(dfs_runs[state].iloc[:,run_selection].sample(frac=subsample), ax=ax)
            fig = kdeplot.get_figure()
            kdeplot.set(title=f"e_{range(0,len(dfs_runs))[state]+1}")

            if standalone:
                fig.savefig(f"e_{range(0,len(dfs_runs))[state]+1}.png")
                fig.close()

        gc.collect()


    def kde_ees(self, states: list = None, ax = None, threshold: list = [0.15, 0.85], refstate: int = -1, subsample: float=0.01):
        """
        Generates kernel density estimate plots for the given states and plots them onto a single plot, concatenating the given energy states

        Args:
            states (list, optional): Which energies to plot - 0 based list. Default plots all energies.
            ax (_type_, optional): Ax object onto which to plot the kde. Default generates a new figure and saved it to a file in the cwd.
        """
        # Get the indices of the interesting runs (those where the fraction of the refstate is between the set threshold values)
        fractions = self.getFractions(refstate)
        run_selection = [i > threshold[0] and i < threshold[1] for i in fractions]

       # dfs is a list containing dataframes of each run with the states in the columns
        dfs = [run.loc[:, run.columns.str.startswith('e')] for run in self.energies]

        if states is None:
            states = range(0, len(dfs[0].columns))

        # dfs_combined_e is a DataFrame containing the concatenated energies of the selected states separated by the run number.
        df_combined_e = pd.DataFrame()
        for run, df in enumerate(dfs):
            comb_e = df.iloc[:,states].values
            comb_e = comb_e.reshape(comb_e.size)
            colname = f'run{run+1}_e{[s + 1 for s in states]}'
            df_combined_e[colname] = comb_e

        standalone = False
        if ax is None:
            standalone = True

        # plotting
        kdeplot = sns.kdeplot(df_combined_e.iloc[:,run_selection].sample(frac=subsample), ax=ax)
        fig = kdeplot.get_figure()
        kdeplot.set(title=f"e_{[s + 1 for s in states]}")

        if standalone:
            fig.savefig(f"e_{[s + 1 for s in states]}.png")

        gc.collect()

        """
        This method is not yet implemented, but on the To-Do list.
def kde_ridge_plot(df,x,y="runs"):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    df_t1 = pd.read_csv(df)
    df_t_melt,hue = merge_columns(df_t1,['runs', 'states'])
    pal = sns.cubehelix_palette(len(df_t_melt[y].unique()), start=1.4, rot=-.25, light=.7, dark=.4)
    g = sns.FacetGrid(df_t_melt, row=y, hue=y, aspect=20, height=.5, palette=pal)
    g.map(sns.kdeplot, x, bw_adjust=.6, cut=5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, x, bw_adjust=.6, cut=5, clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .1, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, y)
    g.fig.subplots_adjust(hspace=-.7)
    g.set(yticks=[], xlabel=x, ylabel="", xlim=(None, 0), title="")
    g.despine(bottom=True, left=True)
    plt.savefig(f'./kde_states_{x}.png') 
    plt.close('all')
    gc.collect()
        """