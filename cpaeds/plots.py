import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cpaeds.algorithms import ph_curve, log_fit, logistic_curve, inverse_log_curve
from scipy.stats import linregress

class Plot():
    def __init__(self, datafile) -> None:
        """
        Reads in the datafile which consists of a yaml file with the items "energies", "results", "vmix" and contains the corresponding filenames for the post-processing output files.
        Also reads in the pKa from this file.

        Args:
            datafile (_type_): _description_
        """
        with open(datafile, "r") as f:
            self.filenames = yaml.safe_load(f) 
    
        categories = ['energies', 'results', 'vmix']
        
        self.data = dict()
        for cat in categories:
            self.data[cat] = list()
            for f in self.filenames[cat]:
               self.data[cat].append(pd.read_csv(f))
    
        self.pka = self.filenames['pka']

        return
    
class StdPlot(Plot):
    def __init__(self, datafile) -> None:
        super().__init__(datafile)

        return
    
    def offset_fraction(self, ax=None):
        """
        Generates a plot with the offset on the x-axis and the fraction of the states on the y-axis.
        If there is only two states, it will only plot the fraction of the first state (assuming that fraction(1) + fraction(2) = 1)

        Args:
        """
        df = self.data['results'][0]
        x = df['OFFSET']
        ys = df.loc[:, df.columns.str.startswith('FRACTION')]

        standalone = False
        if ax == None:
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

    def offset_pH(self, state: int = -1, ax = None, linfit_subset: list = [0,-1]):
        """
        Generates a plot with the offset on the x-axis and the computed pH (from the pKa) on the y-axis.
        If no state is given, then the last state is assumed to be the fully deprotonated state.
        Currently only works for molecules with a single deprotonated state.

        Args:
        state: int, index of state of the fully deprotonated form.
        linfit_subset: list, can be used to subset the datapoints used for the linear fit. Defaults to all datapoints (from 0 to -1)
        """

        df = self.data['results'][0]
        x = df['OFFSET']
        fractions = df.loc[:, df.columns.str.startswith('FRACTION')].iloc[:,state]

        pH = ph_curve(self.pka, fractions)

        # Fitting to the data
        x_subset = x[linfit_subset[0]:linfit_subset[1]]
        pH_subset = pH[linfit_subset[0]:linfit_subset[1]]
        self.linfit = linregress(x_subset, pH_subset) 

        # Logfit
        self.logfit = log_fit(x, pH)

        standalone = False
        if ax == None:
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

    def offset_pH_fraction(self, state: int = -1, ax = None, linfit_subset: list = [0,-1], fit = 'log'):
        """
        Generates a plot with the offset on the lower x-axis, the computed correspondign pH on the upper x-axis and the fraction of states on the y-axis.

        Args:
            state (int, optional): index of state of the fully deprotonated form. Defaults to -1.
            fit (str, optional): either 'log' for logistic fit or 'lin' for linear fit
        """
        df = self.data['results'][0]
        offset = df['OFFSET']
        fractions = df.loc[:, df.columns.str.startswith('FRACTION')].iloc[:,state]
        x = offset
        
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
        if ax == None:
            standalone = True # Flag which indicates if the plot is passed to an ax object or not.
            fig, ax = plt.subplots(1,1, dpi=200)

        #plotpoints = np.linspace(x.min(), x.max())
        #ax.plot(plotpoints, logistic_curve(plotpoints, *self.logfit))
        #ax.plot(x, self.linfit.intercept + x * self.linfit.slope)
        ax.scatter(x, fractions)
        ax.set_ylabel("fraction of time")
        ax.set_xlabel("Offset [kJ/mol]")
        secxax = ax.secondary_xaxis('top', functions=(offsetToPH, pHToOffset))
        secxax.set_xlabel(secondary_label)

        if standalone:
            ax.plot()
            plt.savefig("offset_pH_fraction.png")

        gc.collect() 



    def kde_vmix(self):
        pass

    def kde_e(self, state):
        pass

    def kde_e_to_e(self, states):
        pass
