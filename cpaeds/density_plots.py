import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logger import LoggerFactory

logger = LoggerFactory.get_logger("density_plots.py", log_level="DEBUG", file_name = "debug.log")

def read_energyfile(efile):
    etraj = []
    ttraj = []
    with open(efile, 'r') as inp:
        for line in inp:
            if line.startswith('#'):
                continue
            fields = line.split()
            etraj.append(float(fields[1]))
            ttraj.append(float(fields[0]))
    return etraj, ttraj
logger.info("Test message")
state = f'e2'
logger.info("setting e1")
logger.debug("Test message2")
erun1,trun1=read_energyfile(f'/Users/bene/Desktop/energy_files/{state}_s7_3.dat')
logger.error("no file")
erun2,trun2=read_energyfile(f'/Users/bene/Desktop/energy_files/{state}_s7_4.dat')
erun3,trun2=read_energyfile(f'/Users/bene/Desktop/energy_files/{state}_s7_5.dat')
erun4,trun2=read_energyfile(f'/Users/bene/Desktop/energy_files/{state}_s7_6.dat')
df = pd.DataFrame(list(zip(erun1, erun2,erun3,erun4)), index = trun1,
               columns =['run3', 'run4', 'run5','run6'])
print (df)
kde_plot=sns.kdeplot(data=df, shade=False)
fig = kde_plot.get_figure()
fig.savefig(f'/Users/bene/Desktop/kde_{state}.png') 
