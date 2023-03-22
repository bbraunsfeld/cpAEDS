import os
import shutil
import yaml
import sys
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.polynomial.polynomial import polyfit 
from numpy.polynomial import Polynomial
import seaborn as sns
import subprocess
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd 
from cpaeds.algorithms import natural_keys, offset_steps, ph_curve
from cpaeds.file_factory import build_ene_ana
from cpaeds.context_manager import set_directory
from cpaeds.logger import LoggerFactory

logger = LoggerFactory.get_logger("utils.py", log_level="INFO", file_name = "debug.log")

def load_config_yaml(config) -> dict:
    with open(f"{config}", "r") as stream:
        try:
            settings_loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return settings_loaded
    
def dumb_full_config_yaml(settings_loaded):
    with open(f"final_settings.yaml", 'w') as file:
        yaml.dump(settings_loaded, file)

def read_last_line(f) -> str:
    f.seek(-2, 2)              # Jump to the second last byte.
    while f.read(1) != b"\n":  # Until EOL is found ...
        f.seek(-2, 1)          # ... jump back, over the read byte plus one more.
    return f.read() 

def get_dir_list():
    current_path = os.getcwd()
    contents = os.listdir(current_path)
    dir_list = []
    for item in contents:
        if os.path.isdir(item):
            dir_list.append(item)
    dir_list.sort(key=natural_keys)
    return dir_list

def get_file_list(file_ext: str):
    current_path = os.getcwd()
    contents = os.listdir(current_path)
    file_list = []
    for file in contents:
    # check only text files
        if file.endswith(file_ext):
            file_list.append(file)
    file_list.sort(key=natural_keys)
    return file_list

### pp ###
def check_finished(settings_loaded):
    omd_list = []
    run_complete = False
    for file in os.listdir(os.getcwd()):
        if file.endswith('.omd'):
            omd_list.append(file)

    if len(omd_list) == settings_loaded['simulation']['parameters']['NRUN']:
        run_complete = True

    return run_complete, len(omd_list)

def copy_lib_file(destination,name):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/{name}"))
    if os.path.exists(f"{destination}/{name}"):
        pass
    else:
        shutil.copyfile(path, f"{destination}/{name}")

def write_file(input_string,name) -> str:
    check_file = Path(f"{os.getcwd()}/{name}")
    if os.path.exists(check_file):
        logger.info(f"{name} exists in {os.getcwd()}.")
        pass
    else:
        with open(check_file,'w+') as file:
            file.write(input_string)
        logger.info(f"{name} created in {os.getcwd()}.")

def write_file2(input_string,name) -> str:
    check_file = Path(f"{os.getcwd()}/{name}")
    with open(check_file,'w+') as file:
        file.write(input_string)
    logger.info(f"{name} created in {os.getcwd()}.")

def create_ana_dir(settings_loaded):
    try:
        os.mkdir('ene_ana')
    except FileExistsError:
        pass
    parent = os.getcwd()
    with set_directory(f"{parent}/ene_ana"):
        copy_lib_file(f"{parent}/ene_ana",'ene_ana.md++.lib')
        ene_ana_body =  build_ene_ana(settings_loaded,settings_loaded['simulation']['parameters']['NRUN'])
        write_file(ene_ana_body,'ene_ana.arg')

#### analysis ####
def read_df(file):
    df = 'NaN'
    with open(file, "r") as inn:
         for line in inn:
            if f"DF_2_1" in line:
                fields = line.split()
                df = float(fields[1])
    return df

def read_rmsd(file):
    one_before_last = None
    last_line = None
    with open(file, "r") as inn:
        for line in inn:
            one_before_last=last_line
            last_line = line
        fields = one_before_last.split()
        rmsd = float(fields[1])
    return rmsd

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

def read_state_file(file):
    state_traj = []
    ttraj = []
    with open(file, 'r') as inp:
        for line in inp:
            if line.startswith('#'):
                continue
            fields = line.split()
            state_traj.append(float(fields[1]))
            ttraj.append(float(fields[0]))
    return state_traj, ttraj

def read_output(file):
    fraction_list = []
    offset_list = []
    with open(file, "r") as inn:
        next(inn)
        for line in inn: 
            line_splitted = line.split(',')
            fraction_list.append(float(line_splitted[2]))
            offset_list.append(float(line_splitted[1]))
    return fraction_list,offset_list

def plot_offset_ratio(offsets,fractions,order,settings_loaded):
    x=np.array(offsets)
    y=np.array(fractions)
    #fitting
    fit = polyfit(x, y, order)
    fit_p = Polynomial(fit)
    
    plt.plot(x,y, 'o')
    plt.plot(x, fit_p(x))
    roots = (fit_p - 0.5).roots()
    print(roots) 
    roots_new = roots[np.isclose(roots.imag, 0)]
    print(roots_new) 
    upper = offsets[-1]
    lower = offsets[0]
    position_in_array=np.where(np.logical_and(roots_new>=lower, roots_new<=upper))
    print(position_in_array)
    if len(position_in_array[0]) >= 1:
        pKa = round(roots_new[position_in_array[0][-1]].real,2)
        plt.plot(pKa,0.5, 'o')
        plt.legend(["data", "polyfit",f"eq_offset = {pKa}"], loc ="upper right")
    else:
        plt.legend(["data", "polyfit"], loc ="upper right")
    plt.title(f"{settings_loaded['system']['name']} with emin: {settings_loaded['simulation']['parameters']['EMIN']} kJ/mol; emax: {settings_loaded['simulation']['parameters']['EMAX']} kJ/mol")
    plt.xlabel("offset [kJ/mol]")
    plt.ylabel("fraction of time")
    plt.savefig(f"offset_ratio_order_{order}.png")
    plt.close('all')
    gc.collect()

def plot_offst_dG(offsets,dG,order,settings_loaded):
    x=np.array(offsets)
    y=np.array(dG)
    #fitting
    fit = polyfit(x, y, order)
    fit_p = Polynomial(fit)
    
    plt.plot(x,y, 'o')
    plt.plot(x, fit_p(x))
    plt.title(f"{settings_loaded['system']['name']} with emin: {settings_loaded['simulation']['parameters']['EMIN']} kJ/mol; emax: {settings_loaded['simulation']['parameters']['EMAX']} kJ/mol")
    plt.xlabel("offset [kJ/mol]")
    plt.ylabel("dG [kJ/mol]")
    plt.legend(["data", "polyfit"], loc ="upper right")
    plt.savefig(f"offset_dG_order_{order}.png")
    plt.close('all')
    gc.collect()

def plot_offset_pH_fraction(offsets,fractions,settings_loaded):
    pka=4.89
    x=np.array(offsets)
    y=np.array(fractions)
    ph = ph_curve(pka,fractions)
    model = linear_regression(x,ph)
    #fit = polyfit(x, y, 5)
    #fit_p = Polynomial(fit)

    fig, ax1 = plt.subplots()

    #roots = (fit_p - 0.5).roots()
    #roots_new = roots[np.isclose(roots.imag, 0)]
    #upper = offsets[-1]
    #lower = offsets[0]
    #position_in_array=np.where(np.logical_and(roots_new>=lower, roots_new<=upper))
    #if len(position_in_array[0]) >= 1:
        #pKa = round(roots_new[position_in_array[0][-1]].real,2)
        #ax1.plot(pKa,0.5, 'D',color='#4daf4a',zorder=100)
    
    color = '#377eb8'
    ax1.set_xlabel('offset [kJ/mol]', color=color)
    ax1.set_ylabel('fraction of time')
    ax1.plot(x, y, 'o', color=color,zorder=0)
    ax1.tick_params(axis='x', labelcolor=color)

    ax2 = ax1.twiny()  
    color = '#ff7f00'
    ax2.set_xlabel('pH', color=color)
    plt.axvline(x=pka, color='#ff7f00') 
    #plt.axhline(y=0.5,color='r')
    #ax2.plot(pka,0.5, 'D',color='#4daf4a',zorder=100) 
    ax2.plot(model, y, color=color, linestyle='None',zorder=0)
    ax2.tick_params(axis='x', labelcolor=color)
    fig.tight_layout() 
    red_patch = mpatches.Patch(color='#377eb8', label='AEDS-data')
    #green_patch = mpatches.Patch(color='#4daf4a', label='eq_offset')
    blue_patch = mpatches.Patch(color='#ff7f00', label='exp_pKa')
    plt.legend(handles=[red_patch,blue_patch]) #green_patch,blue_patch])
    plt.title(f"{settings_loaded['system']['name']} with emin: {settings_loaded['simulation']['parameters']['EMIN']}; emax: {settings_loaded['simulation']['parameters']['EMAX']}") 
    plt.savefig(f"offset_pH_fraction.png",bbox_inches='tight')
    plt.close('all')
    gc.collect()

def linear_regression(x,y):
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    def lin_func(x):
        return slope * x + intercept
    model = list(map(lin_func, x))
    return model

def plot_offset_pH(offsets,fractions,settings_loaded):
    pka=4.89
    x=np.array(offsets)
    y=np.array(fractions)
    ph = ph_curve(pka,fractions)
    model = linear_regression(x,ph)
    fig, ax1 = plt.subplots()
    
    color1 = '#377eb8'
    ax1.set_xlabel('offset [kJ/mol]', color=color1)
    color2 = '#ff7f00'
    ax1.set_ylabel('pH', color=color2)
    ax1.scatter(x, ph)
    ax1.plot(x, model)
    ax1.tick_params(axis='x', labelcolor=color1)
    ax1.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout() 
    plt.title(f"{settings_loaded['system']['name']} with emin: {settings_loaded['simulation']['parameters']['EMIN']}; emax: {settings_loaded['simulation']['parameters']['EMAX']}") 
    plt.savefig(f"offset_pH_reg.png",bbox_inches='tight')
    plt.close('all')
    gc.collect()

def density_map_to_df(density_map,column_name):
    energies = []
    time_steps = []
    for lst in density_map:
        energies.append(lst[0])
        time_steps.append(lst[1])  
    df = pd.DataFrame(list(zip(*energies)), index = time_steps[0],
               columns = column_name) 
    return df

def density_map_to_concat_df(density_map,column_name):
    energies = []
    time_steps = []
    for lst in density_map:
        energies.extend(lst[0])
        time_steps.extend(lst[1])  
    df = pd.DataFrame(energies, index = time_steps,
               columns = column_name) 
    df.to_csv(f'./{column_name[0]}.csv') 

def density_plot(density_map_e1,density_map_e2,density_map_emix,column_name):
    df1=density_map_to_df(density_map_e1,column_name)
    kde_plot=sns.kdeplot(data=df1, shade=False)
    fig = kde_plot.get_figure()
    fig.savefig(f'kde_e1.png') 
    df1.to_csv(f'./e1.csv')  
    plt.close('all')
    gc.collect()

    df2=density_map_to_df(density_map_e2,column_name)
    kde_plot=sns.kdeplot(data=df2, shade=False)
    fig = kde_plot.get_figure()
    fig.savefig(f'kde_e2.png') 
    df2.to_csv(f'./e2.csv')  
    plt.close('all')
    gc.collect()

    df3=density_map_to_df(density_map_emix,column_name)
    kde_plot=sns.kdeplot(data=df3, shade=False)
    fig = kde_plot.get_figure()
    fig.savefig(f'kde_vmix.png')  
    df3.to_csv(f'./vmix.csv') 
    plt.close('all')
    gc.collect()

    ax=sns.kdeplot(data=df1, shade=False)
    ax=sns.kdeplot(data=df2, shade=False)
    # Get the two lines from the axes to generate shading
    l1 = ax.lines[0]
    l2 = ax.lines[1]

    # Get the xy data from the lines so that we can shade
    x1, y1 = l1.get_xydata().T
    x2, y2 = l2.get_xydata().T

    xmin = max(x1.min(), x2.min())
    xmax = min(x1.max(), x2.max())
    x = np.linspace(xmin, xmax, 100)
    y1 = np.interp(x, x1, y1)
    y2 = np.interp(x, x2, y2)
    #y = np.minimum(y1, y2)
    #ax.fill_between(x, y, color="red", alpha=0.3)
    fig = ax.get_figure()
    fig.savefig(f'kde_e1_e2.png') 
    plt.close('all')
    gc.collect()

def merge_columns(df,flds):
    df['_'.join(flds)] =  pd.Series(df.reindex(flds, axis='columns')
                                     .astype('str')
                                     .values.tolist()
                                  ).str.join('_')
    hue = '_'.join(flds)
    return df, hue

def state_density_csv(density_map_e1,density_map_e2,density_map_state,column_name):    
    df_columns = ['e1','e2','runs','states']
    energies_e1 = []
    time_steps = []
    energies_e2 = []
    states = []
    run_lst = []
    lst_count = 0
    for lst in density_map_e1:
        for i in lst[0]:
            energies_e1.append(i)
        for t in lst[1]:
            time_steps.append(t)
        for i in range(len(lst[0])):
            run_lst.append(column_name[lst_count])  
        lst_count += 1
    for lst in density_map_e2:
        for i in lst[0]:
            energies_e2.append(i)
    for lst in density_map_state:
        for i in lst[0]:
            states.append(i) 
    data_list = [energies_e1,energies_e2,run_lst,states]
    df = pd.DataFrame(data=data_list, index = df_columns)
    df_t= df.T
    df_t.to_csv(f'./e_state.csv') 

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