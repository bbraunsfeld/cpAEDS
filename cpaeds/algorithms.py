import re
import math
import numpy as np
from scipy.optimize import curve_fit

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def offset_steps(EIR_start,EIR_range,EIR_step_size,EIR_groups,cpAEDS_type):
    """
    Creates a list of offsets for different applications.
        cpAEDS_type = 1 -> search type application used with bigger EIR_step_size and has equal spacing between offset steps.
        cpAEDS_type = 2 -> production type application with smaller EIR_step_size close to EIR_start and bigger steps further away. 
        cpAEDS_type = 3 -> production type application with smaller EIR_step_size close to EIR_start and bigger steps further away.
                           Te offsets are averaged around 0.
        cpAEDS_type = 4 -> The offset of state is increased until the mid point and the the offset of the other state groups is decreased.
    """
    offsets=[[]] * len(EIR_start)
    if EIR_range == 0:
        offsets = [[i] for i in EIR_start]
    else:
        for i, group in enumerate(EIR_groups):
            if i == 0:
                pass
            else:
                for state in group:
                    EIR =  EIR_start[state]
                    if cpAEDS_type == 1:
                        offset_list = [*range(int(EIR-EIR_range/2),int(EIR+EIR_range/2+EIR_step_size),EIR_step_size)]
                        offset_list.sort()
                        offsets[state].append(offset_list)
                    elif cpAEDS_type == 2 or cpAEDS_type == 3 or  cpAEDS_type == 4:
                        offset_close_eq = [*range(int(EIR-8),int(EIR+8+EIR_step_size),EIR_step_size)]
                        offset_upper_limit = [*range(int(EIR-(EIR_range-8)/2-4*EIR_step_size),int(EIR-8),EIR_step_size*4)]
                        offset_lower_limit = [*range(int(EIR+8+4*EIR_step_size),int(EIR+EIR_range/2+EIR_step_size),EIR_step_size*4)]
                        offset_list = offset_close_eq + offset_upper_limit + offset_lower_limit
                        offset_list.sort()
                        offsets[state] = offset_list                    

        n_offsets = len(offsets[EIR_groups[-1][-1]])
        for i,lst in enumerate(offsets):
            if len(lst) == 0:
                offsets[i] = [EIR_start[i]] * n_offsets
    if cpAEDS_type == 3:
        offset_arr = np.array(offsets)    
        offset_cumsum = np.cumsum(offset_arr, axis = 0)
        for state in offsets:
            for i,eir in enumerate(state):
                state[i] = float(round(state[i] - offset_cumsum[-1][i]/len(EIR_start),2))
    if cpAEDS_type == 4:
        offset_arr = np.array(offsets)
        num_rows, num_cols = offset_arr.shape
        diff_arr = np.zeros([1,num_cols])
        for i,eir in enumerate(offset_arr[EIR_groups[-1][-1], :]):
            if eir < EIR_start[EIR_groups[-1][-1]]:
                diff_arr[0][i] = EIR_start[EIR_groups[-1][-1]] - eir
        updated_arr = offset_arr + diff_arr
        offsets = updated_arr.tolist()
    return offsets

def pKa_from_df(df,temp):
    if df == 'NaN':
        pKa = 'NaN'
        return pKa
    else:    
        k = 0.00831451
        Ka = math.exp(-(df/(k*temp)))
        pKa = -math.log10(Ka)
        return pKa


def ph_curve(pka,fraction):
    """
    The idea is to create an artifical titration curve comming from an experimantel pKa value and linking it to the offset curve.
    pka_exp = experimentel pka value
    pH_exp = values connected to experimental pka
    offsets = values used to create certain frations of time
    fraction = time spent in a state divided by the total time of sampling. Basically related to the protonation/deprotonation of a molecule
    """
    ph_list = []
    for i in fraction:
        if i == 1 or i == 0:
            ph_list.append(np.NaN)
        else:
            ph = pka - np.log10(i/(1-i))
            ph_list.append(ph)
    return ph_list

def logistic_curve(x, a, b, c, d):
    """
    Logistic function with parameters a, b, c, d
    a is the curve's maximum value (top asymptote)
    b is the curve's minimum value (bottom asymptote)
    c is the logistic growth rate or steepness of the curve
    d is the x value of the sigmoid's midpoint
    """
    return ((a-b) / (1 + np.exp(-c * (x - d)))) + b

def inverse_log_curve(y, a, b, c, d):
    """
    Computes the inverse of the logisitic function

    Args:
        x (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """

    return - (np.log((a-b)/(y-b) -1))/c + d

def log_fit(x, y):
    """
    Tries to fit a given dataset to the logisitc_curve function
    returns the optimized parameters of the fit.
    Does not work if there are NaNs in the x values.
    """

    # Data cleanup, because curve_fit does not play well with NaNs.
    x = np.array(x)
    y = np.array(y)
    nans = np.isnan(y)
    x = x[~nans]
    y = y[~nans]

    initial_guess = [np.max(y), np.min(y), 1, np.median(x)]
    popt, pcov = curve_fit(logistic_curve, x, y, p0=initial_guess, method='dogbox',maxfev=10000) 
    
    return popt
