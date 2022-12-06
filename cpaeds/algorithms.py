import re
import math
import numpy as np

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def offset_steps(EIR_start,EIR_range,EIR_step_size,cpAEDS_type):
    """
    Creates a list of offsets for different applications.
        cpAEDS_type = 1 -> search type application used with bigger EIR_step_size and has equal spacing between offset steps.
        cpAEDS_type = 2 -> production type application with smaller EIR_step_size close to EIR_start and bigger steps further away. 
    """
    if cpAEDS_type == 1:
        offset_list = [*range(int(EIR_start-EIR_range/2),int(EIR_start+EIR_range/2+EIR_step_size),EIR_step_size)]
        offset_list.sort()
    else:
        offset_close_eq = [*range(int(EIR_start-8),int(EIR_start+8+EIR_step_size),EIR_step_size)]
        offset_upper_limit = [*range(int(EIR_start-(EIR_range-8)/2-4*EIR_step_size),int(EIR_start-8),EIR_step_size*4)]
        offset_lower_limit = [*range(int(EIR_start+8+4*EIR_step_size),int(EIR_start+EIR_range/2+EIR_step_size),EIR_step_size*4)]
        offset_list = offset_close_eq + offset_upper_limit + offset_lower_limit
        offset_list.sort()
    return offset_list

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
        if i == 1:
            i = 0.9999999999
        elif i == 0:
            i = 0.00000001
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
