import numpy as np
import glob
import argparse
import statsmodels.api as sm
#from statsmodels.graphics import tsaplots
#import matplotlib.pyplot as plt
#------------------------------
# Script written by Oriol Gracia Carmona 
# email: oriol.gracar@gmail.com 
# Adapted and configured for constant pH calculations with AEDS
#------------------------------
# Calculates total sampling based on minimum energy and probability of being sampled
# Some parameters can be given by argparser but the ones regarding file names have to be changed manually if desired
# Work in progress
#------------------------------

# Internal variables ----------

G_efile_template = "e%ss.dat"
G_emin_outfile = "statetser.dat"
G_prob_outfile = "prob_statetser.dat"

#------------------------------


# Function definitions-----------------

def read_energyfile(efile):
    etraj = []
    ttraj = []
    with open(efile, "r") as inn:
        for line in inn:
            if line.startswith("#"):
                continue
            fields = line.split()
            etraj.append(float(fields[1]))
            ttraj.append(float(fields[0]))
    return etraj, ttraj

def calculate_probabilities(energys, temp, prob_state):
    # Straight away calculation
    beta = 0.008314463 * temp
    exponential_term = [np.e**(-x/beta) for x in energys]
    total_e = sum(exponential_term)
    for i,e in enumerate(exponential_term):
        prob_state[i+1].append(e/total_e)

def calculate_statesampled(energys_list, temp):
    emin_state = []
    prob_state = {x+1:[] for x in range(len(energys_list))} 
    for step in zip(*energys_list):
        lowest_e_state = step.index(min(step)) + 1
        emin_state.append(lowest_e_state)
        calculate_probabilities(step, temp, prob_state)
    return emin_state, prob_state

def calc_prob_sampling(num_states, prob_state, time_step):

    # calculate total sampling times
    print("\n# Total Sampling times")
    total_sampling = 0
    for i in range(num_states):
        sn = i+1
        s_steps = np.sum(prob_state[sn])
        total_sampling += s_steps
        print("State %s:\t%s Steps\t%s ps" % (sn, round(s_steps,2), round(s_steps * time_step,2)))
    print("\nTotal sampling: %s steps %s ps" % (round(total_sampling,0),round(total_sampling * time_step,0)))

    # calculate avg lifetimes using autocorrelation function and seeing at  with lag decays to 0
    print("\n# Endstates avg lifetimes")
    for i in range(num_states):
        sn = i+1
        autocorr, confidence = sm.tsa.acf(prob_state[sn], nlags=10000, alpha=0.05)
        lifetime = "NA"
        n = 0
        while(lifetime == "NA" and n < len(autocorr)):
            if (confidence[n][1] - autocorr[n]) > autocorr[n]:
                lifetime = n+1
            n += 1

        print("State %s lifetime: %s Steps %s ps" % (sn, lifetime, round(lifetime * time_step,2)))
            
        #fig = tsaplots.plot_acf(prob_state[sn], lags=1000)
        #plt.show()
        

def calc_sampling(emin_states, num_states, time_step):
    transitions = {x+1:{y+1:0 for y in range(num_states)} for x in range(num_states)}
    lifetimes = {x+1:[] for x in range(num_states)}
    num_transitions = 0
    current_state = emin_states[0]
    sampling_steps = 1
    for state in emin_states[1::]:
        if state == current_state:
            sampling_steps += 1
        else:
            lifetimes[current_state].append(sampling_steps)
            transitions[current_state][state] += 1
            num_transitions += 1
            current_state = state
            sampling_steps = 1
    # save the last state
    lifetimes[current_state].append(sampling_steps)

    # Format and print the data

    print("\n# Total Sampling times")
    total_sampling = 0
    sampling_time = []
    fraction_of_time = []
    for i in range(num_states):
        sn = i+1
        s_steps = np.sum(lifetimes[sn])
        print("State %s:\t%s Steps\t%s ps" % (sn, s_steps, round(s_steps * time_step,2)))
        sampling_time.append(round(s_steps * time_step,2)) 
        total_sampling += s_steps  
    for i in sampling_time:
        fraction_of_time.append(i/round(total_sampling * time_step,2))
    print("Total sampling time:\t%s Steps\t%s ps\n" % (total_sampling,  round(total_sampling * time_step,2)))


    print("\n# Endstates avg lifetimes")
    for i in range(num_states):
        sn = i+1
        s_steps = round(np.sum(lifetimes[sn])/len(lifetimes[sn]),2)
        print("State %s:\t%s Steps\t%s ps" % (sn, s_steps, round(s_steps * time_step,2)))

    print("\n# Transitions between states")
    header = ""
    for i in range(num_states):
        header += "\t%s" % str(i+1)
    print(header)
    for i in range(num_states):
        new_line = str(i+1)
        for j in range(num_states):
            new_line += "\t%s" % (transitions[i+1][j+1])
        print(new_line)
    print("\nNumber of transitions: %s" % num_transitions)
    return fraction_of_time

def write_sampling(emin_outfile, emin_state, itime, step):
    time = itime
    with open(emin_outfile,"w") as out:
        for element in emin_state:       
            out.write("%s\t%s\n" % (round(time,3), element))
            time += step
 
def write_prob_sampling(prob_outfile, prob_state, itime, step):
    time = itime
    e_holder = []
    for key in sorted(prob_state):
        e_holder.append(prob_state[key])        
    with open(prob_outfile,"w") as out:
        for element in zip(*e_holder):
            out_string = "%s" % str(round(time,3))
            for element2 in element:
                out_string += "\t%s" % str(round(element2,6))
            out_string += "\n"       
            out.write(out_string)
            time += step
 

