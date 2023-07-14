import numpy as np
import glob
import statsmodels.api as sm
from natsort import natsorted

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

    # calculate avg lifetimes using autocorrelation function and seeing at with lag decays to 0
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
        
        if lifetime == "NA":
            print("State %s was not sampled" % (sn))
        else:
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

##################################################################
######## New sampling based on contribution to free energy #######
##################################################################

class sampling():
    def __init__(self, config, offsets, dfs):
        self.boltzmann = 0.00831441
        self.OFFSETS = offsets
        self.FREE = dfs
        self.config = config
        self.temp = self.config['simulation']['parameters']['temp']
        self.REFERENCE = "eds_vr.dat"
        self.VMIX = "eds_vmix.dat"
        self.BETA = 1.0/(self.temp * self.boltzmann)

    @staticmethod
    def read_energy_file(file_name):
        values_array = []
        with open(file_name) as in_file:
            #skip first line
            in_file.readline()
            for line in in_file:
                line = line.rstrip()
                value = float(line.split(" ")[-1]) 
                values_array.append(value)
        return np.array(values_array, dtype=np.float64)

    def main(self):
        #read files
        efiles = natsorted(glob.glob("e*[0-9].dat"))
        endstates_e = [self.read_energy_file(x) for x in efiles]
        endstates_totals = np.empty_like(endstates_e)
        reference = self.read_energy_file(self.REFERENCE)
        vmix = self.read_energy_file(self.VMIX)

        if len(endstates_e) != len(self.OFFSETS):
            raise ValueError("endstates_e and self.OFFSETS must have the same length")
        #compute the energies for each endstate
        for i, hi in enumerate(endstates_e):
            de = (hi - self.OFFSETS[i]) * self.BETA * -1.0
            endstates_totals[i] = np.exp(de)

        contributions = np.zeros(len(endstates_e))
        lowest_energy = np.zeros(len(endstates_e))
        # Calculates the number of frames where the endstate minus the reference energy is lower than the free energy plus kT (These are the frames that are contributing to the free energy)
        dG_diff = [np.sum(endstates_e[i] - reference < self.FREE[i] + (self.temp*self.boltzmann)) for i in range(len(endstates_e))]

        minstates = np.argmax(endstates_totals, axis=0) # Lists the state with the minimum energy in the given frame
        states, min_counts = np.unique(minstates, return_counts=True) # counts how often a state is has the minimum energy
        # Puts the results in the correct order in the correct array
        print(lowest_energy, states, min_counts)
        for n, s in enumerate(states):
            lowest_energy[s] = min_counts[n]

        final_e = sum(endstates_totals) # The final_e is the sum of the totals of all endstates, used for normalization
        frame_contribution = endstates_totals / final_e # states contributing to each frame
        contributions = np.sum(frame_contribution, axis=1) # sum over all contributing frames for each state

        # sum up some things for normalization
        tot_con_frames = sum(contributions)
        tot_en_frames = sum(lowest_energy)

        # Fractions of the states with the lowest energy averaged over the whole trajectory
        fractions = lowest_energy / tot_en_frames

        for i in range(len(efiles)):
            print("Endstates    %s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (i, round(contributions[i],2), round(contributions[i]*100/tot_con_frames,2), 
                                                                lowest_energy[i], round(fractions[i],2),
                                                                dG_diff[i]))
        print("")

        # Accumulative average of the contributing frames
        contrib_accum = np.cumsum(frame_contribution.T, axis=0) / np.cumsum(np.ones_like(frame_contribution.T), axis=0)

        energies = [vmix, reference] + endstates_e

        return fractions, energies, contrib_accum, frame_contribution.T