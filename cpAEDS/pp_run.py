from contextlib import redirect_stderr
import os
import sys
import subprocess
from cpAEDS.utils import load_config_yaml,get_dir_list,write_file,read_df,plot_offset_ratio,plot_offst_dG,check_finished,write_file2,read_output
from cpAEDS.file_factory import build_dfmult_file,build_ene_ana,build_output
from cpAEDS.AEDS_sampling import read_energyfile, calculate_statesampled,calc_prob_sampling,write_prob_sampling,calc_sampling,write_sampling
from cpAEDS.algorithms import pKa_from_df
#'C:/Users/Bene/Documents/PhD/scripts/const_pH/tests/test_data/input_parameter.yaml')

G_efile_template = "e%ss.dat"
G_emin_outfile = "statetser.dat"
G_prob_outfile = "prob_statetser.dat"

def sampling_main(efile_template, emin_outfile, prob_outfile, numstates, temp, itime, step):
    energys = []
    for i in range(numstates):
        e, t = read_energyfile(efile_template % str(i+1))
        energys.append(e)
    emin_state, prob_state = calculate_statesampled(energys, temp)
    print("#########################################")
    print("# Estimation based on probabilities")
    calc_prob_sampling(numstates, prob_state, step)
    write_prob_sampling(prob_outfile, prob_state, itime, step)
    print("\n\n#########################################")
    print("# Estimation based on minimum energy")
    fraction_list = calc_sampling(emin_state, numstates, step) 
    write_sampling(emin_outfile, emin_state, itime, step)
    return fraction_list

if __name__ == "__main__":
        settings_loaded = load_config_yaml(
                config= './final_settings.yaml')

        dir_list = get_dir_list()
        fraction_list = []
        pka_dG_list = [] 
        pka_offset_list = []
        dG_list = []
        for dir in dir_list:
                if dir == 'results':
                        continue
                else:
                        os.chdir(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/{dir}")
                        run_finished, NOMD = check_finished(settings_loaded)
                        
                        os.chdir(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/{dir}/ene_ana")
                        df_file_body = build_dfmult_file(settings_loaded)
                        write_file(df_file_body,'df.arg')
                        if run_finished == True:
                                print('Running ene_ana for finished run...')
                                exe = subprocess.run(
                                ['ene_ana', '@f', 'ene_ana.arg'],
                                check=True,
                                capture_output=True,
                                text=True
                                )
                                exe.check_returncode()
                        else:
                                print('Running ene_ana for unfinished run...')
                                ene_ana_body =  build_ene_ana(settings_loaded,NOMD)
                                write_file2(ene_ana_body,'ene_ana_temp.arg')
                                print('Running ene_ana for unfinished run...')
                                exe = subprocess.run(
                                ['ene_ana', '@f', 'ene_ana_temp.arg'],
                                check=True,
                                capture_output=True,
                                text=True
                                )
                                exe.check_returncode()
                        print('Finished running ene_ana.') 

                        fraction_list.append(sampling_main(G_efile_template, G_emin_outfile, G_prob_outfile,
                                                                2, int(settings_loaded['simulation']['parameters']['temp']), 
                                                                0, settings_loaded['simulation']['parameters']['dt']*settings_loaded['simulation']['parameters']['NTWE']))

                        print('Running dfmult...')
                        with open('df.out', 'w') as sp:
                                exe = subprocess.run(
                                ['dfmult', '@f', 'df.arg'], 
                                stdout=sp)

                        dG_list.append(read_df('./df.out'))
                        pka_dG_list.append(pKa_from_df(read_df('./df.out'),int(settings_loaded['simulation']['parameters']['temp'])))
        for offset in  settings_loaded['simulation']['parameters']['EIR_list']:    
                pka_offset_list.append(pKa_from_df(offset,int(settings_loaded['simulation']['parameters']['temp'])))
        os.chdir(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/")
        try:
                os.mkdir('results')
        except FileExistsError:
                pass
        parent = os.getcwd()
        os.chdir(f"{parent}/results")
        output_body = build_output(settings_loaded,fraction_list,dG_list,pka_dG_list,pka_offset_list)
        write_file2(output_body,'results.out')
        fraction_state1,offset = read_output('./results.out')
        plot_offset_ratio(offset,fraction_state1,5,settings_loaded)
        plot_offst_dG(offset,dG_list,1,settings_loaded)