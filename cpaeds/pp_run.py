from contextlib import redirect_stderr
import os
import sys
import subprocess
from cpaeds.utils import (load_config_yaml,get_dir_list,write_file,read_energyfile,read_state_file,read_df,plot_offset_ratio,plot_offst_dG,
                        check_finished,write_file2,read_output,plot_offset_pH,plot_offset_pH_fraction,density_plot,
                        state_density_csv,kde_ridge_plot,read_rmsd)
from cpaeds.file_factory import build_dfmult_file,build_ene_ana,build_output,build_rmsd
from cpaeds.aeds_sampling import calculate_statesampled,calc_prob_sampling,write_prob_sampling,calc_sampling,write_sampling
from cpaeds.algorithms import pKa_from_df, natural_keys


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

def main():
        settings_loaded = load_config_yaml(
                config= './final_settings.yaml')

        pdir_list = []
        stat_run_path = f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}"[:-1]
        starting_number = int(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}"[-1:])
        if int(settings_loaded['simulation']['NSTATS']) > 1: 
                for i in range(starting_number, settings_loaded['simulation']['NSTATS'] + 1):
                        pdir_list.append(f"{stat_run_path}{i}")
                        pdir_list.sort(key=natural_keys)
        else:
                pdir_list.append(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}")
        equilibrate= settings_loaded['simulation']['equilibrate']
        for pdir in pdir_list:
                os.chdir(f"{pdir}")
                dir_list = get_dir_list()
                fraction_list = []
                pka_dG_list = [] 
                pka_offset_list = []
                dG_list = []
                rmsd_list = []
                density_map_e1 = []
                density_map_e2 = []
                density_map_emix = []
                density_map_state = []
                column_name = []
                for i, dir in enumerate(dir_list):
                        if dir == 'results':
                                continue
                        else:
                                os.chdir(f"{pdir}/{dir}")
                                run_finished, NOMD = check_finished(settings_loaded)
                                try:
                                        os.mkdir('rmsd')
                                except FileExistsError:
                                        pass
                                os.chdir(f"{pdir}/{dir}/rmsd")
                                if equilibrate[0] == True and run_finished == True:
                                        print('Running rmsd for finished run...')
                                        print(f'Discarding {equilibrate[1]}% for equilibration...')
                                        rmsd_body = build_rmsd(settings_loaded,settings_loaded['simulation']['parameters']['NRUN'])
                                        write_file2(rmsd_body,'rmsd_eq.arg')
                                        with open('rmsd.out', 'w') as sp:
                                                exe = subprocess.run(
                                                ['rmsd', '@f', 'rmsd_eq.arg'], 
                                                stdout=sp)
                                        exe.check_returncode()
                                        rmsd_list.append(read_rmsd('rmsd.out'))
                                elif run_finished == True:
                                        print('Running rmsd for finished run...')
                                        rmsd_body = build_rmsd(settings_loaded,settings_loaded['simulation']['parameters']['NRUN'])
                                        write_file(rmsd_body,'rmsd.arg')
                                        with open('rmsd.out', 'w') as sp:
                                                exe = subprocess.run(
                                                ['rmsd', '@f', 'rmsd.arg'], 
                                                stdout=sp)
                                        exe.check_returncode()
                                        rmsd_list.append(read_rmsd('rmsd.out'))
                                else:
                                        print('Running rmsd for unfinished run...')
                                        rmsd_body = build_rmsd(settings_loaded,NOMD)
                                        write_file2(rmsd_body,'rmsd_temp.arg')
                                        with open('rmsd.out', 'w') as sp:
                                                exe = subprocess.run(
                                                ['rmsd', '@f', 'rmsd_temp.arg'], 
                                                stdout=sp)
                                        exe.check_returncode()
                                        rmsd_list.append(read_rmsd('rmsd.out'))
                                os.chdir(f"{pdir}/{dir}/ene_ana")
                                df_file_body = build_dfmult_file(settings_loaded)
                                write_file(df_file_body,'df.arg')
                                if equilibrate[0] == True and run_finished == True:
                                        print('Running ene_ana for finished run...')
                                        print(f'Discarding {equilibrate[1]}% for equilibration...')
                                        ene_ana_body =  build_ene_ana(settings_loaded,settings_loaded['simulation']['parameters']['NRUN'])
                                        write_file2(ene_ana_body,'ene_ana_temp.arg')
                                        exe = subprocess.run(
                                        ['ene_ana', '@f', 'ene_ana_temp.arg'],
                                        check=True,
                                        capture_output=True,
                                        text=True
                                        )
                                        exe.check_returncode()
                                elif run_finished == True:
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
                                
                                if fraction_list[-1][0] > 0.15 and fraction_list[-1][0] < 0.85:
                                        e1,t1 = read_energyfile(f'./e1.dat')
                                        e2,t2 = read_energyfile(f'./e2.dat')
                                        emix,tmix = read_energyfile(f'./eds_vmix.dat')
                                        state,tstate = read_state_file(f'./statetser.dat')
                                        column_name.append(f'run{i+1}[{dG_list[-1]}]')
                                        density_map_e1.append([e1,t1])
                                        density_map_e2.append([e2,t2])
                                        density_map_emix.append([emix,tmix])
                                        density_map_state.append([state,tstate])
                                else:
                                        pass

                for offset in  settings_loaded['simulation']['parameters']['EIR_list']:    
                        pka_offset_list.append(pKa_from_df(offset,int(settings_loaded['simulation']['parameters']['temp'])))
                os.chdir(f"{pdir}")
                try:
                        os.mkdir('results')
                except FileExistsError:
                        pass
                os.chdir(f"{pdir}/results")
                output_body = build_output(settings_loaded,fraction_list,dG_list,rmsd_list)
                write_file2(output_body,'results.out')
                fraction_state1,offset = read_output('./results.out')
                plot_offset_ratio(offset,fraction_state1,5,settings_loaded)
                #plot_offst_dG(offset,dG_list,1,settings_loaded)
                plot_offset_pH(offset,fraction_state1,settings_loaded)
                plot_offset_pH_fraction(offset,fraction_state1,settings_loaded)
                if len(column_name) == 0:
                        pass
                else:
                        density_plot(density_map_e1,density_map_e2,density_map_emix,column_name)
                        state_density_csv(density_map_e1,density_map_e2,density_map_state,column_name)
                        kde_ridge_plot('e_state.csv','e1')
                        kde_ridge_plot('e_state.csv','e2')
if __name__ == "__main__":
        main()