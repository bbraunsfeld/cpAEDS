
import os
from cpaeds.logger import LoggerFactory
from cpaeds.utils import (load_config_yaml,get_dir_list,get_file_list,write_file,read_energyfile,read_state_file,read_df,plot_offset_ratio,plot_offst_dG,
                        check_finished,write_file2,read_output,plot_offset_pH,plot_offset_pH_fraction,density_plot,state_density_csv,kde_ridge_plot)
from cpaeds.algorithms import natural_keys

logger = LoggerFactory.get_logger("system.py", log_level="INFO", file_name = "debug.log")

"""
This is just a quick hack around to create a continuous run. 
Same setup as a normal run but taking the cnf from the previous run. 
Also this script is meant for a search from offset 0 to i.e. -100.
"""

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
        for pdir in pdir_list:
                os.chdir(f"{pdir}")
                dir_list = get_dir_list()
                print(len(dir_list))
                if settings_loaded['simulation']['parameters']['EIR_list'][-1] == 0:
                    for i, dir in enumerate(dir_list):
                            if dir == 'results':
                                    continue
                            elif dir == f"{settings_loaded['system']['output_dir_name']}_{len(dir_list)}":
                                os.chdir(f"{pdir}/{dir}")
                                file_list = get_file_list('.run')
                                with open(file_list[-1], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                            replacement = replacement + line
                                    changes = f"cd ${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{len(dir_list)-1}" +  '\n'
                                    changes = changes +  f"sbatch --ntasks ${{SLURM_NTASKS}} --cpus-per-task ${{SLURM_CPUS_PER_TASK}} --gres=mps:$(scontrol show job ${{SLURM_JOB_ID}} | grep gres/mps | sed \"s/^.*gres\/mps=//\") --mem ${{SLURM_MEM_PER_NODE}} --partition ${{SLURM_JOB_PARTITION}} --time $(scontrol show job ${{SLURM_JOB_ID}} | grep TimeLimit | sed \"s/^.*TimeLimit=//\" | cut -d\" \" -f1) {file_list[0]}"
                                    replacement = replacement + changes
                                with open(file_list[-1],'w+') as file:
                                    file.write(replacement)
                            else:
                                os.chdir(f"{pdir}/{dir}")
                                file_list = get_file_list('.run')
                                with open(file_list[0], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                        if f"INPUTCRD=$" in line:
                                            line = line.strip()
                                            changes = line.replace(line, f"INPUTCRD=${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{i+2}/aeds_{settings_loaded['system']['name']}_{settings_loaded['simulation']['parameters']['NRUN']}.cnf")
                                            replacement = replacement + changes + '\n'
                                        else:
                                            replacement = replacement + line
                                with open(file_list[0],'w+') as file:
                                    file.write(replacement)
                                with open(file_list[-1], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                            replacement = replacement + line
                                    if i == 0:
                                        changes = ""
                                    else:
                                        changes = f"cd ${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{i}" +  '\n'
                                        changes = changes +  f"sbatch --ntasks ${{SLURM_NTASKS}} --cpus-per-task ${{SLURM_CPUS_PER_TASK}} --gres=mps:$(scontrol show job ${{SLURM_JOB_ID}} | grep gres/mps | sed \"s/^.*gres\/mps=//\") --mem ${{SLURM_MEM_PER_NODE}} --partition ${{SLURM_JOB_PARTITION}} --time $(scontrol show job ${{SLURM_JOB_ID}} | grep TimeLimit | sed \"s/^.*TimeLimit=//\" | cut -d\" \" -f1) {file_list[0]}"
                                    replacement = replacement + changes
                                with open(file_list[-1],'w+') as file:
                                    file.write(replacement)

                                    
                elif settings_loaded['simulation']['parameters']['EIR_list'][0] == 0:
                        for i, dir in enumerate(dir_list):
                            if dir == 'results':
                                    continue
                            elif dir == f"{settings_loaded['system']['output_dir_name']}_1":
                                os.chdir(f"{pdir}/{dir}")
                                file_list = get_file_list('.run')
                                with open(file_list[-1], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                            replacement = replacement + line
                                    changes = f"cd ${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{i+2}" +  '\n'
                                    changes = changes +  f"sbatch --ntasks ${{SLURM_NTASKS}} --cpus-per-task ${{SLURM_CPUS_PER_TASK}} --gres=mps:$(scontrol show job ${{SLURM_JOB_ID}} | grep gres/mps | sed \"s/^.*gres\/mps=//\") --mem ${{SLURM_MEM_PER_NODE}} --partition ${{SLURM_JOB_PARTITION}} --time $(scontrol show job ${{SLURM_JOB_ID}} | grep TimeLimit | sed \"s/^.*TimeLimit=//\" | cut -d\" \" -f1) {file_list[0]}"
                                    replacement = replacement + changes
                                with open(file_list[-1],'w+') as file:
                                    file.write(replacement)
                            else:
                                os.chdir(f"{pdir}/{dir}")
                                file_list = get_file_list('.run')
                                with open(file_list[0], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                        if f"INPUTCRD=$" in line:
                                            line = line.strip()
                                            changes = line.replace(line, f"INPUTCRD=${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{i}/aeds_{settings_loaded['system']['name']}_{settings_loaded['simulation']['parameters']['NRUN']}.cnf")
                                            replacement = replacement + changes + '\n'
                                        else:
                                            replacement = replacement + line
                                with open(file_list[0],'w+') as file:
                                    file.write(replacement)
                                with open(file_list[-1], 'r') as inp:
                                    replacement = ""
                                    for line in inp:
                                            replacement = replacement + line
                                    if i == len(dir_list)-1:
                                        changes = ""
                                    else:
                                        changes = f"cd ${{SIMULDIR}}/../{settings_loaded['system']['output_dir_name']}_{i+2}" +  '\n'
                                        changes = changes +  f"sbatch --ntasks ${{SLURM_NTASKS}} --cpus-per-task ${{SLURM_CPUS_PER_TASK}} --gres=mps:$(scontrol show job ${{SLURM_JOB_ID}} | grep gres/mps | sed \"s/^.*gres\/mps=//\") --mem ${{SLURM_MEM_PER_NODE}} --partition ${{SLURM_JOB_PARTITION}} --time $(scontrol show job ${{SLURM_JOB_ID}} | grep TimeLimit | sed \"s/^.*TimeLimit=//\" | cut -d\" \" -f1) {file_list[0]}"
                                    replacement = replacement + changes
                                with open(file_list[-1],'w+') as file:
                                    file.write(replacement)
                else:
                    logger.critical("Not started at zero")
