import os
import sys
import copy
import subprocess
from cpaeds.utils import (load_config_yaml,check_system_settings,check_system_dir,check_dirs,check_input_files,check_simulation_settings,
                        create_offsets,create_folders,get_dir_list,copy_lib_file,
                        write_file,create_ana_dir,dumb_full_config_yaml)
from cpaeds.file_factory import build_mk_script_file,build_job_file,build_imd_file
from cpaeds.algorithms import natural_keys

def create_prod_run(settings_loaded):
    create_offsets(settings_loaded)
    create_folders(settings_loaded)
    pdir_list = []
    for i in range(1, settings_loaded['simulation']['NSTATS'] + 1):
        pdir_list.append(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}_{i}")
    pdir_list.sort(key=natural_keys)
    random_seed = 1
    for pdir in pdir_list:
        os.chdir(f"{pdir}")
        dir_list = get_dir_list()
        eir_counter = 0
        for dir in dir_list:
            os.chdir(f"{pdir}/{dir}")
            if settings_loaded['system']['lib_type'] == f"cuda":
                copy_lib_file(os.getcwd(),'mk_script_cuda_8_slurm.lib')
            elif settings_loaded['system']['lib_type'] == f"cuda_local":
                  copy_lib_file(os.getcwd(),'mk_script_cuda_8.lib')
            mk_script_body =  build_mk_script_file(settings_loaded,os.getcwd())
            write_file(mk_script_body,'aeds_mk_script.arg')
            job_file_body = build_job_file(settings_loaded)
            write_file(job_file_body,'aeds.job')
            EIR = settings_loaded['simulation']['parameters']['EIR_list'][eir_counter]
            imd_file_body = build_imd_file(settings_loaded,EIR,random_seed) 
            write_file(imd_file_body,'aeds.imd')
            create_ana_dir(settings_loaded)

            if os.path.exists(f"{pdir}/{dir}/aeds_{settings_loaded['system']['name']}_1.imd"):
                print("imd and run files exist")
                pass
            else:
                print('Running mk_script...')
                exe = subprocess.run(
                    [settings_loaded['system']['path_to_mk_script'], '@f', 'aeds_mk_script.arg'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                exe.check_returncode()
                print('Finished running mk_script.')   
            eir_counter += 1
        settings_updated = copy.deepcopy(settings_loaded)
        settings_updated['system']['output_dir_name']=f"{settings_updated['system']['output_dir_name']}_{random_seed}"
        dumb_full_config_yaml(settings_updated)
        random_seed += 1
    """
    for dir in dir_list:
        start_prod_run(settings_loaded,dir)
    """

if __name__ == "__main__":
        settings = load_config_yaml(
                config= sys.argv[1])
        
        check_system_settings(settings)
        check_system_dir(settings)
        check_dirs(settings)
        check_input_files(settings)
        check_simulation_settings(settings)
        create_prod_run(settings)
 