import os
import sys
import subprocess
from cpAEDS.utils import (load_config_yaml,check_settings,check_system_dir,check_dirs,check_input_files,
                        create_offsets,create_folders,get_dir_list,copy_lib_file,
                        write_file,create_ana_dir,dumb_full_config_yaml)
from cpAEDS.file_factory import build_mk_script_file,build_job_file,build_imd_file

def create_prod_run(settings_loaded):
    create_offsets(settings_loaded)
    create_folders(settings_loaded)
    dir_list = get_dir_list()
    eir_counter = 0
    for dir in dir_list:
        os.chdir(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/{dir}")
        copy_lib_file(os.getcwd(),'mk_script_cuda_8_slurm.lib')
        mk_script_body =  build_mk_script_file(settings_loaded,os.getcwd())
        write_file(mk_script_body,'aeds_mk_script.arg')
        job_file_body = build_job_file(settings_loaded)
        write_file(job_file_body,'aeds.job')
        EIR = settings_loaded['simulation']['parameters']['EIR_list'][eir_counter]
        imd_file_body = build_imd_file(settings_loaded,EIR) 
        write_file(imd_file_body,'aeds.imd')
        create_ana_dir(settings_loaded)

        if os.path.exists(f"{settings_loaded['system']['aeds_dir']}/{settings_loaded['system']['output_dir_name']}/{dir}/aeds_{settings_loaded['system']['name']}_1.imd"):
            print("imd and run files exist")
            pass
        else:
            print('Running mk_script...')
            exe = subprocess.run(
                ['mk_script', '@f', 'aeds_mk_script.arg'],
                check=True,
                capture_output=True,
                text=True
            )
            exe.check_returncode()
            print('Finished running mk_script.')   
        eir_counter += 1
    """
    for dir in dir_list:
        start_prod_run(settings_loaded,dir)
    """

#'C:/Users/Bene/Documents/PhD/scripts/const_pH/tests/test_data/input_parameter.yaml')

if __name__ == "__main__":
        settings = load_config_yaml(
                config= sys.argv[1])
        
        check_settings(settings)
        check_system_dir(settings)
        check_dirs(settings)
        check_input_files(settings)
        create_prod_run(settings)
        dumb_full_config_yaml(settings) 