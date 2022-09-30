import os
import subprocess
from cpaeds.utils import get_dir_list, load_config_yaml, check_job_running
from cpaeds.algorithms import natural_keys

def initialise():
    settings_loaded = load_config_yaml(
                config= './final_settings.yaml')
    dir_list=get_dir_list()
    dir_list.sort(key=natural_keys)
    path = os.getcwd()
    path_to_sh = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/run.sh'))
    for dir in dir_list:
        os.chdir(f"{path}/{dir}")
        files = [f for f in os.listdir(f"{path}/{dir}") if os.path.isfile(os.path.join(f"{path}/{dir}", f))]
        if check_job_running('bbraun', dir=os.getcwd()):
            pass
        else:
            run_list=[]
            omd_list=[]
            for file in files:
                #logic error to fix. Errors: omd 2 has shake error. Run is still added and omd position is at the file with shake failure.
                if file.endswith('.run'):
                    run_list.append(file)
                if file.endswith('.omd'):
                    with open(file, 'r') as omd:
                        for line in omd:
                            if line.startswith(f"MD++ finished successfully"):
                                omd_list.append(file)
                            #elif "ERROR Shake::solute : SHAKE error. vectors orthogonal" in line:
                                #stuff    
                            elif "ERROR" in line:
                                print(f"Error in file {dir}/{file}")
            if len(omd_list) == settings_loaded['simulation']['parameters']['NRUN']:
                pass
            else:
                run_list.sort(key=natural_keys)
                exe = subprocess.run(
                    ['bash', path_to_sh, str(run_list[len(omd_list)])],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                exe.check_returncode()
                