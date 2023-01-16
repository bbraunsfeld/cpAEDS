import glob
import os
import subprocess
import yaml
from cpaeds.algorithms import natural_keys
from cpaeds.context_manager import set_directory
from cpaeds.utils import get_dir_list, load_config_yaml, check_job_running,dumb_full_config_yaml


def initialise():
    settings_loaded = load_config_yaml(
                config= './final_settings.yaml')
    dir_list=get_dir_list()
    dir_list.sort(key=natural_keys)
    path = os.getcwd()
    path_to_sh = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/run.sh'))
    status={}
    try: 
        status_loaded=load_config_yaml(
                config= './status.yaml')
        status.update(status_loaded)
    except IOError:
        print(f"status.yaml not found.")
    for i, dir in enumerate(dir_list):
        with set_directory(f"{path}/{dir}"):
            files = [f for f in os.listdir(f"{path}/{dir}") if os.path.isfile(os.path.join(f"{path}/{dir}", f))]
            if check_job_running(os.getlogin(), status, i,dir=os.getcwd()):
                pass
            else:
                run_list=[]
                omd_list=[]
                for file in files:
                    if file.endswith('.run'):
                        run_list.append(file)
                    if file.endswith('.omd'):
                        with open(file, 'r') as omd:
                            for line in omd:
                                if line.startswith(f"MD++ finished successfully"):
                                    omd_list.append(file)
                                elif "ERROR" in line:
                                    status[f"run_{i+1}"][5]="ERR"
                if len(omd_list) == settings_loaded['simulation']['parameters']['NRUN']:
                    status[f"run_{i+1}"][5]="FIN"
                """else:
                    run_list.sort(key=natural_keys)
                    exe = subprocess.run(
                        ['bash', path_to_sh, str(run_list[len(omd_list)])],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    exe.check_returncode()"""
    with open(f"status.yaml", 'w+') as file:
        yaml.dump(status, file)

                
def first_submit():
    path_to_sh = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/run.sh'))
    for runfile in glob.glob(f"./*/*_1.run"):
        exe = subprocess.run(
            ['bash', path_to_sh, runfile],
            check=True,
            capture_output=True,
            text=True,
        )
        exe.check_returncode()
        
    
