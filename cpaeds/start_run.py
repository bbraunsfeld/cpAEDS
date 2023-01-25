import glob
import os
import subprocess
import yaml
from cpaeds.algorithms import natural_keys
from cpaeds.context_manager import set_directory
from cpaeds.utils import get_dir_list, load_config_yaml
from cpaeds.ssh_connection import SSHConnection

def check_job_running(user,status,run,dir):
    """_summary_

    Args:
        user (String): Name of the user
        status (dict): Status of all runs 
        run (Int): Number of run 
        dir (String): Target directory

    Returns:
        dict: creates or updates status.yaml
    """
    with open('running_jobs.out', 'w+') as outfile:
        exe = subprocess.run(
                ['squeue', '-u', user],
                check=True,
                stdout=outfile,
                capture_output= False,
                text=True,
            )
    exe.check_returncode()
    with open('running_jobs.out', 'r') as paths:
        for line in paths:
            if f"{dir} " in line:
                line = line.split()
                status[f"run_{run+1}"] = line
                return True
    os.remove('running_jobs.out')  

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
                    if f"run_{i+1}" in status:
                        status[f"run_{i+1}"][5]="FIN"
                    else:
                         status[f"run_{i+1}"]="FIN"
                else:
                    print("reached submit")
                    run_list.sort(key=natural_keys)
                    exe = subprocess.run(
                        ['bash', path_to_sh, str(run_list[len(omd_list)])],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    exe.check_returncode()
    with open(f"status.yaml", 'w+') as file:
        yaml.dump(status, file)

def sumbit_jobs_ssh(ssh_connection=None):
    """
    Submits all _1.run jobs to the queue using a provided ssh connection. If no connection is given, it will be created upon runtime.

    connection: SSHConnection object

    returns: SSHConnection object
    """
    if ssh_connection == None:
        ssh_connection = SSHConnection()
    
    run_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/run.sh'))
    run_path = os.path.abspath(".")

    for runfile in glob.glob("./*/*_1.run"):
        stdout, stderr = ssh_connection.exec_command(command=f"bash {run_script} {runfile}", path=run_path)
        print(stdout, stderr)  

    return ssh_connection  