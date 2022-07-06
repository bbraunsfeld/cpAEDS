import os
import subprocess
from cpAEDS.utils import get_dir_list
from cpAEDS.algorithms import natural_keys

if __name__ == "__main__":
    dir_list=get_dir_list()
    dir_list.sort(key=natural_keys)
    path = os.getcwd()
    path_to_sh = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/run.sh'))
    for dir in dir_list:
        os.chdir(f"{path}/{dir}")
        files = [f for f in os.listdir(f"{path}/{dir}") if os.path.isfile(os.path.join(f"{path}/{dir}", f))]
        run_list=[]
        for file in files:
            if file.endswith('.run'):
                run_list.append(file)
        run_list.sort(key=natural_keys)
        exe = subprocess.run(
            ['bash', path_to_sh, str(run_list[0])],
            check=True,
            capture_output=True,
            text=True,
        )
        exe.check_returncode()
