import os
from tqdm import tqdm
import subprocess
import numpy as np
##cpaeds 
from cpaeds.algorithms import natural_keys
from cpaeds.context_manager import set_directory
from cpaeds.logger import LoggerFactory
from cpaeds.file_factory import build_ene_ana, build_rmsd, build_dfmult_file, build_output
from cpaeds.utils import get_dir_list, write_file,  write_file2, copy_lib_file, check_finished
from cpaeds.aeds_sampling import sampling

# setting logger name and log level
logger = LoggerFactory.get_logger("postprocessing.py", log_level="INFO", file_name = "debug.log")

class postprocessing(object):
    def __init__(self, settings: dict) -> None:
        self.config = settings
        self.equilibrate = self.config['simulation']['equilibrate']
        self.fraction_list = []
        self.dF_list = []
        self.rmsd_list = []
        self.energy_map = self.initialise_energy_map()

    def create_ana_dir(self):
        try:
            os.mkdir('ene_ana')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/ene_ana"):
            copy_lib_file(f"{parent}/ene_ana",'ene_ana.md++.lib')
            ene_ana_body =  build_ene_ana(self.config,self.config['simulation']['parameters']['NRUN'])
            write_file(ene_ana_body,'ene_ana.arg')
            df_file_body = build_dfmult_file(self.config)
            write_file(df_file_body,'df.arg')

    def create_rmsd_dir(self):
        try:
            os.mkdir('rmsd')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsd"):
            rmsd_body = build_rmsd(self.config,self.config['simulation']['parameters']['NRUN'])
            write_file(rmsd_body,'rmsd_eq.arg')

    def create_output_dir(self):
        try:
            os.mkdir('results')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/results"):
            output_body = build_output(self.config,self.fraction_list,self.dF_list,self.rmsd_list)
            write_file2(output_body,'results.out')
            np.save('energies.npy', np.array(self.energy_map), allow_pickle=False)

    def initialise_energy_map(self):
        map = []
        for i in range(self.config['simulation']['NSTATES']+2):
            map.append(np.array([], dtype=np.float64))
        return map
    
    def update_energy_mapping(self,map, appendix):
        if len(map) == len(appendix):
            temp_map = []
            for i in range(len(map)):
                temp_map.append(np.append(map[i],appendix[i]))
        else:
            logger.critical(f"Energymap shapes do not match")
        return temp_map      
         
    def run_ene_ana(self):
        parent = os.getcwd()
        with set_directory(f"{parent}/ene_ana"):
            exe = subprocess.run(
                                ['ene_ana', '@f', 'ene_ana.arg'],
                                check=True,
                                capture_output=True,
                                text=True
                                )
            exe.check_returncode()
            with open('df.out', 'w') as sp:
                        exe = subprocess.run(
                                            ['dfmult', '@f', 'df.arg'], 
                                            stdout=sp)
            df = self.read_df(f'./df.out')
            self.dF_list.append(df)
            samples = sampling(self.config,self.offsets,df)
            fractions, energies = samples.main()
            self.fraction_list.append(fractions)
            self.energy_map = self.update_energy_mapping(self.energy_map, energies)

    def run_rmsd(self):
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsd"):
            with open('rmsd.out', 'w') as sp:
                exe = subprocess.run(
                                    ['rmsd', '@f', 'rmsd.arg'], 
                                    stdout=sp)
                exe.check_returncode()
                self.rmsd_list.append(self.read_rmsd('rmsd.out'))

    def read_df(self,file):
        dfs = []
        with open(file, "r") as inn:
            for line in inn:
                if f"DF_" and "_R" in line:
                    fields = line.split()
                    dfs.append(float(fields[1]))
        return dfs

    def read_rmsd(self,file):
        one_before_last = None
        last_line = None
        with open(file, "r") as inn:
            for line in inn:
                one_before_last=last_line
                last_line = line
            fields = one_before_last.split()
            rmsd = float(fields[1])
        return rmsd

    def read_output(self, file):
        self.fraction_list = []
        offset_list = []
        with open(file, "r") as inn:
            next(inn)
            for line in inn: 
                line_splitted = line.split(',')
                self.fraction_list.append(float(line_splitted[2]))
                offset_list.append(float(line_splitted[1]))
        return self.fraction_list,offset_list
    
    def offsets_sp(self,depth):
        self.offsets = []
        for i in range(len(self.config['simulation']['parameters']['EIR_list'])):
            self.offsets.append(self.config['simulation']['parameters']['EIR_list'][i][depth]) 
    
    def run_postprocessing(self):
            pdir_list = []
            stat_run_path = f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[:-1]
            starting_number = int(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[-1:])
            if int(self.config['simulation']['NSTATS']) > 1: 
                    for i in range(starting_number, self.config['simulation']['NSTATS'] + 1):
                            pdir_list.append(f"{stat_run_path}{i}")
                            pdir_list.sort(key=natural_keys)
            else:
                    pdir_list.append(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}")
            
            for pdir in pdir_list:
                    with set_directory(f"{pdir}"):
                        dir_list = get_dir_list()
                        for i, dir in enumerate(dir_list):
                                if dir == 'results':
                                        continue
                                else:
                                    with set_directory(f"{pdir}/{dir}"):
                                        self.run_finished, NOMD = check_finished(self.config)
                                        if self.run_finished == True:
                                            logger.info(f"Run in {dir} finished.")
                                        else:
                                            logger.info(f"Run in {dir} unfinished.")
                                        #rmsd & ene ana
                                        self.offsets_sp(i)
                                        self.create_ana_dir()
                                        self.run_ene_ana()
                                        self.create_rmsd_dir()
                                        self.run_rmsd()
                                        """ 
                                        ### remove hard coding
                                        if self.fraction_list[-1][0] > 0.15 and self.fraction_list[-1][0] < 0.85:
                                            emix,tmix = read_energyfile(f'./eds_vmix.dat')
                                            state,tstate = read_state_file(f'./statetser.dat')
                                            self.column_name.append(f'run{i+1}[{self.dF_list[-1]}]')
                                            self.density_map_emix.append([emix,tmix])
                                            self.density_map_state.append([state,tstate])
                                        else:
                                            pass
"""  
                    self.create_output_dir()