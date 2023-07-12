import os
from tqdm import tqdm
import subprocess
import numpy as np
from glob import glob
import multiprocessing
##cpaeds 
from cpaeds.algorithms import natural_keys
from cpaeds.context_manager import set_directory
from cpaeds.logger import LoggerFactory
from cpaeds.file_factory import build_ene_ana, build_rmsd, build_dfmult_file, build_output, build_rmsf
from cpaeds.utils import write_file2, copy_lib_file, check_finished
from cpaeds.aeds_sampling import sampling

# setting logger name and log level
logger = LoggerFactory.get_logger("postprocessing.py", log_level="INFO", file_name = "debug.log")

class postprocessing_parallel(object):
    """Reads in and stores all the data needed for the postprocessing. Calculates dFs and sampling related values.
       Stores results in an np-array
    Args:
        object (dict): finalsettings of the system setup.
    """
    def __init__(self, settings: dict) -> None:
        self.config = settings
        self.equilibrate = self.config['simulation']['equilibrate']
        self.fraction_list = []
        self.dF_list = []
        self.rmsd_list = []
        self.energy_map = self.initialise_energy_map()
        self.energy_runs = []
        self.__check_overwrite()

    def __check_overwrite(self):
        if 'overwrite' in self.config['system']:
            if self.config['system']['overwrite'] == True:
                self.overwrite = True
                logger.info(f"Overwritting cpAEDS files")  
        else:
            self.overwrite = False

    def create_ana_dir(self, NOMD):
        """
        Creates a folder (ene_ana) with the argument files for gromos++ ene_ana and dfmult.
        """
        try:
            os.mkdir('ene_ana')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/ene_ana"):
            if self.config['system']['lib_type'] == f"cuda":
                copy_lib_file(f"{parent}/ene_ana",'ene_ana.md++.lib','2018_12_10',self.overwrite)
            elif self.config['system']['lib_type'] == f"cuda_local":
                copy_lib_file(f"{parent}/ene_ana",'ene_ana.md++.lib','2018_12_10',self.overwrite)
            elif self.config['system']['lib_type'] == f"cuda_new":
                copy_lib_file(f"{parent}/ene_ana",'ene_ana.md++.lib','2023_04_15',self.overwrite)
            ene_ana_body =  build_ene_ana(self.config,NOMD)
            write_file2(ene_ana_body,'ene_ana.arg')
            df_file_body = build_dfmult_file(self.config)
            write_file2(df_file_body,'df.arg')

    def create_rmsd_dir(self, NOMD):
        """
        Creates a folder (rmsd) with the argument file for gromos++ rmsd.
        """
        try:
            os.mkdir('rmsd')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsd"):
            rmsd_body = build_rmsd(self.config,NOMD)
            write_file2(rmsd_body,'rmsd.arg')

    def create_rmsf_dir(self, NOMD):
        """
        creates the rmsf folder and places an rmsf argument file in it.

        Args:
            NOMD (_type_): Number of finished runs
        """
        try:
            os.mkdir('rmsf')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsf"):
            rmsf_body = build_rmsf(self.config,NOMD)
            write_file2(rmsf_body,'rmsf.arg')

    def create_output_dir(self):
        """
        Creates a folder (results) with the collected energies.npy and a results.out in csv-format.
        """
        try:
            os.mkdir('results')
        except FileExistsError:
            pass
        parent = os.getcwd()
        with set_directory(f"{parent}/results"):
            output_body = build_output(self.config,self.fraction_list,self.dF_list,self.rmsd_list)
            write_file2(output_body,'results.out')
            harmonizedEnergyArray = self.harmonizeEnergyArray(self.energy_runs)
            np.save('energies.npy', harmonizedEnergyArray, allow_pickle=False)

    def harmonizeEnergyArray(self, l: list):
        """
        Harmonizes the length of a given list of lists containing energies by padding the 3rd dimension of the list with np.nan.

        This is needed to use np.save on unfinished runs if there are some runs which are more finished than others.

        Args:
            l (list): list of lists with three dimensions: (runs, states, energies)
        """

        array_list = []

        for run in l:
            array_list.append(np.array(run))

        maxLen = max([len(a[-1]) for a in array_list])

        for i, run in enumerate(array_list):
            paddedArray = np.pad(run, [(0,0), (0, maxLen - len(run[0]))], 'constant', constant_values=np.nan)
            array_list[i] = paddedArray

        return np.array(array_list)

    def initialise_energy_map(self):
        """
        Initialises a list with empty np.arrays to store energies in.
        """
        map = []
        for i in range(self.config['simulation']['NSTATES']+2):
            map.append(np.array([], dtype=np.float64))
        return map
    
    def update_energy_mapping(self,map, appendix):
        """
        Updates np.arrays with energies from sampling.
        """
        if len(map) == len(appendix):
            temp_map = []
            for i in range(len(map)):
                temp_map.append(np.append(map[i],appendix[i]))
        else:
            logger.critical(f"Energymap shapes do not match")
            logger.critical(f"{len(map)} != {len(appendix)}")
        return temp_map      
         
    def run_ene_ana(self):
        """
        executing gromos++ ene_ana and dfmult in the ene_ana dir. Executing sampling class.  
        """
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

    def run_rmsd(self):
        """
        Running gromos++ rmsd in rmsd folder.
        """
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsd"):
            with open('rmsd.out', 'w') as sp:
                exe = subprocess.run(
                                    ['rmsd', '@f', 'rmsd.arg'], 
                                    stdout=sp)
                exe.check_returncode()

    def run_rmsf(self):
        """
        Running gromos++ rmsf in rmsf folder.
        """
        parent = os.getcwd()
        with set_directory(f"{parent}/rmsf"):
            with open('rmsf.out', 'w') as sp:
                exe = subprocess.run(
                                    ['rmsf', '@f', 'rmsf.arg'], 
                                    stdout=sp)
                exe.check_returncode()

    def read_df(self,file):
        """
        Reads in df.out created by run_ene_ana.
        Args:
            file (txt): df.out

        Returns:
            list: list of free energies to reference state for each endstate. Sorted like 1,2,3,..,11,12.
        """
        dfs = []
        with open(file, "r") as inn:
            for line in inn:
                if f"DF_" and "_R" in line:
                    fields = line.split()
                    dfs.append(float(fields[1]))
        return dfs

    def read_rmsd(self,file):
        """
        Reads rmsd.out.
        Args:
            file (txt): rmsd over time.

        Returns:
            float: rmsd between first and last snap.
        """
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
        """
        Reads results.out.
        Needs update.
        """
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
        """
        Deconstructs offsets from final_setting.yaml to get offsets for each single point calculation.
        Args:
            depth (_type_): _description_
        """
        self.offsets = []
        for i in range(len(self.config['simulation']['parameters']['EIR_list'])):
            self.offsets.append(self.config['simulation']['parameters']['EIR_list'][i][depth]) 

    def run_postprocessing_parallel(self, tasks: int=1):
        """
        Runs the postprocessing in parallel with *tasks* tasks

        Args:
            tasks (int, optional): Number of tasks to run in parallel. Defaults to 1.
        """
        pdir_list = []
        stat_run_path = f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[:-1]
        starting_number = int(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[-1:])
        #For statistical run
        if int(self.config['simulation']['NSTATS']) > 1: 
                for i in range(starting_number, self.config['simulation']['NSTATS'] + 1):
                        pdir_list.append(f"{stat_run_path}{i}")
                        pdir_list.sort(key=natural_keys)
        else:
                pdir_list.append(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}")

        for pdir in pdir_list:
            with set_directory(pdir):
                dir_list = glob(f"{pdir}/{self.config['system']['output_dir_name']}_*/")

                with multiprocessing.Pool(tasks) as pool:
                    pool.starmap(self.postprocessing_single_run, enumerate(dir_list))

        self.get_results()

    def postprocessing_single_run(self, i: int, dir: str):
        """
        Internal function used for parallel processing of post-processing in the method "run_postprocessing_parallel)
        Performs the actual post-processing tasks

        Args:
            i (int): n from enumerate
            dir (str): directory to work in 
        """
        with set_directory(dir):
            #Checking for finished runs.
            self.run_finished, NOMD = check_finished(self.config)
            if self.run_finished == True:
                logger.info(f"Run in {dir} finished.")
            else:
                logger.info(f"Run in {dir} unfinished.")
            #rmsd & ene ana
            self.offsets_sp(i)
            self.create_ana_dir(NOMD=NOMD)
            self.run_ene_ana()
            self.create_rmsd_dir(NOMD=NOMD)
            self.run_rmsd()
            if self.config['simulation']['rmsf'] is True:
                self.create_rmsf_dir(NOMD=NOMD)
                self.run_rmsf()

    def get_results(self):
        # loops through folders and only gets the results from the outputs
        # fraction_list, dF_list, rmsd_list, energy_runs
        pdir_list = []
        stat_run_path = f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[:-1]
        starting_number = int(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}"[-1:])
        #For statistical run
        if int(self.config['simulation']['NSTATS']) > 1: 
                for i in range(starting_number, self.config['simulation']['NSTATS'] + 1):
                        pdir_list.append(f"{stat_run_path}{i}")
                        pdir_list.sort(key=natural_keys)
        else:
                pdir_list.append(f"{self.config['system']['aeds_dir']}/{self.config['system']['output_dir_name']}")

        for pdir in tqdm(pdir_list):
            with set_directory(pdir):
                dir_list = glob(f"{pdir}/{self.config['system']['output_dir_name']}_*/")
                for n, dir in tqdm(enumerate(dir_list), desc="subdir", total=len(dir_list)):
                    with set_directory(dir+"/ene_ana"):
                        #logger.info(f"processing {dir}")
                        df = self.read_df('./df.out')
                        self.dF_list.append(df)
                        self.offsets_sp(n)
                        samples = sampling(self.config, self.offsets, df)
                        fractions, energies = samples.main()
                        self.fraction_list.append(fractions)
                        self.energy_runs.append(energies)
                    with set_directory(dir+"/rmsd"):
                        self.rmsd_list.append(self.read_rmsd('rmsd.out'))

                self.create_output_dir()