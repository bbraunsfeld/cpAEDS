# python modules
import copy
import os
import subprocess
import sys
from tqdm import tqdm
# cpaeds modules
from cpaeds.algorithms import natural_keys, offset_steps
from cpaeds.logger import LoggerFactory
from cpaeds.utils import get_dir_list, copy_lib_file, write_file, create_ana_dir, dumb_full_config_yaml
from cpaeds.context_manager import set_directory
from cpaeds.file_factory import build_mk_script_file,build_job_file,build_imd_file

# setting logger name and log level
logger = LoggerFactory.get_logger("system.py", log_level="INFO", file_name = "debug.log")

class SetupSystem(object):
    def __init__(self, settings: dict):
        """
        A class that contains all informations for the system to be constructed.
        Everything is needed to start the construction of the simulation folders is checked here.
        Parameters
        ----------
        settings: dict
            the configuration dictionary obtained with utils.load_config_yaml
        """
        self.config: dict = settings
        self.md_dir: str = self.config['system']['md_dir']
        self.output_dir: str = self.config['system']['output_dir_name']
        self.topology_file: str = self.config['system']['topo_file']
        self.pertubation_file: str = self.config['system']['pert_file']
        self.sys_dir: str = self.config['system']['system_dir']
        self.dir_list = None

    def __check_engines(self):
        """
        Checking for a path given towards md binary.
        Checking for path to mk_script binary. If not given set a default.[/home/common/gromos_20210129/gromos++/UBUNTU/bin//mk_script]
        """
        if 'path_to_md_engine' in self.config['system']:
            pass
        else:
            logger.critical(f"No path to md program.")

        if 'path_to_mk_script' in self.config['system']:
            pass
        else:
            self.config['system']['path_to_mk_script'] = f"/home/common/gromos_20210129/gromos++/UBUNTU/bin//mk_script"

    def __check_input_settings(self):
        """
        A function that checks for the existence of the input files for a simulation.
        ----------
        md_dir: Path to the MD directory containing the minimized coordinates. [.cnf]
        output_dir: Name of the directory to create the simulation files in.
        topo_file: Path to the topology file. (same as for the MDs) [.top] 
        pert_file: Path to the pertubation file. [.ptp]
        """
        if self.md_dir:
            logger.info(f"MD folder set to {self.md_dir}")
        else:
            logger.critical(f"Missing MD folder argument.")
            sys.exit(f"Error changing to output folder directory.")
        if self.output_dir:
            logger.info(f"Output folder name set to {self.output_dir}.")
        else:
            self.output_dir: str = 'prod_runs'
            logger.info(f"Output folder name set to {self.output_dir}.")
        if self.topology_file:
            logger.debug(os.path.dirname(self.topology_file))
            self.topo_dir: str = os.path.dirname(self.topology_file)
            print(self.topo_dir)
            logger.info(f"Topo folder set to {self.topo_dir}.")
            self.topology_file: str = os.path.basename(os.path.normpath(self.topology_file))
            logger.info(f"Topo file set to {self.topology_file}.")
        else:
            logger.critical(f"Missing topo file argument.")
            sys.exit(f"Error changing to output folder directory.")
        if self.pertubation_file:
            self.pertubation_file: str = os.path.basename(os.path.normpath(self.pertubation_file))
            logger.info(f"Pertubation file set to {self.pertubation_file}.")
        else:
            logger.critical(f"No pertubation file (.ptp) argument.")
            sys.exit(f"Error changing to output folder directory.")

    """
    def __check_system_dir(self):
        """"""Checking if current dir is simulation dir""""""
        if self.sys_dir == os.path.dirname(os.path.abspath(__file__)):
            pass
        else:
            os.chdir(self.sys_dir)
    """

    def __check_input_dirs(self):
        """Checking if path to input dir exist"""
        with set_directory(f"{self.sys_dir}"):
            self.dir_list = get_dir_list()
        if os.path.basename(os.path.normpath(self.topo_dir)) in self.dir_list:
            logger.info(f"Topo folder found.")
        else:
            logger.critical(f"Missing topo folder.")
        if os.path.basename(os.path.normpath(self.md_dir)) in self.dir_list:
            logger.info(f"MD folder found.")
        else:
            logger.critical(f"Missing MD folder.")
    
    def __check_aeds_dir(self):
        """Checking if aeds dir exist. Creates aeds dir if it does not exist"""
        with set_directory(f"{self.sys_dir}"):
            self.dir_list = get_dir_list()
        if 'aeds' in self.dir_list:
            self.aeds_dir = f"{self.sys_dir}/aeds"
            self.config['system']['aeds_dir'] = self.aeds_dir 
            logger.info(f"aeds folder already exists.")
        else:
            with set_directory(f"{self.sys_dir}"):
                os.mkdir(f"aeds")
            self.aeds_dir = f"{self.sys_dir}/aeds"
            self.config['system']['aeds_dir'] = self.aeds_dir
            logger.info(f"aeds folder created.")

    def __check_input_files(self):
        """Checking if the input files .topo, .ptp, .cnf & .imd exist"""
        files = [f for f in os.listdir(self.topo_dir) if os.path.isfile(os.path.join(self.topo_dir, f))]
        if  self.topology_file in files:
            logger.info(f"Topology file found.")
        else:
            logger.critical(f"No topology file exists.")
        if self.pertubation_file in files:
            logger.info(f"Pertubation file found.")
        else:
            logger.critical(f"No pertubation file exists.")

        files = [f for f in os.listdir(self.md_dir) if os.path.isfile(os.path.join(self.md_dir, f))]
        cnf_list=[]
        for file in files:
            if file.endswith('.cnf'):
                cnf_list.append(file)
        cnf_list.sort(key=natural_keys)
        self.config['system']['cnf_file'] = f"{cnf_list[-1]}"
        logger.info(f"{self.config['system']['cnf_file']} found.")
        self.config['system']['ref_imd'] = f"{cnf_list[-1][:-4]}.imd" 
        logger.info(f"{self.config['system']['ref_imd']} found.")

    def __check_simulation_settings(self):
        """Checks if the set of input parameter in the settings.yaml is complete"""
        if 'NSTATS' in self.config['simulation']:
            if int(self.config['simulation']['NSTATS']) > 1:
                logger.info(f"Statistic mode with {self.config['simulation']['NSTATS']} repetitions")  
        else:
            self.config['simulation']['NSTATS'] = 1
            logger.info(f"Single run mode")
        if 'NSTATES' in self.config['simulation']:
            if int(self.config['simulation']['NSTATES']) > 2:
                logger.info(f"Multi-endstate run with {self.config['simulation']['NSTATES']} endstates")  
        else:
            self.config['simulation']['NSTATES'] = 2
            logger.info(f"Default run with 2 endstates")
        if 'EIR_start' in self.config['simulation']['parameters']:
            if isinstance(self.config['simulation']['parameters']['EIR_start'], list):
                logger.info(f"Starting offset: {self.config['simulation']['parameters']['EIR_start']}")
            elif isinstance(self.config['simulation']['parameters']['EIR_start'], float):
                self.config['simulation']['parameters']['EIR_start'] = [self.config['simulation']['parameters']['EIR_start']]
                logger.info(f"Starting offset: {self.config['simulation']['parameters']['EIR_start']}")
            elif isinstance(self.config['simulation']['parameters']['EIR_start'], int):
                self.config['simulation']['parameters']['EIR_start'] = [self.config['simulation']['parameters']['EIR_start']]
                logger.info(f"Starting offset: {self.config['simulation']['parameters']['EIR_start']}")
        else:
            logger.critical(f"No value for EIR_start in input yaml.")
            sys.exit()
        if 'equilibrate' in self.config['simulation']:
            if int(self.config['simulation']['equilibrate'][0]) == True:
                logger.info(f"Standard equilibration set to {self.config['simulation']['equilibrate'][1]}")  
        else:
            self.config['simulation']['equilibrate'] = [False]
            logger.info(f"No extra equilibration found.")
        if 'cpAEDS_type' in self.config['simulation']:
            if int(self.config['simulation']['cpAEDS_type']) > 1:
                logger.info(f"cpAEDS_type set to {self.config['simulation']['cpAEDS_type']}")  
        else:
            self.config['simulation']['cpAEDS_type'] = 1
            logger.info(f"No cpAEDS_type found. Default set to 1")
        if 'temp' in self.config['simulation']['parameters']:
            pass
        else:
            self.config['simulation']['parameters']['temp'] = 300
            logger.info(f"Setting simulation temperature to 300 K")
        if  ( 
            self.config['simulation']['parameters'].get('NRUN') == None or
            self.config['simulation']['parameters'].get('NSTLIM') == None or 
            self.config['simulation']['parameters'].get('NTPR') == None or
            self.config['simulation']['parameters'].get('NTWX') == None or
            self.config['simulation']['parameters'].get('NTWE') == None or
            self.config['simulation']['parameters'].get('dt') == None or 
            self.config['simulation']['parameters'].get('EMIN') == None or
            self.config['simulation']['parameters'].get('EMAX') == None or
            self.config['simulation']['parameters'].get('EIR_start') == None or
            self.config['simulation']['parameters'].get('EIR_range') == None or
            self.config['simulation']['parameters'].get('EIR_step_size') == None
        ):
            sys.exit(f"Error missing simulation parameter.")

    def run_checks(self):
        self.__check_engines()
        self.__check_input_settings()
        #self.__check_system_dir()
        self.__check_input_dirs()
        self.__check_aeds_dir()
        self.__check_input_files()
        self.__check_simulation_settings()

    def __create_offsets(self):
        offsets = offset_steps(self.config['simulation']['parameters']['EIR_start'],
                                    self.config['simulation']['parameters']['EIR_range'],
                                    self.config['simulation']['parameters']['EIR_step_size'],
                                    self.config['simulation']['cpAEDS_type'])
        self.config['simulation']['parameters']['EIR_list'] = offsets
        self.config['simulation']['parameters']['n_runs'] = len(offsets[0])
        for i in range(len(offsets)):
            logger.info(f"List of offsets {offsets[i]} for state {i+1}.")


    def __create_folders(self):
        with set_directory(self.aeds_dir):
            for i in range(1, self.config['simulation']['NSTATS'] + 1):
                try:
                    os.makedirs(f"{self.aeds_dir}/{self.output_dir}_{i}")
                except FileExistsError:
                    pass
            
            with set_directory(f"{self.aeds_dir}/{self.output_dir}_{i}"):
                for j in range(1,self.config['simulation']['parameters']['n_runs'] + 1):
                    try:
                        os.mkdir(f"{self.output_dir}_{i}_{j}")
                    except FileExistsError:
                        pass
    
    def create_input_files(self):
        pdir_list = []
        for i in range(1, self.config['simulation']['NSTATS'] + 1):
            pdir_list.append(f"{self.aeds_dir}/{self.output_dir}_{i}")
        pdir_list.sort(key=natural_keys)
        random_seed = 1
        for pdir in pdir_list:
            with set_directory(f"{pdir}"):
                dir_list = get_dir_list()
                eir_counter = 0
                for dir in tqdm(dir_list):
                    with set_directory(f"{pdir}/{dir}"):
                        if self.config['system']['lib_type'] == f"cuda":
                            copy_lib_file(os.getcwd(),'mk_script_cuda_8_slurm.lib')
                        elif self.config['system']['lib_type'] == f"cuda_local":
                            copy_lib_file(os.getcwd(),'mk_script_cuda_8.lib')
                        mk_script_body =  build_mk_script_file(self.config,os.getcwd())
                        write_file(mk_script_body,'aeds_mk_script.arg')
                        job_file_body = build_job_file(self.config)
                        write_file(job_file_body,'aeds.job')
                        ### parses list of EIRS on same level
                        EIR = [item[eir_counter] for item in self.config['simulation']['parameters']['EIR_list']]
                        logger.info(f"EIR parsed to build_imd {EIR}.")
                        imd_file_body = build_imd_file(self.config,EIR,random_seed) 
                        write_file(imd_file_body,'aeds.imd')
                        create_ana_dir(self.config)

                        if os.path.exists(f"{pdir}/{dir}/aeds_{self.config['system']['name']}_1.imd"):
                            logger.info(f"imd and run files exist")
                            pass
                        else:
                            logger.info(f'Running mk_script...')
                            exe = subprocess.run(
                                [self.config['system']['path_to_mk_script'], '@f', 'aeds_mk_script.arg'],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            exe.check_returncode()
                            logger.info(f'Finished running mk_script.')   
                        eir_counter += 1
                settings_updated = copy.deepcopy(self.config)
                settings_updated['system']['output_dir_name']=f"{self.output_dir}_{random_seed}"
                dumb_full_config_yaml(settings_updated)
                random_seed += 1

    def create_prod_run(self):
        self.__create_offsets()
        self.__create_folders()
        self.create_input_files()
        