# python modules
import os
import sys
# cpaeds modules
from cpaeds.algorithms import natural_keys
from cpaeds.logger import LoggerFactory
from cpaeds.utils import get_dir_list

# setting logger name and log level
logger = LoggerFactory.get_logger("system.py", log_level="DEBUG", file_name = "debug.log")

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

    def __check_system_dir(self):
        """Checking if current dir is simulation dir"""
        if self.sys_dir == os.path.dirname(os.path.abspath(__file__)):
            pass
        else:
            os.chdir(self.sys_dir)

    def __check_input_dirs(self):
        """Checking if path to input dir exist"""
        self.dir_list = get_dir_list()
        if os.path.basename(os.path.normpath(self.topo_dir)) in self.dir_list:
            logger.info(f"Topo folder found.")
        else:
            logger.critical(f"Missing topo folder.")
        if os.path.basename(os.path.normpath(self.md_dir)) in self.dir_list:
            logger.info(f"MD folder found.")
        else:
            logger.critical(f"Missing MD folder.")
        if 'aeds' in self.dir_list:
            self.aeds_dir = f"{self.sys_dir}/aeds"
            logger.info(f"aeds folder already exists.")
        else:
            try:
                os.mkdir('aeds')
            except FileExistsError:
                pass
            self.aeds_dir = f"{self.sys_dir}/aeds"
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
        imd_list=[]
        for file in files:
            if file.endswith('.cnf'):
                cnf_list.append(file)
            if file.endswith('.imd'):
                imd_list.append(file)
        cnf_list.sort(key=natural_keys)
        imd_list.sort(key=natural_keys)
        self.cnf_file = f"{cnf_list[-1]}"
        logger.info(f"{self.cnf_file} found.")
        self.ref_imd = f"{imd_list[-1]}" 
        logger.info(f"{self.ref_imd} found.")

    def __check_simulation_settings(self):
        """Checks if the set of input parameter in the settings.yaml is complete"""
        if self.config['simulation']['NSTATS']:
            if int(self.config['simulation']['NSTATS']) > 1:
                logger.info(f"Statistic mode with {self.config['simulation']['NSTATS']} repetitions")  
        else:
            self.config['simulation']['NSTATS'] = 1
            logger.info(f"Single run mode")
        if self.config['simulation']['NSTATES']:
            if int(self.config['simulation']['NSTATES']) > 2:
                logger.info(f"Multi-endstate run with {self.config['simulation']['NSTATES']} endstates")  
        else:
            self.config['simulation']['NSTATES'] = 2
            logger.info(f"Default run with 2 endstates")
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
        self.__check_input_settings()
        self.__check_system_dir()
        self.__check_input_dirs()
        self.__check_input_files()
        self.__check_simulation_settings()