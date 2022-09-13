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
        self.md_dir: str = settings['system']['md_dir']
        self.output_dir: str = settings['system']['output_dir_name']
        self.topology_file: str = settings['system']['topo_file']
        self.pertubation_file: str = settings['system']['pert_file']
        self.sys_dir: str = settings['system']['system_dir']

    def check_input_settings(self):
        """
        A function that checks for the existence of the input files for a simulation.
        ----------
        md_dir: Path to the MD directory containing the minimized coordinates. [.cnf]
        output_dir: Name of the directory to create the simulation files in.
        topo_file: Path to the topology file. (same as for the MDs) [.top] 
        pert_file: Path to the pertubation file. [.ptp]
        """
        outdir_exists = False
        topo_exists = False
        pert_exists = False
        if self.md_dir:
            logger.info(f"MD folder set to {self.md_dir}")
        else:
            logger.critical(f"Missing MD folder argument.")
            sys.exit(f"Error changing to output folder directory.")
        if self.output_dir:
            outdir_exists = True
            if outdir_exists == True:
                logger.info(f"Output folder name set to {self.output_dir}.")
            else:
                self.output_dir: str = 'prod_runs'
                logger.info(f"Output folder name set to {self.output_dir}.")
        if self.topology_file:
            topo_exists = True
            if topo_exists == True:
                self.topo_dir: str = os.path.dirname(self.topology_file)
                logger.info(f"Topo folder set to {self.topo_dir}.")
                self.topology_file: str = os.path.basename(os.path.normpath(self.topology_file))
                logger.info(f"Topo file set to {self.topology_file}.")
            else:
                logger.critical("Missing topo file argument.")
                sys.exit("Error changing to output folder directory.")
        if self.pertubation_file:
            pert_exists = True
            if pert_exists == True:
                self.pertubation_file: str = os.path.basename(os.path.normpath(self.pertubation_file))
                logger.info(f"Pertubation file set to {self.pertubation_file}.")
            else:
                logger.critical("No pertubation file (.ptp) argument.")
                sys.exit("Error changing to output folder directory.")

    def check_system_dir(self):
        if self.sys_dir == os.path.dirname(os.path.abspath(__file__)):
            pass
        else:
            os.chdir(self.sys_dir)



    def check_dirs(settings_loaded):
        dir_list = get_dir_list()
        if os.path.basename(os.path.normpath(settings_loaded['system']['topo_dir'])) in dir_list:
            print("Topo folder found.")
        else:
            print("Missing topo folder.")
        if os.path.basename(os.path.normpath(settings_loaded['system']['md_dir'])) in dir_list:
            print("MD folder found.")
        else:
            print("Missing MD folder.")
        if 'aeds' in dir_list:
            settings_loaded['system']['aeds_dir'] = f"{settings_loaded['system']['system_dir']}/aeds"
            print("aeds folder already exists.")
        else:
            try:
                os.mkdir('aeds')
            except FileExistsError:
                pass
            settings_loaded['system']['aeds_dir'] = f"{settings_loaded['system']['system_dir']}/aeds"
            print("AEDS folder created.")

    def check_input_files(settings_loaded):
        files = [f for f in os.listdir(settings_loaded['system']['topo_dir']) if os.path.isfile(os.path.join(settings_loaded['system']['topo_dir'], f))]
        if  settings_loaded['system']['topo_file'] in files:
            print(f"{settings_loaded['system']['name']}.top found.")
        else:
            print("No topo file exists.")
        if settings_loaded['system']['pert_file'] in files:
            print(f"{settings_loaded['system']['pert_file']} found.")
        else:
            print("No ptp file exists.")

        files = [f for f in os.listdir(settings_loaded['system']['md_dir']) if os.path.isfile(os.path.join(settings_loaded['system']['md_dir'], f))]
        cnf_list=[]
        imd_list=[]
        for file in files:
            if file.endswith('.cnf'):
                cnf_list.append(file)
            if file.endswith('.imd'):
                imd_list.append(file)
        cnf_list.sort(key=natural_keys)
        imd_list.sort(key=natural_keys)
        settings_loaded['system']['cnf_file'] = f"{cnf_list[-1]}"
        print(f"{settings_loaded['system']['cnf_file']} found.")
        settings_loaded['system']['ref_imd'] = f"{imd_list[-1]}" 
        print(f"{settings_loaded['system']['ref_imd']} found.")

    def check_simulation_settings(settings_loaded):
        nstats_exists = False
        if settings_loaded['simulation']['NSTATS']:
            nstats_exists = True
            if nstats_exists == True and int(settings_loaded['simulation']['NSTATS']) > 1:
                print(f"Statistic mode with {settings_loaded['simulation']['NSTATS']} repetitions")  
            else:
                settings_loaded['simulation']['NSTATS'] = 1
                print(f"Single run mode")
        if  ( 
            settings_loaded['simulation']['parameters'].get('NRUN') == None or
            settings_loaded['simulation']['parameters'].get('NSTLIM') == None or 
            settings_loaded['simulation']['parameters'].get('NTPR') == None or
            settings_loaded['simulation']['parameters'].get('NTWX') == None or
            settings_loaded['simulation']['parameters'].get('NTWE') == None or
            settings_loaded['simulation']['parameters'].get('dt') == None or 
            settings_loaded['simulation']['parameters'].get('EMIN') == None or
            settings_loaded['simulation']['parameters'].get('EMAX') == None or
            settings_loaded['simulation']['parameters'].get('EIR_start') == None or
            settings_loaded['simulation']['parameters'].get('EIR_range') == None or
            settings_loaded['simulation']['parameters'].get('EIR_step_size') == None
        ):
            raise KeyError("Parameterset is not complete")