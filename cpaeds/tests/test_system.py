"""
Unit and regression test for the SetupSystem class.
"""

# Import package, test suite, and other packages as needed
import shutil
import os
import pytest
from cpaeds.utils import load_config_yaml, read_last_line
from cpaeds.system import SetupSystem
from cpaeds.context_manager import set_directory

def test_check_input_settings():
        """Sample test, will pass so long as all inputs are given in settings.yaml."""
        path = os.getcwd()
        settings = load_config_yaml(
            config= f'{path}/cpaeds/tests/test_data/test_settings.yaml')
    
        system = SetupSystem(settings)
        system.run_checks()
        log_file = f"{path}/debug.log"
        with open(log_file, "rb") as log_f:
                last = str(read_last_line(log_f))

        assert f"No extra equilibration found." in last
        assert os.path.isdir(f'{path}/cpaeds/tests/test_data/example_system/aeds') == True
        with set_directory(f'{path}/cpaeds/tests/test_data/example_system/'):
                os.rmdir('aeds')

def test_check_input_systems_wrong():
        """Sample test, will pass if sys.exit for missing input parameter works."""
        path = os.getcwd()
        settings = load_config_yaml(
                config= f'{path}/cpaeds/tests/test_data/wrong_test_settings.yaml')
    
        system = SetupSystem(settings)
        with pytest.raises(SystemExit) as excinfo:
                system.run_checks()

        assert excinfo.value.code == f"Error changing to output folder directory."

def test_check_input_simulation_wrong():
        """Sample test, will pass if sys.exit for missing input parameter works."""
        path = os.getcwd()
        settings = load_config_yaml(
                config= f'{path}/cpaeds/tests/test_data/wrong_test_settings2.yaml')
    
        system = SetupSystem(settings)
        with pytest.raises(SystemExit) as excinfo:
                system.run_checks()

        assert excinfo.value.code == f"Error missing simulation parameter."

def test_folder_structure():
        """Sample test, will pass so long as all inputs are given in settings.yaml."""
        path = os.getcwd()
        settings = load_config_yaml(
            config= f'{path}/cpaeds/tests/test_data/test_settings.yaml')
    
        system = SetupSystem(settings)
        system.run_checks()
        system.create_prod_run()

        with set_directory(f'{path}/cpaeds/tests/test_data/example_system/'):            
                shutil.rmtree('./aeds')

def test_cpAEDS_type_2():
        """Sample test, will pass so long as all inputs are given in settings.yaml."""
        path = os.getcwd()
        settings = load_config_yaml(
            config= f'{path}/cpaeds/tests/test_data/test_settings_offsets.yaml')
    
        system = SetupSystem(settings)
        system.run_checks()
        system.create_prod_run()

        with set_directory(f'{path}/cpaeds/tests/test_data/example_system/'):            
                shutil.rmtree('./aeds')