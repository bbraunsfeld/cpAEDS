"""
Unit and regression test for the SetupSystem class.
"""

# Import package, test suite, and other packages as needed
import sys
import os
import pytest
from cpaeds.utils import load_config_yaml, read_last_line
from cpaeds.system import SetupSystem

def test_check_input_settings():
        """Sample test, will pass so long as all inputs are given in settings.yaml."""
        path = os.getcwd()
        settings = load_config_yaml(
            config= f'{path}/cpaeds/tests/test_data/test_settings.yaml')
    
        system = SetupSystem(settings)
        SetupSystem.run_checks(system)
        log_file = f"{path}/debug.log"
        with open(log_file, "rb") as log_f:
                last = str(read_last_line(log_f))

        assert f"md_propionic_acid_20.imd found" in last
        assert os.getcwd() == f"{path}/cpaeds/tests/test_data"
        assert os.path.isdir('aeds') == True

        os.rmdir('aeds')
        os.chdir(path)

def test_check_input_systems_wrong():
        """Sample test, will pass if sys.exit for missing input parameter works."""
        path = os.getcwd()
        settings = load_config_yaml(
                config= f'{path}/cpaeds/tests/test_data/wrong_test_settings.yaml')
    
        system = SetupSystem(settings)
        with pytest.raises(SystemExit) as excinfo:
                SetupSystem.run_checks(system)

        assert excinfo.value.code == f"Error changing to output folder directory."

def test_check_input_simulation_wrong():
        """Sample test, will pass if sys.exit for missing input parameter works."""
        path = os.getcwd()
        settings = load_config_yaml(
                config= f'{path}/cpaeds/tests/test_data/wrong_test_settings2.yaml')
    
        system = SetupSystem(settings)
        with pytest.raises(SystemExit) as excinfo:
                SetupSystem.run_checks(system)

        assert excinfo.value.code == f"Error missing simulation parameter."