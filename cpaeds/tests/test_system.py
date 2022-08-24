"""
Unit and regression test for the SetupSystem class.
"""

# Import package, test suite, and other packages as needed
import sys
import os
import pytest
from cpaeds.utils import load_config_yaml
from cpaeds.system import SetupSystem


def test_check_input_settings():
    """Sample test, will pass so long as all inputs are given in settings.yaml."""
    path = os.getcwd()
    settings = load_config_yaml(
            config= f'{path}/cpaeds/tests/test_data/test_settings.yaml')
    
    system = SetupSystem(settings)
    SetupSystem.check_input_settings(system)
