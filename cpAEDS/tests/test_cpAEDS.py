"""
Unit and regression test for the cpAEDS package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import cpAEDS


def test_cpAEDS_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cpAEDS" in sys.modules
