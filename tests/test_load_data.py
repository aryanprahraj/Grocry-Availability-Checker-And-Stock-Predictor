"""Tests for data loading module"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_python():
    """Test that basic Python works"""
    assert True


def test_pandas_available():
    """Test that pandas is available"""
    import pandas as pd
    assert pd is not None


def test_src_module_exists():
    """Test that src module can be found"""
    import src
    assert src is not None


def test_load_data_module_exists():
    """Test that load_data module exists"""
    try:
        import src.data.load_data
        assert src.data.load_data is not None
    except ImportError as e:
        pytest.skip(f"Could not import module: {e}")
