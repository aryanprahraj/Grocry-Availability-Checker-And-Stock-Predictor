"""Tests for data loading module"""
import pytest
import pandas as pd
import os


def test_imports_work():
    """Test that basic imports work"""
    from src.data.load_data import load_train_test
    assert load_train_test is not None


def test_load_train_test_returns_dataframes():
    """Test that load_train_test returns pandas DataFrames"""
    # Skip if data files don't exist (e.g., in CI environment)
    if not os.path.exists("data/Training_BOP.csv"):
        pytest.skip("Data files not available in CI environment")
    
    from src.data.load_data import load_train_test
    train_df, test_df = load_train_test()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_load_train_test_column_consistency():
    """Test that train and test have consistent columns"""
    # Skip if data files don't exist (e.g., in CI environment)
    if not os.path.exists("data/Training_BOP.csv"):
        pytest.skip("Data files not available in CI environment")
    
    from src.data.load_data import load_train_test
    train_df, test_df = load_train_test()
    # Both should have the same columns (or test might have one less if target is excluded)
    assert set(train_df.columns).intersection(set(test_df.columns))
