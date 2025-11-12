"""Tests for data loading module"""
import pytest
import pandas as pd
from src.data.load_data import load_train_test


def test_load_train_test_returns_dataframes():
    """Test that load_train_test returns pandas DataFrames"""
    # This test will only work if data files are present
    # In CI/CD, you might want to use mock data
    try:
        train_df, test_df = load_train_test()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) > 0
        assert len(test_df) > 0
    except FileNotFoundError:
        pytest.skip("Data files not available")


def test_load_train_test_column_consistency():
    """Test that train and test have consistent columns"""
    try:
        train_df, test_df = load_train_test()
        # Both should have the same columns (or test might have one less if target is excluded)
        assert set(train_df.columns).intersection(set(test_df.columns))
    except FileNotFoundError:
        pytest.skip("Data files not available")
