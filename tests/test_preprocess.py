"""Tests for preprocessing module"""
import pytest
import pandas as pd
import numpy as np


def test_imports_work():
    """Test that preprocessing imports work"""
    from src.data.preprocess import preprocess_data
    assert preprocess_data is not None


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    train_data = pd.DataFrame({
        'sku': ['SKU1', 'SKU2', 'SKU3', 'SKU4', 'SKU5'],
        'national_inv': [100, 200, 150, 300, 250],
        'potential_issue': ['Yes', 'No', 'No', 'Yes', 'No'],
        'deck_risk': ['No', 'No', 'Yes', 'No', 'Yes'],
        'oe_constraint': ['No', 'Yes', 'No', 'No', 'Yes'],
        'ppap_risk': ['No', 'No', 'No', 'Yes', 'No'],
        'stop_auto_buy': ['Yes', 'No', 'No', 'No', 'No'],
        'rev_stop': ['No', 'No', 'Yes', 'No', 'No'],
        'went_on_backorder': ['No', 'Yes', 'No', 'No', 'Yes']
    })
    
    test_data = pd.DataFrame({
        'sku': ['SKU6', 'SKU7'],
        'national_inv': [175, 225],
        'potential_issue': ['No', 'Yes'],
        'deck_risk': ['No', 'No'],
        'oe_constraint': ['Yes', 'No'],
        'ppap_risk': ['No', 'No'],
        'stop_auto_buy': ['No', 'Yes'],
        'rev_stop': ['No', 'No']
    })
    
    return train_data, test_data


def test_preprocess_data_returns_correct_shapes(sample_data):
    """Test that preprocessing returns correct output shapes"""
    from src.data.preprocess import preprocess_data
    train_df, test_df = sample_data
    X_train, y_train, X_test = preprocess_data(train_df, test_df, use_rfe=False)
    
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(test_df)
    assert X_train.shape[1] == X_test.shape[1]


def test_preprocess_data_target_encoding(sample_data):
    """Test that target variable is properly encoded"""
    from src.data.preprocess import preprocess_data
    train_df, test_df = sample_data
    X_train, y_train, X_test = preprocess_data(train_df, test_df, use_rfe=False)
    
    assert set(y_train.unique()).issubset({0, 1})
    assert y_train.dtype in [np.int64, np.int32, int]
