"""
Unit tests for prediction module
"""
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.models.predict import (
    BackorderPredictor,
    load_model,
    predict_stock_availability,
    predict_with_probability
)


def test_predictor_class_exists():
    """Test that BackorderPredictor class exists"""
    assert BackorderPredictor is not None


def test_load_model_function_exists():
    """Test that load_model function exists"""
    assert callable(load_model)


def test_predict_stock_availability_function_exists():
    """Test that predict_stock_availability function exists"""
    assert callable(predict_stock_availability)


def test_predict_with_probability_function_exists():
    """Test that predict_with_probability function exists"""
    assert callable(predict_with_probability)


def test_predictor_initialization():
    """Test that predictor can be initialized"""
    # Should handle missing model gracefully
    predictor = BackorderPredictor(model_path='nonexistent.pkl')
    assert predictor is not None
    assert predictor.model is None  # Model shouldn't load if file doesn't exist


def test_predict_returns_error_when_no_model():
    """Test that prediction returns error when model not loaded"""
    predictor = BackorderPredictor(model_path='nonexistent.pkl')
    result = predictor.predict_single([1.0, 2.0, 3.0])
    assert 'error' in result or result['prediction'] == -1


@pytest.mark.skipif(
    not os.path.exists('saved_models/final_model.pkl'),
    reason="Trained model not available"
)
def test_predictor_with_real_model():
    """Test predictor with actual trained model"""
    predictor = BackorderPredictor()
    
    # Model should load successfully
    assert predictor.model is not None
    
    # Should have feature information
    if predictor.n_features:
        # Create test features
        test_features = [100.0] * predictor.n_features
        
        # Make prediction
        result = predictor.predict_single(test_features)
        
        # Check result structure
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert result['prediction'] in [0, 1]
        assert result['prediction_label'] in ['Available', 'Backorder']


@pytest.mark.skipif(
    not os.path.exists('saved_models/final_model.pkl'),
    reason="Trained model not available"
)
def test_batch_prediction():
    """Test batch prediction functionality"""
    predictor = BackorderPredictor()
    
    if predictor.model is not None and predictor.n_features:
        # Create multiple test samples
        test_batch = [[100.0] * predictor.n_features for _ in range(3)]
        
        # Make batch prediction
        results = predictor.predict_batch(test_batch)
        
        # Check results
        assert len(results) == 3
        assert all('prediction' in r for r in results)


def test_standalone_predict_without_model():
    """Test standalone predict function without model"""
    result = predict_stock_availability([1.0, 2.0], None)
    assert result == -1


def test_get_model_info_without_model():
    """Test get_model_info when no model loaded"""
    predictor = BackorderPredictor(model_path='nonexistent.pkl')
    info = predictor.get_model_info()
    assert 'error' in info
