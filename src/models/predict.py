"""
Prediction Script for Grocery Availability
Provides core prediction logic for the API and applications
"""

import pickle
import numpy as np
import pandas as pd
import os
from typing import Union, List, Dict, Optional
from pathlib import Path

# Default paths
MODEL_PATH = 'saved_models/final_model.pkl'
METADATA_PATH = 'saved_models/final_model_metadata.pkl'


class BackorderPredictor:
    """
    Main predictor class for backorder predictions.
    Handles model loading, validation, and predictions.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.n_features = None
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Loads the saved machine learning model and metadata.
        
        Args:
            path (str, optional): Custom path to model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if path is None:
            path = self.model_path
            
        try:
            # Load model
            with open(path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"✅ Model loaded successfully from {path}")
            
            # Try to load metadata
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as file:
                    self.metadata = pickle.load(file)
                self.feature_names = self.metadata.get('feature_names', [])
                self.n_features = self.metadata.get('n_features', None)
                print(f"✅ Model metadata loaded")
                print(f"   Trained: {self.metadata.get('trained_date', 'Unknown')}")
                print(f"   Accuracy: {self.metadata.get('validation_accuracy', 0):.4f}")
                print(f"   Features: {self.n_features}")
            else:
                print(f"⚠️ No metadata found. Predictions will work but with limited info.")
                
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {path}")
            print(f"   Please run: python src/models/train.py")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def validate_features(self, features: Union[list, np.ndarray, pd.DataFrame]) -> bool:
        """
        Validates input features.
        
        Args:
            features: Input features (list, array, or DataFrame)
            
        Returns:
            bool: True if valid, False otherwise
        """
        if self.model is None:
            print("❌ Model not loaded!")
            return False
        
        # Convert to array for validation
        if isinstance(features, pd.DataFrame):
            feature_array = features.values
        elif isinstance(features, list):
            feature_array = np.array(features)
        else:
            feature_array = features
            
        # Check dimensions
        if len(feature_array.shape) == 1:
            feature_array = feature_array.reshape(1, -1)
        
        # Validate feature count
        if self.n_features and feature_array.shape[1] != self.n_features:
            print(f"❌ Expected {self.n_features} features, got {feature_array.shape[1]}")
            if self.feature_names:
                print(f"   Expected features: {self.feature_names}")
            return False
            
        return True
    
    def predict_single(self, features: list) -> Dict:
        """
        Makes a prediction on a single product instance.
        
        Args:
            features (list): A list of features for a single product.
                           Must match the number of features used in training.
        
        Returns:
            dict: Prediction result with class, probability, and interpretation
                  Example: {
                      'prediction': 0,
                      'prediction_label': 'Available',
                      'backorder_probability': 0.15,
                      'confidence': 0.85,
                      'risk_level': 'Low'
                  }
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': -1
            }
        
        # Validate features
        if not self.validate_features(features):
            return {
                'error': 'Invalid features',
                'prediction': -1
            }
        
        try:
            # Convert to numpy array with correct shape
            input_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_array)[0]
                backorder_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.0
                confidence = float(max(probabilities))
            else:
                # If no probability available, use binary prediction
                backorder_prob = float(prediction)
                confidence = 1.0
            
            # Determine risk level
            if backorder_prob < 0.3:
                risk_level = 'Low'
            elif backorder_prob < 0.6:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            return {
                'prediction': int(prediction),
                'prediction_label': 'Backorder' if prediction == 1 else 'Available',
                'backorder_probability': round(backorder_prob, 4),
                'confidence': round(confidence, 4),
                'risk_level': risk_level,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': -1,
                'status': 'error'
            }
    
    def predict_batch(self, features_list: Union[List[list], np.ndarray, pd.DataFrame]) -> List[Dict]:
        """
        Makes predictions on multiple product instances.
        
        Args:
            features_list: Multiple feature sets (list of lists, array, or DataFrame)
        
        Returns:
            list: List of prediction dictionaries
        """
        if self.model is None:
            return [{'error': 'Model not loaded', 'prediction': -1}]
        
        # Convert to appropriate format
        if isinstance(features_list, pd.DataFrame):
            feature_array = features_list.values
        elif isinstance(features_list, list):
            feature_array = np.array(features_list)
        else:
            feature_array = features_list
        
        # Validate
        if not self.validate_features(feature_array):
            return [{'error': 'Invalid features', 'prediction': -1}]
        
        results = []
        for i, features in enumerate(feature_array):
            result = self.predict_single(features.tolist())
            result['index'] = i
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Returns information about the loaded model.
        
        Returns:
            dict: Model information including metadata
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        info = {
            'model_loaded': True,
            'model_type': str(type(self.model).__name__),
            'model_path': self.model_path,
        }
        
        if self.metadata:
            info.update({
                'trained_date': self.metadata.get('trained_date', 'Unknown'),
                'validation_accuracy': self.metadata.get('validation_accuracy', 'Unknown'),
                'n_features': self.metadata.get('n_features', 'Unknown'),
                'feature_names': self.metadata.get('feature_names', []),
                'training_samples': self.metadata.get('n_training_samples', 'Unknown'),
            })
        
        return info


# Standalone functions for backward compatibility and simple usage

def load_model(path: str = MODEL_PATH):
    """
    Loads the saved machine learning model (standalone function).
    
    Args:
        path (str): Path to the model file
        
    Returns:
        model: Loaded model object or None
    """
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print(f"✅ Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {path}")
        print(f"   Run 'python src/models/train.py' to train the model first.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None


def predict_stock_availability(features: list, model) -> int:
    """
    Makes a prediction on a single instance (standalone function).
    
    Args:
        features (list): List of features for a single product
        model: The loaded ML model object
    
    Returns:
        int: Predicted class (0 for Available, 1 for Backorder, -1 for error)
    """
    if model is None:
        return -1
    
    try:
        # Convert input to numpy array with correct shape
        input_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        return int(prediction)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return -1


def predict_with_probability(features: list, model) -> Dict:
    """
    Makes a prediction with probability scores (standalone function).
    
    Args:
        features (list): List of features for a single product
        model: The loaded ML model object
    
    Returns:
        dict: Prediction result with probabilities
    """
    if model is None:
        return {'prediction': -1, 'error': 'Model not loaded'}
    
    try:
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Backorder' if prediction == 1 else 'Available'
        }
        
        # Add probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_array)[0]
            result['backorder_probability'] = float(probabilities[1])
            result['available_probability'] = float(probabilities[0])
        
        return result
        
    except Exception as e:
        return {'prediction': -1, 'error': str(e)}


if __name__ == '__main__':
    """
    Test the prediction functionality
    """
    print("=" * 80)
    print("🔮 BACKORDER PREDICTION - TEST")
    print("=" * 80)
    
    # Initialize predictor
    predictor = BackorderPredictor()
    
    if predictor.model is not None:
        # Show model info
        print("\n📊 Model Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            if key != 'feature_names':
                print(f"   {key}: {value}")
        
        # Example prediction (dummy data matching feature count)
        print("\n🧪 Testing Prediction...")
        
        if predictor.n_features:
            # Create dummy features
            test_features = [100.0] * predictor.n_features  # Dummy values
            
            result = predictor.predict_single(test_features)
            
            print(f"\n✅ Prediction Result:")
            print(f"   Status: {result['prediction_label']}")
            print(f"   Backorder Probability: {result.get('backorder_probability', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
        else:
            print("⚠️ Cannot test without feature information")
            print("   Train the model first: python src/models/train.py")
    else:
        print("\n❌ Model not loaded. Cannot run predictions.")
        print("   Please train the model first: python src/models/train.py")
    
    print("\n" + "=" * 80)
