# api.py
from flask import Flask, request, jsonify
import logging
import os

# Import the actual prediction module (Aditya's implementation)
from src.models.predict import BackorderPredictor

# Define constants
MODEL_FILE_PATH = 'saved_models/final_model.pkl'
# This matches the preprocessing RFE output (10 features selected)
EXPECTED_FEATURE_COUNT = 10

# --- Setup and Initialization ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global predictor instance
GLOBAL_PREDICTOR = None

# Load the predictor on startup
try:
    GLOBAL_PREDICTOR = BackorderPredictor(model_path=MODEL_FILE_PATH)
    if GLOBAL_PREDICTOR.model is not None:
        logging.info(f"✅ Model loaded successfully with {GLOBAL_PREDICTOR.n_features} features")
    else:
        logging.error(f"❌ Model failed to load from {MODEL_FILE_PATH}")
except Exception as e:
    logging.error(f"❌ Error initializing predictor: {e}")
    GLOBAL_PREDICTOR = None


# --- API Endpoint Definition ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive product features (10) and return stock prediction.
    Expected JSON input: {"features": [f1, f2, f3, ..., f10]}
    
    The 10 features should be:
    1. sku
    2. national_inv
    3. forecast_3_month
    4. forecast_6_month
    5. forecast_9_month
    6. sales_3_month
    7. sales_6_month
    8. sales_9_month
    9. perf_6_month_avg
    10. perf_12_month_avg
    """
    
    if GLOBAL_PREDICTOR is None or GLOBAL_PREDICTOR.model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model file exists.',
            'model_path': MODEL_FILE_PATH
        }), 500
        
    data = request.get_json(silent=True)
    
    if not data or 'features' not in data:
        return jsonify({
            'error': 'Invalid request format. Expected JSON with a "features" key.',
            'example': {'features': [12345, 500.0, 1200.0, 2400.0, 3600.0, 800.0, 1600.0, 2400.0, 0.95, 0.92]}
        }), 400
        
    features = data.get('features')
    
    # Input validation: check for 10 features
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_COUNT:
        return jsonify({
            'error': f'Invalid features. Expected {EXPECTED_FEATURE_COUNT} features, got {len(features) if isinstance(features, list) else "non-list"}.',
            'expected_features': GLOBAL_PREDICTOR.feature_names if GLOBAL_PREDICTOR.feature_names else 'Unknown'
        }), 400

    # Make prediction using BackorderPredictor
    try:
        result = GLOBAL_PREDICTOR.predict_single(features)
        
        if result.get('status') == 'error':
            return jsonify({
                'error': result.get('error', 'Prediction failed'),
                'prediction_value': -1
            }), 500
        
        # Return comprehensive prediction result
        return jsonify({
            'success': True,
            'product_status': result['prediction_label'],
            'status_code': result['prediction'],  # 0 = Available, 1 = Backorder
            'backorder_probability': result['backorder_probability'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level'],
            'message': 'Stock status predicted successfully.'
        })
    except Exception as e:
        logging.error(f"Prediction processing failed: {e}")
        return jsonify({
            'error': f'Prediction processing failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    logging.info("Starting Flask API server...")
    # This configuration is suitable for local development/testing
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
