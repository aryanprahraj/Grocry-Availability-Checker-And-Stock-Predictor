# api.py
from flask import Flask, request, jsonify
import logging
import os

# --- Placeholder Imports and Constants (Will be replaced by Kashish's files later) ---
# NOTE: We assume Kashish's prediction logic will be located here:
# from src.models.predict import load_model, predict_stock_availability

# Define constants for prediction logic we expect to be available
MODEL_FILE_PATH = 'saved_models/final_model.pkl'
# NOTE: This should match the number of features after preprocessing (RFE selection)
# Current preprocessing uses 10 features by default - coordinate with data team
EXPECTED_FEATURE_COUNT = 10  # Updated to match preprocessing RFE output

# --- Setup and Initialization ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global variable to hold the loaded ML model
GLOBAL_MODEL = None

# Placeholder function for loading the model (Will be replaced by Kashish's import)
def load_model_placeholder():
    """Mocks the model loading for initial API testing."""
    if os.path.exists(MODEL_FILE_PATH):
        logging.info(f"Placeholder: Found model file at {MODEL_FILE_PATH}. Assuming load successful.")
        # In the final version, this will load the model object. For now, a string placeholder is fine.
        return "MOCK_LOADED_MODEL" 
    else:
        logging.error(f"Error: Model file not found at {MODEL_FILE_PATH}. Cannot proceed with deployment setup.")
        return None

# Placeholder function for making a prediction (Will be replaced by Kashish's import)
def predict_stock_availability_placeholder(features, model):
    """Mocks the prediction logic."""
    # This mock logic assumes a product is 'Available' if the first feature is 0,
    # and 'Backorder' if the first feature is 1 (just for basic testing).
    if model and features:
        mock_prediction_value = 1 if features[0] > 0.5 else 0
        return mock_prediction_value
    return -1 # Error state

# Load the model on startup
GLOBAL_MODEL = load_model_placeholder()


# --- API Endpoint Definition ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive product features (17) and return stock prediction.
    Expected JSON input: {"features": [f1, f2, f3, ..., f17]}
    """
    
    if GLOBAL_MODEL is None:
        return jsonify({'error': 'Model failed to load on startup. Check model file path.'}), 500
        
    data = request.get_json(silent=True)
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid request format. Expected JSON with a "features" key.'}), 400
        
    features = data.get('features')
    
    # Input validation: check for 17 features
    if not isinstance(features, list) or len(features) != EXPECTED_FEATURE_COUNT:
        return jsonify({
            'error': f'Invalid features. Expected {EXPECTED_FEATURE_COUNT} features, got {len(features) if isinstance(features, list) else "non-list"}.'
        }), 400

    # Make prediction using the placeholder (or real) function
    try:
        # In final code, we'll use the imported function: predict_stock_availability(features, GLOBAL_MODEL)
        prediction_value = predict_stock_availability_placeholder(features, GLOBAL_MODEL)
        
        # Map integer prediction back to a meaningful label
        prediction_label = "Went on Backorder (Stock Out)" if prediction_value == 1 else "Available"

        # Return the result
        return jsonify({
            'product_status': prediction_label,
            'status_code': prediction_value, # 0 = Available, 1 = Backorder
            'message': 'Stock status predicted successfully.'
        })
    except Exception as e:
        logging.error(f"Prediction processing failed: {e}")
        return jsonify({'error': f'Prediction processing failed due to an internal error: {e}'}), 500


if __name__ == '__main__':
    logging.info("Starting Flask API server...")
    # This configuration is suitable for local development/testing
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
