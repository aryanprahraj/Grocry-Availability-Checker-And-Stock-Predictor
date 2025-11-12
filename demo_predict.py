"""
Demo script for prediction functionality
Shows how to use the prediction module
"""
import sys
sys.path.insert(0, '.')

from src.models.predict import BackorderPredictor, load_model, predict_stock_availability
import numpy as np

print("=" * 80)
print("🔮 PREDICTION MODULE DEMO")
print("=" * 80)

# Method 1: Using the BackorderPredictor class (Recommended)
print("\n📦 Method 1: Using BackorderPredictor Class")
print("-" * 80)

predictor = BackorderPredictor()

if predictor.model is not None:
    # Get model information
    print("\n📊 Model Information:")
    info = predictor.get_model_info()
    print(f"   Model Type: {info.get('model_type', 'Unknown')}")
    print(f"   Trained: {info.get('trained_date', 'Unknown')}")
    print(f"   Accuracy: {info.get('validation_accuracy', 'Unknown')}")
    print(f"   Features Required: {info.get('n_features', 'Unknown')}")
    
    if info.get('feature_names'):
        print(f"\n   Feature Names:")
        for i, name in enumerate(info['feature_names'], 1):
            print(f"      {i}. {name}")
    
    # Example 1: Single Prediction
    print("\n\n🎯 Example 1: Single Product Prediction")
    print("-" * 80)
    
    if predictor.n_features:
        # Create sample features (in real scenario, these would be actual product data)
        sample_features = [
            12345,      # sku
            500.0,      # national_inv
            1200.0,     # forecast_3_month
            2400.0,     # forecast_6_month
            3600.0,     # forecast_9_month
            800.0,      # sales_3_month
            1600.0,     # sales_6_month
            2400.0,     # sales_9_month
            0.95,       # perf_6_month_avg
            0.92        # perf_12_month_avg
        ]
        
        print(f"Input Features ({len(sample_features)} values):")
        if predictor.feature_names:
            for name, value in zip(predictor.feature_names, sample_features):
                print(f"   {name}: {value}")
        
        result = predictor.predict_single(sample_features)
        
        print(f"\n✅ Prediction Result:")
        print(f"   Status: {result.get('prediction_label', 'Unknown')}")
        print(f"   Backorder Probability: {result.get('backorder_probability', 0):.2%}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Risk Level: {result.get('risk_level', 'Unknown')}")
        
        # Example 2: Batch Prediction
        print("\n\n🎯 Example 2: Batch Prediction (Multiple Products)")
        print("-" * 80)
        
        # Create multiple sample products
        batch_features = [
            sample_features,  # Product 1
            [i * 1.5 for i in sample_features],  # Product 2 (scaled up)
            [i * 0.5 for i in sample_features],  # Product 3 (scaled down)
        ]
        
        results = predictor.predict_batch(batch_features)
        
        print(f"Predicted {len(results)} products:\n")
        for i, result in enumerate(results, 1):
            print(f"Product {i}:")
            print(f"   Status: {result.get('prediction_label', 'Unknown')}")
            print(f"   Backorder Prob: {result.get('backorder_probability', 0):.2%}")
            print(f"   Risk: {result.get('risk_level', 'Unknown')}")
            print()
    
    else:
        print("⚠️ Feature information not available")

else:
    print("\n❌ Model not loaded!")
    print("   Train the model first: python src/models/train.py")

# Method 2: Using standalone functions
print("\n\n📦 Method 2: Using Standalone Functions")
print("-" * 80)

model = load_model()

if model is not None:
    # Simple prediction
    if predictor.n_features:
        test_features = [100.0] * predictor.n_features
        
        prediction = predict_stock_availability(test_features, model)
        print(f"\n✅ Simple Prediction: {prediction}")
        print(f"   (0 = Available, 1 = Backorder, -1 = Error)")

# API Integration Example
print("\n\n🌐 Example 3: API Integration Pattern")
print("-" * 80)
print("""
# This is how Aryan can integrate in the API:

from src.models.predict import BackorderPredictor

# Initialize once (at app startup)
predictor = BackorderPredictor()

# In API endpoint
@app.post("/predict")
def predict_endpoint(features: list):
    result = predictor.predict_single(features)
    return {
        "success": result.get('status') == 'success',
        "prediction": result.get('prediction_label'),
        "probability": result.get('backorder_probability'),
        "risk_level": result.get('risk_level')
    }

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch_endpoint(features_list: list):
    results = predictor.predict_batch(features_list)
    return {"predictions": results}
""")

print("\n" + "=" * 80)
print("✨ Demo Complete!")
print("=" * 80)
print("\n💡 Tips:")
print("   - Use BackorderPredictor class for production")
print("   - Standalone functions available for simple use cases")
print("   - All functions handle errors gracefully")
print("   - Batch predictions more efficient for multiple items")
