"""
Test script for the Flask API /predict endpoint
"""
import requests
import json

# API endpoint
url = "http://127.0.0.1:5001/predict"

# Sample test data - 10 features as expected
# Features: sku, national_inv, forecast_3_month, forecast_6_month, forecast_9_month,
#           sales_3_month, sales_6_month, sales_9_month, perf_6_month_avg, perf_12_month_avg
test_cases = [
    {
        "name": "Test Case 1: High inventory, good performance",
        "features": [12345, 500.0, 1200.0, 2400.0, 3600.0, 800.0, 1600.0, 2400.0, 0.95, 0.92]
    },
    {
        "name": "Test Case 2: Low inventory, low sales",
        "features": [67890, 50.0, 100.0, 200.0, 300.0, 80.0, 160.0, 240.0, 0.65, 0.70]
    },
    {
        "name": "Test Case 3: Medium inventory, medium performance",
        "features": [11111, 250.0, 600.0, 1200.0, 1800.0, 400.0, 800.0, 1200.0, 0.80, 0.85]
    }
]

print("=" * 80)
print("Testing Flask API /predict endpoint")
print("=" * 80)

for test_case in test_cases:
    print(f"\n{test_case['name']}")
    print("-" * 80)
    
    payload = {"features": test_case["features"]}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"   Product Status: {result.get('product_status')}")
            print(f"   Status Code: {result.get('status_code')}")
            print(f"   Backorder Probability: {result.get('backorder_probability', 0):.2%}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Risk Level: {result.get('risk_level')}")
        else:
            print(f"❌ Error: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to API at {url}")
        print("   Make sure the Flask server is running (python api.py)")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
