"""
Test the API with sample data
"""
import requests
import numpy as np
import json

# API URL
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("\n📍 Testing root endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict_valid():
    """Test prediction with valid data"""
    print("\n🔮 Testing prediction endpoint (valid data)...")
    
    # Generate random HR and respiration sequences (128 values each)
    hr_sequence = np.random.uniform(60, 120, 128).tolist()
    resp_sequence = np.random.uniform(10, 20, 128).tolist()
    
    # Send request with CORRECT format
    payload = {
        "hr_values": hr_sequence,
        "resp_values": resp_sequence
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.2%}")
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_predict_invalid():
    """Test prediction with invalid data (should fail gracefully)"""
    print("\n❌ Testing prediction with invalid data (should fail)...")
    
    # Only 3 values instead of 128
    payload = {
        "hr_values": [60, 61, 62],
        "resp_values": [12, 13, 14]
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 400  # Should fail

def test_info():
    """Test info endpoint"""
    print("\n📋 Testing info endpoint...")
    response = requests.get(f"{API_URL}/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing LSTM Stress Detection API")
    print("="*60)
    
    results = {
        "Health Check": test_health(),
        "Root Endpoint": test_root(),
        "Info Endpoint": test_info(),
        "Valid Prediction": test_predict_valid(),
        "Invalid Input": test_predict_invalid()
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)