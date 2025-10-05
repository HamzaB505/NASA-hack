#!/usr/bin/env python3
"""
Test script to verify frontend-backend connection
"""

import requests
import json

def test_backend_connection():
    """Test if the backend is running and accessible"""
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Backend is running!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running on http://localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    try:
        # Create sample CSV data
        sample_data = """koi_max_sngle_ev,koi_depth,koi_insol,koi_max_mult_ev,koi_dikco_msky,koi_insol_err2,koi_incl,koi_ror,koi_smet_err2,koi_prad_err1
0.000123,0.000045,1.2,0.000098,0.000156,0.1,89.5,0.0123,0.05,0.001"""
        
        # Prepare form data
        files = {
            'file': ('test_data.csv', sample_data, 'text/csv')
        }
        data = {
            'telescope': 'kepler',
            'model': 'logistic_regression'
        }
        
        response = requests.post("http://localhost:8000/api/predict", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Prediction endpoint working!")
            result = response.json()
            print(f"Prediction: {result.get('prediction', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 'Unknown')}")
            return True
        else:
            print(f"❌ Prediction endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing ExoPlanet AI Backend Connection...")
    print("=" * 50)
    
    # Test backend connection
    if test_backend_connection():
        print("\n🧪 Testing prediction endpoint...")
        test_prediction_endpoint()
    
    print("\n" + "=" * 50)
    print("Test completed!")
