"""
Test script to verify the /api/dashboard/stats endpoint
"""
import requests
import json

# Test the dashboard stats endpoint
try:
    response = requests.get('http://localhost:8000/api/dashboard/stats')
    if response.status_code == 200:
        stats = response.json()
        print("✓ Dashboard stats endpoint working!")
        print("\nKEPLER Stats:")
        print(f"  Accuracy: {stats['kepler']['accuracy']}%")
        print(f"  Training Samples: {stats['kepler']['training_samples']}")
        print(f"  Test Samples: {stats['kepler']['test_samples']}")
        print(f"  Features: {stats['kepler']['features']}")
        print(f"  Status: {stats['kepler']['status']}")
        
        print("\nTESS Stats:")
        print(f"  Accuracy: {stats['tess']['accuracy']}%")
        print(f"  Training Samples: {stats['tess']['training_samples']}")
        print(f"  Test Samples: {stats['tess']['test_samples']}")
        print(f"  Features: {stats['tess']['features']}")
        print(f"  Status: {stats['tess']['status']}")
    else:
        print(f"✗ Endpoint returned status code: {response.status_code}")
        print(f"Response: {response.text}")
except requests.exceptions.ConnectionError:
    print("✗ Could not connect to server. Make sure the backend is running on http://localhost:8000")
except Exception as e:
    print(f"✗ Error: {e}")
