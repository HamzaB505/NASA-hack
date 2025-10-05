#!/usr/bin/env python3
"""
Test script to verify the new API endpoints for telescope-specific figures
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(url, description):
    """Test an API endpoint and print results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ Status: {response.status_code} OK")
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"✗ Status: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server")
        print("   Make sure the server is running: python backend/main.py")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print("\n" + "="*60)
    print("API Endpoint Test Suite")
    print("="*60)
    
    tests = [
        (f"{BASE_URL}/api/confusion_matrix/kepler", "KEPLER Confusion Matrix"),
        (f"{BASE_URL}/api/confusion_matrix/tess", "TESS Confusion Matrix"),
        (f"{BASE_URL}/api/feature_importance/kepler", "KEPLER Feature Importance"),
        (f"{BASE_URL}/api/feature_importance/tess", "TESS Feature Importance"),
        (f"{BASE_URL}/api/analytics?telescope=kepler", "KEPLER Analytics"),
        (f"{BASE_URL}/api/analytics?telescope=tess", "TESS Analytics"),
    ]
    
    results = []
    for url, description in tests:
        results.append(test_endpoint(url, description))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
