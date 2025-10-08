# API endpoint tests
#!/usr/bin/env python3
"""
Quick API test client for Marketplace Validator
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_validate_compliant():
    """Test validation with compliant listing"""
    print("Testing compliant listing...")
    
    payload = {
        "listing": """Premium Rewards Card

Earn 2% cashback on all purchases.

Features:
- 0% intro APR for 12 months, then 18.9% APR representative (variable)
- Annual fee: £95
- Late payment fee: up to £12
- Requires good to excellent credit (score 881-999)

Subject to status and credit approval.""",
        "listing_id": "test-001",
        "lender_id": "bank-xyz"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/validate", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_validate_non_compliant():
    """Test validation with non-compliant listing"""
    print("Testing non-compliant listing...")
    
    payload = {
        "listing": "BEST CARD EVER! Guaranteed approval! No credit check required! Instant approval!"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/validate", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_batch_validation():
    """Test batch validation"""
    print("Testing batch validation...")
    
    payload = {
        "listings": [
            {
                "listing": "Card 1: 0% APR for 12 months, then 18.9% APR. Annual fee: £95.",
                "listing_id": "batch-001"
            },
            {
                "listing": "Card 2: Best rates! Guaranteed approval!",
                "listing_id": "batch-002"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/batch", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_with_curl_command():
    """Print curl command for testing"""
    print("Curl command for testing:")
    print("""
curl -X POST "http://localhost:8000/api/v1/validate" \\
  -H "Content-Type: application/json" \\
  -d '{
    "listing": "Premium Card: 0% intro APR for 12 months, then 18.9% APR representative (variable). Annual fee: £95.",
    "listing_id": "test-curl"
  }'
""")

if __name__ == "__main__":
    print(" Testing Marketplace Validator API\n")
    print("="*60 + "\n")
    
    try:
        test_health()
        test_validate_compliant()
        test_validate_non_compliant()
        test_batch_validation()
        test_with_curl_command()
        
        print(" All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print(" Error: Cannot connect to API. Is it running?")
        print("Run: uvicorn main:app --reload")
    except Exception as e:
        print(f" Error: {e}")