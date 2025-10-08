#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://localhost:8000"

print("1. Testing health endpoint...")
r = requests.get(f"{BASE_URL}/health")
print(f"Status: {r.status_code}\n")

print("2. Testing validation...")
payload = {
    "listing": "Premium Rewards Card: Earn 2% cashback on all purchases. 0% intro APR for 12 months, then 18.9% APR representative (variable). Annual fee: £95. Late payment fee: up to £12. Requires good to excellent credit (score 881-999). Subject to status and credit approval"
}
r = requests.post(f"{BASE_URL}/api/v1/validate", json=payload)
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))