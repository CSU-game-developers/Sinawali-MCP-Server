"""Diagnostic script to check API status"""
import requests
import time

print("Testing API endpoints...")

# Test health endpoint
print("\n1. Testing /health endpoint...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.Timeout:
    print("   ❌ Timeout - Server not responding")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test status endpoint
print("\n2. Testing /status endpoint...")
try:
    response = requests.get("http://localhost:8000/status", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.Timeout:
    print("   ❌ Timeout - Server not responding")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test check-connection endpoint
print("\n3. Testing /check-connection endpoint...")
try:
    response = requests.post("http://localhost:8000/check-connection", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.Timeout:
    print("   ❌ Timeout - Server not responding")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\nDiagnostics complete!")
