"""Quick test with optimized settings"""
import requests
import json
import time

print("=" * 60)
print("Testing API with Optimized Settings")
print("=" * 60)

# Test 1: Fast query (no tools)
print("\n1. Testing /query-fast endpoint (no tools, fastest)...")
start = time.time()
try:
    response = requests.post(
        "http://localhost:8000/query-fast",
        json={"query": "What is a warrior character?"},
        timeout=15
    )
    elapsed = time.time() - start
    print(f"   ✅ Response in {elapsed:.2f}s")
    if response.status_code == 200:
        data = response.json()
        print(f"   Response preview: {data['response'][:100]}...")
    else:
        print(f"   Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Regular query with reduced steps
print("\n2. Testing /query endpoint with max_steps=3, timeout=20...")
start = time.time()
try:
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": "Create a warrior named Juan with brave trait",
            "max_steps": 3,
            "timeout": 20
        },
        timeout=25
    )
    elapsed = time.time() - start
    print(f"   ✅ Response in {elapsed:.2f}s")
    if response.status_code == 200:
        data = response.json()
        print(f"   Processing time: {data.get('processing_time', 'N/A'):.2f}s")
        print(f"   Max steps used: {data.get('max_steps_used', 'N/A')}")
        if 'warning' in data:
            print(f"   ⚠️  Warning: {data['warning']}")
    else:
        print(f"   Status: {response.status_code}")
        print(f"   Error: {response.text}")
except requests.exceptions.Timeout:
    print(f"   ❌ Timeout after {time.time() - start:.2f}s")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Connection check
print("\n3. Checking connection status...")
try:
    response = requests.post("http://localhost:8000/check-connection", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Server: {data.get('server_status', 'unknown')}")
        print(f"   Message: {data['message']}")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Tests complete!")
print("=" * 60)
