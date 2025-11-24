"""Test the timeout fixes with various scenarios"""
import requests
import json
import time

print("=" * 70)
print("Testing Timeout Fixes")
print("=" * 70)

base_url = "http://localhost:8000"

def test_endpoint(name, method, url, data=None, timeout=30):
    """Helper to test an endpoint"""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")
    start = time.time()
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=data, timeout=timeout)
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ SUCCESS in {elapsed:.2f}s")
            result = response.json()
            
            # Print relevant info
            if 'processing_time' in result:
                print(f"   Processing time: {result['processing_time']:.2f}s")
            if 'used_fallback' in result:
                print(f"   Used fallback: {result['used_fallback']}")
            if 'warning' in result:
                print(f"   ⚠️  Warning: {result['warning']}")
            if 'response' in result:
                print(f"   Response preview: {result['response'][:100]}...")
            elif 'raw_response' in result:
                resp = result['raw_response']
                if isinstance(resp, str):
                    print(f"   Response preview: {resp[:100]}...")
                elif isinstance(resp, dict) and 'output' in resp:
                    print(f"   Response preview: {resp['output'][:100]}...")
            
            return True, elapsed, result
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False, elapsed, None
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"❌ TIMEOUT after {elapsed:.2f}s")
        return False, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ ERROR after {elapsed:.2f}s: {str(e)[:100]}")
        return False, elapsed, None

# Test 1: Health check
test_endpoint("Health Check", "GET", f"{base_url}/health", timeout=5)

# Test 2: Status check
test_endpoint("Status Check", "GET", f"{base_url}/status", timeout=5)

# Test 3: Fast query (should work)
test_endpoint(
    "Fast Query (no tools)",
    "POST",
    f"{base_url}/query-fast",
    {"query": "What is a warrior?"},
    timeout=10
)

# Test 4: Regular query with minimal settings
test_endpoint(
    "Regular Query (max_steps=2, timeout=10)",
    "POST",
    f"{base_url}/query",
    {
        "query": "Create a character named Test",
        "max_steps": 2,
        "timeout": 10
    },
    timeout=15
)

# Test 5: Regular query with default settings
test_endpoint(
    "Regular Query (defaults: max_steps=3, timeout=15)",
    "POST",
    f"{base_url}/query",
    {"query": "Create a warrior named Juan with brave trait"},
    timeout=20
)

# Test 6: Connection check
test_endpoint("Connection Check", "POST", f"{base_url}/check-connection", timeout=5)

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)
print("\nRecommendations:")
print("- If regular queries still timeout, use /query-fast endpoint")
print("- Reduce max_steps to 1-2 for faster responses")
print("- Reduce timeout to 5-10 seconds")
print("- Check if MCP server is responding slowly")
