"""Test the agent API with a simple query"""
import requests
import json

url = "http://localhost:8000/query"
payload = {
    "query": "Create a warrior character named Juan with brave and proud traits. Keep it brief."
}

print("Sending query to agent...")
try:
    response = requests.post(url, json=payload, timeout=70)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Response received:")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Error: {response.text}")
except requests.exceptions.Timeout:
    print("❌ Request timed out after 70 seconds")
except Exception as e:
    print(f"❌ Error: {e}")
