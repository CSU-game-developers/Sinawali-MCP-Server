"""Test Groq LLM response time"""
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

print("Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Using a different, faster model
    temperature=0.0,
    max_retries=2,
)

print("Sending test query...")
start = time.time()

try:
    response = llm.invoke("Hello, please respond with just 'Hi' and nothing else.")
    elapsed = time.time() - start
    print(f"✅ Response received in {elapsed:.2f}s")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
