import os
import sys
from pathlib import Path
import requests
import json

def test_gemini():
    try:
        api_key = "AIzaSyBTDWNA7KFxsv6yDS3s9OVb8IEUmX00VE4"
        if not api_key:
            print("Error: No API key provided")
            return False
            
        print(f"Testing Gemini API connection...")
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts":[{
                    "text": "Hello! Please introduce yourself briefly."
                }]
            }]
        }
        
        print("Sending test request...")
        response = requests.post(
            f"{url}?key={api_key}",
            headers=headers,
            json=data
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini()

# 1. Run the script:
#    python config/test_gemini.py

# 2. You should see output like:
#    Testing Gemini API connection...
#    Sending test request...
#    Response status code: 200
#    Response content: { ... }

# 3. If you get status code 200 and a sensible response, your API key and endpoint are working.
#    If you get 401/403 or another error, check your API key and network.