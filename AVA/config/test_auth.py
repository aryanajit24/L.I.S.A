import os
from google.cloud import aiplatform
from google.cloud import storage
import json

def test_auth():
    # Get credentials path
    creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "dogwood-boulder-458622-t1-839b69a572b8.json")
    
    if not os.path.exists(creds_path):
        print(f"Error: Credentials file not found at {creds_path}")
        return False
    
    # Set credentials environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    try:
        # Load and print service account info
        with open(creds_path) as f:
            creds = json.load(f)
            print(f"\nService Account Email: {creds.get('client_email')}")
            print(f"Project ID: {creds.get('project_id')}")
        
        print("\nTesting Storage API access...")
        storage_client = storage.Client()
        try:
            buckets = list(storage_client.list_buckets())
            print(f"Successfully listed {len(buckets)} buckets")
        except Exception as e:
            print(f"Storage API error: {str(e)}")
        
        print("\nTesting Vertex AI API access...")
        aiplatform.init(project=creds.get('project_id'), location="us-central1")
        
        # List available models
        print("\nListing available models...")
        models = aiplatform.Model.list()
        print(f"Found {len(list(models))} models")
        
        # List model endpoints
        print("\nListing model endpoints...")
        endpoints = aiplatform.Endpoint.list()
        print(f"Found {len(list(endpoints))} endpoints")
        
        return True
        
    except Exception as e:
        print(f"\nError during authentication test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting authentication test...")
    test_auth()