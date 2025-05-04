import os
import vertexai
from google.cloud import aiplatform

# Set credentials path
credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dogwood-boulder-458622-t1-839b69a572b8.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

def test_connection():
    try:
        print(f"Using credentials from: {credentials_path}")
        print("Initializing Vertex AI...")
        vertexai.init(project="dogwood-boulder-458622-t1", location="us-central1")
        print("Successfully initialized Vertex AI")
        
        # Try to list models
        print("Attempting to list models...")
        aiplatform.Model.list()
        print("Successfully listed models")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()