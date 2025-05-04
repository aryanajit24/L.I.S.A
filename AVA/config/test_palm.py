import os
import vertexai
from vertexai.generative_models import GenerativeModel

# Set credentials path
creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dogwood-boulder-458622-t1-839b69a572b8.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

def test_palm():
    try:
        print(f"Using credentials from: {creds_path}")
        print("Initializing Vertex AI...")
        
        # Initialize Vertex AI with project and location
        vertexai.init(
            project="dogwood-boulder-458622-t1",
            location="us-central1"
        )
        
        print("Creating model instance...")
        
        # Try Gemini Pro model
        try:
            print("\nTrying Gemini Pro model...")
            model = GenerativeModel("gemini-pro")
            chat = model.start_chat()
            response = chat.send_message(
                "Say hello and introduce yourself briefly.",
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 256,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            print("Success with Gemini Pro!")
            print("Model Response:", response.text)
            return True
        except Exception as e:
            print(f"Failed with Gemini Pro: {str(e)}")
            
        print("\nAll model attempts failed.")
        return False
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    test_palm()