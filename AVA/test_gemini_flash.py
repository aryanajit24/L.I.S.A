import os
import sys
import logging
import json
import time
import requests
from dotenv import load_dotenv
from config.config import ModelConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gemini_flash_api():
    """Test the Gemini 2.0 Flash API connection directly"""
    logger.info("==== Testing Gemini 2.0 Flash API ====")
    
    # Load config
    config = ModelConfig()
    api_key = config.api_key
    
    if not api_key:
        logger.error("No API key found in configuration")
        return False
    
    logger.info(f"Testing Gemini model: {config.model_name}")
    
    # Check if model name is in the proper format
    if not config.model_name.startswith("models/"):
        logger.warning(f"Model name '{config.model_name}' might not be in the correct format. "
                      f"It should start with 'models/'")
    
    try:
        # Extract the model name from the config
        model_name = config.model_name
        
        # Construct the API URL
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/{model_name}:generateContent?key={api_key}"
        
        logger.info(f"API URL: {base_url}/{model_name}:generateContent")
        
        # Prepare request payload
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{
                    "text": "Hello! Please verify that you're the Gemini 2.0 Flash model and explain your capabilities briefly."
                }]
            }],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_output_tokens,
                "topP": config.top_p,
                "topK": config.top_k
            }
        }
        
        # Make the API request
        logger.info("Sending request to Gemini API...")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # Calculate and log response time
        response_time = end_time - start_time
        logger.info(f"Response received in {response_time:.2f} seconds")
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    text = parts[0]["text"]
                    logger.info(f"Gemini response: {text[:200]}...")
                    logger.info("API test SUCCESSFUL ✅")
                    return True
                else:
                    logger.error("No text found in response parts")
            else:
                logger.error("No candidates found in response")
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Error response: {response.text}")
        
        return False
    
    except Exception as e:
        logger.error(f"Error testing Gemini API: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_sdk():
    """Test Gemini Flash model using the official Google SDK"""
    logger.info("\n==== Testing Gemini 2.0 Flash with Google SDK ====")
    
    try:
        import google.generativeai as genai
        from config.config import ModelConfig
        
        # Load config
        config = ModelConfig()
        
        # Configure the SDK
        genai.configure(api_key=config.api_key)
        
        # Extract model name without 'models/' prefix
        model_name = config.model_name
        if model_name.startswith("models/"):
            model_name = model_name[7:]  # Remove "models/" prefix for the GenerativeModel constructor
        
        logger.info(f"Using model name: {model_name}")
        
        # List available models
        try:
            models = genai.list_models()
            flash_models = [m.name for m in models if 'flash' in m.name.lower()]
            logger.info(f"Available Flash models: {flash_models}")
        except Exception as e:
            logger.warning(f"Could not list models: {str(e)}")
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Test with simple prompt
        start_time = time.time()
        response = model.generate_content(
            "Please tell me what Gemini model version you are, and what your capabilities are."
        )
        end_time = time.time()
        
        # Log response time
        response_time = end_time - start_time
        logger.info(f"Response received in {response_time:.2f} seconds")
        
        # Log the response
        if hasattr(response, 'text'):
            logger.info(f"Response: {response.text[:200]}...")
            logger.info("SDK test SUCCESSFUL ✅")
            return True
        else:
            logger.error("No text in response")
            return False
            
    except ImportError:
        logger.error("Google Generative AI SDK not installed")
        logger.error("Install with: pip install google-generativeai")
        return False
    except Exception as e:
        logger.error(f"Error testing with SDK: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_chat_session():
    """Test chat session functionality with Gemini Flash model"""
    logger.info("\n==== Testing Chat Session with Gemini 2.0 Flash ====")
    
    try:
        import google.generativeai as genai
        from config.config import ModelConfig
        
        # Load config
        config = ModelConfig()
        
        # Configure the SDK
        genai.configure(api_key=config.api_key)
        
        # Extract model name without 'models/' prefix
        model_name = config.model_name
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Start a chat session
        chat = model.start_chat(history=[])
        
        # Send a series of messages to test context retention
        messages = [
            "Hello, I need to ask you a series of questions to test your capabilities.",
            "What's your name and model version?",
            "Can you remember what I asked you first?",
            "What are your strengths compared to other Gemini model versions?"
        ]
        
        for i, message in enumerate(messages):
            logger.info(f"\nSending message {i+1}: {message}")
            start_time = time.time()
            response = chat.send_message(message)
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"Response time: {response_time:.2f} seconds")
            logger.info(f"Response: {response.text[:150]}...")
        
        logger.info("Chat session test SUCCESSFUL ✅")
        return True
            
    except ImportError:
        logger.error("Google Generative AI SDK not installed")
        return False
    except Exception as e:
        logger.error(f"Error in chat session test: {str(e)}")
        return False

def check_model_listing():
    """List all available Gemini models"""
    logger.info("\n==== Checking Available Gemini Models ====")
    
    try:
        import google.generativeai as genai
        from config.config import ModelConfig
        
        # Load config
        config = ModelConfig()
        
        # Configure the SDK
        genai.configure(api_key=config.api_key)
        
        # List all models
        models = genai.list_models()
        
        # Group models by type
        model_types = {}
        for model in models:
            name = model.name
            if 'gemini' in name.lower():
                type_key = None
                if 'flash' in name.lower():
                    if '2.0' in name or '2-0' in name:
                        type_key = 'gemini-2.0-flash'
                    elif '1.5' in name or '1-5' in name:
                        type_key = 'gemini-1.5-flash'
                elif 'pro' in name.lower():
                    if '2.0' in name or '2-0' in name:
                        type_key = 'gemini-2.0-pro'
                    elif '1.5' in name or '1-5' in name:
                        type_key = 'gemini-1.5-pro'
                
                if type_key:
                    if type_key not in model_types:
                        model_types[type_key] = []
                    model_types[type_key].append(name)
        
        # Log model groups
        logger.info(f"Found {len(models)} total models, grouped by type:")
        for type_name, model_names in model_types.items():
            logger.info(f"\n{type_name}:")
            for name in model_names:
                logger.info(f"  - {name}")
        
        # Check the current model
        current_model = config.model_name
        logger.info(f"\nCurrently configured model: {current_model}")
        
        # Suggest the best model
        suggested_model = None
        if 'models/gemini-2.0-flash' in model_types.get('gemini-2.0-flash', []):
            suggested_model = 'models/gemini-2.0-flash'
        elif any('flash' in name and '2.0' in name for name in model_types.get('gemini-2.0-flash', [])):
            for name in model_types.get('gemini-2.0-flash', []):
                if 'flash' in name and '2.0' in name:
                    suggested_model = name
                    break
                    
        if suggested_model and suggested_model != current_model:
            logger.info(f"\nSuggested model: {suggested_model}")
            logger.info("To use this model, update the model_name in config.py")
        
        return True
        
    except ImportError:
        logger.error("Google Generative AI SDK not installed")
        return False
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return False

def get_system_status():
    """Get system information and dependencies status"""
    logger.info("\n==== System Information ====")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check for required packages
    packages = {
        "google-generativeai": "Google Generative AI SDK",
        "requests": "HTTP requests library",
        "PyMuPDF": "PDF processing library",
        "pdf2image": "PDF to image conversion",
        "pytesseract": "OCR library",
        "nltk": "Natural Language Toolkit"
    }
    
    for package, description in packages.items():
        try:
            if package == "PyMuPDF":
                import fitz
                logger.info(f"✅ {package} ({description}): Installed")
            else:
                module = __import__(package.replace("-", "_"))
                logger.info(f"✅ {package} ({description}): Installed")
        except ImportError:
            logger.warning(f"❌ {package} ({description}): Not installed")
    
    return True

if __name__ == "__main__":
    logger.info("==== Gemini 2.0 Flash Model Test ====")
    
    # Get system status
    get_system_status()
    
    # Check model listing
    check_model_listing()
    
    # Test REST API connection
    rest_api_test = test_gemini_flash_api()
    
    # Test SDK
    sdk_test = test_with_sdk()
    
    # Test chat session
    chat_test = test_chat_session()
    
    # Overall test result
    overall_result = rest_api_test and sdk_test and chat_test
    
    if overall_result:
        logger.info("\n==== ALL TESTS PASSED ✅ ====")
        logger.info("Gemini 2.0 Flash model is working correctly")
    else:
        logger.error("\n==== SOME TESTS FAILED ❌ ====")
        logger.error("Check the logs above for details on which tests failed")
    
    logger.info("\n==== Test Complete ====")