import os
import sys
import time
import traceback
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from dotenv import load_dotenv

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.model import AVAModel
from src.data_processor import DataProcessor
from config.config import ModelConfig

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="AVA API Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AVA_PORT", 8000)),
                        help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default=os.environ.get("AVA_HOST", "127.0.0.1"),
                        help="Host to bind the server to (default: 127.0.0.1)")
    return parser.parse_args()

# Initialize FastAPI app
app = FastAPI(title="AVA API", description="Text-based Language Model API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize configuration
config = ModelConfig()

# Set Google Cloud credentials
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not creds_path:
    creds_path = os.path.join(project_root, "dogwood-boulder-458622-t1-839b69a572b8.json")
if not os.path.exists(creds_path):
    raise FileNotFoundError(f"Google Cloud credentials not found at {creds_path}")
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
logger.info(f"Using Google Cloud credentials from: {creds_path}")

# Initialize global components
model = None
is_initialized = False
data_processor = DataProcessor(config)  # Pass config to data processor

def initialize_components():
    """Initialize Vertex AI components"""
    global model, is_initialized
    
    try:
        logger.info("Initializing AVA model...")
        model = AVAModel(config)
        is_initialized = True
        logger.info("Successfully initialized AVA model")
        return True
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.post("/api/chat")
async def chat_with_ava(message: str = Form(...)):
    """Chat endpoint that handles text messages"""
    global is_initialized, model
    
    try:
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        logger.info(f"Received chat message: {message[:50]}...")
        
        # Try to initialize if not already initialized
        if not is_initialized:
            logger.info("System not initialized, attempting initialization...")
            is_initialized = initialize_components()
        
        if not is_initialized or not model:
            error_msg = "AVA is not properly initialized. Please check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
        
        # Generate response
        try:
            logger.info("Generating response...")
            response = model.generate_response(message)
            logger.info(f"Generated response: {response[:50]}...")
            return {"status": "success", "response": response}
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/process_document")
async def process_document(file: UploadFile = File(...)):
    """Process a document (PDF, DOCX) and extract text/content."""
    try:
        # Save uploaded file to a temporary location
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process the document
        result = data_processor.process_document(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/analyze_document")
async def analyze_document(file: UploadFile = File(...), prompt: str = Form(None)):
    global is_initialized, model
    try:
        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        supported_formats = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf'],
            'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'spreadsheets': ['.csv', '.xlsx', '.xls'],
            'videos': ['.mp4', '.avi', '.mov']
        }

        # Validate file format
        valid_format = any(ext in formats for formats in supported_formats.values())
        if not valid_format:
            return {
                "status": "error",
                "message": f"Unsupported file format: {ext}. Supported formats: {', '.join([f for formats in supported_formats.values() for f in formats])}"
            }

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded file to a temporary location with unique name
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{file.filename}")
        try:
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)

            # Initialize components if needed
            if not is_initialized:
                is_initialized = initialize_components()

            if not is_initialized or not model:
                raise HTTPException(status_code=503, detail="Model not initialized")

            # Process the file based on its type
            try:
                doc_result = data_processor.process_document(temp_path)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                return {"status": "error", "message": f"Error processing document: {str(e)}"}

            if "error" in doc_result:
                return {"status": "error", "message": doc_result["error"]}

            # Generate response using the model with document data
            default_prompt = "Please analyze this content and provide key insights, main topics, and important information."
            try:
                response = model.generate_response(
                    prompt or default_prompt,
                    document_data=doc_result
                )
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return {"status": "error", "message": f"Error generating response: {str(e)}"}

            return {
                "status": "success",
                "response": response,
                "metadata": {
                    "filename": file.filename,
                    "file_type": ext[1:].upper(),
                    "total_pages": doc_result.get('total_pages', 1),
                    "processed_text_length": len(doc_result.get('extracted_text', '')),
                }
            }

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {str(e)}")

    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/status")
async def get_model_status():
    """Get the current status of the model"""
    global is_initialized, model
    
    try:
        if not is_initialized:
            is_initialized = initialize_components()
        
        return {
            "status": "ready" if is_initialized and model else "unavailable",
            "model_type": "gemini-pro",  # Changed from text-bison to gemini-pro
            "message": "AVA is ready" if is_initialized and model else "AVA is initializing"
        }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "model_type": "gemini-pro",  # Changed from text-bison to gemini-pro
            "message": f"Error: {str(e)}"
        }

# Mount static files directory
static_dir = os.path.join(project_root, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """Serve the frontend index.html at root."""
    return FileResponse(os.path.join(static_dir, "index.html"))

# 1. Start your backend:
#    python src/main.py
#    or
#    uvicorn src.main:app --reload --port 8000

# 2. Open your browser and go to:
#    http://127.0.0.1:8000/
#    Try sending a message in the chat interface.

# 3. Alternatively, test the backend directly:
#    curl -X POST -F "message=hello" http://127.0.0.1:8000/api/chat

# 4. Check the backend terminal/logs for:
#    - "Successfully initialized AVA model"
#    - "Generated response: ..." (when you send a message)
#    - No errors about missing API key or credentials

# 5. If you still see "AVA is not properly initialized":
#    - Check for errors in the backend logs.
#    - Make sure your API key in config/config.py is valid and not expired.
#    - Make sure your credentials file exists and is correct.
#    - Try running the Gemini API test script (see below).

def handle_shutdown(signum, frame):
    """Handle shutdown gracefully"""
    logger.info("Received shutdown signal, cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Parse command line arguments
        args = parse_args()
        
        # Initialize components on startup
        if initialize_components():
            logger.info("Server initialized successfully")
        else:
            logger.warning("Server initialization failed - will retry on first request")
            
        # Start the server with debug logging
        logger.info(f"Starting server on {args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise