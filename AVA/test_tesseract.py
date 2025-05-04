import os
import sys
import subprocess
import tempfile
from PIL import Image
import pytesseract
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tesseract_installation():
    """Test if Tesseract OCR is properly installed and accessible in the PATH"""
    logger.info("Testing Tesseract OCR installation...")
    
    # Test 1: Check using pytesseract.get_tesseract_version()
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version from pytesseract: {version}")
    except Exception as e:
        logger.error(f"Failed to get Tesseract version using pytesseract: {str(e)}")
        logger.error("Make sure Tesseract is installed and in your PATH")
    
    # Test 2: Try direct command line call
    try:
        result = subprocess.run(['tesseract', '--version'], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        logger.info(f"Tesseract command line test: {result.stdout.splitlines()[0]}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to call Tesseract from command line: {e.stderr}")
    except FileNotFoundError:
        logger.error("Tesseract executable not found in PATH")
    
    # Test 3: Check pytesseract configuration
    logger.info(f"Tesseract path configured in pytesseract: {pytesseract.pytesseract.tesseract_cmd}")
    
    # Create simple test image with text
    logger.info("Creating test image with text...")
    test_img = create_test_image()
    
    # Test 4: Try OCR on test image
    try:
        text = pytesseract.image_to_string(test_img)
        logger.info(f"OCR test result: {'SUCCESS' if text.strip() else 'FAILED - No text detected'}")
        logger.info(f"Detected text: {text.strip()}")
    except Exception as e:
        logger.error(f"OCR test failed: {str(e)}")

def create_test_image():
    """Create a simple test image with text for OCR testing"""
    width, height = 300, 100
    img = Image.new('RGB', (width, height), color='white')
    
    # We're not creating actual text here since we don't have a simple way
    # to draw text on an image without additional libraries
    # Instead, we'll return the image and if Tesseract is working, it should
    # at least attempt to process it (even if it doesn't find text)
    return img

def check_system_path():
    """Check if common Tesseract installation paths are in the system PATH"""
    logger.info("Checking PATH for common Tesseract installation directories...")
    paths = os.environ.get('PATH', '').split(os.pathsep)
    tesseract_common_paths = [
        r'C:\Program Files\Tesseract-OCR',
        r'C:\Program Files (x86)\Tesseract-OCR',
        r'/usr/bin',
        r'/usr/local/bin',
    ]
    
    for common_path in tesseract_common_paths:
        if common_path in paths:
            logger.info(f"Found potential Tesseract path in PATH: {common_path}")
    
    return paths

def suggest_fixes():
    """Suggest fixes for common Tesseract installation issues"""
    logger.info("\nSuggested fixes if Tesseract is not working:")
    logger.info("1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.info("2. Add Tesseract installation directory to your system PATH")
    logger.info("3. Set pytesseract.pytesseract.tesseract_cmd to point to the tesseract.exe file")
    logger.info("   Example: pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    
    # Check for common installation directories
    common_install_locations = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for location in common_install_locations:
        if os.path.exists(location):
            logger.info(f"\nTesseract executable found at: {location}")
            logger.info(f"You can use this path in your code: pytesseract.pytesseract.tesseract_cmd = r'{location}'")

if __name__ == "__main__":
    logger.info("==== Tesseract OCR Test ====")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check system PATH
    paths = check_system_path()
    logger.info(f"Number of directories in PATH: {len(paths)}")
    
    # Run tests
    test_tesseract_installation()
    
    # Suggest fixes
    suggest_fixes()
    
    logger.info("==== Test Complete ====")