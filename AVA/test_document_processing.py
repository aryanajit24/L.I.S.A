#!/usr/bin/env python
# test_document_processing.py - AVA Document Processing Test Script

import os
import sys
import logging
from pathlib import Path
import time
import traceback
from dotenv import load_dotenv

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ava_test_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import required modules
from src.data_processor import DataProcessor
from config.config import ModelConfig

def check_tesseract_installation():
    """Check if Tesseract OCR is properly installed and in PATH, or in common locations"""
    try:
        # First try using the explicit path set in data_processor
        import pytesseract
        
        # Common Tesseract installation paths to check
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\yasin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            r'C:\Users\yasin\AppData\Local\Tesseract-OCR\tesseract.exe'
        ]
        
        # Check each path and use the first one that exists
        tesseract_path = None
        for path in tesseract_paths:
            if os.path.exists(path):
                tesseract_path = path
                logger.info(f"✅ Found Tesseract OCR at: {tesseract_path}")
                break
        
        if tesseract_path:
            # Set the path directly for future use
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test if it works
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract OCR is correctly installed (version {version})")
            return True
        else:
            # Try default path as a last resort
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract OCR is correctly installed via PATH (version {version})")
            return True
            
    except Exception as e:
        logger.error(f"❌ Tesseract OCR test failed: {str(e)}")
        logger.error("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.error("And make sure it's in your system PATH")
        return False

def check_google_apis():
    """Check if Google API credentials are properly set up"""
    # Load environment variables
    load_dotenv()
    
    # Check Google Cloud credentials
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        creds_path = os.path.join(project_root, "dogwood-boulder-458622-t1-839b69a572b8.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    if not os.path.exists(creds_path):
        logger.error(f"❌ Google Cloud credentials not found at {creds_path}")
        return False
    
    logger.info(f"✅ Google Cloud credentials found at {creds_path}")
    
    # Check Gemini API key
    config = ModelConfig()
    if not config.api_key:
        logger.error("❌ Missing Gemini API key in ModelConfig")
        return False
    
    logger.info(f"✅ Gemini API key is configured")
    return True

def check_installed_packages():
    """Check if required packages are installed"""
    required_packages = [
        "PyMuPDF", "pdf2image", "pytesseract", "google-generativeai", 
        "google-cloud-vision", "google-cloud-videointelligence"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_").split(".")[0])
            logger.info(f"✅ Package {package} is installed")
        except ImportError:
            missing.append(package)
            logger.error(f"❌ Package {package} is not installed")
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def create_test_pdf():
    """Create a simple test PDF to verify processing"""
    try:
        from fpdf import FPDF
        
        # Create test directory
        test_dir = os.path.join(project_root, "test_files")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create simple PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="AVA Test Document", ln=True, align="C")
        
        # Add content
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt="This is a test document for AVA's PDF processing capabilities.\n\nIt contains simple text to verify that OCR and text extraction are working correctly.")
        
        # Add some test data
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Test Data Table:", ln=True)
        
        # Add simple table
        data = [
            ["ID", "Name", "Value"],
            ["1", "Alpha", "100"],
            ["2", "Beta", "200"],
            ["3", "Gamma", "300"]
        ]
        
        # Table
        line_height = 10
        col_widths = [20, 60, 30]
        for row in data:
            for i, item in enumerate(row):
                if i == 0:
                    pdf.set_font("Arial", "B", 12)
                else:
                    pdf.set_font("Arial", size=12)
                pdf.cell(col_widths[i], line_height, txt=item, border=1)
            pdf.ln(line_height)
        
        # Save the PDF
        file_path = os.path.join(test_dir, "test_document.pdf")
        pdf.output(file_path)
        logger.info(f"✅ Created test PDF at {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"❌ Failed to create test PDF: {str(e)}")
        logger.error("Installing fpdf: pip install fpdf")
        return None

def test_pdf_processing():
    """Test PDF processing capabilities"""
    try:
        # Create a test file if needed
        pdf_path = create_test_pdf()
        if not pdf_path:
            # Try using an existing PDF if available
            test_dir = os.path.join(project_root, "data", "raw", "documents")
            pdfs = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]
            if pdfs:
                pdf_path = os.path.join(test_dir, pdfs[0])
                logger.info(f"Using existing PDF: {pdf_path}")
            else:
                logger.error("No test PDF available")
                return False
        
        # Initialize DataProcessor
        logger.info("Initializing DataProcessor...")
        config = ModelConfig()
        processor = DataProcessor(config)
        
        # Process PDF file
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        result = processor.process_document(pdf_path)
        processing_time = time.time() - start_time
        
        # Analyze results
        if 'error' in result:
            logger.error(f"❌ PDF processing failed: {result['error']}")
            return False
        
        # Standard processor results
        if result.get('format') == 'pdf':
            logger.info(f"✅ Standard PDF processing succeeded in {processing_time:.2f} seconds:")
            logger.info(f"  - Total pages: {result.get('total_pages', 0)}")
            text_length = len(result.get('extracted_text', ''))
            logger.info(f"  - Extracted text length: {text_length} characters")
            
            # Check if text extraction was successful
            if text_length < 10:
                logger.warning("⚠️ Very little text extracted, possible OCR issue")
        
        # Try Gemini processing if available
        if hasattr(processor, 'gemini_client') and processor.gemini_client:
            try:
                logger.info("Testing Gemini PDF processing...")
                gemini_result = processor._process_pdf_gemini(pdf_path)
                if gemini_result:
                    matches = gemini_result.get('gemini_matches', [])
                    logger.info(f"✅ Gemini PDF processing succeeded, found {len(matches)} matches")
                    text_elements = len(gemini_result.get('text_elements', []))
                    table_elements = len(gemini_result.get('table_elements', []))
                    image_elements = len(gemini_result.get('image_elements', []))
                    logger.info(f"  - Text elements: {text_elements}")
                    logger.info(f"  - Table elements: {table_elements}")
                    logger.info(f"  - Image elements: {image_elements}")
            except Exception as e:
                logger.error(f"❌ Gemini PDF processing failed: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ PDF processing test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests and provide comprehensive diagnostics"""
    logger.info("==== AVA Document Processing Test ====")
    
    # Run each test and track results
    test_results = {}
    
    # Check dependencies
    test_results["tesseract"] = check_tesseract_installation()
    test_results["packages"] = check_installed_packages()
    test_results["google_apis"] = check_google_apis()
    
    # Test document processing
    test_results["pdf_processing"] = test_pdf_processing()
    
    # Summary
    logger.info("\n==== Test Summary ====")
    for test, result in test_results.items():
        status = "PASSED ✅" if result else "FAILED ❌"
        logger.info(f"{test.replace('_', ' ').title()}: {status}")
    
    if all(test_results.values()):
        logger.info("\n✅ All tests passed! AVA should be able to process documents correctly.")
    else:
        logger.info("\n❌ Some tests failed. Please fix the issues reported above.")
        logger.info("Most common issues:")
        logger.info("1. Tesseract OCR not installed or not in PATH")
        logger.info("2. Missing dependencies (run 'pip install -r requirements.txt')")
        logger.info("3. Google API credentials not properly configured")
    
    return all(test_results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)