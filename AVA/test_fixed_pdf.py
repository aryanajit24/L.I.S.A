import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import after path is set up
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    
    logger.info("Successfully imported all required libraries")
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Run fix_document_processing.py to install missing dependencies")
    sys.exit(1)

# Import the fixed PDF processor
from fix_document_processing import fix_pdf_processing

def test_pdf_loading():
    """Test PDF loading with PyMuPDF (fitz)"""
    logger.info("Testing PDF loading with PyMuPDF...")
    
    # Create a test PDF file
    test_pdf = os.path.join(current_dir, "test_pdf.pdf")
    
    try:
        # Create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        
        # Add some text
        text_point = fitz.Point(50, 50)
        page.insert_text(text_point, "Hello, this is a test PDF document for AVA!")
        
        # Add a simple shape
        rect = fitz.Rect(50, 100, 200, 150)
        page.draw_rect(rect, color=(0, 0, 1), fill=(1, 0, 0))
        
        # Add an image if possible
        try:
            # Create a small image
            img = Image.new('RGB', (100, 100), color=(0, 0, 255))
            img_path = os.path.join(current_dir, "test_image.png")
            img.save(img_path)
            
            # Insert the image
            rect = fitz.Rect(50, 200, 150, 300)
            page.insert_image(rect, filename=img_path)
            
            # Clean up
            os.remove(img_path)
        except Exception as e:
            logger.warning(f"Could not add image to test PDF: {e}")
        
        # Save the document
        doc.save(test_pdf)
        doc.close()
        logger.info(f"Created test PDF at {test_pdf}")
        
        # Now open and read the document
        with fitz.open(test_pdf) as doc:
            logger.info(f"Test PDF has {len(doc)} pages")
            
            # Read text from the first page
            page = doc[0]
            text = page.get_text()
            logger.info(f"Extracted text: {text.strip()}")
            
            # Check if the page has images
            image_list = page.get_images()
            logger.info(f"Found {len(image_list)} images in the test PDF")
        
        logger.info("PDF loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"PDF loading test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_pdf):
            os.remove(test_pdf)

def test_pdf_to_image():
    """Test PDF to image conversion with pdf2image (poppler)"""
    logger.info("Testing PDF to image conversion with pdf2image...")
    
    # Create a test PDF file
    test_pdf = os.path.join(current_dir, "test_pdf.pdf")
    
    try:
        # Create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(fitz.Point(50, 50), "Testing PDF to image conversion")
        doc.save(test_pdf)
        doc.close()
        
        # Convert PDF to images
        images = convert_from_path(test_pdf)
        
        logger.info(f"Successfully converted PDF to {len(images)} images")
        
        # Save the first image
        if images:
            img_path = os.path.join(current_dir, "test_pdf_image.png")
            images[0].save(img_path)
            logger.info(f"Saved PDF image to {img_path}")
            
            # Test OCR on the image
            try:
                text = pytesseract.image_to_string(images[0])
                logger.info(f"OCR result: {text.strip()}")
                logger.info("Tesseract OCR is working correctly")
            except Exception as e:
                logger.error(f"OCR test failed: {e}")
            
            # Clean up
            os.remove(img_path)
        
        logger.info("PDF to image conversion test passed!")
        return True
        
    except Exception as e:
        logger.error(f"PDF to image conversion test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_pdf):
            os.remove(test_pdf)

def test_opencv():
    """Test OpenCV image processing"""
    logger.info("Testing OpenCV image processing...")
    
    try:
        # Create a test image
        img_path = os.path.join(current_dir, "test_opencv.png")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Add some text and shapes for testing
        cv2.putText(img, "OpenCV Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.circle(img, (150, 150), 30, (0, 0, 255), -1)
        
        # Save the image
        cv2.imwrite(img_path, img)
        logger.info(f"Created test OpenCV image at {img_path}")
        
        # Read the image back
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate the percentage of edge pixels
        edge_percentage = np.count_nonzero(edges) / edges.size
        logger.info(f"Edge percentage: {edge_percentage:.4f}")
        
        # Calculate image blurriness (lower is sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blurriness = 1.0 / (laplacian.var() + 1e-5)  # Avoid division by zero
        logger.info(f"Image blurriness: {blurriness:.4f}")
        
        # Test face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            logger.warning("Failed to load face cascade classifier")
        else:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            logger.info(f"Found {len(faces)} faces in test image")
        
        logger.info("OpenCV image processing test passed!")
        
        # Clean up
        os.remove(img_path)
        return True
        
    except Exception as e:
        logger.error(f"OpenCV image processing test failed: {e}")
        return False

def test_pdf_processing():
    """Test the fixed PDF processing implementation."""
    print("\n==== Testing Fixed PDF Processing ====")
    
    # Initialize the fixed data processor
    FixedDataProcessor = fix_pdf_processing()
    processor = FixedDataProcessor()
    
    # Test with sample PDF
    test_pdf = os.path.join('test_files', 'test_document.pdf')
    if not os.path.exists(test_pdf):
        print(f"Error: Test PDF not found at {test_pdf}")
        return False
    
    print(f"Processing PDF: {test_pdf}")
    start_time = time.time()
    
    try:
        # Process the PDF with our fixed implementation
        result = processor.process_document(test_pdf)
        
        # Check the results
        if result.get("success", False):
            print("✅ PDF processing successful")
            print(f"  - Pages: {result.get('metadata', {}).get('pages', 0)}")
            print(f"  - Text extracted: {len(result.get('extracted_text', ''))} characters")
            if result.get('metadata', {}).get('title'):
                print(f"  - Title: {result.get('metadata', {}).get('title')}")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
            return True
        else:
            print("❌ PDF processing failed")
            if "error" in result:
                print(f"  - Error: {result['error']}")
            return False
    
    except Exception as e:
        print(f"❌ Error during PDF processing: {str(e)}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    # Check PyMuPDF
    try:
        import fitz
        print("✅ PyMuPDF (fitz) installed")
    except ImportError:
        print("❌ PyMuPDF not installed")
        missing_deps.append("PyMuPDF")
    
    # Check pdf2image
    try:
        import pdf2image
        print("✅ pdf2image installed")
    except ImportError:
        print("❌ pdf2image not installed")
        missing_deps.append("pdf2image")
    
    # Check pytesseract
    try:
        import pytesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR installed (version {version})")
        except Exception:
            print("❌ Tesseract OCR not properly installed or not in PATH")
            missing_deps.append("Tesseract OCR (executable)")
    except ImportError:
        print("❌ pytesseract not installed")
        missing_deps.append("pytesseract")
    
    return missing_deps

def main():
    print("==== AVA Fixed Document Processing Test ====")
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print("\nMissing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing dependencies using:")
        print("  python fix_document_processing.py")
        
        proceed = input("\nContinue with available dependencies? (y/n): ")
        if proceed.lower() not in ['y', 'yes']:
            print("Exiting...")
            return
    
    # Run the PDF processing test
    success = test_pdf_processing()
    
    if success:
        print("\n🎉 Fixed PDF processing is working correctly!")
        print("You can now use this implementation in your application.")
    else:
        print("\n⚠️ The test was not successful.")
        print("Please make sure all dependencies are installed correctly:")
        print("1. Run 'python fix_document_processing.py'")
        print("2. Install Tesseract OCR if not already installed")
        print("3. Install Poppler if not already installed")

if __name__ == "__main__":
    main()
