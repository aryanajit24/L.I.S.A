# Project AVA

A comprehensive document processing and analysis system with support for multiple document types, OCR, and AI-powered analysis.

## Features

- **Multi-format Document Processing**: PDF, DOCX, TXT, CSV, XLSX, images, and videos
- **Advanced PDF Processing**: Extract text, images, and structure from PDFs
- **OCR Capabilities**: Extract text from images and scanned documents
- **AI-powered Analysis**: 
  - Natural Language Processing for text analysis
  - Google Cloud Vision API integration for image analysis
  - Google Cloud Video Intelligence for video content analysis
  - Gemini API integration for advanced document understanding

## Installation

### Prerequisites

- Python 3.8+ 
- Tesseract OCR
- Poppler (for PDF processing)
- (Optional) Google Cloud credentials for Vision API, Video Intelligence, and Gemini

### Automated Setup

Run the following script to automatically install and configure all dependencies:

```bash
python fix_document_processing.py
```

This script will:
1. Install required Python packages
2. Check and install Tesseract OCR
3. Check and install Poppler
4. Fix any document processing issues

### Manual Setup

1. **Install Python Packages**:

```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR**:

For Windows, run:
```
powershell -ExecutionPolicy Bypass -File InstallTesseractOCR.ps1
```

Or download and install from: https://github.com/UB-Mannheim/tesseract/wiki

3. **Install Poppler**:

For Windows, run:
```
powershell -ExecutionPolicy Bypass -File install_poppler.ps1
```

Or download from: https://github.com/oschwartz10612/poppler-windows/releases

## Configuration

1. Set up Google Cloud credentials (if using cloud services):
   - Place your Google Cloud credentials JSON file in the project root
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
     ```
     set GOOGLE_APPLICATION_CREDENTIALS=path\to\credentials.json
     ```

2. Configure Gemini API (if using):
   - Edit `config/config.py` and update your API key

## Usage

### Process Documents

```python
from src.data_processor import DataProcessor
from config.config import ModelConfig

# Initialize
config = ModelConfig()
processor = DataProcessor(config)

# Process a document
result = processor.process_document("path/to/document.pdf")
print(result['summary'])
```

### Test PDF Processing

To test the PDF document processing functionality:

```bash
python test_fixed_pdf.py
```

## Troubleshooting

### Common Issues

1. **PDF Processing Fails**:
   - Ensure Tesseract OCR and Poppler are correctly installed
   - Check PATH environment variables include Tesseract and Poppler bins
   - Run `python fix_document_processing.py` to repair the installation

2. **OCR Not Working**:
   - Verify Tesseract is installed with `tesseract --version`
   - Ensure the path in `data_processor.py` points to your Tesseract installation

3. **Google Cloud APIs Not Working**:
   - Check if your credentials are valid and have the required permissions
   - Verify your API keys have access to the required services
   - Check network connectivity if using cloud services

## System Design

- `data_processor.py`: Main document processing class
- `model.py`: ML model integration
- `config/`: Configuration settings
- `utils/`: Utility functions for data management and storage

## License

This project is proprietary and confidential. All rights reserved.

## Contributors

- Project AVA Team