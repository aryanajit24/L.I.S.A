# Script to install Tesseract OCR on Windows
# Run with elevated privileges (as Administrator)

# Configuration
$tesseractVersion = "5.3.1"
$tesseractUrl = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-$tesseractVersion.exe"
$installerPath = "$env:TEMP\tesseract-installer.exe"
$installDir = "C:\Program Files\Tesseract-OCR"

Write-Host "AVA - Tesseract OCR Installer" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Check if Tesseract is already installed
if (Test-Path $installDir) {
    Write-Host "Tesseract appears to be already installed at: $installDir" -ForegroundColor Green
    
    # Try to get version
    try {
        $version = & "$installDir\tesseract.exe" --version 2>$null
        if ($version) {
            Write-Host "Detected Tesseract version: $version" -ForegroundColor Green
            Write-Host "Installation will be skipped. Delete the directory if you want to reinstall."
            exit 0
        }
    } catch {}
    
    # Version detection failed, could be corrupt installation
    Write-Host "Could not verify Tesseract version." -ForegroundColor Yellow
    Write-Host "Would you like to reinstall? (Y/N)" -ForegroundColor Yellow
    $confirm = Read-Host
    if ($confirm -ne 'Y' -and $confirm -ne 'y') {
        Write-Host "Installation cancelled. Exiting."
        exit 0
    }
    
    Write-Host "Removing previous installation..." -ForegroundColor Yellow
    Remove-Item -Path $installDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Create directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

# Download Tesseract installer
Write-Host "Downloading Tesseract OCR $tesseractVersion..." -ForegroundColor Cyan
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($tesseractUrl, $installerPath)
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    Write-Host "Please download manually from: $tesseractUrl" -ForegroundColor Yellow
    Write-Host "And install with the following options:" -ForegroundColor Yellow
    Write-Host "  - Install to: $installDir" -ForegroundColor Yellow
    Write-Host "  - Select 'Additional language data' for better OCR" -ForegroundColor Yellow
    exit 1
}

# Install Tesseract
Write-Host "Installing Tesseract OCR..." -ForegroundColor Cyan
try {
    # Run installer silently with default components
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
    
    # Verify installation
    if (Test-Path "$installDir\tesseract.exe") {
        Write-Host "Tesseract OCR installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Installation may have failed. Tesseract executable not found." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Installation failed: $_" -ForegroundColor Red
    exit 1
}

# Add to PATH if not already present
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if (-not $currentPath.Contains($installDir)) {
    Write-Host "Adding Tesseract to system PATH..." -ForegroundColor Cyan
    try {
        $newPath = "$currentPath;$installDir"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Host "Tesseract added to PATH successfully!" -ForegroundColor Green
        Write-Host "NOTE: You may need to restart your terminal or applications for the PATH change to take effect." -ForegroundColor Yellow
    } catch {
        Write-Host "Failed to update PATH: $_" -ForegroundColor Red
        Write-Host "Please add $installDir to your system PATH manually." -ForegroundColor Yellow
    }
}

# Test Tesseract
Write-Host "Testing Tesseract installation..." -ForegroundColor Cyan
try {
    $version = & "$installDir\tesseract.exe" --version 2>$null
    if ($version) {
        Write-Host "Tesseract is working correctly!" -ForegroundColor Green
        Write-Host $version -ForegroundColor Green
    } else {
        Write-Host "Could not verify Tesseract installation. Please check manually." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error testing Tesseract: $_" -ForegroundColor Red
}

# Cleanup
Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Tesseract OCR installation completed." -ForegroundColor Cyan
Write-Host "You can now use Tesseract for OCR in your Python applications." -ForegroundColor Cyan
Write-Host ""
