# Script to install Poppler on Windows
# Run with elevated privileges (as Administrator)

# Configuration
$popplerVersion = "23.11.0-0"
$popplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v$popplerVersion/Release-$popplerVersion.zip"
$zipPath = "$env:TEMP\poppler.zip"
$installDir = "C:\Poppler"

Write-Host "AVA - Poppler Installer" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will install Poppler $popplerVersion for Windows"
Write-Host "Poppler is needed for PDF to image conversion and text extraction"
Write-Host ""

# Check if Poppler is already installed
if (Test-Path $installDir) {
    Write-Host "Poppler appears to be already installed at: $installDir" -ForegroundColor Green
    
    # Try to get version
    try {
        $versionOutput = & "$installDir\bin\pdftoppm.exe" -v 2>&1
        if ($versionOutput -match '\d+\.\d+\.\d+') {
            Write-Host "Detected Poppler version: $($matches[0])" -ForegroundColor Green
            Write-Host "Installation will be skipped. Delete the directory if you want to reinstall."
            exit 0
        }
    } catch {}
    
    # Version detection failed, could be corrupt installation
    Write-Host "Could not verify Poppler version." -ForegroundColor Yellow
    Write-Host "Would you like to reinstall? (Y/N)" -ForegroundColor Yellow
    $confirm = Read-Host
    if ($confirm -ne 'Y' -and $confirm -ne 'y') {
        Write-Host "Installation cancelled. Exiting."
        exit 0
    }
    
    Write-Host "Removing previous installation..." -ForegroundColor Yellow
    Remove-Item -Path $installDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Create temp directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

# Download Poppler
Write-Host "Downloading Poppler $popplerVersion..." -ForegroundColor Cyan
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($popplerUrl, $zipPath)
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    Write-Host "Please download manually from: $popplerUrl" -ForegroundColor Yellow
    Write-Host "Extract to: $installDir" -ForegroundColor Yellow
    Write-Host "And add $installDir\bin to your system PATH" -ForegroundColor Yellow
    exit 1
}

# Extract Poppler
Write-Host "Extracting Poppler files..." -ForegroundColor Cyan
try {
    # Create the installation directory
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    
    # Extract the ZIP file
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $installDir)
    Write-Host "Extraction complete!" -ForegroundColor Green
} catch {
    Write-Host "Extraction failed: $_" -ForegroundColor Red
    exit 1
}

# Fix directory structure if needed
if (-not (Test-Path "$installDir\bin\pdftoppm.exe") -and (Test-Path "$installDir\poppler-$popplerVersion\bin\pdftoppm.exe")) {
    Write-Host "Fixing directory structure..." -ForegroundColor Cyan
    # Copy from nested directory to main install directory
    Copy-Item -Path "$installDir\poppler-$popplerVersion\*" -Destination $installDir -Recurse -Force
    # Remove the now-redundant nested directory
    Remove-Item -Path "$installDir\poppler-$popplerVersion" -Recurse -Force -ErrorAction SilentlyContinue
}

# Verify installation
if (Test-Path "$installDir\bin\pdftoppm.exe") {
    Write-Host "Poppler files extracted successfully!" -ForegroundColor Green
} else {
    Write-Host "Installation may have failed. Poppler executables not found." -ForegroundColor Red
    exit 1
}

# Add to PATH if not already present
$binDir = "$installDir\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if (-not $currentPath.Contains($binDir)) {
    Write-Host "Adding Poppler to system PATH..." -ForegroundColor Cyan
    try {
        $newPath = "$currentPath;$binDir"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Host "Poppler added to PATH successfully!" -ForegroundColor Green
        Write-Host "NOTE: You may need to restart your terminal or applications for the PATH change to take effect." -ForegroundColor Yellow
    } catch {
        Write-Host "Failed to update PATH: $_" -ForegroundColor Red
        Write-Host "Please add $binDir to your system PATH manually." -ForegroundColor Yellow
    }
}

# Test Poppler
Write-Host "Testing Poppler installation..." -ForegroundColor Cyan
try {
    $versionOutput = & "$binDir\pdftoppm.exe" -v 2>&1
    if ($versionOutput) {
        Write-Host "Poppler is working correctly!" -ForegroundColor Green
        Write-Host $versionOutput -ForegroundColor Green
    } else {
        Write-Host "Could not verify Poppler installation. Please check manually." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error testing Poppler: $_" -ForegroundColor Red
}

# Cleanup
Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Poppler installation completed." -ForegroundColor Cyan
Write-Host "You can now use Poppler for PDF processing in your Python applications." -ForegroundColor Cyan
Write-Host ""