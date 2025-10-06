# Tesseract OCR Installer Script
$url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.5.0.20241111.exe"
$output = "$env:TEMP\tesseract-installer.exe"

Write-Host "Downloading Tesseract OCR installer..." -ForegroundColor Green
try {
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
    Write-Host "Downloaded to: $output" -ForegroundColor Green
    Write-Host "Starting installer..." -ForegroundColor Yellow
    Write-Host "IMPORTANT: During installation, make sure to select 'Add to PATH' option!" -ForegroundColor Yellow
    Start-Process -FilePath $output -Wait
    Write-Host "Installation complete!" -ForegroundColor Green
    Write-Host "Verifying installation..." -ForegroundColor Green

    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    # Test if tesseract is available
    try {
        $tesseractVersion = & tesseract --version 2>&1
        Write-Host "Tesseract installed successfully!" -ForegroundColor Green
        Write-Host $tesseractVersion
    } catch {
        Write-Host "Tesseract may require a system restart or manual PATH configuration." -ForegroundColor Yellow
        Write-Host "Default installation path: C:\Program Files\Tesseract-OCR" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error during download or installation: $_" -ForegroundColor Red
}
