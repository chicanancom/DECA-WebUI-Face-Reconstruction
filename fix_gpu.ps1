Write-Host "Cleaning up old processes..." -ForegroundColor Yellow
taskkill /F /IM python.exe /T 2>$null

Write-Host "Updating critical dependencies..." -ForegroundColor Cyan
pip install --upgrade typing-extensions pillow "numpy<2.0.0"

Write-Host "Reinstalling Torch with CUDA support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-cache-dir

Write-Host "Configuring CUDA for compilation..." -ForegroundColor Cyan
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
if (Test-Path $cudaBin) {
    $env:PATH = "$cudaBin;" + $env:PATH
    Write-Host "Found CUDA at $cudaBin" -ForegroundColor Green
}

Write-Host "Reinstalling llama-cpp-python with CUDA support (using pre-built wheel)..." -ForegroundColor Cyan
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu129 --force-reinstall --no-cache-dir

Write-Host "Done! Try running main.py again." -ForegroundColor Green
