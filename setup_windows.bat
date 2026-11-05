@echo off
setlocal
python --version
if errorlevel 1 (
  echo Python is not installed or not on PATH. Install Python 3.11 first.
  exit /b 1
)
python -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install torch torchvision
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo Setup complete. Activate with: .venv\Scripts\activate
 echo Then download models with: python scripts\download_models.py --models grounding-dino dinov2
endlocal
