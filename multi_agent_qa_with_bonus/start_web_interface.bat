@echo off
echo.
echo ========================================
echo  Multi-Agent QA System Web Interface
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import flask, flask_socketio" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install flask flask-socketio eventlet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Create necessary directories
if not exist "history" mkdir history
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "config" mkdir config

echo.
echo Starting Multi-Agent QA System...
echo.
echo 🚀 Web Interface will be available at:
echo    http://localhost:5000
echo.
echo 📋 Available pages:
echo    • Dashboard:     http://localhost:5000/
echo    • Setup API:     http://localhost:5000/setup  
echo    • Execute Tests: http://localhost:5000/execute
echo    • View History:  http://localhost:5000/history
echo.
echo 💡 Press Ctrl+C to stop the server
echo.

REM Start the Flask application
python app.py

pause
