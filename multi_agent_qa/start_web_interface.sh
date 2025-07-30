#!/bin/bash

echo ""
echo "========================================"
echo " Multi-Agent QA System Web Interface"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if required packages are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import flask, flask_socketio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install flask flask-socketio eventlet 2>/dev/null || pip install flask flask-socketio eventlet
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p history output logs config

echo ""
echo "Starting Multi-Agent QA System..."
echo ""
echo "🚀 Web Interface will be available at:"
echo "   http://localhost:5000"
echo ""
echo "📋 Available pages:"
echo "   • Dashboard:     http://localhost:5000/"
echo "   • Setup API:     http://localhost:5000/setup"  
echo "   • Execute Tests: http://localhost:5000/execute"
echo "   • View History:  http://localhost:5000/history"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
$PYTHON_CMD app.py
