#!/bin/bash

# Launch Load Scheduler Web App
# Runs the Streamlit web application

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy matplotlib streamlit plotly kaleido
else
    source venv/bin/activate
fi

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing..."
    pip install streamlit plotly kaleido
fi

echo "=========================================="
echo "Starting Load Scheduler Web App..."
echo "=========================================="
echo ""
echo "The app will open in your browser automatically."
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py
