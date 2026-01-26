#!/bin/bash

# Schedule Visualization Runner
# Activates virtual environment and runs the visualization script

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy matplotlib
else
    source venv/bin/activate
fi

# Check if matplotlib is installed
if ! python -c "import matplotlib" 2>/dev/null; then
    echo "Installing matplotlib..."
    pip install matplotlib
fi

# Run the visualization with provided arguments
python visualize_schedule.py "$@"
