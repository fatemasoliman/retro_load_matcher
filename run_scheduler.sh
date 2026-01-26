#!/bin/bash

# Load Scheduler Runner
# Activates virtual environment and runs the scheduler

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy
else
    source venv/bin/activate
fi

# Run the scheduler with provided arguments
python schedule_loads.py "$@"
