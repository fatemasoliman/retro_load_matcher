#!/bin/bash

# Run All - Schedule and Compare
# Runs the scheduler and then generates comparison visualizations

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

echo "=========================================="
echo "STEP 1: Running scheduler..."
echo "=========================================="
echo ""

python schedule_loads.py --all-months --input inputs/loads.csv

if [ $? -ne 0 ]; then
    echo "Error: Scheduler failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 2: Generating comparisons..."
echo "=========================================="
echo ""

python compare_actual_vs_simulated.py --actual inputs/actuals.csv --simulated outputs/monthly_summary.csv

if [ $? -ne 0 ]; then
    echo "Error: Comparison failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Schedules saved to: outputs/schedules/"
echo "Visualizations saved to: outputs/visualizations/"
