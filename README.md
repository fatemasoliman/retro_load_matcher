# Load Scheduler

A Python script that optimizes load scheduling across multiple vehicles to maximize total revenue while respecting time and travel constraints.

## Features

- **Multi-Objective Optimization**: Balances revenue maximization with deadmile minimization
- **Deadmile Minimization**: Reduces empty miles traveled between loads
- **Time Constraint Handling**: Ensures loads don't overlap and pickup times are respected
- **Travel Time Calculation**: Uses Haversine distance formula to calculate travel time between locations
- **Configurable Speed**: Adjust average vehicle speed for travel time calculations
- **CSV Output**: Generates detailed schedule with vehicle assignments in organized folders
- **Visual Gantt Charts**: Generate timeline visualizations of your schedule
- **Multi-Month Processing**: Process all months separately with comprehensive statistics

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib (for visualization)

## Installation

The dependencies are automatically installed in a virtual environment when you use the run script.

## Usage

### Basic Usage

```bash
./run_scheduler.sh <number_of_vehicles>
```

Example:
```bash
./run_scheduler.sh 5
```

### Advanced Options

```bash
./run_scheduler.sh <number_of_vehicles> [OPTIONS]

Options:
  --input FILE          Input CSV file path or URL (default: loads.csv)
  --output FILE         Output CSV file for schedule (default: schedules/schedule_output.csv)
  --speed SPEED         Average vehicle speed in km/h (default: 60)
  --month MONTH         Filter loads by specific month (e.g., "December 2025")
  --all-months          Process all months separately and generate summary statistics
  --deadmile-weight W   Weight for deadmile minimization (0-1, default: 0.3)
```

Examples:
```bash
# Schedule for 10 vehicles with 80 km/h average speed
./run_scheduler.sh 10 --speed 80

# Process all months separately
./run_scheduler.sh 10 --speed 80 --all-months

# Process specific month only
./run_scheduler.sh 5 --month "December 2025" --speed 80

# Use data from URL
./run_scheduler.sh 10 --input "http://example.com/loads.csv" --speed 80

# Use custom input/output files
./run_scheduler.sh 5 --input my_loads.csv --output my_schedule.csv

# Combine multiple options
./run_scheduler.sh 8 --input loads.csv --output schedule.csv --speed 70
```

### Running Directly with Python

If you prefer to run the Python script directly:

```bash
source venv/bin/activate
python schedule_loads.py <number_of_vehicles> [OPTIONS]
```

## Visualization

After generating a schedule, you can create visual Gantt charts to see the timeline for each vehicle.

### Basic Visualization

```bash
./visualize.sh
```

This creates a Gantt chart showing:
- Each vehicle's timeline
- Load assignments (blue bars with revenue labels)
- Travel time between loads (gray bars)
- Total loads and revenue statistics

The output is saved as `visualizations/schedule_gantt.png`.

### Visualization Options

```bash
./visualize.sh [OPTIONS]

Options:
  --input FILE      Input schedule CSV file (default: schedule_output.csv)
  --output FILE     Output image file (default: visualizations/schedule_gantt.png)
  --speed SPEED     Average vehicle speed in km/h (default: 60)
  --detailed        Create detailed timeline view with more info
  --summary         Print text summary of schedule
  --width WIDTH     Figure width in inches (default: 16)
  --height HEIGHT   Figure height in inches (default: 10)
  --dpi DPI         Image resolution (default: 100)
```

### Visualization Examples

**Basic Gantt chart:**
```bash
./visualize.sh
```

**Detailed timeline with labels:**
```bash
./visualize.sh --detailed --output detailed_schedule.png
```

**High-resolution chart:**
```bash
./visualize.sh --dpi 300 --width 20 --height 12
```

**Visualize custom schedule with summary:**
```bash
./visualize.sh --input my_schedule.csv --output my_chart.png --summary --speed 80
```

### Complete Workflow Example

```bash
# 1. Generate schedule for 5 vehicles
./run_scheduler.sh 5 --speed 80

# 2. Visualize the schedule
./visualize.sh --speed 80 --detailed --summary

# 3. Open the visualization
open visualizations/schedule_gantt.png  # macOS
# or: xdg-open visualizations/schedule_gantt.png  # Linux
# or: start visualizations/schedule_gantt.png  # Windows
```

## Actual vs Simulated Comparison

Compare your actual revenue performance against the simulated scheduler results to validate the model and identify optimization opportunities.

### Prepare Actual Data

Create an `actual.csv` file with your actual revenue data:

```csv
month,actual_gb_per_vehicle
January 2025,22500
February 2025,18000
March 2025,20000
...
```

The script automatically detects columns containing "month" and "gb"/"revenue"/"per_vehicle".

### Run Comparison

```bash
# Basic comparison (creates bar chart)
./compare.sh

# With line plot
./compare.sh --line-plot

# Custom files
./compare.sh --actual my_actual.csv --simulated monthly_summary.csv

# High resolution output
./compare.sh --dpi 300 --width 16 --height 10
```

### Comparison Options

```bash
./compare.sh [OPTIONS]

Options:
  --actual FILE     Input CSV with actual data (default: actual.csv)
  --simulated FILE  Input CSV with simulated data (default: monthly_summary.csv)
  --output FILE     Output image file (default: visualizations/actual_vs_simulated.png)
  --line-plot       Also create a line plot comparison
  --width WIDTH     Figure width in inches (default: 14)
  --height HEIGHT   Figure height in inches (default: 8)
  --dpi DPI         Image resolution (default: 150)
```

### Output

The comparison generates:
- **Bar chart**: Side-by-side comparison with difference percentages
- **Line plot** (optional): Trend analysis over time
- **Console table**: Detailed month-by-month comparison
- **Summary statistics**: Average performance and variance

Example output:
```
ACTUAL VS SIMULATED COMPARISON
================================================================================
Month          Actual (SAR /veh)  Simulated (SAR /veh)  Difference (SAR )  Difference (%)
January 2025        22,500.00          24,187.58        1,687.58            +7.5
February 2025       18,000.00          16,707.89       -1,292.11            -7.2
...

Summary Statistics:
Average Actual: SAR 23,461.54/vehicle
Average Simulated: SAR 23,849.65/vehicle
Average Difference: SAR 388.11/vehicle
Average Difference %: +1.7%
```

This helps you:
- Validate the scheduler's accuracy
- Identify months where simulated results differ significantly
- Understand if the optimization is realistic
- Calibrate model parameters for better accuracy

## Output Folder Structure

The scheduler organizes all output files into folders:

```
retro_load_matcher/
├── schedules/              # All schedule CSV files
│   ├── schedule_output.csv
│   ├── schedule_january_1,_2025.csv
│   ├── schedule_february_1,_2025.csv
│   └── ...
├── visualizations/         # All visualization images
│   ├── schedule_gantt.png
│   ├── actual_vs_simulated.png
│   └── ...
└── monthly_summary.csv    # Summary statistics across all months
```

This structure keeps your workspace organized and makes it easy to:
- Find schedules for specific months
- Manage visualization outputs
- Share or archive results

## Input CSV Format

The input CSV file should have the following columns:

- `key`: Unique identifier for the load
- `id`: Load ID
- `entity`: Shipper/entity name
- `shipper_price`: Revenue from the load
- `distance`: Distance in kilometers
- `pickup_date`: When the load should be picked up
- `pickup_lat`: Pickup latitude
- `pickup_lng`: Pickup longitude
- `dropoff_lat`: Drop-off latitude
- `dropoff_lng`: Drop-off longitude
- `number_of_addresses`: Number of stops
- `median_duration_days`: Duration of load in days

## Output Format

The output CSV contains:

- `vehicle_id`: Vehicle assigned to the load
- `load_sequence`: Order of load for this vehicle
- `load_key`: Unique load identifier
- `load_id`: Load ID
- `entity`: Shipper name
- `revenue`: Revenue from this load
- `pickup_date`: Pickup date and time
- `dropoff_date`: Drop-off date and time
- `pickup_lat`, `pickup_lng`: Pickup coordinates
- `dropoff_lat`, `dropoff_lng`: Drop-off coordinates
- `duration_hours`: Load duration in hours

## Multi-Month Processing

When using the `--all-months` flag, the scheduler processes each month separately and generates:

### Individual Month Schedules
- Separate CSV files for each month in the `schedules/` folder (e.g., `schedules/schedule_december_1,_2025.csv`)
- Each file contains the optimized schedule for that specific month

### Monthly Summary Statistics
A `monthly_summary.csv` file containing comprehensive statistics for all months:

- `month`: Month name and year
- `num_vehicles`: Number of vehicles used
- `num_loads`: Total loads assigned
- `total_loaded_km`: Distance traveled while carrying loads
- `total_unloaded_km`: Empty travel distance between loads
- `total_km`: Total distance (loaded + unloaded)
- `loaded_ratio`: Percentage of distance that is revenue-generating
- `total_revenue`: Total revenue for the month
- `revenue_per_vehicle`: Mean revenue per vehicle
- `median_revenue_per_vehicle`: Median revenue per vehicle
- `variance_revenue_per_vehicle`: Variance in revenue across vehicles
- `avg_revenue_per_load`: Average revenue per load

This allows you to:
- Compare performance across different months
- Identify seasonal patterns
- Optimize fleet size per month
- Track efficiency metrics over time

## Algorithm

The scheduler uses a greedy algorithm with multi-objective optimization:

1. **Load Processing**: Sorts loads chronologically by pickup date
2. **Vehicle Compatibility**: For each load, finds compatible vehicles that can:
   - Be available before the pickup time
   - Reach the pickup location in time from their last drop-off
3. **Composite Scoring**: Assigns each load using a weighted score that balances:
   - **Revenue Distribution** (70% default): Assigns to vehicles with lower revenue to balance workload
   - **Deadmile Minimization** (30% default): Prioritizes vehicles closer to pickup location to reduce empty miles
4. **Optimization Goals**:
   - Maximize total revenue across all vehicles
   - Minimize unloaded distance (deadmiles) between loads
   - Balance revenue distribution across the fleet

### Deadmile Weight Parameter

The `--deadmile-weight` parameter (0-1) controls the balance:
- **0.0**: Pure revenue balancing (ignore proximity)
- **0.3** (default): Balanced approach (70% revenue, 30% deadmiles)
- **0.5**: Equal weight to both objectives
- **1.0**: Pure deadmile minimization (assign to nearest vehicle)

## Constraints and Limitations

### Time and Distance Constraints
- Loads must be picked up at their exact pickup time (not before or after)
- Vehicles must have enough time to travel from the previous drop-off to the next pickup
- Loads cannot overlap on the same vehicle
- Travel time is calculated using the Haversine distance formula

### Fleet Size Limitations
- **Fixed fleet size**: The scheduler assumes a constant number of vehicles across all months and within each month
- All vehicles are assumed to be available for the entire month
- In reality, fleet sizes may vary seasonally or vehicles may have downtime, but the current model does not account for this
- To model variable fleet sizes, run the scheduler separately for different periods with appropriate vehicle counts

## Examples

### Example 1: Basic Scheduling

```bash
./run_scheduler.sh 5
```

Output:
```
Loading loads from loads.csv...
Found 2968 loads
Scheduling for 5 vehicles...

============================================================
SCHEDULING SUMMARY
============================================================
Total vehicles: 5
Total loads assigned: 57
Total revenue: SAR 152,802.04
Average revenue per vehicle: SAR 30,560.41

Per-vehicle breakdown:
------------------------------------------------------------
  Vehicle 1: 12 loads, SAR 40,726.76 revenue
  Vehicle 2: 12 loads, SAR 21,551.76 revenue
  Vehicle 3: 12 loads, SAR 36,960.00 revenue
  Vehicle 4: 11 loads, SAR 26,168.52 revenue
  Vehicle 5: 10 loads, SAR 27,395.00 revenue
============================================================

Schedule saved to schedule_output.csv
```

### Example 2: Higher Speed

```bash
./run_scheduler.sh 5 --speed 100
```

Using a higher average speed allows vehicles to travel faster between locations, potentially enabling more loads to be assigned.

### Example 3: Multi-Month Analysis

```bash
./run_scheduler.sh 10 --speed 80 --all-months
```

Processes all months separately and generates:
- Individual schedule files: `schedule_january_2025.csv`, `schedule_february_2025.csv`, etc.
- Monthly summary file: `monthly_summary.csv` with comparative statistics

Output includes summary table:
```
         month  num_vehicles  num_loads  total_loaded_km  total_unloaded_km  total_revenue  avg_revenue_per_vehicle
  January 2025            10        132        106,441.57          23,830.67     SAR 258,270.68               SAR 25,827.07
 February 2025            10        140         68,231.79          26,829.27     SAR 173,128.84               SAR 17,312.88
    March 2025            10        144         81,899.52          26,079.64     SAR 237,730.68               SAR 23,773.07
```

This allows you to compare efficiency, revenue, and utilization across different time periods.

## Troubleshooting

**Issue**: Few loads are being assigned

**Possible causes**:
- Time constraints are too tight (vehicles can't reach next pickup in time)
- Average speed is too low
- Loads have conflicting time windows

**Solutions**:
- Increase the `--speed` parameter
- Add more vehicles
- Check that your loads.csv has realistic time windows

## License

MIT
