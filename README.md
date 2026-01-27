# Load Scheduler

A Python script that optimizes load scheduling across multiple vehicles to maximize total revenue while respecting time and travel constraints.

## Features

- **Real Vehicle Availability**: Uses actual vehicle availability data to only assign loads to vehicles that were active on the pickup date
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

When `inputs/active_vehicles.csv` is available, vehicles are automatically detected:

```bash
./run_scheduler.sh
```

Or with options:
```bash
./run_scheduler.sh --speed 80 --all-months
```

You can still specify a number of vehicles if `active_vehicles.csv` is not available:
```bash
./run_scheduler.sh 5
```

### Advanced Options

```bash
./run_scheduler.sh [num_vehicles] [OPTIONS]

Arguments:
  num_vehicles            Number of vehicles (optional if active_vehicles.csv exists)

Options:
  --input FILE            Input CSV file path or URL (default: inputs/loads.csv)
  --output FILE           Output CSV file for schedule (default: outputs/schedules/schedule_output.csv)
  --speed SPEED           Average vehicle speed in km/h (default: 60)
  --month MONTH           Filter loads by specific month (e.g., "December 2025")
  --all-months            Process all months separately and generate summary statistics
  --deadmile-weight W     Weight for deadmile minimization (0-1, default: 0.3)
  --active-vehicles FILE  Path to active vehicles CSV (default: inputs/active_vehicles.csv)
```

Examples:
```bash
# Basic usage (auto-detect vehicles from active_vehicles.csv)
./run_scheduler.sh --speed 80

# Process all months separately
./run_scheduler.sh --speed 80 --all-months

# Process specific month only
./run_scheduler.sh --month "December 2025" --speed 80

# Use data from URL
./run_scheduler.sh --input "http://example.com/loads.csv" --speed 80

# Use custom input/output files
./run_scheduler.sh --input my_loads.csv --output my_schedule.csv

# If active_vehicles.csv is not available, specify vehicle count
./run_scheduler.sh 10 --speed 80

# Combine multiple options
./run_scheduler.sh --input loads.csv --output schedule.csv --speed 70
```

### Running Directly with Python

If you prefer to run the Python script directly:

```bash
source venv/bin/activate
python schedule_loads.py [OPTIONS]

# Or with specific vehicle count if active_vehicles.csv not available
python schedule_loads.py 10 [OPTIONS]
```

## Visualization

After generating a schedule, you can create visual Gantt charts to see the timeline for each vehicle.

### Monthly Gantt Charts (Recommended)

For better readability, generate separate Gantt charts for each month:

```bash
source venv/bin/activate
python visualize_monthly_gantts.py
```

This creates individual Gantt charts for each month showing:
- Each vehicle's timeline for that month
- Load assignments (blue bars with revenue labels)
- Travel time between loads (gray bars)
- Monthly statistics (vehicles, loads, revenue)
- Automatic height adjustment based on number of vehicles

The outputs are saved as `outputs/visualizations/monthly_gantts/gantt_[month].png`.

**Options:**
```bash
python visualize_monthly_gantts.py [OPTIONS]

Options:
  --schedules-dir DIR   Directory with monthly schedules (default: outputs/schedules)
  --output-dir DIR      Output directory for charts (default: outputs/visualizations/monthly_gantts)
  --speed SPEED         Average vehicle speed in km/h (default: 60)
  --width WIDTH         Figure width in inches (default: 20)
  --dpi DPI            Image resolution (default: 150)
```

### Basic Visualization (All Loads)

For a single comprehensive view (may be hard to read with many vehicles):

```bash
./visualize.sh
```

This creates a Gantt chart showing:
- Each vehicle's timeline
- Load assignments (blue bars with revenue labels)
- Travel time between loads (gray bars)
- Total loads and revenue statistics

The output is saved as `outputs/visualizations/schedule_gantt.png`.

### Visualization Options

```bash
./visualize.sh [OPTIONS]

Options:
  --input FILE      Input schedule CSV file (default: outputs/schedules/schedule_output.csv)
  --output FILE     Output image file (default: outputs/visualizations/schedule_gantt.png)
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
# 1. Generate schedule (vehicles auto-detected from active_vehicles.csv)
./run_scheduler.sh --speed 80 --all-months

# 2. Visualize the schedule
./visualize.sh --speed 80 --detailed --summary

# 3. Open the visualization
open outputs/visualizations/schedule_gantt.png  # macOS
# or: xdg-open outputs/visualizations/schedule_gantt.png  # Linux
# or: start outputs/visualizations/schedule_gantt.png  # Windows
```

## Actual vs Simulated Comparison

Compare your actual revenue performance against the simulated scheduler results to validate the model and identify optimization opportunities.

### Prepare Actual Data

Create an `inputs/actual.csv` file with your actual revenue data:

```csv
month,vehicles,loads,gb,gb_per_vehicle
07/01/25,71,490,1499602.0,21121.15
08/01/25,73,521,1331136.0,18234.74
09/01/25,73,574,1405523.0,19253.74
...
```

The script automatically detects columns containing "month", "vehicles", and "gb_per_vehicle"/"revenue_per_vehicle".

### Run Comparison

```bash
# Basic comparison
python compare_actual_vs_simulated.py

# Or use the compare script
./compare.sh

# Custom files
python compare_actual_vs_simulated.py --actual my_actual.csv --simulated monthly_summary.csv

# High resolution output
python compare_actual_vs_simulated.py --dpi 300 --width 16 --height 10
```

### Comparison Options

```bash
python compare_actual_vs_simulated.py [OPTIONS]

Options:
  --actual FILE     Input CSV with actual data (default: inputs/actual.csv)
  --simulated FILE  Input CSV with simulated data (default: outputs/monthly_summary.csv)
  --output FILE     Output image file (default: outputs/visualizations/actual_vs_simulated.png)
  --width WIDTH     Figure width in inches (default: 14)
  --height HEIGHT   Figure height in inches (default: 8)
  --dpi DPI         Image resolution (default: 150)
```

### Output

The comparison generates:
- **Line plot**: Shows actual vs simulated revenue per vehicle trends over time
- **Difference bars**: Green/red bars showing the absolute difference (Sim - Actual) on the same axis
- **Vehicle count line**: Orange dashed line showing actual vehicle usage (secondary y-axis)
- **Console table**: Detailed month-by-month comparison with vehicle counts
- **Summary statistics**: Average performance and variance

The chart includes:
- **Green line** (circles): Actual GB/vehicle
- **Blue line** (squares): Simulated GB/vehicle
- **Green/Red bars**: Difference in SAR (positive = simulator better, negative = actual better)
- **Orange dashed line** (diamonds): Actual vehicle count
- **Value labels**: SAR difference amounts on bars

Example output:
```
ACTUAL VS SIMULATED COMPARISON
====================================================================================================
Month          Actual Veh  Sim Veh  Actual (SAR/veh)  Simulated (SAR/veh)  Difference (SAR)  Difference (%)
July 2025            71       74          21,121.15             22,612.15           1,490.99        +7.1
August 2025          73      100          18,234.74             23,004.62           4,769.88       +26.2
...

Summary Statistics:
Average Actual: SAR 17,708.36/vehicle
Average Simulated: SAR 21,724.58/vehicle
Average Difference: SAR 4,016.21/vehicle
Average Difference %: +27.9%
```

This helps you:
- Validate the scheduler's accuracy
- Identify months where simulated results differ significantly
- Understand optimization potential
- Calibrate model parameters for better accuracy
- Compare actual vs simulated vehicle usage efficiency

**Important Notes:**
- **Fair Comparison**: Simulated revenue is divided by the **actual** number of vehicles used (not the simulated vehicle count) to ensure an apples-to-apples comparison
- **Actual Veh**: Number of vehicles actually used in operations
- **Sim Veh**: Number of vehicles that received loads in the simulation
- The simulator may use more vehicles to optimize the schedule across all available loads

## Project Structure

The project is organized with separate input and output folders:

```
retro_load_matcher/
├── inputs/                 # All input data files
│   ├── loads.csv          # Load data with pickup/dropoff info
│   └── actual.csv         # Actual performance data
├── outputs/               # All generated outputs
│   ├── schedules/         # Schedule CSV files
│   │   ├── schedule_output.csv
│   │   ├── schedule_january_1,_2025.csv
│   │   ├── schedule_february_1,_2025.csv
│   │   └── ...
│   ├── visualizations/    # Charts and graphs
│   │   ├── schedule_gantt.png
│   │   ├── actual_vs_simulated.png
│   │   └── ...
│   └── monthly_summary.csv  # Summary statistics across all months
├── schedule_loads.py      # Main scheduling algorithm
├── visualize_schedule.py  # Visualization tool
├── compare_actual_vs_simulated.py  # Comparison tool
└── *.sh                   # Shell scripts for easy execution
```

This structure keeps your workspace organized and makes it easy to:
- Separate input data from generated outputs
- Find schedules for specific months
- Manage visualization outputs
- Share or archive results
- Version control input data separately

## Input CSV Format

### loads.csv

The main input CSV file should have the following columns:

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

### active_vehicles.csv

The active vehicles file tracks which vehicles were available on each date:

- `active_date`: Date in MM/DD/YY format
- `active_vehicle_count`: Number of active vehicles on that date
- `vehicle_keys`: Comma-separated list of vehicle keys (e.g., "vch12ee26e3c19adf9c,vch1894e7e32036b0a8,...")

The scheduler will only assign loads to vehicles that were active on the pickup date of the load. This ensures the simulation matches real-world vehicle availability.

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
- Separate CSV files for each month in the `outputs/schedules/` folder (e.g., `outputs/schedules/schedule_december_1,_2025.csv`)
- Each file contains the optimized schedule for that specific month

### Monthly Summary Statistics
An `outputs/monthly_summary.csv` file containing comprehensive statistics for all months:

- `month`: Month name and year
- `num_vehicles`: Total number of vehicles available
- `num_vehicles_used`: Number of vehicles that actually received loads
- `num_loads`: Total loads assigned
- `total_loaded_km`: Distance traveled while carrying loads
- `total_unloaded_km`: Empty travel distance between loads
- `total_km`: Total distance (loaded + unloaded)
- `loaded_ratio`: Percentage of distance that is revenue-generating
- `total_revenue`: Total revenue for the month
- `revenue_per_vehicle`: Mean revenue per vehicle **that received loads** (total_revenue / num_vehicles_used)
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
2. **Vehicle Availability**: Loads active vehicles data to determine which vehicles were available on each date
3. **Vehicle Compatibility**: For each load, finds compatible vehicles that:
   - Were active on the pickup date (from active_vehicles.csv)
   - Are available before the pickup time
   - Can reach the pickup location in time from their last drop-off
4. **Composite Scoring**: Assigns each load using a weighted score that balances:
   - **Revenue Distribution** (70% default): Assigns to vehicles with lower revenue to balance workload
   - **Deadmile Minimization** (30% default): Prioritizes vehicles closer to pickup location to reduce empty miles
5. **Optimization Goals**:
   - Maximize total revenue across all vehicles
   - Minimize unloaded distance (deadmiles) between loads
   - Balance revenue distribution across the fleet
   - Respect real-world vehicle availability constraints

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

### Vehicle Availability
- **Real Vehicle Availability**: When `active_vehicles.csv` is provided, the scheduler uses actual vehicle availability data
- Vehicles are only assigned loads on dates when they were marked as active in the input file
- This accounts for real-world scenarios like vehicle maintenance, driver availability, and fleet size variations
- If `active_vehicles.csv` is not available, the scheduler falls back to using a fixed number of vehicles

## Examples

### Example 1: Basic Scheduling

```bash
./run_scheduler.sh --speed 80
```

Output:
```
Loading loads from inputs/loads.csv...
Found 2968 loads
Using average speed of 80.0 km/h for travel time calculations

Loaded 223 vehicles from active_vehicles.csv

============================================================
SCHEDULING SUMMARY
============================================================
Month: December 2025
Number of vehicles: 223
Number of vehicles used: 84
Number of loads assigned: 786
Total revenue: SAR 1,810,284.16
Revenue per vehicle used (mean): SAR 21,551.82

Per-vehicle breakdown:
------------------------------------------------------------
  Vehicle vch13408e38bd45a681: 10 loads, SAR 27,782.12 revenue
  Vehicle vch144dc1f103e2c8ca: 12 loads, SAR 26,088.12 revenue
  ...
============================================================

Schedule saved to outputs/schedules/schedule_output.csv
```

### Example 2: Higher Speed

```bash
./run_scheduler.sh --speed 100
```

Using a higher average speed allows vehicles to travel faster between locations, potentially enabling more loads to be assigned.

### Example 3: Multi-Month Analysis

```bash
./run_scheduler.sh --speed 80 --all-months
```

Processes all months separately and generates:
- Individual schedule files: `outputs/schedules/schedule_august_2025.csv`, `schedule_september_2025.csv`, etc.
- Monthly summary file: `outputs/monthly_summary.csv` with comparative statistics

Output includes summary table:
```
         month  num_vehicles  num_vehicles_used  num_loads  total_loaded_km  total_unloaded_km  total_revenue  revenue_per_vehicle
   August 2025           223                100        807       754,218.73      186,427.66  SAR 1,699,617.76      SAR 16,793.38
September 2025           223                 81        715       691,080.28      162,195.26  SAR 1,562,885.84      SAR 18,993.91
  October 2025           223                 74        698       651,564.11      214,042.82  SAR 1,454,418.28      SAR 19,315.79
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
