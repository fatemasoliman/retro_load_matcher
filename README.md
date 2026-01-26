# Load Scheduler

A Python script that optimizes load scheduling across multiple vehicles to maximize total revenue while respecting time and travel constraints.

## Features

- **Revenue Maximization**: Assigns loads to vehicles to maximize total revenue
- **Time Constraint Handling**: Ensures loads don't overlap and pickup times are respected
- **Travel Time Calculation**: Uses Haversine distance formula to calculate travel time between locations
- **Configurable Speed**: Adjust average vehicle speed for travel time calculations
- **CSV Output**: Generates detailed schedule with vehicle assignments
- **Visual Gantt Charts**: Generate beautiful timeline visualizations of your schedule

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
  --input FILE      Input CSV file with loads (default: loads.csv)
  --output FILE     Output CSV file for schedule (default: schedule_output.csv)
  --speed SPEED     Average vehicle speed in km/h (default: 60)
```

Examples:
```bash
# Schedule for 10 vehicles with 80 km/h average speed
./run_scheduler.sh 10 --speed 80

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

The output is saved as `schedule_gantt.png`.

### Visualization Options

```bash
./visualize.sh [OPTIONS]

Options:
  --input FILE      Input schedule CSV file (default: schedule_output.csv)
  --output FILE     Output image file (default: schedule_gantt.png)
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
open schedule_gantt.png  # macOS
# or: xdg-open schedule_gantt.png  # Linux
# or: start schedule_gantt.png  # Windows
```

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

## Algorithm

The scheduler uses a greedy algorithm that:

1. Sorts loads chronologically by pickup date
2. For each load, finds compatible vehicles that can:
   - Be available before the pickup time
   - Reach the pickup location in time from their last drop-off
3. Assigns each load to the compatible vehicle with the lowest current revenue to balance the load distribution
4. Maximizes total revenue across all vehicles

## Constraints

- Loads must be picked up at their exact pickup time (not before or after)
- Vehicles must have enough time to travel from the previous drop-off to the next pickup
- Loads cannot overlap on the same vehicle
- Travel time is calculated using the Haversine distance formula

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
Total revenue: $152,802.04
Average revenue per vehicle: $30,560.41

Per-vehicle breakdown:
------------------------------------------------------------
  Vehicle 1: 12 loads, $40,726.76 revenue
  Vehicle 2: 12 loads, $21,551.76 revenue
  Vehicle 3: 12 loads, $36,960.00 revenue
  Vehicle 4: 11 loads, $26,168.52 revenue
  Vehicle 5: 10 loads, $27,395.00 revenue
============================================================

Schedule saved to schedule_output.csv
```

### Example 2: Higher Speed

```bash
./run_scheduler.sh 5 --speed 100
```

Using a higher average speed allows vehicles to travel faster between locations, potentially enabling more loads to be assigned.

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
