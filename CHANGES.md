# Changes Summary - Retro Load Matcher Updates

## Session Date: January 27, 2026

### 1. Project Reorganization
- Created separate `inputs/` and `outputs/` folders for better organization
- Moved all input data files to `inputs/` (loads.csv, actual.csv, active_vehicles.csv)
- Moved all generated outputs to `outputs/` (schedules/, visualizations/, monthly_summary.csv)
- Updated all scripts to use new folder structure
- Updated .gitignore to handle new structure
- Updated README with new folder structure documentation

**Benefits:**
- Clean separation of inputs vs outputs
- Easier version control of input data
- Better organization for sharing/archiving

### 2. Real Vehicle Availability Integration
- Added support for `active_vehicles.csv` file tracking daily vehicle availability
- Modified scheduler to only assign loads to vehicles active on pickup date
- Uses actual vehicle keys (e.g., "vch12ee26e3c19adf9c") instead of sequential IDs
- Ignores transit time and load duration - only checks pickup date availability

**Key Changes:**
- New function: `load_active_vehicles()` to parse daily vehicle availability
- Updated `Vehicle.can_assign_load()` to check active date constraints
- Modified `schedule_loads()` to auto-detect vehicles from CSV
- Loads 223 unique vehicles automatically from active_vehicles.csv

**Files Modified:**
- schedule_loads.py: Added vehicle availability logic
- README.md: Documented new vehicle availability feature

### 3. Made num_vehicles Optional
- `num_vehicles` is now optional when active_vehicles.csv exists
- Script auto-detects all available vehicles from the CSV
- Falls back to numbered vehicles if CSV not available and num_vehicles specified
- Provides clear error message if neither is available

**New Usage:**
```bash
# Simple usage - auto-detect vehicles
./run_scheduler.sh --speed 80 --all-months

# Fallback if active_vehicles.csv not available
./run_scheduler.sh 50 --speed 80
```

### 4. Added Vehicle Count Line to Comparison Chart
- Enhanced actual vs simulated comparison chart with vehicle count trend line
- Shows vehicle counts on secondary y-axis (orange line)
- Updated comparison table to include both actual and simulated vehicle counts
- Added documentation explaining vehicle count differences

**Chart Features:**
- Green bars: Actual revenue per vehicle
- Blue bars: Simulated revenue per vehicle  
- Orange line with labels: Number of vehicles
- Percentage differences above bars

**Table Output:**
```
Month          Actual Veh  Sim Veh  Actual (SAR/veh)  Simulated (SAR/veh)  Difference
July 2025            71      223          21,121.15              7,192.07      -65.9%
August 2025          73      223          18,234.74              7,621.60      -58.2%
```

**Files Modified:**
- compare_actual_vs_simulated.py: Added vehicle count line and table columns
- README.md: Documented new visualization features

### Technical Details

**Algorithm Changes:**
- Vehicle compatibility now checks: active on pickup date → available before pickup → can reach in time
- Only the pickup date matters for availability check
- Load duration and transit time don't affect availability constraints
- This matches real-world operations where vehicle assignments are made at pickup time

**Data Flow:**
1. Load active_vehicles.csv → parse daily vehicle availability
2. Load loads.csv → parse load details with pickup dates
3. For each load, filter to vehicles active on that pickup date
4. Apply existing time and distance constraints
5. Assign to best-scoring available vehicle

### Test Results

**Scheduler Output:**
- Successfully loads 223 vehicles from active_vehicles.csv
- Assigns 485-884 loads per month depending on availability
- Maintains 75-81% loaded ratio (good efficiency)
- Revenue per vehicle: SAR 5,230 - 8,562/month

**Comparison Output:**
- Shows actual vehicles used: 55-83 per month
- Shows available vehicles: 223 (all unique vehicles in dataset)
- Difference explained: actual = used, simulated = available
- Clear visualization of both metrics

### Files Changed
- schedule_loads.py
- compare_actual_vs_simulated.py
- visualize_schedule.py (path updates)
- run_scheduler.sh
- compare.sh (path updates)
- visualize.sh (path updates)
- README.md
- .gitignore

### Migration Notes
No breaking changes for users. Old commands still work if you provide num_vehicles.
Recommended to transition to new simplified syntax without num_vehicles parameter.
