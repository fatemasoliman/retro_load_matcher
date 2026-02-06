#!/usr/bin/env python3
"""
Load Scheduling Optimizer
Schedules loads across multiple vehicles to maximize total revenue while respecting time constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import argparse
import sys
import os


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth (in kilometers).

    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2

    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r


def calculate_travel_time(lat1, lon1, lat2, lon2, avg_speed_kmh=60):
    """
    Calculate travel time between two points in hours.

    Args:
        lat1, lon1: Starting point coordinates
        lat2, lon2: Ending point coordinates
        avg_speed_kmh: Average speed in km/h (default: 60)

    Returns:
        Travel time in hours
    """
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    return distance / avg_speed_kmh


def parse_number(value):
    """Parse number from string, removing commas."""
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)


def load_csv(filepath_or_url, month_filter=None):
    """
    Load and parse the loads CSV file from a file path or URL.

    Args:
        filepath_or_url: Path to the loads.csv file or URL to fetch data from
        month_filter: Optional month filter (e.g., "December 2025", "Dec 2025", or "2025-12")

    Returns:
        DataFrame with parsed load data
    """
    # Load CSV from file or URL
    df = pd.read_csv(filepath_or_url)

    # Parse shipper_price (remove commas)
    df['shipper_price'] = df['shipper_price'].apply(parse_number)

    # Parse distance (remove commas)
    df['distance'] = df['distance'].apply(parse_number)

    # Parse pickup_date
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])

    # Convert duration from days to hours
    df['duration_hours'] = df['median_duration_days'] * 24

    # Calculate dropoff time
    df['dropoff_date'] = df['pickup_date'] + pd.to_timedelta(df['duration_hours'], unit='h')

    # Filter by month if specified
    if month_filter:
        # Handle month column if it exists
        if 'month' in df.columns:
            df = df[df['month'].str.contains(month_filter, case=False, na=False)]
        else:
            # Fallback: filter by pickup_date
            try:
                # Try to parse as YYYY-MM format
                filter_date = pd.to_datetime(month_filter)
                df = df[
                    (df['pickup_date'].dt.year == filter_date.year) &
                    (df['pickup_date'].dt.month == filter_date.month)
                ]
            except:
                # Try month name matching
                df = df[df['pickup_date'].dt.strftime('%B %Y').str.contains(month_filter, case=False)]

    return df


def load_active_vehicles(filepath='inputs/active_vehicles.csv'):
    """
    Load active vehicles data from CSV.

    Args:
        filepath: Path to the active_vehicles.csv file

    Returns:
        Tuple of (active_vehicles_by_date, vehicle_types, vehicle_plates, vehicle_locations)
        - active_vehicles_by_date: Dictionary mapping date -> set of active vehicle keys
        - vehicle_types: Dictionary mapping vehicle_key -> vehicle_type
        - vehicle_plates: Dictionary mapping vehicle_key -> license plate
        - vehicle_locations: Dictionary mapping (vehicle_key, date) -> (lat, lng)
    """
    try:
        df = pd.read_csv(filepath)

        # Parse the active_date column
        df['active_date'] = pd.to_datetime(df['active_date'], format='%m/%d/%y')

        # Create dictionaries for vehicle data
        active_vehicles_by_date = {}
        vehicle_types = {}
        vehicle_plates = {}
        vehicle_locations = {}

        for _, row in df.iterrows():
            date = row['active_date'].date()  # Just the date, no time
            vehicle_key = row.get('VehicleKey', row.get('vehicle_key'))
            vehicle_type = row.get('vehicle_type', None)
            vehicle_plate = row.get('vehicle_plate', None)
            dropoff_lat = row.get('dropoff_lat', None)
            dropoff_lng = row.get('dropoff_lng', None)

            # Add vehicle to this date
            if date not in active_vehicles_by_date:
                active_vehicles_by_date[date] = set()
            active_vehicles_by_date[date].add(vehicle_key)

            # Map vehicle key to its vehicle type (keep first seen)
            if vehicle_type and vehicle_key not in vehicle_types:
                vehicle_types[vehicle_key] = vehicle_type

            # Map vehicle key to its license plate (keep first seen)
            if vehicle_plate and vehicle_key not in vehicle_plates:
                vehicle_plates[vehicle_key] = vehicle_plate

            # Store vehicle location for this date
            if dropoff_lat is not None and dropoff_lng is not None:
                vehicle_locations[(vehicle_key, date)] = (float(dropoff_lat), float(dropoff_lng))

        return active_vehicles_by_date, vehicle_types, vehicle_plates, vehicle_locations
    except FileNotFoundError:
        print(f"Warning: Active vehicles file not found at {filepath}")
        print("Proceeding without vehicle availability constraints.")
        return None, None, None, None
    except Exception as e:
        print(f"Warning: Error loading active vehicles: {e}")
        print("Proceeding without vehicle availability constraints.")
        return None, None, None, None


class Load:
    """Represents a single load with all its attributes."""

    def __init__(self, row):
        self.key = row['key']
        self.id = row['id']
        self.entity = row['entity']
        self.revenue = row['shipper_price']
        self.distance = row['distance']
        self.pickup_date = row['pickup_date']
        self.dropoff_date = row['dropoff_date']
        self.pickup_lat = row['pickup_lat']
        self.pickup_lng = row['pickup_lng']
        self.dropoff_lat = row['dropoff_lat']
        self.dropoff_lng = row['dropoff_lng']
        self.duration_hours = row['duration_hours']
        self.vehicle_type = row.get('vehicle_type', None)  # Vehicle type requirement
        # Additional fields for display
        self.pickup_city = row.get('pickup_city', None)
        self.destination_city = row.get('destination_city', None)
        self.status = row.get('status', None)
        self.rental = row.get('rental', None)

    def __repr__(self):
        return f"Load({self.key}, revenue=${self.revenue}, pickup={self.pickup_date}, type={self.vehicle_type})"


class Vehicle:
    """Represents a vehicle with its assigned loads."""

    def __init__(self, vehicle_id, vehicle_type=None, license_plate=None):
        self.id = vehicle_id  # Vehicle key (e.g., vch12ee26e3c19adf9c)
        self.vehicle_type = vehicle_type  # Vehicle type (e.g., Flatbed, Curtain Sides)
        self.license_plate = license_plate  # License plate for display (e.g., "1174 VRA")
        self.loads = []
        self.total_revenue = 0
        self.current_lat = None
        self.current_lng = None
        self.available_at = None
        self.initial_lat = None  # Initial location for first load
        self.initial_lng = None  # Initial location for first load

    def can_assign_load(self, load, avg_speed_kmh=60, active_vehicles_by_date=None):
        """
        Check if a load can be assigned to this vehicle.

        Args:
            load: Load object to check
            avg_speed_kmh: Average speed for travel time calculation
            active_vehicles_by_date: Dict mapping date -> set of active vehicle keys (optional)

        Returns:
            True if load can be assigned, False otherwise
        """
        # Check vehicle type compatibility
        if load.vehicle_type and self.vehicle_type:
            if load.vehicle_type != self.vehicle_type:
                return False

        # If active vehicles data is provided, check if this vehicle was active on pickup date
        if active_vehicles_by_date is not None:
            pickup_date = load.pickup_date.date()  # Just the date, no time
            if pickup_date in active_vehicles_by_date:
                if self.id not in active_vehicles_by_date[pickup_date]:
                    return False
            # If date not in active_vehicles_by_date, we can't verify, so allow it

        # If no loads assigned yet, check if vehicle can reach from initial location
        if not self.loads:
            # If vehicle has an initial location, check travel time from there
            if self.initial_lat is not None and self.initial_lng is not None:
                travel_time_hours = calculate_travel_time(
                    self.initial_lat, self.initial_lng,
                    load.pickup_lat, load.pickup_lng,
                    avg_speed_kmh
                )
                # Assume vehicle is available at start of pickup date
                # Vehicle needs to be able to reach pickup by pickup time
                # This is a simplified check - in reality we'd need to know when vehicle finished previous load
                return True  # Allow assignment if vehicle was active that day
            return True

        # Check if vehicle is available before pickup time
        if self.available_at > load.pickup_date:
            return False

        # Calculate travel time from last dropoff to new pickup
        travel_time_hours = calculate_travel_time(
            self.current_lat, self.current_lng,
            load.pickup_lat, load.pickup_lng,
            avg_speed_kmh
        )

        # Check if vehicle can reach pickup location in time
        arrival_time = self.available_at + timedelta(hours=travel_time_hours)

        return arrival_time <= load.pickup_date

    def assign_load(self, load):
        """
        Assign a load to this vehicle.

        Args:
            load: Load object to assign
        """
        self.loads.append(load)
        self.total_revenue += load.revenue
        self.current_lat = load.dropoff_lat
        self.current_lng = load.dropoff_lng
        self.available_at = load.dropoff_date

    def __repr__(self):
        type_str = f" ({self.vehicle_type})" if self.vehicle_type else ""
        display_id = self.license_plate if self.license_plate else self.id
        return f"Vehicle {display_id}{type_str}: {len(self.loads)} loads, ${self.total_revenue} revenue"


def schedule_loads(loads_df, num_vehicles=None, avg_speed_kmh=60, deadmile_weight=0.3,
                   active_vehicles_file='inputs/active_vehicles.csv'):
    """
    Schedule loads across vehicles to maximize revenue and minimize deadmiles.

    Uses a greedy algorithm that:
    1. Sorts loads chronologically by pickup date
    2. For each load, finds compatible vehicles (available on pickup date + can reach in time)
    3. Assigns to vehicle with best score balancing:
       - Revenue distribution (prefer underutilized vehicles)
       - Deadmile minimization (prefer vehicles closer to pickup)

    Args:
        loads_df: DataFrame with load data
        num_vehicles: Number of vehicles (ignored if active_vehicles_file is provided)
        avg_speed_kmh: Average speed for travel time calculation
        deadmile_weight: Weight for deadmile penalty (0-1, default 0.3)
                        Higher = more emphasis on reducing deadmiles
                        Lower = more emphasis on revenue balance
        active_vehicles_file: Path to active vehicles CSV file

    Returns:
        List of Vehicle objects with assigned loads
    """
    # Load active vehicles data
    active_vehicles_by_date, vehicle_types, vehicle_plates, vehicle_locations = load_active_vehicles(active_vehicles_file)

    # Convert DataFrame rows to Load objects
    loads = [Load(row) for _, row in loads_df.iterrows()]

    # Filter out loads with zero or negative duration
    loads = [l for l in loads if l.duration_hours > 0]

    # Sort loads chronologically by pickup_date, then by revenue (descending) as tiebreaker
    loads.sort(key=lambda l: (l.pickup_date, -l.revenue))

    # Initialize vehicles based on active vehicles data
    if active_vehicles_by_date:
        # Get all unique vehicle keys across all dates
        all_vehicle_keys = set()
        for vehicle_set in active_vehicles_by_date.values():
            all_vehicle_keys.update(vehicle_set)

        # Create Vehicle objects using actual vehicle keys, their types, and license plates
        vehicles = []
        for vehicle_key in sorted(all_vehicle_keys):
            vehicle = Vehicle(
                vehicle_key,
                vehicle_types.get(vehicle_key),
                vehicle_plates.get(vehicle_key)
            )
            vehicles.append(vehicle)
        print(f"Loaded {len(vehicles)} vehicles from active_vehicles.csv")
    else:
        # Fallback to numbered vehicles if active vehicles file not available
        if num_vehicles is None:
            raise ValueError("num_vehicles must be provided if active_vehicles.csv is not available")
        vehicles = [Vehicle(f"vehicle_{i + 1}") for i in range(num_vehicles)]
        print(f"Using {num_vehicles} numbered vehicles (active_vehicles.csv not available)")

    # Try to assign each load to a vehicle
    for load in loads:
        # Find all vehicles that can take this load
        compatible_vehicles = [v for v in vehicles if v.can_assign_load(load, avg_speed_kmh, active_vehicles_by_date)]

        if compatible_vehicles:
            # Score each compatible vehicle based on:
            # 1. Revenue balance (prefer vehicles with lower current revenue)
            # 2. Deadmiles (prefer vehicles closer to pickup location)

            best_vehicle = None
            best_score = float('-inf')

            for vehicle in compatible_vehicles:
                # Calculate deadmiles (distance from last dropoff to new pickup)
                if vehicle.loads:
                    deadmiles = haversine_distance(
                        vehicle.current_lat, vehicle.current_lng,
                        load.pickup_lat, load.pickup_lng
                    )
                else:
                    # No previous load - check if vehicle has initial location for this date
                    if vehicle_locations:
                        pickup_date = load.pickup_date.date()
                        location_key = (vehicle.id, pickup_date)
                        if location_key in vehicle_locations:
                            initial_lat, initial_lng = vehicle_locations[location_key]
                            deadmiles = haversine_distance(
                                initial_lat, initial_lng,
                                load.pickup_lat, load.pickup_lng
                            )
                            # Store initial location in vehicle for use in assignment
                            if vehicle.initial_lat is None:
                                vehicle.initial_lat = initial_lat
                                vehicle.initial_lng = initial_lng
                        else:
                            # No location data for this date, assume zero deadmiles
                            deadmiles = 0
                    else:
                        # No location data available, assume zero deadmiles
                        deadmiles = 0

                # Scoring function:
                # - Negative revenue balance (prefer lower revenue vehicles for distribution)
                # - Negative deadmiles (prefer shorter deadmiles)
                # Weight deadmiles by typical revenue per km to make them comparable
                avg_revenue_per_km = load.revenue / load.distance if load.distance > 0 else 0
                deadmile_penalty = deadmiles * avg_revenue_per_km

                # Score: balance load distribution and deadmile minimization
                # Higher deadmile_weight = more emphasis on reducing deadmiles
                revenue_weight = 1.0 - deadmile_weight
                score = -(revenue_weight * vehicle.total_revenue) - (deadmile_weight * deadmile_penalty)

                if score > best_score:
                    best_score = score
                    best_vehicle = vehicle

            if best_vehicle:
                best_vehicle.assign_load(load)

    return vehicles


def generate_schedule_output(vehicles, output_file='schedule_output.csv'):
    """
    Generate CSV output file with the schedule.

    Args:
        vehicles: List of Vehicle objects with assigned loads
        output_file: Path to output CSV file
    """
    rows = []

    for vehicle in vehicles:
        for i, load in enumerate(vehicle.loads):
            rows.append({
                'vehicle_id': vehicle.license_plate if vehicle.license_plate else vehicle.id,
                'vehicle_key': vehicle.id,  # Keep vehicle key for reference
                'load_sequence': i + 1,
                'load_key': load.key,
                'load_id': load.id,
                'entity': load.entity,
                'revenue': load.revenue,
                'pickup_date': load.pickup_date,
                'dropoff_date': load.dropoff_date,
                'pickup_city': load.pickup_city,
                'destination_city': load.destination_city,
                'pickup_lat': load.pickup_lat,
                'pickup_lng': load.pickup_lng,
                'dropoff_lat': load.dropoff_lat,
                'dropoff_lng': load.dropoff_lng,
                'duration_hours': load.duration_hours,
                'status': load.status,
                'rental': load.rental
            })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False)

    return output_df


def print_summary(vehicles, avg_speed_kmh=60, active_vehicles_by_date=None, loads_df=None):
    """Print summary statistics of the schedule."""
    total_revenue = sum(v.total_revenue for v in vehicles)
    total_loads = sum(len(v.loads) for v in vehicles)

    # Calculate total loaded and unloaded kilometers
    total_loaded_km = 0
    total_unloaded_km = 0

    # Get month from first load if available
    month = None

    for vehicle in vehicles:
        if vehicle.loads:
            # Get month from first load
            if month is None:
                month = vehicle.loads[0].pickup_date.strftime('%B %Y')

            # Sum loaded kilometers (distance of each load)
            for load in vehicle.loads:
                total_loaded_km += load.distance

            # Calculate unloaded kilometers (travel between loads)
            for i in range(len(vehicle.loads) - 1):
                current_load = vehicle.loads[i]
                next_load = vehicle.loads[i + 1]

                # Calculate distance from current dropoff to next pickup
                unloaded_distance = haversine_distance(
                    current_load.dropoff_lat, current_load.dropoff_lng,
                    next_load.pickup_lat, next_load.pickup_lng
                )
                total_unloaded_km += unloaded_distance

    total_km = total_loaded_km + total_unloaded_km

    # Calculate per-vehicle revenue statistics
    vehicle_revenues = [v.total_revenue for v in vehicles]
    median_revenue = np.median(vehicle_revenues) if vehicle_revenues else 0
    variance_revenue = np.var(vehicle_revenues) if vehicle_revenues else 0

    # Count vehicles with loads
    vehicles_with_loads = sum(1 for v in vehicles if len(v.loads) > 0)

    # Count vehicles active in this month
    num_vehicles_active_in_month = len(vehicles)  # default
    if active_vehicles_by_date and loads_df is not None and not loads_df.empty:
        # Get all unique vehicles that were active during any date in this month
        unique_vehicles_in_month = set()
        for date in loads_df['pickup_date'].dt.date.unique():
            if date in active_vehicles_by_date:
                unique_vehicles_in_month.update(active_vehicles_by_date[date])
        num_vehicles_active_in_month = len(unique_vehicles_in_month)

    print("\n" + "=" * 80)
    print("SCHEDULING SUMMARY")
    print("=" * 80)

    if month:
        print(f"Month: {month}")
    print(f"Number of vehicles: {num_vehicles_active_in_month}")
    print(f"Number of vehicles used: {vehicles_with_loads}")
    print(f"Number of loads assigned: {total_loads}")
    print(f"Total loaded kilometers: {total_loaded_km:,.2f} km")
    print(f"Total unloaded kilometers: {total_unloaded_km:,.2f} km")
    print(f"Total kilometers (loaded + unloaded): {total_km:,.2f} km")
    print(f"Loaded/Total ratio: {(total_loaded_km/total_km*100) if total_km > 0 else 0:.1f}%")
    print(f"\nTotal revenue: SAR {total_revenue:,.2f}")
    print(f"Revenue per vehicle used (mean): SAR {total_revenue / vehicles_with_loads if vehicles_with_loads > 0 else 0:,.2f}")
    print(f"Revenue per vehicle (median): SAR {median_revenue:,.2f}")
    print(f"Revenue per vehicle (variance): {variance_revenue:,.2f}")
    if total_loads > 0:
        print(f"Average revenue per load: SAR {total_revenue / total_loads:,.2f}")

    print("\nPer-vehicle breakdown:")
    print("-" * 80)

    for vehicle in vehicles:
        if vehicle.loads:
            type_str = f" ({vehicle.vehicle_type})" if vehicle.vehicle_type else ""
            print(f"  Vehicle {vehicle.id}{type_str}: {len(vehicle.loads)} loads, "
                  f"SAR {vehicle.total_revenue:,.2f} revenue")
        else:
            type_str = f" ({vehicle.vehicle_type})" if vehicle.vehicle_type else ""
            print(f"  Vehicle {vehicle.id}{type_str}: No loads assigned")

    print("=" * 80 + "\n")


def calculate_month_statistics_by_vehicle_type(vehicles, avg_speed_kmh=60, active_vehicles_by_date=None, loads_df=None):
    """
    Calculate statistics for a month's schedule, broken down by vehicle type.

    Args:
        vehicles: List of Vehicle objects with assigned loads
        avg_speed_kmh: Average speed for travel time calculation
        active_vehicles_by_date: Dictionary mapping date -> set of active vehicle keys
        loads_df: DataFrame with loads for this month (to determine date range)

    Returns:
        List of dictionaries with statistics, one per vehicle type
    """
    # Group vehicles by type
    vehicles_by_type = {}
    for vehicle in vehicles:
        vtype = vehicle.vehicle_type or 'Unknown'
        if vtype not in vehicles_by_type:
            vehicles_by_type[vtype] = []
        vehicles_by_type[vtype].append(vehicle)

    # Calculate stats for each vehicle type
    all_stats = []

    for vtype, vtype_vehicles in vehicles_by_type.items():
        stats = calculate_month_statistics(vtype_vehicles, avg_speed_kmh, active_vehicles_by_date, loads_df)
        stats['vehicle_type'] = vtype
        all_stats.append(stats)

    return all_stats


def calculate_month_statistics(vehicles, avg_speed_kmh=60, active_vehicles_by_date=None, loads_df=None):
    """
    Calculate statistics for a month's schedule.

    Args:
        vehicles: List of Vehicle objects with assigned loads
        avg_speed_kmh: Average speed for travel time calculation
        active_vehicles_by_date: Dictionary mapping date -> set of active vehicle keys
        loads_df: DataFrame with loads for this month (to determine date range)

    Returns:
        Dictionary with statistics
    """
    total_revenue = sum(v.total_revenue for v in vehicles)
    total_loads = sum(len(v.loads) for v in vehicles)

    total_loaded_km = 0
    total_unloaded_km = 0
    month = None

    for vehicle in vehicles:
        if vehicle.loads:
            if month is None:
                month = vehicle.loads[0].pickup_date.strftime('%B %Y')

            for load in vehicle.loads:
                total_loaded_km += load.distance

            for i in range(len(vehicle.loads) - 1):
                current_load = vehicle.loads[i]
                next_load = vehicle.loads[i + 1]
                unloaded_distance = haversine_distance(
                    current_load.dropoff_lat, current_load.dropoff_lng,
                    next_load.pickup_lat, next_load.pickup_lng
                )
                total_unloaded_km += unloaded_distance

    total_km = total_loaded_km + total_unloaded_km

    # Calculate per-vehicle revenue statistics
    vehicle_revenues = [v.total_revenue for v in vehicles]
    median_revenue = np.median(vehicle_revenues) if vehicle_revenues else 0
    variance_revenue = np.var(vehicle_revenues) if vehicle_revenues else 0

    # Count only vehicles that actually got loads assigned
    vehicles_with_loads = sum(1 for v in vehicles if len(v.loads) > 0)

    # Count vehicles active in this month
    num_vehicles_active_in_month = len(vehicles)  # default
    if active_vehicles_by_date and loads_df is not None and not loads_df.empty:
        # Get all unique vehicles that were active during any date in this month
        unique_vehicles_in_month = set()
        for date in loads_df['pickup_date'].dt.date.unique():
            if date in active_vehicles_by_date:
                unique_vehicles_in_month.update(active_vehicles_by_date[date])
        num_vehicles_active_in_month = len(unique_vehicles_in_month)

    return {
        'month': month,
        'num_vehicles': num_vehicles_active_in_month,
        'num_vehicles_used': vehicles_with_loads,
        'num_loads': total_loads,
        'total_loaded_km': total_loaded_km,
        'total_unloaded_km': total_unloaded_km,
        'total_km': total_km,
        'loaded_ratio': (total_loaded_km / total_km * 100) if total_km > 0 else 0,
        'total_revenue': total_revenue,
        'revenue_per_vehicle': total_revenue / vehicles_with_loads if vehicles_with_loads > 0 else 0,
        'median_revenue_per_vehicle': median_revenue,
        'variance_revenue_per_vehicle': variance_revenue,
        'avg_revenue_per_load': total_revenue / total_loads if total_loads > 0 else 0
    }


def process_single_month(loads_df, num_vehicles, avg_speed_kmh, output_file, deadmile_weight=0.3,
                        active_vehicles_file='inputs/active_vehicles.csv'):
    """
    Process a single month's loads.

    Args:
        loads_df: DataFrame with load data
        num_vehicles: Number of vehicles (optional if using active_vehicles_file)
        avg_speed_kmh: Average speed
        output_file: Output CSV file path
        deadmile_weight: Weight for deadmile penalty (0-1)
        active_vehicles_file: Path to active vehicles CSV

    Returns:
        Statistics dictionary
    """
    # Load active vehicles data
    active_vehicles_by_date, vehicle_types, _, _ = load_active_vehicles(active_vehicles_file)

    vehicles = schedule_loads(loads_df, num_vehicles, avg_speed_kmh, deadmile_weight, active_vehicles_file)
    generate_schedule_output(vehicles, output_file)
    return calculate_month_statistics(vehicles, avg_speed_kmh, active_vehicles_by_date, loads_df)


def main():
    """Main function to run the load scheduler."""
    parser = argparse.ArgumentParser(
        description='Schedule loads across vehicles to maximize revenue'
    )
    parser.add_argument(
        'num_vehicles',
        type=int,
        nargs='?',
        default=None,
        help='Number of vehicles (optional, auto-detected from active_vehicles.csv if available)'
    )
    parser.add_argument(
        '--input',
        default='inputs/loads.csv',
        help='Input CSV file path or URL (default: inputs/loads.csv)'
    )
    parser.add_argument(
        '--output',
        default='outputs/schedules/schedule_output.csv',
        help='Output CSV file for schedule (default: outputs/schedules/schedule_output.csv)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=60,
        help='Average vehicle speed in km/h (default: 60)'
    )
    parser.add_argument(
        '--month',
        help='Filter loads by month (e.g., "December 2025", "Dec 2025", or "2025-12")'
    )
    parser.add_argument(
        '--all-months',
        action='store_true',
        help='Process all months separately and generate summary statistics'
    )
    parser.add_argument(
        '--deadmile-weight',
        type=float,
        default=0.3,
        help='Weight for deadmile minimization (0-1, default: 0.3). Higher values prioritize reducing empty miles.'
    )
    parser.add_argument(
        '--active-vehicles',
        default='inputs/active_vehicles.csv',
        help='Path to active vehicles CSV file (default: inputs/active_vehicles.csv)'
    )

    args = parser.parse_args()

    # Check if active_vehicles file exists
    active_vehicles_exists = os.path.exists(args.active_vehicles)

    # Validate inputs
    if not active_vehicles_exists and args.num_vehicles is None:
        print("Error: Either provide num_vehicles or ensure active_vehicles.csv exists", file=sys.stderr)
        sys.exit(1)

    if args.num_vehicles is not None and args.num_vehicles < 1:
        print("Error: Number of vehicles must be at least 1", file=sys.stderr)
        sys.exit(1)

    print(f"Loading loads from {args.input}...")
    try:
        # Load all data first (without month filter if processing all months)
        loads_df = load_csv(args.input, args.month if not args.all_months else None)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Process all months separately
    if args.all_months:
        print(f"Found {len(loads_df)} total loads")
        print(f"Processing all months separately...\n")

        # Get unique months
        if 'month' in loads_df.columns:
            unique_months = loads_df['month'].unique()
        else:
            loads_df['month_temp'] = loads_df['pickup_date'].dt.strftime('%B %Y')
            unique_months = loads_df['month_temp'].unique()

        all_month_stats = []

        for month in sorted(unique_months):
            print(f"\n{'=' * 80}")
            print(f"Processing {month}")
            print('=' * 80)

            # Filter data for this month
            if 'month' in loads_df.columns:
                month_df = loads_df[loads_df['month'] == month].copy()
            else:
                month_df = loads_df[loads_df['month_temp'] == month].copy()

            print(f"Found {len(month_df)} loads for {month}")
            if args.num_vehicles:
                print(f"Max vehicles: {args.num_vehicles} (actual availability from active_vehicles.csv)")
            print(f"Using average speed of {args.speed} km/h for travel time calculations\n")

            # Generate month-specific output file
            month_slug = month.replace(' ', '_').lower()
            os.makedirs('outputs/schedules', exist_ok=True)
            output_file = f"outputs/schedules/schedule_{month_slug}.csv"

            # Process this month
            vehicles = schedule_loads(month_df, args.num_vehicles, args.speed, args.deadmile_weight, args.active_vehicles)
            generate_schedule_output(vehicles, output_file)

            # Load active vehicles for stats calculation
            active_vehicles_by_date_temp, _, _, _ = load_active_vehicles(args.active_vehicles)

            # Get stats broken down by vehicle type
            stats_by_type = calculate_month_statistics_by_vehicle_type(vehicles, args.speed, active_vehicles_by_date_temp, month_df)
            all_month_stats.extend(stats_by_type)

            # Also calculate overall stats for printing
            stats = stats_by_type[0] if len(stats_by_type) == 1 else calculate_month_statistics(vehicles, args.speed, active_vehicles_by_date_temp, month_df)

            # Print this month's summary
            print(f"\nMonth: {stats['month']}")
            print(f"Number of vehicles: {stats['num_vehicles']}")
            print(f"Number of loads assigned: {stats['num_loads']}")
            print(f"Total loaded kilometers: {stats['total_loaded_km']:,.2f} km")
            print(f"Total unloaded kilometers: {stats['total_unloaded_km']:,.2f} km")
            print(f"Total kilometers: {stats['total_km']:,.2f} km")
            print(f"Loaded/Total ratio: {stats['loaded_ratio']:.1f}%")
            print(f"Total revenue: SAR {stats['total_revenue']:,.2f}")
            print(f"Revenue per vehicle (mean): SAR {stats['revenue_per_vehicle']:,.2f}")
            print(f"Revenue per vehicle (median): SAR {stats['median_revenue_per_vehicle']:,.2f}")
            print(f"Revenue per vehicle (variance): {stats['variance_revenue_per_vehicle']:,.2f}")
            if stats['num_loads'] > 0:
                print(f"Average revenue per load: SAR {stats['avg_revenue_per_load']:,.2f}")
            print(f"\nSchedule saved to {output_file}")

        # Generate summary CSV
        summary_df = pd.DataFrame(all_month_stats)
        summary_file = 'outputs/monthly_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        print(f"\n{'=' * 80}")
        print("ALL MONTHS SUMMARY")
        print('=' * 80)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to {summary_file}")
        print("Done!")

    else:
        # Process single month
        if args.month:
            print(f"Filtering for month: {args.month}")

        print(f"Found {len(loads_df)} loads")
        if args.num_vehicles:
            print(f"Max vehicles: {args.num_vehicles} (actual availability from active_vehicles.csv)")
        print(f"Using average speed of {args.speed} km/h for travel time calculations\n")

        # Load active vehicles data
        active_vehicles_by_date, vehicle_types, _, _ = load_active_vehicles(args.active_vehicles)

        # Run scheduling algorithm
        vehicles = schedule_loads(loads_df, args.num_vehicles, args.speed, args.deadmile_weight, args.active_vehicles)

        # Generate output
        print(f"Writing schedule to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        output_df = generate_schedule_output(vehicles, args.output)

        # Print summary
        print_summary(vehicles, args.speed, active_vehicles_by_date, loads_df)

        print(f"Schedule saved to {args.output}")
        print("Done!")


if __name__ == '__main__':
    main()
