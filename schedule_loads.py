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

    def __repr__(self):
        return f"Load({self.key}, revenue=${self.revenue}, pickup={self.pickup_date})"


class Vehicle:
    """Represents a vehicle with its assigned loads."""

    def __init__(self, vehicle_id):
        self.id = vehicle_id
        self.loads = []
        self.total_revenue = 0
        self.current_lat = None
        self.current_lng = None
        self.available_at = None

    def can_assign_load(self, load, avg_speed_kmh=60):
        """
        Check if a load can be assigned to this vehicle.

        Args:
            load: Load object to check
            avg_speed_kmh: Average speed for travel time calculation

        Returns:
            True if load can be assigned, False otherwise
        """
        # If no loads assigned yet, can always assign
        if not self.loads:
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
        return f"Vehicle {self.id}: {len(self.loads)} loads, ${self.total_revenue} revenue"


def schedule_loads(loads_df, num_vehicles, avg_speed_kmh=60, deadmile_weight=0.3):
    """
    Schedule loads across vehicles to maximize revenue and minimize deadmiles.

    Uses a greedy algorithm that:
    1. Sorts loads chronologically by pickup date
    2. For each load, finds compatible vehicles (available + can reach in time)
    3. Assigns to vehicle with best score balancing:
       - Revenue distribution (prefer underutilized vehicles)
       - Deadmile minimization (prefer vehicles closer to pickup)

    Args:
        loads_df: DataFrame with load data
        num_vehicles: Number of vehicles available
        avg_speed_kmh: Average speed for travel time calculation
        deadmile_weight: Weight for deadmile penalty (0-1, default 0.3)
                        Higher = more emphasis on reducing deadmiles
                        Lower = more emphasis on revenue balance

    Returns:
        List of Vehicle objects with assigned loads
    """
    # Convert DataFrame rows to Load objects
    loads = [Load(row) for _, row in loads_df.iterrows()]

    # Filter out loads with zero or negative duration
    loads = [l for l in loads if l.duration_hours > 0]

    # Sort loads chronologically by pickup_date, then by revenue (descending) as tiebreaker
    loads.sort(key=lambda l: (l.pickup_date, -l.revenue))

    # Initialize vehicles
    vehicles = [Vehicle(i + 1) for i in range(num_vehicles)]

    # Try to assign each load to a vehicle
    for load in loads:
        # Find all vehicles that can take this load
        compatible_vehicles = [v for v in vehicles if v.can_assign_load(load, avg_speed_kmh)]

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
                    # No previous load, assume starting from pickup (zero deadmiles)
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
                'vehicle_id': vehicle.id,
                'load_sequence': i + 1,
                'load_key': load.key,
                'load_id': load.id,
                'entity': load.entity,
                'revenue': load.revenue,
                'pickup_date': load.pickup_date,
                'dropoff_date': load.dropoff_date,
                'pickup_lat': load.pickup_lat,
                'pickup_lng': load.pickup_lng,
                'dropoff_lat': load.dropoff_lat,
                'dropoff_lng': load.dropoff_lng,
                'duration_hours': load.duration_hours
            })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False)

    return output_df


def print_summary(vehicles, avg_speed_kmh=60):
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

    print("\n" + "=" * 80)
    print("SCHEDULING SUMMARY")
    print("=" * 80)

    if month:
        print(f"Month: {month}")
    print(f"Number of vehicles: {len(vehicles)}")
    print(f"Number of loads assigned: {total_loads}")
    print(f"Total loaded kilometers: {total_loaded_km:,.2f} km")
    print(f"Total unloaded kilometers: {total_unloaded_km:,.2f} km")
    print(f"Total kilometers (loaded + unloaded): {total_km:,.2f} km")
    print(f"Loaded/Total ratio: {(total_loaded_km/total_km*100) if total_km > 0 else 0:.1f}%")
    print(f"\nTotal revenue: SAR {total_revenue:,.2f}")
    print(f"Revenue per vehicle (mean): SAR {total_revenue / len(vehicles):,.2f}")
    print(f"Revenue per vehicle (median): SAR {median_revenue:,.2f}")
    print(f"Revenue per vehicle (variance): {variance_revenue:,.2f}")
    if total_loads > 0:
        print(f"Average revenue per load: SAR {total_revenue / total_loads:,.2f}")

    print("\nPer-vehicle breakdown:")
    print("-" * 80)

    for vehicle in vehicles:
        if vehicle.loads:
            print(f"  Vehicle {vehicle.id}: {len(vehicle.loads)} loads, "
                  f"SAR {vehicle.total_revenue:,.2f} revenue")
        else:
            print(f"  Vehicle {vehicle.id}: No loads assigned")

    print("=" * 80 + "\n")


def calculate_month_statistics(vehicles, avg_speed_kmh=60):
    """
    Calculate statistics for a month's schedule.

    Args:
        vehicles: List of Vehicle objects with assigned loads
        avg_speed_kmh: Average speed for travel time calculation

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

    return {
        'month': month,
        'num_vehicles': len(vehicles),
        'num_loads': total_loads,
        'total_loaded_km': total_loaded_km,
        'total_unloaded_km': total_unloaded_km,
        'total_km': total_km,
        'loaded_ratio': (total_loaded_km / total_km * 100) if total_km > 0 else 0,
        'total_revenue': total_revenue,
        'revenue_per_vehicle': total_revenue / len(vehicles) if len(vehicles) > 0 else 0,
        'median_revenue_per_vehicle': median_revenue,
        'variance_revenue_per_vehicle': variance_revenue,
        'avg_revenue_per_load': total_revenue / total_loads if total_loads > 0 else 0
    }


def process_single_month(loads_df, num_vehicles, avg_speed_kmh, output_file, deadmile_weight=0.3):
    """
    Process a single month's loads.

    Args:
        loads_df: DataFrame with load data
        num_vehicles: Number of vehicles
        avg_speed_kmh: Average speed
        output_file: Output CSV file path
        deadmile_weight: Weight for deadmile penalty (0-1)

    Returns:
        Statistics dictionary
    """
    vehicles = schedule_loads(loads_df, num_vehicles, avg_speed_kmh, deadmile_weight)
    generate_schedule_output(vehicles, output_file)
    return calculate_month_statistics(vehicles, avg_speed_kmh)


def main():
    """Main function to run the load scheduler."""
    parser = argparse.ArgumentParser(
        description='Schedule loads across vehicles to maximize revenue'
    )
    parser.add_argument(
        'num_vehicles',
        type=int,
        help='Number of vehicles available'
    )
    parser.add_argument(
        '--input',
        default='loads.csv',
        help='Input CSV file path or URL (default: loads.csv)'
    )
    parser.add_argument(
        '--output',
        default='schedules/schedule_output.csv',
        help='Output CSV file for schedule (default: schedules/schedule_output.csv)'
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

    args = parser.parse_args()

    # Validate inputs
    if args.num_vehicles < 1:
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
            print(f"Scheduling for {args.num_vehicles} vehicles...")
            print(f"Using average speed of {args.speed} km/h for travel time calculations\n")

            # Generate month-specific output file
            month_slug = month.replace(' ', '_').lower()
            os.makedirs('schedules', exist_ok=True)
            output_file = f"schedules/schedule_{month_slug}.csv"

            # Process this month
            stats = process_single_month(month_df, args.num_vehicles, args.speed, output_file, args.deadmile_weight)
            all_month_stats.append(stats)

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
        summary_file = 'monthly_summary.csv'
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
        print(f"Scheduling for {args.num_vehicles} vehicles...")
        print(f"Using average speed of {args.speed} km/h for travel time calculations\n")

        # Run scheduling algorithm
        vehicles = schedule_loads(loads_df, args.num_vehicles, args.speed, args.deadmile_weight)

        # Generate output
        print(f"Writing schedule to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        output_df = generate_schedule_output(vehicles, args.output)

        # Print summary
        print_summary(vehicles, args.speed)

        print(f"Schedule saved to {args.output}")
        print("Done!")


if __name__ == '__main__':
    main()
