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


def load_csv(filepath):
    """
    Load and parse the loads CSV file.

    Args:
        filepath: Path to the loads.csv file

    Returns:
        DataFrame with parsed load data
    """
    df = pd.read_csv(filepath)

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


def schedule_loads(loads_df, num_vehicles, avg_speed_kmh=60):
    """
    Schedule loads across vehicles to maximize revenue.

    Uses a greedy algorithm that sorts loads by revenue/duration ratio
    and assigns them to the first available vehicle.

    Args:
        loads_df: DataFrame with load data
        num_vehicles: Number of vehicles available
        avg_speed_kmh: Average speed for travel time calculation

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
            # Assign to the vehicle with lowest current revenue to maximize total revenue
            # by distributing loads more evenly
            best_vehicle = min(compatible_vehicles, key=lambda v: v.total_revenue)
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


def print_summary(vehicles):
    """Print summary statistics of the schedule."""
    total_revenue = sum(v.total_revenue for v in vehicles)
    total_loads = sum(len(v.loads) for v in vehicles)

    print("\n" + "=" * 60)
    print("SCHEDULING SUMMARY")
    print("=" * 60)
    print(f"Total vehicles: {len(vehicles)}")
    print(f"Total loads assigned: {total_loads}")
    print(f"Total revenue: ${total_revenue:,.2f}")
    print(f"Average revenue per vehicle: ${total_revenue / len(vehicles):,.2f}")
    print("\nPer-vehicle breakdown:")
    print("-" * 60)

    for vehicle in vehicles:
        if vehicle.loads:
            print(f"  Vehicle {vehicle.id}: {len(vehicle.loads)} loads, "
                  f"${vehicle.total_revenue:,.2f} revenue")
        else:
            print(f"  Vehicle {vehicle.id}: No loads assigned")

    print("=" * 60 + "\n")


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
        help='Input CSV file with loads (default: loads.csv)'
    )
    parser.add_argument(
        '--output',
        default='schedule_output.csv',
        help='Output CSV file for schedule (default: schedule_output.csv)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=60,
        help='Average vehicle speed in km/h (default: 60)'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.num_vehicles < 1:
        print("Error: Number of vehicles must be at least 1", file=sys.stderr)
        sys.exit(1)

    print(f"Loading loads from {args.input}...")
    try:
        loads_df = load_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(loads_df)} loads")
    print(f"Scheduling for {args.num_vehicles} vehicles...")
    print(f"Using average speed of {args.speed} km/h for travel time calculations\n")

    # Run scheduling algorithm
    vehicles = schedule_loads(loads_df, args.num_vehicles, args.speed)

    # Generate output
    print(f"Writing schedule to {args.output}...")
    output_df = generate_schedule_output(vehicles, args.output)

    # Print summary
    print_summary(vehicles)

    print(f"Schedule saved to {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
