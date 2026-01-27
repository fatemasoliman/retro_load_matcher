#!/usr/bin/env python3
"""
Monthly Gantt Chart Generator
Creates separate Gantt chart visualizations for each month's schedule.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from datetime import datetime, timedelta
import argparse
import sys
import os
import glob
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth (in kilometers)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def calculate_travel_time(lat1, lon1, lat2, lon2, avg_speed_kmh=60):
    """Calculate travel time between two points in hours."""
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    return distance / avg_speed_kmh


def load_schedule(filepath):
    """Load and parse the schedule CSV file."""
    df = pd.read_csv(filepath)
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df['dropoff_date'] = pd.to_datetime(df['dropoff_date'])
    return df


def create_monthly_gantt(schedule_df, month_name, avg_speed_kmh=60, output_file='schedule_gantt.png',
                        figsize=(20, 12), dpi=150):
    """
    Create a Gantt chart for a specific month.

    Args:
        schedule_df: DataFrame with schedule data
        month_name: Name of the month for the title
        avg_speed_kmh: Average speed for travel time calculation
        output_file: Path to save the output image
        figsize: Figure size in inches (width, height)
        dpi: Resolution of output image
    """
    # Sort by vehicle and sequence
    schedule_df = schedule_df.sort_values(['vehicle_id', 'load_sequence'])

    # Get unique vehicles (only those with loads)
    vehicles = sorted(schedule_df['vehicle_id'].unique())
    num_vehicles = len(vehicles)

    if num_vehicles == 0:
        print(f"No vehicles with loads for {month_name}")
        return None

    # Calculate figure height based on number of vehicles
    # More vehicles = taller chart
    height = max(12, num_vehicles * 0.4)
    figsize = (figsize[0], height)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color palette
    load_color = '#3498db'      # Blue for active loads
    travel_color = '#95a5a6'    # Gray for travel time

    # Plot each vehicle's schedule
    for idx, vehicle_id in enumerate(vehicles):
        vehicle_loads = schedule_df[schedule_df['vehicle_id'] == vehicle_id]
        y_position = idx

        prev_dropoff_date = None
        prev_dropoff_lat = None
        prev_dropoff_lng = None

        for _, load in vehicle_loads.iterrows():
            # If there's a previous load, calculate and show travel time
            if prev_dropoff_date is not None:
                travel_time_hours = calculate_travel_time(
                    prev_dropoff_lat, prev_dropoff_lng,
                    load['pickup_lat'], load['pickup_lng'],
                    avg_speed_kmh
                )
                travel_duration = timedelta(hours=travel_time_hours)

                # Draw travel time bar
                ax.barh(y_position, travel_duration.total_seconds() / 3600 / 24,
                       left=prev_dropoff_date, height=0.6,
                       color=travel_color, alpha=0.5, edgecolor='black', linewidth=0.5)

            # Calculate load duration
            load_duration = load['dropoff_date'] - load['pickup_date']

            # Draw load bar
            bar = ax.barh(y_position, load_duration.total_seconds() / 3600 / 24,
                         left=load['pickup_date'], height=0.6,
                         color=load_color, alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add load label (only if bar is wide enough)
            bar_width_days = load_duration.total_seconds() / 3600 / 24
            if bar_width_days > 0.5:  # Only show label if load is longer than half a day
                bar_center = load['pickup_date'] + load_duration / 2
                load_label = f"SAR {load['revenue']:,.0f}"
                ax.text(bar_center, y_position, load_label,
                       ha='center', va='center', fontsize=6, fontweight='bold',
                       color='white')

            # Update previous position
            prev_dropoff_date = load['dropoff_date']
            prev_dropoff_lat = load['dropoff_lat']
            prev_dropoff_lng = load['dropoff_lng']

    # Formatting
    ax.set_yticks(range(num_vehicles))
    # Truncate vehicle IDs to make them more readable
    vehicle_labels = [f'{str(v)[:15]}...' if len(str(v)) > 15 else str(v) for v in vehicles]
    ax.set_yticklabels(vehicle_labels, fontsize=8)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vehicle', fontsize=12, fontweight='bold')
    ax.set_title(f'Load Schedule - {month_name}', fontsize=16, fontweight='bold', pad=20)

    # Format x-axis to show dates
    ax.xaxis.set_major_locator(DayLocator(interval=2))  # Every 2 days
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(DayLocator())

    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add legend
    load_patch = mpatches.Patch(color=load_color, alpha=0.8, label='Load (Active)')
    travel_patch = mpatches.Patch(color=travel_color, alpha=0.5, label='Travel Time')
    ax.legend(handles=[load_patch, travel_patch], loc='upper right', fontsize=10)

    # Add statistics box
    total_loads = len(schedule_df)
    total_revenue = schedule_df['revenue'].sum()
    avg_revenue = total_revenue / num_vehicles if num_vehicles > 0 else 0
    stats_text = (f'Vehicles: {num_vehicles}\n'
                 f'Total Loads: {total_loads}\n'
                 f'Total Revenue: SAR {total_revenue:,.0f}\n'
                 f'Avg Revenue/Vehicle: SAR {avg_revenue:,.0f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Close to free memory

    return output_file


def main():
    """Main function to generate monthly Gantt charts."""
    parser = argparse.ArgumentParser(
        description='Generate separate Gantt charts for each month'
    )
    parser.add_argument(
        '--schedules-dir',
        default='outputs/schedules',
        help='Directory containing monthly schedule CSV files (default: outputs/schedules)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/visualizations/monthly_gantts',
        help='Directory to save monthly Gantt charts (default: outputs/visualizations/monthly_gantts)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=60,
        help='Average vehicle speed in km/h for travel time (default: 60)'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=20,
        help='Figure width in inches (default: 20)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Image resolution in DPI (default: 150)'
    )

    args = parser.parse_args()

    # Find all monthly schedule files
    schedule_pattern = os.path.join(args.schedules_dir, 'schedule_*.csv')
    schedule_files = glob.glob(schedule_pattern)

    # Filter out the generic schedule_output.csv if it exists
    schedule_files = [f for f in schedule_files if not f.endswith('schedule_output.csv')]

    if not schedule_files:
        print(f"No monthly schedule files found in {args.schedules_dir}")
        print("Run the scheduler with --all-months first to generate monthly schedules.")
        sys.exit(1)

    # Sort files by name
    schedule_files.sort()

    print(f"Found {len(schedule_files)} monthly schedule files")
    print(f"Generating Gantt charts in {args.output_dir}...\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each month
    successful = 0
    for schedule_file in schedule_files:
        # Extract month name from filename
        # e.g., "schedule_july_1,_2025.csv" -> "July 2025"
        basename = os.path.basename(schedule_file)
        month_slug = basename.replace('schedule_', '').replace('.csv', '')

        # Try to parse the month name
        # Handle formats like "july_1,_2025" or "july_2025"
        try:
            # Clean up the month slug
            month_slug = month_slug.replace('_1,_', '_').replace('_', ' ').title()
            month_name = month_slug
        except:
            month_name = month_slug

        print(f"Processing {month_name}...")

        try:
            # Load schedule
            schedule_df = load_schedule(schedule_file)

            if len(schedule_df) == 0:
                print(f"  No loads found in {basename}, skipping.")
                continue

            # Generate output filename
            output_filename = f"gantt_{basename.replace('schedule_', '').replace('.csv', '.png')}"
            output_path = os.path.join(args.output_dir, output_filename)

            # Create Gantt chart
            result = create_monthly_gantt(
                schedule_df,
                month_name=month_name,
                avg_speed_kmh=args.speed,
                output_file=output_path,
                figsize=(args.width, 12),
                dpi=args.dpi
            )

            if result:
                print(f"  ✓ Saved to {output_path}")
                successful += 1
            else:
                print(f"  ✗ Failed to create chart")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"Completed: {successful}/{len(schedule_files)} Gantt charts generated")
    print(f"Location: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
