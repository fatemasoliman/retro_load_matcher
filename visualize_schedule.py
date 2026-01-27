#!/usr/bin/env python3
"""
Schedule Visualization Tool
Creates a Gantt chart visualization of the vehicle load schedule.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, HourLocator, DayLocator
from datetime import datetime, timedelta
import argparse
import sys
import os
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


def create_gantt_chart(schedule_df, avg_speed_kmh=60, output_file='schedule_gantt.png',
                       figsize=(16, 10), dpi=100):
    """
    Create a Gantt chart visualization of the schedule.

    Args:
        schedule_df: DataFrame with schedule data
        avg_speed_kmh: Average speed for travel time calculation
        output_file: Path to save the output image
        figsize: Figure size in inches (width, height)
        dpi: Resolution of output image
    """
    # Sort by vehicle and sequence
    schedule_df = schedule_df.sort_values(['vehicle_id', 'load_sequence'])

    # Get unique vehicles
    vehicles = sorted(schedule_df['vehicle_id'].unique())
    num_vehicles = len(vehicles)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color palette for loads
    load_color = '#3498db'      # Blue for active loads
    travel_color = '#95a5a6'    # Gray for travel time

    # Track vehicle positions for calculating travel times
    vehicle_last_position = {}

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

            # Add load label
            bar_center = load['pickup_date'] + load_duration / 2
            load_label = f"SAR {load['revenue']:,.0f}"
            ax.text(bar_center, y_position, load_label,
                   ha='center', va='center', fontsize=7, fontweight='bold',
                   color='white')

            # Update previous position
            prev_dropoff_date = load['dropoff_date']
            prev_dropoff_lat = load['dropoff_lat']
            prev_dropoff_lng = load['dropoff_lng']

    # Formatting
    ax.set_yticks(range(num_vehicles))
    ax.set_yticklabels([f'Vehicle {v}' for v in vehicles])
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vehicle', fontsize=12, fontweight='bold')
    ax.set_title('Load Schedule Gantt Chart', fontsize=16, fontweight='bold', pad=20)

    # Format x-axis to show dates
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(HourLocator(interval=6))

    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

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
    stats_text = f'Total Loads: {total_loads}\nTotal Revenue: SAR {total_revenue:,.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return fig, ax


def create_detailed_timeline(schedule_df, avg_speed_kmh=60, output_file='schedule_timeline.png',
                            max_vehicles_per_chart=10, figsize=(18, 12), dpi=100):
    """
    Create a detailed timeline view with more information.

    Args:
        schedule_df: DataFrame with schedule data
        avg_speed_kmh: Average speed for travel time calculation
        output_file: Path to save the output image
        max_vehicles_per_chart: Maximum vehicles to show per chart
        figsize: Figure size in inches (width, height)
        dpi: Resolution of output image
    """
    # Sort by vehicle and sequence
    schedule_df = schedule_df.sort_values(['vehicle_id', 'load_sequence'])

    # Get unique vehicles
    vehicles = sorted(schedule_df['vehicle_id'].unique())
    num_vehicles = len(vehicles)

    # Create multiple charts if needed
    num_charts = (num_vehicles + max_vehicles_per_chart - 1) // max_vehicles_per_chart

    for chart_idx in range(num_charts):
        start_idx = chart_idx * max_vehicles_per_chart
        end_idx = min(start_idx + max_vehicles_per_chart, num_vehicles)
        chart_vehicles = vehicles[start_idx:end_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot each vehicle
        for plot_idx, vehicle_id in enumerate(chart_vehicles):
            vehicle_loads = schedule_df[schedule_df['vehicle_id'] == vehicle_id]
            y_position = plot_idx

            prev_dropoff_date = None
            prev_dropoff_lat = None
            prev_dropoff_lng = None

            for _, load in vehicle_loads.iterrows():
                # If there's a previous load, show travel
                if prev_dropoff_date is not None:
                    travel_time_hours = calculate_travel_time(
                        prev_dropoff_lat, prev_dropoff_lng,
                        load['pickup_lat'], load['pickup_lng'],
                        avg_speed_kmh
                    )
                    travel_duration = timedelta(hours=travel_time_hours)

                    # Draw travel segment
                    ax.barh(y_position, travel_duration.total_seconds() / 3600 / 24,
                           left=prev_dropoff_date, height=0.5,
                           color='#95a5a6', alpha=0.5, edgecolor='black', linewidth=0.5)

                    # Add travel time label
                    travel_center = prev_dropoff_date + travel_duration / 2
                    ax.text(travel_center, y_position, f'{travel_time_hours:.1f}h',
                           ha='center', va='center', fontsize=6, style='italic')

                # Draw load
                load_duration = load['dropoff_date'] - load['pickup_date']

                bar = ax.barh(y_position, load_duration.total_seconds() / 3600 / 24,
                             left=load['pickup_date'], height=0.5,
                             color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)

                # Add detailed load label
                bar_center = load['pickup_date'] + load_duration / 2
                duration_days = load['duration_hours'] / 24
                load_label = f"SAR {load['revenue']:,.0f}\n{duration_days:.1f}d"
                ax.text(bar_center, y_position, load_label,
                       ha='center', va='center', fontsize=6, fontweight='bold')

                prev_dropoff_date = load['dropoff_date']
                prev_dropoff_lat = load['dropoff_lat']
                prev_dropoff_lng = load['dropoff_lng']

            # Add vehicle statistics on the right
            vehicle_revenue = vehicle_loads['revenue'].sum()
            vehicle_load_count = len(vehicle_loads)
            ax.text(1.01, y_position, f'{vehicle_load_count} loads | SAR {vehicle_revenue:,.0f}',
                   transform=ax.get_yaxis_transform(), fontsize=8, va='center')

        # Formatting
        ax.set_yticks(range(len(chart_vehicles)))
        ax.set_yticklabels([f'Vehicle {v}' for v in chart_vehicles])
        ax.set_xlabel('Date and Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vehicle', fontsize=12, fontweight='bold')

        title = 'Detailed Load Schedule Timeline'
        if num_charts > 1:
            title += f' (Part {chart_idx + 1}/{num_charts})'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Format x-axis
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %d\n%H:%M'))
        ax.xaxis.set_minor_locator(HourLocator(interval=12))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        if num_charts > 1:
            base_name = output_file.rsplit('.', 1)[0]
            ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'png'
            save_path = f'{base_name}_part{chart_idx + 1}.{ext}'
        else:
            save_path = output_file

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Timeline visualization saved to: {save_path}")


def print_schedule_summary(schedule_df):
    """Print a text summary of the schedule."""
    vehicles = sorted(schedule_df['vehicle_id'].unique())

    print("\n" + "=" * 80)
    print("SCHEDULE SUMMARY")
    print("=" * 80)

    for vehicle_id in vehicles:
        vehicle_loads = schedule_df[schedule_df['vehicle_id'] == vehicle_id].sort_values('load_sequence')
        total_revenue = vehicle_loads['revenue'].sum()

        print(f"\nVehicle {vehicle_id} - {len(vehicle_loads)} loads, SAR {total_revenue:,.2f} revenue")
        print("-" * 80)

        for _, load in vehicle_loads.iterrows():
            print(f"  {load['load_sequence']}. {load['entity'][:30]:30} | "
                  f"SAR {load['revenue']:>8,.2f} | "
                  f"{load['pickup_date'].strftime('%b %d %H:%M')} → "
                  f"{load['dropoff_date'].strftime('%b %d %H:%M')}")

    print("\n" + "=" * 80)


def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize vehicle load schedule as a Gantt chart'
    )
    parser.add_argument(
        '--input',
        default='outputs/schedules/schedule_output.csv',
        help='Input CSV file with schedule (default: outputs/schedules/schedule_output.csv)'
    )
    parser.add_argument(
        '--output',
        default='outputs/visualizations/schedule_gantt.png',
        help='Output image file (default: outputs/visualizations/schedule_gantt.png)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=60,
        help='Average vehicle speed in km/h for travel time (default: 60)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Create detailed timeline view instead of simple Gantt chart'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=16,
        help='Figure width in inches (default: 16)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=10,
        help='Figure height in inches (default: 10)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='Image resolution in DPI (default: 100)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print text summary of schedule'
    )

    args = parser.parse_args()

    # Load schedule
    print(f"Loading schedule from {args.input}...")
    try:
        schedule_df = load_schedule(args.input)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        print("Please run the scheduler first to generate a schedule file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading schedule: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(schedule_df)} loads across {schedule_df['vehicle_id'].nunique()} vehicles")

    # Print summary if requested
    if args.summary:
        print_schedule_summary(schedule_df)

    # Create visualization
    print(f"\nCreating {'detailed timeline' if args.detailed else 'Gantt chart'} visualization...")

    try:
        if args.detailed:
            create_detailed_timeline(
                schedule_df,
                avg_speed_kmh=args.speed,
                output_file=args.output,
                figsize=(args.width, args.height),
                dpi=args.dpi
            )
        else:
            create_gantt_chart(
                schedule_df,
                avg_speed_kmh=args.speed,
                output_file=args.output,
                figsize=(args.width, args.height),
                dpi=args.dpi
            )

        print("\nVisualization complete!")
        print(f"Open {args.output} to view the schedule.")

    except Exception as e:
        print(f"Error creating visualization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
