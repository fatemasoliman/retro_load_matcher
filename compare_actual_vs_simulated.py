#!/usr/bin/env python3
"""
Actual vs Simulated Comparison Tool
Compares actual revenue/vehicle with simulated scheduler results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from datetime import datetime


def load_actual_data(filepath):
    """
    Load actual performance data from CSV.

    Expected columns: month, actual_gb_per_vehicle (or similar), vehicles
    """
    df = pd.read_csv(filepath)

    # Try to identify the month, revenue, and vehicles columns
    month_col = None
    revenue_col = None
    vehicles_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'month' in col_lower or 'date' in col_lower:
            month_col = col
        elif 'gb_per_vehicle' in col_lower or 'per_vehicle' in col_lower or 'revenue_per_vehicle' in col_lower:
            revenue_col = col
        elif col_lower == 'vehicles' or 'vehicle' in col_lower and 'per' not in col_lower:
            vehicles_col = col

    if month_col is None or revenue_col is None:
        print(f"Error: Could not identify month and revenue columns in {filepath}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Rename for consistency
    rename_dict = {month_col: 'month', revenue_col: 'actual_gb_per_vehicle'}
    if vehicles_col:
        rename_dict[vehicles_col] = 'actual_vehicles'
    df = df.rename(columns=rename_dict)

    # Parse the month column - handle various date formats
    try:
        # Try to parse as datetime
        df['month_parsed'] = pd.to_datetime(df['month'])
        # Convert to "Month Year" format
        df['month'] = df['month_parsed'].dt.strftime('%B %Y')
        df = df.drop('month_parsed', axis=1)
    except:
        # If it fails, assume it's already in the correct format
        pass

    columns_to_return = ['month', 'actual_gb_per_vehicle']
    if 'actual_vehicles' in df.columns:
        columns_to_return.append('actual_vehicles')

    return df[columns_to_return]


def load_simulated_data(filepath):
    """Load simulated data from monthly summary."""
    df = pd.read_csv(filepath)
    columns_to_return = ['month', 'total_revenue']
    # Use num_vehicles_used if available, fallback to num_vehicles
    if 'num_vehicles_used' in df.columns:
        df['num_vehicles'] = df['num_vehicles_used']
    if 'num_vehicles' in df.columns:
        columns_to_return.append('num_vehicles')
    return df[columns_to_return]


def merge_and_prepare_data(actual_df, simulated_df):
    """Merge actual and simulated data."""
    # Merge on month
    merged = pd.merge(actual_df, simulated_df, on='month', how='outer')

    # Recalculate simulated revenue per vehicle using ACTUAL vehicle count for fair comparison
    if 'total_revenue' in merged.columns and 'actual_vehicles' in merged.columns:
        merged['revenue_per_vehicle'] = merged['total_revenue'] / merged['actual_vehicles']

    # Convert month to datetime for proper sorting
    try:
        merged['month_date'] = pd.to_datetime(merged['month'], format='%B %Y')
    except:
        # Try with mixed format
        merged['month_date'] = pd.to_datetime(merged['month'], format='mixed')

    merged = merged.sort_values('month_date')

    return merged


def create_comparison_plot(merged_df, output_file='visualizations/actual_vs_simulated.png',
                           figsize=(14, 8), dpi=150):
    """
    Create a comparison plot of actual vs simulated revenue per vehicle.

    Args:
        merged_df: DataFrame with merged actual and simulated data
        output_file: Path to save the output image
        figsize: Figure size in inches (width, height)
        dpi: Resolution of output image
    """
    # Check if we have vehicle count data
    has_vehicle_data = 'actual_vehicles' in merged_df.columns or 'num_vehicles' in merged_df.columns

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Prepare data
    months = merged_df['month']
    actual = merged_df['actual_gb_per_vehicle']
    simulated = merged_df['revenue_per_vehicle']
    difference = simulated - actual

    # Create x-axis positions
    x = range(len(months))

    # Create bars for the difference on the same axis
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in difference]
    bars = ax.bar(x, difference, alpha=0.25, color=colors, width=0.4,
                  label='Difference (Sim - Actual)', zorder=2, edgecolor='black', linewidth=0.5)

    # Add value labels on difference bars
    for i, diff in enumerate(difference):
        if pd.notna(diff) and abs(diff) > 100:  # Only show if difference is significant
            label_y = diff
            va = 'bottom' if diff > 0 else 'top'
            offset = 400 if diff > 0 else -400
            ax.text(i, label_y + offset, f'{diff:,.0f}',
                   ha='center', va=va, fontsize=7, color='#34495e', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # Create line plots (on top of bars)
    line1 = ax.plot(x, actual, marker='o', linewidth=2.5, markersize=8,
                    label='Actual', color='#2ecc71', alpha=0.9, zorder=4)
    line2 = ax.plot(x, simulated, marker='s', linewidth=2.5, markersize=8,
                    label='Simulated', color='#3498db', alpha=0.9, zorder=4)

    # Add a zero reference line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3, zorder=1)

    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('GB/vehicle', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Simulated Revenue per Vehicle', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on lines (simplified to avoid clutter)
    # Only show values at the end points
    if len(actual) > 0:
        first_idx, last_idx = 0, len(actual) - 1
        if pd.notna(actual.iloc[first_idx]):
            ax.text(first_idx - 0.3, actual.iloc[first_idx], f'{actual.iloc[first_idx]:,.0f}',
                   ha='right', va='center', fontsize=8, color='#2ecc71', fontweight='bold')
        if pd.notna(actual.iloc[last_idx]):
            ax.text(last_idx + 0.3, actual.iloc[last_idx], f'{actual.iloc[last_idx]:,.0f}',
                   ha='left', va='center', fontsize=8, color='#2ecc71', fontweight='bold')
        if pd.notna(simulated.iloc[first_idx]):
            ax.text(first_idx - 0.3, simulated.iloc[first_idx], f'{simulated.iloc[first_idx]:,.0f}',
                   ha='right', va='center', fontsize=8, color='#3498db', fontweight='bold')
        if pd.notna(simulated.iloc[last_idx]):
            ax.text(last_idx + 0.3, simulated.iloc[last_idx], f'{simulated.iloc[last_idx]:,.0f}',
                   ha='left', va='center', fontsize=8, color='#3498db', fontweight='bold')

    # Add vehicle count line if data is available
    if has_vehicle_data:
        # Create secondary y-axis for vehicle count
        ax2 = ax.twinx()

        # Get vehicle count data (prefer actual, fallback to simulated)
        if 'actual_vehicles' in merged_df.columns:
            vehicle_counts = merged_df['actual_vehicles']
            label = 'Actual Vehicles'
            color = '#e67e22'
        elif 'num_vehicles' in merged_df.columns:
            vehicle_counts = merged_df['num_vehicles']
            label = 'Simulated Vehicles'
            color = '#e67e22'

        # Plot vehicle count line
        line = ax2.plot(x, vehicle_counts, color=color, marker='D', linewidth=2.5,
                       markersize=6, label=label, zorder=5, alpha=0.9, linestyle='--')

        # Style the vehicle count axis
        ax2.set_ylabel('Number of Vehicles', fontsize=12, fontweight='bold', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)

        # Add vehicle count labels on the line
        for i, count in enumerate(vehicle_counts):
            if pd.notna(count):
                ax2.text(i, count + max(vehicle_counts) * 0.02, f'{int(count)}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=color, bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', edgecolor=color, alpha=0.8))

        # Combine all legends and place below plot
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10, framealpha=0.9)
    else:
        # Just use main plot legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=3, fontsize=11, framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"\nComparison chart saved to: {output_file}")

    return fig, ax


def create_line_plot(merged_df, output_file='visualizations/actual_vs_simulated_line.png',
                     figsize=(14, 8), dpi=150):
    """
    Create a line plot comparison of actual vs simulated revenue per vehicle.

    Args:
        merged_df: DataFrame with merged actual and simulated data
        output_file: Path to save the output image
        figsize: Figure size in inches (width, height)
        dpi: Resolution of output image
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Prepare data
    months = merged_df['month']
    actual = merged_df['actual_gb_per_vehicle']
    simulated = merged_df['revenue_per_vehicle']

    # Create x-axis positions
    x = range(len(months))

    # Plot lines
    ax.plot(x, actual, marker='o', linewidth=2, markersize=8,
            label='Actual', color='#2ecc71', alpha=0.8)
    ax.plot(x, simulated, marker='s', linewidth=2, markersize=8,
            label='Simulated', color='#3498db', alpha=0.8)

    # Fill area between lines
    ax.fill_between(x, actual, simulated, alpha=0.2, color='gray')

    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue per Vehicle (SAR)', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Simulated Revenue per Vehicle - Trend Analysis',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Line chart saved to: {output_file}")

    return fig, ax


def print_comparison_table(merged_df):
    """Print a detailed comparison table."""
    print("\n" + "=" * 120)
    print("ACTUAL VS SIMULATED COMPARISON")
    print("=" * 120)

    # Calculate difference
    merged_df['difference'] = merged_df['revenue_per_vehicle'] - merged_df['actual_gb_per_vehicle']
    merged_df['difference_pct'] = (merged_df['difference'] / merged_df['actual_gb_per_vehicle'] * 100)

    # Prepare columns for display
    columns_to_show = ['month', 'actual_gb_per_vehicle', 'revenue_per_vehicle',
                       'difference', 'difference_pct']
    column_names = ['Month', 'Actual (SAR/veh)', 'Simulated (SAR/veh)', 'Difference (SAR)', 'Difference (%)']

    # Add vehicle counts if available
    if 'actual_vehicles' in merged_df.columns:
        columns_to_show.insert(1, 'actual_vehicles')
        column_names.insert(1, 'Actual Veh')
    if 'num_vehicles' in merged_df.columns:
        columns_to_show.insert(2 if 'actual_vehicles' in merged_df.columns else 1, 'num_vehicles')
        column_names.insert(2 if 'actual_vehicles' in merged_df.columns else 1, 'Sim Veh')

    # Format for display
    display_df = merged_df[columns_to_show].copy()
    display_df.columns = column_names

    # Round values
    if 'Actual (SAR/veh)' in display_df.columns:
        display_df['Actual (SAR/veh)'] = display_df['Actual (SAR/veh)'].round(2)
    if 'Simulated (SAR/veh)' in display_df.columns:
        display_df['Simulated (SAR/veh)'] = display_df['Simulated (SAR/veh)'].round(2)
    if 'Difference (SAR)' in display_df.columns:
        display_df['Difference (SAR)'] = display_df['Difference (SAR)'].round(2)
    if 'Difference (%)' in display_df.columns:
        display_df['Difference (%)'] = display_df['Difference (%)'].round(1)

    print(display_df.to_string(index=False))
    print("=" * 120)

    # Calculate summary statistics
    valid_data = merged_df[merged_df['actual_gb_per_vehicle'].notna() &
                           merged_df['revenue_per_vehicle'].notna()]

    if len(valid_data) > 0:
        print(f"\nSummary Statistics:")
        print(f"Average Actual: SAR {valid_data['actual_gb_per_vehicle'].mean():,.2f}/vehicle")
        print(f"Average Simulated: SAR {valid_data['revenue_per_vehicle'].mean():,.2f}/vehicle")
        print(f"Average Difference: SAR {valid_data['difference'].mean():,.2f}/vehicle")
        print(f"Average Difference %: {valid_data['difference_pct'].mean():+.1f}%")
        print(f"Months Compared: {len(valid_data)}")
        print("=" * 100)


def main():
    """Main function to run the comparison tool."""
    parser = argparse.ArgumentParser(
        description='Compare actual vs simulated revenue per vehicle'
    )
    parser.add_argument(
        '--actual',
        default='inputs/actual.csv',
        help='Input CSV file with actual data (default: inputs/actual.csv)'
    )
    parser.add_argument(
        '--simulated',
        default='outputs/monthly_summary.csv',
        help='Input CSV file with simulated data (default: outputs/monthly_summary.csv)'
    )
    parser.add_argument(
        '--output',
        default='outputs/visualizations/actual_vs_simulated.png',
        help='Output image file (default: outputs/visualizations/actual_vs_simulated.png)'
    )
    parser.add_argument(
        '--width',
        type=float,
        default=14,
        help='Figure width in inches (default: 14)'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=8,
        help='Figure height in inches (default: 8)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Image resolution in DPI (default: 150)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading actual data from {args.actual}...")
    try:
        actual_df = load_actual_data(args.actual)
    except FileNotFoundError:
        print(f"Error: File '{args.actual}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading actual data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading simulated data from {args.simulated}...")
    try:
        simulated_df = load_simulated_data(args.simulated)
    except FileNotFoundError:
        print(f"Error: File '{args.simulated}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading simulated data: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge data
    print("Merging and preparing data...")
    merged_df = merge_and_prepare_data(actual_df, simulated_df)

    # Print comparison table
    print_comparison_table(merged_df)

    # Create comparison chart with vehicle counts
    print(f"\nCreating comparison chart...")
    create_comparison_plot(merged_df, args.output,
                          figsize=(args.width, args.height), dpi=args.dpi)

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
