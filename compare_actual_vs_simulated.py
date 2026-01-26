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

    Expected columns: month, actual_gb_per_vehicle (or similar)
    """
    df = pd.read_csv(filepath)

    # Try to identify the month and revenue columns
    month_col = None
    revenue_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'month' in col_lower or 'date' in col_lower:
            month_col = col
        elif 'gb_per_vehicle' in col_lower or 'per_vehicle' in col_lower or 'revenue_per_vehicle' in col_lower:
            revenue_col = col

    if month_col is None or revenue_col is None:
        print(f"Error: Could not identify month and revenue columns in {filepath}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Rename for consistency
    df = df.rename(columns={month_col: 'month', revenue_col: 'actual_gb_per_vehicle'})

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

    return df[['month', 'actual_gb_per_vehicle']]


def load_simulated_data(filepath):
    """Load simulated data from monthly summary."""
    df = pd.read_csv(filepath)
    return df[['month', 'revenue_per_vehicle']]


def merge_and_prepare_data(actual_df, simulated_df):
    """Merge actual and simulated data."""
    # Merge on month
    merged = pd.merge(actual_df, simulated_df, on='month', how='outer')

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
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Prepare data
    months = merged_df['month']
    actual = merged_df['actual_gb_per_vehicle']
    simulated = merged_df['revenue_per_vehicle']

    # Create x-axis positions
    x = range(len(months))
    width = 0.35

    # Create bars
    bars1 = ax.bar([i - width/2 for i in x], actual, width,
                    label='Actual', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], simulated, width,
                    label='Simulated', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Calculate difference and add annotations
    for i, (act, sim) in enumerate(zip(actual, simulated)):
        if pd.notna(act) and pd.notna(sim):
            diff = ((sim - act) / act * 100) if act != 0 else 0
            # Add difference percentage on top
            y_pos = max(act, sim)
            color = '#27ae60' if diff >= 0 else '#e74c3c'
            ax.text(i, y_pos + 500, f'{diff:+.1f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue per Vehicle (SAR)', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Simulated Revenue per Vehicle', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'SAR {height:,.0f}',
                       ha='center', va='bottom', fontsize=7, rotation=90)

    # Calculate summary statistics
    valid_data = merged_df[merged_df['actual_gb_per_vehicle'].notna() &
                           merged_df['revenue_per_vehicle'].notna()]

    if len(valid_data) > 0:
        avg_actual = valid_data['actual_gb_per_vehicle'].mean()
        avg_simulated = valid_data['revenue_per_vehicle'].mean()
        overall_diff = ((avg_simulated - avg_actual) / avg_actual * 100) if avg_actual != 0 else 0

        # Add summary text box
        summary_text = (
            f'Summary Statistics:\n'
            f'Avg Actual: SAR {avg_actual:,.2f}/vehicle\n'
            f'Avg Simulated: SAR {avg_simulated:,.2f}/vehicle\n'
            f'Overall Difference: {overall_diff:+.1f}%'
        )

        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

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
    print("\n" + "=" * 100)
    print("ACTUAL VS SIMULATED COMPARISON")
    print("=" * 100)

    # Calculate difference
    merged_df['difference'] = merged_df['revenue_per_vehicle'] - merged_df['actual_gb_per_vehicle']
    merged_df['difference_pct'] = (merged_df['difference'] / merged_df['actual_gb_per_vehicle'] * 100)

    # Format for display
    display_df = merged_df[['month', 'actual_gb_per_vehicle', 'revenue_per_vehicle',
                            'difference', 'difference_pct']].copy()
    display_df.columns = ['Month', 'Actual (SAR/veh)', 'Simulated (SAR/veh)', 'Difference (SAR)', 'Difference (%)']

    # Round values
    display_df['Actual (SAR/veh)'] = display_df['Actual (SAR/veh)'].round(2)
    display_df['Simulated (SAR/veh)'] = display_df['Simulated (SAR/veh)'].round(2)
    display_df['Difference (SAR)'] = display_df['Difference (SAR)'].round(2)
    display_df['Difference (%)'] = display_df['Difference (%)'].round(1)

    print(display_df.to_string(index=False))
    print("=" * 100)

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
        default='actual.csv',
        help='Input CSV file with actual data (default: actual.csv)'
    )
    parser.add_argument(
        '--simulated',
        default='monthly_summary.csv',
        help='Input CSV file with simulated data (default: monthly_summary.csv)'
    )
    parser.add_argument(
        '--output',
        default='visualizations/actual_vs_simulated.png',
        help='Output image file (default: visualizations/actual_vs_simulated.png)'
    )
    parser.add_argument(
        '--line-plot',
        action='store_true',
        help='Also create a line plot comparison'
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

    # Create bar chart
    print(f"\nCreating comparison chart...")
    create_comparison_plot(merged_df, args.output,
                          figsize=(args.width, args.height), dpi=args.dpi)

    # Create line plot if requested
    if args.line_plot:
        line_output = args.output.replace('.png', '_line.png')
        create_line_plot(merged_df, line_output,
                        figsize=(args.width, args.height), dpi=args.dpi)

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
