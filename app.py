#!/usr/bin/env python3
"""
Load Scheduler Web App
Interactive web application for scheduling loads across vehicles with real-time visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta, date
import numpy as np
import platform

# Import core scheduling functions
from schedule_loads import (
    load_csv, load_active_vehicles, schedule_loads,
    calculate_month_statistics, haversine_distance,
    calculate_month_statistics_by_vehicle_type
)


# Page configuration
st.set_page_config(
    page_title="Load Scheduler",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_actuals_data():
    """
    Load actual performance data.
    Tries actuals_per_vehicle.csv first (per-vehicle granular data),
    falls back to actuals.csv (aggregated data).
    Returns DataFrame with columns: month_name, vehicle_type, vehicles, loads, gb, gb_per_vehicle
    """
    try:
        # Try to load per-vehicle actuals first
        actuals_per_veh = pd.read_csv('inputs/actuals_per_vehicle.csv')

        # Load active vehicles to get vehicle_type mapping
        active_vehicles = pd.read_csv('inputs/active_vehicles.csv')

        # Create vehicle_key to vehicle_type mapping
        vehicle_type_map = active_vehicles[['VehicleKey', 'vehicle_type']].drop_duplicates().set_index('VehicleKey')['vehicle_type'].to_dict()

        # Parse active_month (format: MM/DD/YY)
        actuals_per_veh['month_parsed'] = pd.to_datetime(actuals_per_veh['active_month'], format='%m/%d/%y')
        actuals_per_veh['month_name'] = actuals_per_veh['month_parsed'].dt.strftime('%B %Y')

        # Map vehicle_type
        actuals_per_veh['vehicle_type'] = actuals_per_veh['vehicle_key'].map(vehicle_type_map)

        # Drop rows without vehicle_type (vehicles not in active_vehicles.csv)
        actuals_per_veh = actuals_per_veh.dropna(subset=['vehicle_type'])

        # Aggregate by month and vehicle_type
        actuals_agg = actuals_per_veh.groupby(['month_name', 'vehicle_type']).agg({
            'vehicle_key': 'nunique',  # Count unique vehicles
            'total_loads': 'sum',       # Sum loads
            'total_sp': 'sum'           # Sum revenue (total_sp = revenue)
        }).reset_index()

        # Rename columns to match expected format
        actuals_agg.columns = ['month_name', 'vehicle_type', 'vehicles', 'loads', 'gb']

        # Calculate gb_per_vehicle
        actuals_agg['gb_per_vehicle'] = actuals_agg['gb'] / actuals_agg['vehicles']

        # Note: Don't add 'month' column here - it will be renamed from 'month_name' later

        return actuals_agg

    except FileNotFoundError:
        # Fall back to aggregated actuals.csv
        try:
            actuals_df = pd.read_csv('inputs/actuals.csv')
            # Parse month from actuals (format: MM/DD/YY)
            actuals_df['month_parsed'] = pd.to_datetime(actuals_df['month'], format='%m/%d/%y')
            actuals_df['month_name'] = actuals_df['month_parsed'].dt.strftime('%B %Y')
            return actuals_df
        except FileNotFoundError:
            return None
    except Exception as e:
        st.warning(f"Could not load actuals data: {e}")
        return None


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    /* Make metrics font smaller to prevent truncation */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


def create_gantt_chart(schedule_df, month_name, avg_speed_kmh=60):
    """Create interactive Plotly Gantt charts split by vehicle type."""
    if schedule_df.empty:
        return {}

    # Filter out rows with NaN vehicle_id
    schedule_df = schedule_df[schedule_df['vehicle_id'].notna()].copy()

    if schedule_df.empty:
        return {}

    # Sort by vehicle and sequence
    schedule_df = schedule_df.sort_values(['vehicle_id', 'load_sequence'])

    # Get month start and end dates from schedule
    month_start = schedule_df['pickup_date'].min()
    month_end = schedule_df['dropoff_date'].max()

    # Load active vehicles data to determine vehicle availability
    try:
        active_vehicles_df = pd.read_csv('inputs/active_vehicles.csv')
        active_vehicles_df['active_date'] = pd.to_datetime(active_vehicles_df['active_date'], format='%m/%d/%y')

        # Filter to the month we're visualizing
        month_dt = pd.to_datetime(month_name)
        active_vehicles_month = active_vehicles_df[
            (active_vehicles_df['active_date'].dt.year == month_dt.year) &
            (active_vehicles_df['active_date'].dt.month == month_dt.month)
        ]

        # Create dict of VehicleKey -> list of active dates
        vehicle_availability = {}
        for _, row in active_vehicles_month.iterrows():
            vehicle_key = row['VehicleKey']
            date = row['active_date']
            if vehicle_key not in vehicle_availability:
                vehicle_availability[vehicle_key] = []
            vehicle_availability[vehicle_key].append(date)
    except Exception:
        # If we can't load vehicle availability, continue without inactive periods
        vehicle_availability = {}

    # Create mapping of vehicle_id to (license_plate, origin_city) for display
    vehicle_info = {}
    for vehicle_id in schedule_df['vehicle_id'].unique():
        vehicle_data_df = schedule_df[schedule_df['vehicle_id'] == vehicle_id]
        if not vehicle_data_df.empty:
            vehicle_data = vehicle_data_df.iloc[0]
            license_plate = vehicle_data.get('license_plate', '')
            # Use initial_city from active_vehicles.csv (vehicle's location on first active day of month)
            origin_city = vehicle_data.get('initial_city', 'N/A')
            if not origin_city or pd.isna(origin_city):
                origin_city = 'N/A'
            vehicle_info[vehicle_id] = (license_plate, origin_city)
        else:
            vehicle_info[vehicle_id] = ('', 'N/A')

    # Prepare data for Gantt chart
    gantt_data = []

    for vehicle_id in schedule_df['vehicle_id'].unique():
        vehicle_loads = schedule_df[schedule_df['vehicle_id'] == vehicle_id].sort_values('load_sequence')

        # Create vehicle label with key and optionally plate
        license_plate, origin_city = vehicle_info[vehicle_id]
        if license_plate:
            vehicle_label = f"{vehicle_id} ({license_plate}) - {origin_city}"
        else:
            vehicle_label = f"{vehicle_id} - {origin_city}"

        # Track all busy periods (loads + travel) for this vehicle
        prev_dropoff_date = None
        prev_dropoff_lat = None
        prev_dropoff_lng = None

        for _, load in vehicle_loads.iterrows():
            # Add travel time if there's a previous load
            if prev_dropoff_date is not None:
                travel_time_hours = (
                    haversine_distance(
                        prev_dropoff_lat, prev_dropoff_lng,
                        load['pickup_lat'], load['pickup_lng']
                    ) / avg_speed_kmh
                )
                travel_end = prev_dropoff_date + timedelta(hours=travel_time_hours)

                gantt_data.append({
                    'Vehicle': vehicle_label,
                    'Task': f'Travel',
                    'Start': prev_dropoff_date,
                    'Finish': travel_end,
                    'Type': 'Travel',
                    'Revenue': 0,
                    'Load_ID': '',
                    'Entity': '',
                    'Pickup_City': '',
                    'Destination_City': '',
                    'Pickup_Date': None,
                    'Dropoff_Date': None,
                    'Status': '',
                    'Rental': '',
                    'Duration_Hours': 0,
                    'GB_Per_Day_Median': 0
                })

            # Add load
            gantt_data.append({
                'Vehicle': vehicle_label,
                'Task': f'Load {load["load_id"]}',
                'Start': load['pickup_date'],
                'Finish': load['dropoff_date'],
                'Type': 'Load',
                'Revenue': load['revenue'],
                'Load_ID': load['load_id'],
                'Entity': load.get('entity', 'N/A'),
                'Pickup_City': load.get('pickup_city', 'N/A'),
                'Destination_City': load.get('destination_city', 'N/A'),
                'Pickup_Date': load['pickup_date'],
                'Dropoff_Date': load['dropoff_date'],
                'Status': load.get('status', 'N/A'),
                'Rental': 'Yes' if load.get('rental') else 'No',
                'Duration_Hours': load.get('duration_hours', 0),
                'GB_Per_Day_Median': load.get('gb_per_day_median', 0)
            })

            prev_dropoff_date = load['dropoff_date']
            prev_dropoff_lat = load['dropoff_lat']
            prev_dropoff_lng = load['dropoff_lng']

        # Find inactive days (when vehicle is NOT in active_vehicles.csv for this month)
        # Get the month start and end dates
        month_start_date = schedule_df['pickup_date'].min().replace(day=1)
        month_end_date = (month_start_date + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

        # Get all dates in the month
        all_dates_in_month = pd.date_range(start=month_start_date, end=month_end_date, freq='D')

        # Get active dates for this vehicle
        if vehicle_id in vehicle_availability:
            active_dates_set = set(d.date() for d in vehicle_availability[vehicle_id])
        else:
            active_dates_set = set()

        # Find inactive dates (dates in month but NOT in active_vehicles.csv)
        inactive_dates = []
        for date in all_dates_in_month:
            if date.date() not in active_dates_set:
                inactive_dates.append(date)

        # Group consecutive inactive dates
        if inactive_dates:
            current_start = inactive_dates[0]
            prev_date = inactive_dates[0]

            for date in inactive_dates[1:]:
                # If dates are consecutive (within 1 day)
                if (date - prev_date).days <= 1:
                    prev_date = date
                else:
                    # End the current period and start a new one
                    gantt_data.append({
                        'Vehicle': vehicle_label,
                        'Task': 'Not Active',
                        'Start': pd.Timestamp(current_start.date()),
                        'Finish': pd.Timestamp(prev_date.date()) + timedelta(days=1),
                        'Type': 'Not Active',
                        'Revenue': 0,
                        'Load_ID': '',
                        'Entity': '',
                        'Pickup_City': '',
                        'Destination_City': '',
                        'Pickup_Date': None,
                        'Dropoff_Date': None,
                        'Status': '',
                        'Rental': '',
                        'Duration_Hours': 0,
                        'GB_Per_Day_Median': 0
                    })
                    current_start = date
                    prev_date = date

            # Add the last period
            gantt_data.append({
                'Vehicle': vehicle_label,
                'Task': 'Not Active',
                'Start': pd.Timestamp(current_start.date()),
                'Finish': pd.Timestamp(prev_date.date()) + timedelta(days=1),
                'Type': 'Not Active',
                'Revenue': 0,
                'Load_ID': '',
                'Entity': '',
                'Pickup_City': '',
                'Destination_City': '',
                'Pickup_Date': None,
                'Dropoff_Date': None,
                'Status': '',
                'Rental': '',
                'Duration_Hours': 0,
                'GB_Per_Day_Median': 0
            })

    gantt_df = pd.DataFrame(gantt_data)

    # Build custom hover text that only shows non-empty fields
    hover_text = []
    for _, row in gantt_df.iterrows():
        if row['Type'] == 'Travel':
            hover_text.append('<b>Travel</b>')
        elif row['Type'] == 'Not Active':
            hover_text.append('<b>Not Active</b><br>Vehicle not in active system')
        else:
            # Build hover text dynamically based on available data
            parts = []

            # Load ID - always show if available
            if row['Load_ID'] and str(row['Load_ID']).strip() and str(row['Load_ID']) != 'nan':
                parts.append(f"<b>Load: {row['Load_ID']}</b>")
            else:
                parts.append("<b>Load</b>")

            # Entity (without label)
            if row['Entity'] and str(row['Entity']).strip() and str(row['Entity']) not in ['nan', 'N/A', 'None']:
                parts.append(f"{row['Entity']}")

            # Route information
            pickup = row['Pickup_City']
            destination = row['Destination_City']
            if pickup and str(pickup).strip() and str(pickup) not in ['nan', 'N/A', 'None']:
                if destination and str(destination).strip() and str(destination) not in ['nan', 'N/A', 'None']:
                    parts.append(f"{pickup} → {destination}")
                else:
                    parts.append(f"From: {pickup}")
            elif destination and str(destination).strip() and str(destination) not in ['nan', 'N/A', 'None']:
                parts.append(f"To: {destination}")

            # Pickup date with time
            if row.get('Pickup_Date') is not None and not pd.isna(row['Pickup_Date']):
                pickup_date = pd.to_datetime(row['Pickup_Date'])
                parts.append(f"Pickup: {pickup_date.strftime('%b %d, %Y %H:%M')}")

            # Revenue - always show
            if row['Revenue'] and row['Revenue'] > 0:
                parts.append(f"Revenue: SAR {row['Revenue']:,.0f}")

            # Duration
            if row['Duration_Hours'] and row['Duration_Hours'] > 0:
                duration_days = row['Duration_Hours'] / 24
                if duration_days >= 1:
                    parts.append(f"Duration: {duration_days:.1f} days")
                else:
                    parts.append(f"Duration: {row['Duration_Hours']:.1f} hours")

            # GB per day (median)
            if row.get('GB_Per_Day_Median') and row['GB_Per_Day_Median'] > 0:
                parts.append(f"GB/Day: SAR {row['GB_Per_Day_Median']:,.0f}")

            # Status
            if row['Status'] and str(row['Status']).strip() and str(row['Status']) not in ['nan', 'N/A', 'None']:
                parts.append(f"Status: {row['Status']}")

            # Rental
            if row['Rental'] and str(row['Rental']).strip() and str(row['Rental']) not in ['nan', 'N/A', 'None', 'No']:
                parts.append(f"Rental: {row['Rental']}")

            hover_text.append('<br>'.join(parts))

    gantt_df['hover_text'] = hover_text

    # Get vehicle types from original schedule_df (not gantt_df since gantt includes travel/inactive)
    vehicle_types = schedule_df['vehicle_type'].dropna().unique()

    # If no vehicle_type info, return single chart
    if len(vehicle_types) == 0 or 'vehicle_type' not in schedule_df.columns:
        vehicle_types = ['All Vehicles']
        gantt_df['vehicle_type'] = 'All Vehicles'
    else:
        # Map vehicle IDs to their types
        vehicle_type_map = schedule_df[['vehicle_id', 'vehicle_type']].drop_duplicates().set_index('vehicle_id')['vehicle_type'].to_dict()

        # Extract vehicle_id from Vehicle label
        # Format: "VehicleKey (Plate) - City" or "VehicleKey - City"
        def extract_vehicle_id(vehicle_label):
            # Split by ' - ' and take first part
            first_part = vehicle_label.split(' - ')[0]
            # Remove plate in parentheses if present: "VehicleKey (Plate)" -> "VehicleKey"
            if '(' in first_part:
                return first_part.split(' (')[0]
            return first_part

        gantt_df['vehicle_type'] = gantt_df['Vehicle'].map(lambda v: vehicle_type_map.get(extract_vehicle_id(v), 'Unknown'))

    # Create a figure for each vehicle type
    figures = {}

    for vehicle_type in sorted(vehicle_types):
        type_gantt_df = gantt_df[gantt_df['vehicle_type'] == vehicle_type].copy()

        if type_gantt_df.empty:
            continue

        # Create Gantt chart
        fig = px.timeline(
            type_gantt_df,
            x_start='Start',
            x_end='Finish',
            y='Vehicle',
            color='Type',
            title=f'{vehicle_type} - Load Schedule - {month_name}',
            color_discrete_map={
                'Not Active': '#e0e0e0',  # Grey for inactive periods
                'Load': '#3498db',         # Blue for loads
                'Travel': '#FFC107'        # Yellow for travel
            },
            category_orders={'Type': ['Not Active', 'Travel', 'Load']}  # Render Not Active first (background)
        )

        # Update hover text for each trace separately to ensure correct mapping
        for trace in fig.data:
            if trace.name == 'Load':
                # Get indices for Load type
                load_mask = type_gantt_df['Type'] == 'Load'
                load_hover = type_gantt_df[load_mask]['hover_text'].values
                trace.customdata = [[text] for text in load_hover]
                trace.hovertemplate = '%{customdata[0]}<extra></extra>'
                # Add border to make blocks more distinguishable
                trace.marker.line = dict(color='white', width=2)
            elif trace.name == 'Travel':
                # Get indices for Travel type
                travel_mask = type_gantt_df['Type'] == 'Travel'
                travel_hover = type_gantt_df[travel_mask]['hover_text'].values
                trace.customdata = [[text] for text in travel_hover]
                trace.hovertemplate = '%{customdata[0]}<extra></extra>'
                # Add border to make blocks more distinguishable
                trace.marker.line = dict(color='white', width=2)
            elif trace.name == 'Not Active':
                # Get indices for Not Active type
                not_active_mask = type_gantt_df['Type'] == 'Not Active'
                not_active_hover = type_gantt_df[not_active_mask]['hover_text'].values
                trace.customdata = [[text] for text in not_active_hover]
                trace.hovertemplate = '%{customdata[0]}<extra></extra>'
                # Subtle border for background blocks
                trace.marker.line = dict(color='#bdbdbd', width=1)
                # Make it semi-transparent to serve as background
                trace.opacity = 0.5

        # Count unique vehicles in this type
        unique_vehicles_count = type_gantt_df['Vehicle'].nunique()

        fig.update_layout(
            height=max(400, unique_vehicles_count * 40),
            xaxis_title='Date',
            yaxis_title='Vehicle',
            showlegend=True,
            hovermode='closest',
            bargap=0.3,  # Add gap between bars for better visibility
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                dtick=86400000,  # 1 day in milliseconds
                tickformat='%b %d'  # Format: Jan 15
            )
        )

        figures[vehicle_type] = fig

    return figures


def create_comparison_chart(actual_df, simulated_df):
    """Create interactive comparison charts - one for each vehicle type and one total."""
    # Merge data on both month and vehicle_type
    merged = pd.merge(
        actual_df,
        simulated_df[['month', 'vehicle_type', 'total_revenue', 'num_vehicles']],
        on=['month', 'vehicle_type'],
        how='outer'
    )

    # Calculate simulated revenue per vehicle
    if 'actual_vehicles' in merged.columns:
        merged['simulated_revenue_per_vehicle'] = merged['total_revenue'] / merged['actual_vehicles']
    else:
        merged['simulated_revenue_per_vehicle'] = merged['total_revenue'] / merged['num_vehicles']

    # Sort chronologically
    merged['month_dt'] = pd.to_datetime(merged['month'], errors='coerce')
    merged = merged.sort_values('month_dt')

    # Filter out months > now() - 1 month
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    cutoff_date = datetime.now() - relativedelta(months=1)
    merged = merged[merged['month_dt'] <= cutoff_date]

    # Define colors for different vehicle types
    vehicle_type_colors = {
        'Flatbed': {'actual': '#2ecc71', 'simulated': '#f39c12'},  # Green vs Orange
        'Curtain Sides': {'actual': '#3498db', 'simulated': '#9b59b6'},  # Blue vs Purple
        'Lowbed': {'actual': '#e74c3c', 'simulated': '#e67e22'},  # Red vs Dark Orange
    }

    # List to hold all figures
    figures = []

    # Create a chart for each vehicle type
    for vehicle_type in sorted(merged['vehicle_type'].unique()):
        type_data = merged[merged['vehicle_type'] == vehicle_type].copy()
        colors = vehicle_type_colors.get(vehicle_type, {'actual': '#95a5a6', 'simulated': '#7f8c8d'})

        fig = go.Figure()

        # Add actual revenue per vehicle
        fig.add_trace(
            go.Scatter(
                x=type_data['month'],
                y=type_data['actual_gb_per_vehicle'],
                name='Actual',
                mode='lines+markers',
                marker=dict(size=10, color=colors['actual']),
                line=dict(width=3),
            )
        )

        # Add simulated revenue per vehicle
        fig.add_trace(
            go.Scatter(
                x=type_data['month'],
                y=type_data['simulated_revenue_per_vehicle'],
                name='Simulated',
                mode='lines+markers',
                marker=dict(size=10, color=colors['simulated'], symbol='square'),
                line=dict(width=3, dash='dot'),
            )
        )

        # Update layout
        fig.update_layout(
            title=f'{vehicle_type} - Actual vs Simulated Revenue per Vehicle',
            xaxis_title='Month',
            yaxis_title='Revenue per Vehicle (SAR)',
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        figures.append(fig)

    # Create total (all types combined) chart
    # Aggregate by month across all vehicle types
    total_by_month = merged.groupby('month').agg({
        'actual_gb_per_vehicle': 'mean',
        'simulated_revenue_per_vehicle': 'mean',
        'month_dt': 'first'
    }).reset_index().sort_values('month_dt')

    fig_total = go.Figure()

    # Add actual revenue per vehicle (average across all types)
    fig_total.add_trace(
        go.Scatter(
            x=total_by_month['month'],
            y=total_by_month['actual_gb_per_vehicle'],
            name='Actual',
            mode='lines+markers',
            marker=dict(size=10, color='#3498db'),
            line=dict(width=3),
        )
    )

    # Add simulated revenue per vehicle (average across all types)
    fig_total.add_trace(
        go.Scatter(
            x=total_by_month['month'],
            y=total_by_month['simulated_revenue_per_vehicle'],
            name='Simulated',
            mode='lines+markers',
            marker=dict(size=10, color='#9b59b6', symbol='square'),  # Purple for simulated
            line=dict(width=3, dash='dot'),
        )
    )

    # Update layout
    fig_total.update_layout(
        title='Total (All Types) - Actual vs Simulated Revenue per Vehicle',
        xaxis_title='Month',
        yaxis_title='Average Revenue per Vehicle (SAR)',
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return figures, fig_total


def main():
    # Header
    st.markdown('<div class="main-header">🚚 Load Scheduler</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Use default files
    vehicles_path = 'inputs/active_vehicles.csv'

    try:
        loads_df = load_csv('inputs/loads.csv', None)  # Load all months
    except Exception as e:
        st.error(f"⚠️ Could not load inputs/loads.csv: {e}")
        return

    # Get available months from data
    available_months = []
    try:
        if 'month' in loads_df.columns:
            # Get unique months and sort chronologically
            unique_months = loads_df['month'].unique()
            available_months = sorted(unique_months,
                                     key=lambda x: pd.to_datetime(x, errors='coerce'))
        else:
            loads_df['pickup_date'] = pd.to_datetime(loads_df['pickup_date'])
            available_months = sorted(loads_df['pickup_date'].dt.strftime('%B %Y').unique(),
                                     key=lambda x: pd.to_datetime(x, format='%B %Y', errors='coerce'))
    except Exception as e:
        st.warning(f"Could not load months: {e}")
        available_months = []

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Configuration", "📊 Dashboard & Schedule", "📈 Comparison", "📋 Data"])

    with tab1:
        st.header("⚙️ Configuration")

        st.subheader("🔧 Scheduler Parameters")

        col1, col2 = st.columns(2)
        with col1:
            avg_speed = st.slider("Average Speed (km/h)", 30, 100,
                                 st.session_state.get('avg_speed', 40), 5)
            st.session_state['avg_speed'] = avg_speed
        with col2:
            deadmile_weight = st.slider("Deadmile Weight", 0.0, 1.0,
                                       st.session_state.get('deadmile_weight', 0.0), 0.05,
                                       help="Higher values prioritize reducing empty miles")
            st.session_state['deadmile_weight'] = deadmile_weight

        col3, col4, col5 = st.columns(3)
        with col3:
            duration_tolerance = st.slider("Duration Tolerance (days)", 0.0, 5.0,
                                          st.session_state.get('duration_tolerance', 0.25), 0.25,
                                          help="Extra buffer time added to each load duration for scheduling")
            st.session_state['duration_tolerance'] = duration_tolerance
        with col4:
            km_tolerance = st.slider("KM Tolerance Factor", 1.0, 2.0,
                                    st.session_state.get('km_tolerance', 1.25), 0.05,
                                    help="Multiplier for kilometers to account for actual routes vs direct distance")
            st.session_state['km_tolerance'] = km_tolerance
        with col5:
            optimization_objective = st.selectbox(
                "Optimization Objective",
                ["Revenue & Deadmiles", "Price per Duration Day"],
                index=st.session_state.get('optimization_objective_index', 1),
                help="Choose what to optimize: standard (revenue with deadmile penalty) or price per duration day"
            )
            st.session_state['optimization_objective'] = optimization_objective
            st.session_state['optimization_objective_index'] = 0 if optimization_objective == "Revenue & Deadmiles" else 1

        # Important notes
        st.info("""
        **📝 Important Notes:**
        - **Average Speed**: This parameter only affects the calculation of travel time between loads. It does not impact load durations or scheduling logic.
        - **Duration Tolerance**: Adds extra buffer time to each load's duration when calculating vehicle availability. This helps account for delays and ensures more realistic scheduling. The tolerance is added after the load's dropoff time.
        - **KM Tolerance Factor**: Multiplies all calculated kilometers (both loaded and deadmiles) to account for actual routes vs direct distance. For example, 1.25 means actual distance is 25% longer than straight-line distance.
        - **GB (Revenue)**: GB values are net of additional fees and only reflect the selling price to the customer.
        """)

        st.markdown("---")

        # Month selection
        st.subheader("📅 Select Month(s) to Run")

        col1, col2 = st.columns([1, 2])
        with col1:
            run_mode = st.radio(
                "Run Mode",
                ["All Months", "Specific Months"],
                index=1,
                help="Choose to run all months or select specific ones"
            )

        with col2:
            if run_mode == "Specific Months" and len(available_months) > 0:
                # Calculate previous month from now
                from datetime import datetime
                today = datetime.now()

                # Calculate previous month
                if today.month == 1:
                    previous_month_num = 12
                    previous_year = today.year - 1
                else:
                    previous_month_num = today.month - 1
                    previous_year = today.year

                # Try to find a matching month in available_months
                default_month = None
                for month in available_months:
                    try:
                        month_dt = pd.to_datetime(month)
                        if month_dt.year == previous_year and month_dt.month == previous_month_num:
                            default_month = month
                            break
                    except:
                        continue

                # Fallback to first month if previous month not found
                if default_month is None and available_months:
                    default_month = available_months[0]

                selected_months = st.multiselect(
                    "Select Months",
                    options=available_months,
                    default=[default_month] if default_month else [],
                    help="Select one or more months to process"
                )
            else:
                selected_months = available_months

        st.markdown("---")

        if st.button("🚀 Run Scheduler", type="primary", width='stretch'):
            if run_mode == "Specific Months" and len(selected_months) == 0:
                st.error("⚠️ Please select at least one month to run.")
                st.stop()

            with st.spinner(f"Running scheduler for {len(selected_months)} month(s)..."):
                # Process selected months
                if 'month' in loads_df.columns:
                    unique_months = loads_df['month'].unique()
                else:
                    loads_df['month_temp'] = loads_df['pickup_date'].dt.strftime('%B %Y')
                    unique_months = loads_df['month_temp'].unique()

                # Filter to selected months
                months_to_process = [m for m in unique_months if m in selected_months]

                if len(months_to_process) == 0:
                    st.error("⚠️ No months found in data.")
                    st.stop()

                all_stats = []
                all_schedules = {}

                # Sort months chronologically
                def sort_month_key(m):
                    try:
                        return pd.to_datetime(m)
                    except:
                        try:
                            return pd.to_datetime(m, format='%B %Y')
                        except:
                            return pd.to_datetime('2099-12-31')

                sorted_months = sorted(months_to_process, key=sort_month_key)

                progress_bar = st.progress(0)
                for i, month in enumerate(sorted_months):
                    # Filter data for this month
                    if 'month' in loads_df.columns:
                        month_df = loads_df[loads_df['month'] == month].copy()
                    else:
                        month_df = loads_df[loads_df['month_temp'] == month].copy()

                    # Get optimization objective
                    # Run scheduler
                    duration_tolerance = st.session_state.get('duration_tolerance', 0.25)
                    vehicles = schedule_loads(
                        month_df, None, avg_speed, deadmile_weight, vehicles_path, duration_tolerance
                    )

                    # Store schedule
                    schedule_rows = []
                    # Normalize month format to "Month Year" (e.g., "January 2026")
                    month_normalized = pd.to_datetime(month).strftime('%B %Y')

                    # Debug: Check for vehicles with invalid IDs
                    vehicles_with_loads = [v for v in vehicles if len(v.loads) > 0]
                    print(f"\n🔍 Vehicle ID Debug:")
                    print(f"   Total vehicles with loads: {len(vehicles_with_loads)}")

                    # Check different ID scenarios
                    vehicles_by_id_status = {
                        'both_valid': [],
                        'only_id': [],
                        'only_plate': [],
                        'both_missing': []
                    }

                    for v in vehicles_with_loads:
                        has_id = bool(v.id and (not isinstance(v.id, float) or not pd.isna(v.id)))
                        has_plate = bool(v.license_plate and (not isinstance(v.license_plate, float) or not pd.isna(v.license_plate)))

                        if has_id and has_plate:
                            vehicles_by_id_status['both_valid'].append(v)
                        elif has_id and not has_plate:
                            vehicles_by_id_status['only_id'].append(v)
                        elif not has_id and has_plate:
                            vehicles_by_id_status['only_plate'].append(v)
                        else:
                            vehicles_by_id_status['both_missing'].append(v)

                    print(f"   - Both ID and plate valid: {len(vehicles_by_id_status['both_valid'])} vehicles")
                    print(f"   - Only ID (no plate): {len(vehicles_by_id_status['only_id'])} vehicles")
                    print(f"   - Only plate (no ID): {len(vehicles_by_id_status['only_plate'])} vehicles")
                    print(f"   - Both missing: {len(vehicles_by_id_status['both_missing'])} vehicles")

                    if vehicles_by_id_status['both_missing']:
                        print(f"\n   ⚠️  Vehicles with BOTH missing (will be skipped):")
                        for v in vehicles_by_id_status['both_missing'][:5]:
                            print(f"      id='{v.id}' (type: {type(v.id).__name__}), plate='{v.license_plate}' (type: {type(v.license_plate).__name__}), loads={len(v.loads)}")

                    # Show sample of vehicles that should work
                    if vehicles_by_id_status['only_id']:
                        print(f"\n   Sample vehicles using ID (no plate):")
                        for v in vehicles_by_id_status['only_id'][:3]:
                            print(f"      id='{v.id}', loads={len(v.loads)}")

                    for vehicle in vehicles:
                        # Skip vehicles without valid IDs
                        if not vehicle.id or (isinstance(vehicle.id, float) and pd.isna(vehicle.id)):
                            continue

                        for seq, load in enumerate(vehicle.loads):
                            # Always use vehicle key (VehicleKey) as vehicle_id
                            schedule_rows.append({
                                'month': month_normalized,
                                'vehicle_id': vehicle.id,  # Use vehicle key directly
                                'vehicle_key': vehicle.id,
                                'vehicle_type': vehicle.vehicle_type,
                                'license_plate': vehicle.license_plate,  # Keep plate as separate field
                                'initial_city': vehicle.initial_city,  # Starting city from active_vehicles.csv
                                'load_sequence': seq + 1,
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
                                'rental': load.rental,
                                'gb_per_day_median': load.gb_per_day_median
                            })

                    schedule_month_df = pd.DataFrame(schedule_rows)
                    all_schedules[month] = schedule_month_df

                    # Calculate stats based on actual vehicles in the schedule
                    active_vehicles_by_date, _, _, _, _ = load_active_vehicles(vehicles_path)
                    stats_by_type = calculate_month_statistics_by_vehicle_type(
                        vehicles, avg_speed, active_vehicles_by_date, month_df
                    )

                    # Debug: Check for missing or invalid vehicle_ids
                    if not schedule_month_df.empty:
                        total_rows = len(schedule_month_df)
                        null_ids = schedule_month_df['vehicle_id'].isna().sum()
                        unique_ids = schedule_month_df['vehicle_id'].nunique()

                        if null_ids > 0:
                            st.warning(f"⚠️ Found {null_ids} schedule rows with missing vehicle_id out of {total_rows} total rows")

                        print(f"\nSchedule DataFrame stats:")
                        print(f"  Total rows: {total_rows}")
                        print(f"  Null vehicle_ids: {null_ids}")
                        print(f"  Unique vehicle_ids: {unique_ids}")
                        print(f"  By vehicle type:")
                        for vtype in schedule_month_df['vehicle_type'].unique():
                            type_df = schedule_month_df[schedule_month_df['vehicle_type'] == vtype]
                            print(f"    {vtype}: {type_df['vehicle_id'].nunique()} unique vehicles, {len(type_df)} loads")

                        # Don't override - trust the stats from calculate_month_statistics_by_vehicle_type
                        # The stats calculation already accurately counts vehicles with loads
                        # Overriding here can cause mismatches if vehicle_ids are missing/duplicate

                    # Convert stats_by_type dictionary to list of records
                    for key, stats in stats_by_type.items():
                        if key != 'total':  # Skip total, we'll handle per-type stats
                            stats_record = stats.copy()
                            stats_record['month'] = month_normalized
                            all_stats.append(stats_record)

                    progress_bar.progress((i + 1) / len(sorted_months))

                # Store in session state
                st.session_state['all_schedules'] = all_schedules
                st.session_state['all_stats'] = pd.DataFrame(all_stats)
                st.session_state['available_months'] = sorted_months

                if len(sorted_months) == 1:
                    st.success(f"✅ Scheduler completed successfully for {sorted_months[0]}!")
                else:
                    st.success(f"✅ Scheduler completed successfully for {len(sorted_months)} months!")

                st.info("📊 View results in the 'Dashboard & Schedule' tab")

        # Actual vs Simulated Comparison
        if 'all_stats' in st.session_state:
            stats_df = st.session_state['all_stats']

            st.markdown("---")
            try:
                actuals_df = load_actuals_data()

                if actuals_df is None:
                    raise FileNotFoundError("No actuals data available")

                # Merge with simulated stats
                comparison_df = pd.merge(
                    actuals_df[['month_name', 'vehicle_type', 'vehicles', 'loads', 'gb', 'gb_per_vehicle']],
                    stats_df[['month', 'vehicle_type', 'num_vehicles', 'num_loads', 'total_revenue', 'revenue_per_vehicle']],
                    left_on=['month_name', 'vehicle_type'],
                    right_on=['month', 'vehicle_type'],
                    how='inner'
                )

                if not comparison_df.empty:
                    st.markdown("---")
                    st.subheader("📊 Actual vs Simulated Performance")

                    # Calculate differences
                    comparison_df['revenue_diff'] = comparison_df['total_revenue'] - comparison_df['gb']
                    comparison_df['revenue_diff_pct'] = (comparison_df['revenue_diff'] / comparison_df['gb'] * 100)
                    comparison_df['loads_diff'] = comparison_df['num_loads'] - comparison_df['loads']
                    comparison_df['rev_per_veh_diff'] = comparison_df['revenue_per_vehicle'] - comparison_df['gb_per_vehicle']
                    comparison_df['rev_per_veh_diff_pct'] = (comparison_df['rev_per_veh_diff'] / comparison_df['gb_per_vehicle'] * 100)

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    total_actual_revenue = comparison_df['gb'].sum()
                    total_simulated_revenue = comparison_df['total_revenue'].sum()
                    revenue_diff_pct = ((total_simulated_revenue - total_actual_revenue) / total_actual_revenue * 100) if total_actual_revenue > 0 else 0

                    with col1:
                        st.metric(
                            "Actual Revenue",
                            f"SAR {total_actual_revenue:,.0f}",
                            help="Total revenue from actuals data"
                        )

                    with col2:
                        st.metric(
                            "Simulated Revenue",
                            f"SAR {total_simulated_revenue:,.0f}",
                            delta=f"{revenue_diff_pct:+.1f}%",
                            help="Total revenue from simulation"
                        )

                    avg_actual_rev_per_veh = comparison_df['gb_per_vehicle'].mean()
                    avg_simulated_rev_per_veh = comparison_df['revenue_per_vehicle'].mean()
                    rev_per_veh_diff_pct = ((avg_simulated_rev_per_veh - avg_actual_rev_per_veh) / avg_actual_rev_per_veh * 100) if avg_actual_rev_per_veh > 0 else 0

                    with col3:
                        st.metric(
                            "Actual Revenue/Vehicle",
                            f"SAR {avg_actual_rev_per_veh:,.0f}",
                            help="Average revenue per vehicle from actuals"
                        )

                    with col4:
                        st.metric(
                            "Simulated Revenue/Vehicle",
                            f"SAR {avg_simulated_rev_per_veh:,.0f}",
                            delta=f"{rev_per_veh_diff_pct:+.1f}%",
                            help="Average revenue per vehicle from simulation"
                        )

                    # Detailed comparison table
                    st.subheader("📋 Detailed Comparison by Month & Vehicle Type")
                    st.dataframe(
                        comparison_df[[
                            'month', 'vehicle_type',
                            'gb', 'total_revenue', 'revenue_diff', 'revenue_diff_pct',
                            'loads', 'num_loads', 'loads_diff',
                            'gb_per_vehicle', 'revenue_per_vehicle', 'rev_per_veh_diff_pct'
                        ]].rename(columns={
                            'month': 'Month',
                            'vehicle_type': 'Type',
                            'gb': 'Actual Revenue',
                            'total_revenue': 'Simulated Revenue',
                            'revenue_diff': 'Revenue Diff',
                            'revenue_diff_pct': 'Revenue Diff %',
                            'loads': 'Actual Loads',
                            'num_loads': 'Simulated Loads',
                            'loads_diff': 'Loads Diff',
                            'gb_per_vehicle': 'Actual Rev/Veh',
                            'revenue_per_vehicle': 'Simulated Rev/Veh',
                            'rev_per_veh_diff_pct': 'Rev/Veh Diff %'
                        }).style.format({
                            'Actual Revenue': 'SAR {:,.0f}',
                            'Simulated Revenue': 'SAR {:,.0f}',
                            'Revenue Diff': 'SAR {:,.0f}',
                            'Revenue Diff %': '{:.1f}%',
                            'Actual Loads': '{:.0f}',
                            'Simulated Loads': '{:.0f}',
                            'Loads Diff': '{:.0f}',
                            'Actual Rev/Veh': 'SAR {:,.0f}',
                            'Simulated Rev/Veh': 'SAR {:,.0f}',
                            'Rev/Veh Diff %': '{:.1f}%'
                        }),
                        width='stretch'
                    )
            except FileNotFoundError:
                st.info("💡 Add inputs/actuals_per_vehicle.csv or inputs/actuals.csv to see performance comparison")
            except Exception as e:
                st.warning(f"Could not load actuals comparison: {e}")

            # Monthly breakdown
            st.markdown("---")
            st.subheader("📅 Monthly Breakdown")

            # Prepare columns to display - include idle_days if it exists
            columns_to_display = ['month', 'vehicle_type', 'num_vehicles', 'num_loads']
            if 'total_idle_days' in stats_df.columns:
                columns_to_display.append('total_idle_days')
            columns_to_display.extend(['total_revenue', 'revenue_per_vehicle', 'loaded_ratio'])

            format_dict = {
                'total_revenue': 'SAR {:,.0f}',
                'revenue_per_vehicle': 'SAR {:,.0f}',
                'loaded_ratio': '{:.1f}%'
            }
            if 'total_idle_days' in stats_df.columns:
                format_dict['total_idle_days'] = '{:.0f}'

            st.dataframe(
                stats_df[columns_to_display].style.format(format_dict),
                width='stretch'
            )

    with tab2:
        st.header("Dashboard & Schedule")

        # Display overall metrics if available
        if 'all_stats' in st.session_state:
            st.subheader("📊 Overall Metrics")
            stats_df = st.session_state['all_stats']

            # Calculate total days across all months
            unique_months = stats_df['month'].unique()
            total_days = 0
            for month in unique_months:
                try:
                    month_dt = pd.to_datetime(month)
                    # Get number of days in that month
                    total_days += month_dt.days_in_month
                except:
                    total_days += 30  # fallback

            # Load active vehicles to count actual active days
            try:
                active_vehicles_for_stats = pd.read_csv('inputs/active_vehicles.csv')
                active_vehicles_for_stats['active_date'] = pd.to_datetime(active_vehicles_for_stats['active_date'], format='%m/%d/%y')
            except Exception:
                active_vehicles_for_stats = None

            # Load actuals data for comparison
            actuals_data = load_actuals_data()

            # Prepare combined metrics table
            all_metrics_data = []

            # TOTAL (ALL TYPES)
            total_revenue_all = stats_df['total_revenue'].sum()
            total_loads_all = stats_df['num_loads'].sum()
            total_idle_days_all = stats_df['total_idle_days'].sum() if 'total_idle_days' in stats_df.columns else 0

            # Calculate total vehicles and vehicle-days across all types
            total_vehicle_days_all = 0
            total_vehicles_all = 0

            for month in stats_df['month'].unique():
                month_data = stats_df[stats_df['month'] == month]
                month_vehicles = month_data['num_vehicles_used'].sum()
                total_vehicles_all = max(total_vehicles_all, month_vehicles)

                # Count actual active vehicle-days for this month from active_vehicles.csv
                if active_vehicles_for_stats is not None:
                    try:
                        month_dt = pd.to_datetime(month)
                        month_start = month_dt.replace(day=1)
                        month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                        month_active = active_vehicles_for_stats[
                            (active_vehicles_for_stats['active_date'] >= month_start) &
                            (active_vehicles_for_stats['active_date'] <= month_end)
                        ]
                        total_vehicle_days_all += len(month_active)
                    except:
                        month_dt = pd.to_datetime(month)
                        days_in_month = month_dt.days_in_month
                        total_vehicle_days_all += month_vehicles * days_in_month
                else:
                    try:
                        month_dt = pd.to_datetime(month)
                        days_in_month = month_dt.days_in_month
                        total_vehicle_days_all += month_vehicles * days_in_month
                    except:
                        total_vehicle_days_all += month_vehicles * 30

            revenue_per_vehicle_per_day_all = total_revenue_all / total_vehicle_days_all if total_vehicle_days_all > 0 else 0
            loaded_ratio_all = (stats_df['total_loaded_km'].sum() / stats_df['total_km'].sum() * 100) if stats_df['total_km'].sum() > 0 else 0
            total_km_all = stats_df['total_km'].sum()
            avg_revenue_per_load_all = total_revenue_all / total_loads_all if total_loads_all > 0 else 0

            # Add Total row (Simulated)
            all_metrics_data.append({
                'Vehicle Type': 'Total (All Types) - Simulated',
                'Total Revenue': f"SAR {total_revenue_all:,.0f}",
                'Num Vehicles': f"{total_vehicles_all:.0f}",
                'Total Loads': f"{total_loads_all:.0f}",
                'Total Idle Days': f"{total_idle_days_all:.0f}",
                'Avg Revenue/Load': f"SAR {avg_revenue_per_load_all:,.0f}",
                'Revenue/Vehicle/Day': f"SAR {revenue_per_vehicle_per_day_all:,.0f}",
                'Total Kilometers': f"{total_km_all:,.0f}",
                'Loaded/Total Ratio': f"{loaded_ratio_all:.1f}%"
            })

            # Add Total Actual and Difference rows if actuals data available
            if actuals_data is not None:
                # Calculate actuals for Total (All Types)
                actuals_filtered_all = actuals_data[actuals_data['month_name'].isin(stats_df['month'].unique())]
                if not actuals_filtered_all.empty:
                    actual_revenue_all = actuals_filtered_all['gb'].sum()
                    actual_loads_all = actuals_filtered_all['loads'].sum()
                    actual_vehicles_all = actuals_filtered_all['vehicles'].sum()
                    actual_avg_revenue_per_load_all = actual_revenue_all / actual_loads_all if actual_loads_all > 0 else 0

                    # Calculate actual vehicle-days using actual active days from active_vehicles.csv
                    actual_vehicle_days_all = 0
                    for month in stats_df['month'].unique():
                        # Count actual active vehicle-days for this month from active_vehicles.csv
                        if active_vehicles_for_stats is not None:
                            try:
                                month_dt = pd.to_datetime(month)
                                month_start = month_dt.replace(day=1)
                                month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                                month_active = active_vehicles_for_stats[
                                    (active_vehicles_for_stats['active_date'] >= month_start) &
                                    (active_vehicles_for_stats['active_date'] <= month_end)
                                ]
                                actual_vehicle_days_all += len(month_active)
                            except:
                                # Fallback to old calculation
                                month_actuals = actuals_filtered_all[actuals_filtered_all['month_name'] == month]
                                if not month_actuals.empty:
                                    month_dt = pd.to_datetime(month)
                                    days_in_month = month_dt.days_in_month
                                    actual_vehicle_days_all += month_actuals['vehicles'].sum() * days_in_month
                        else:
                            # Fallback if active_vehicles.csv not available
                            month_actuals = actuals_filtered_all[actuals_filtered_all['month_name'] == month]
                            if not month_actuals.empty:
                                try:
                                    month_dt = pd.to_datetime(month)
                                    days_in_month = month_dt.days_in_month
                                    actual_vehicle_days_all += month_actuals['vehicles'].sum() * days_in_month
                                except:
                                    actual_vehicle_days_all += month_actuals['vehicles'].sum() * 30

                    actual_rev_per_vehicle_per_day_all = actual_revenue_all / actual_vehicle_days_all if actual_vehicle_days_all > 0 else 0

                    # Add Actual row
                    all_metrics_data.append({
                        'Vehicle Type': 'Total (All Types) - Actual',
                        'Total Revenue': f"SAR {actual_revenue_all:,.0f}",
                        'Num Vehicles': f"{actual_vehicles_all:.0f}",
                        'Total Loads': f"{actual_loads_all:.0f}",
                        'Total Idle Days': "N/A",
                        'Avg Revenue/Load': f"SAR {actual_avg_revenue_per_load_all:,.0f}",
                        'Revenue/Vehicle/Day': f"SAR {actual_rev_per_vehicle_per_day_all:,.0f}",
                        'Total Kilometers': "N/A",
                        'Loaded/Total Ratio': "N/A"
                    })

                    # Add Difference row (Simulated - Actual)
                    diff_revenue = total_revenue_all - actual_revenue_all
                    diff_vehicles = total_vehicles_all - actual_vehicles_all
                    diff_loads = total_loads_all - actual_loads_all
                    diff_avg_rev_per_load = avg_revenue_per_load_all - actual_avg_revenue_per_load_all
                    diff_rev_per_veh_per_day = revenue_per_vehicle_per_day_all - actual_rev_per_vehicle_per_day_all

                    all_metrics_data.append({
                        'Vehicle Type': 'Total (All Types) - Difference',
                        'Total Revenue': f"SAR {diff_revenue:+,.0f}",
                        'Num Vehicles': f"{diff_vehicles:+.0f}",
                        'Total Loads': f"{diff_loads:+.0f}",
                        'Total Idle Days': "N/A",
                        'Avg Revenue/Load': f"SAR {diff_avg_rev_per_load:+,.0f}",
                        'Revenue/Vehicle/Day': f"SAR {diff_rev_per_veh_per_day:+,.0f}",
                        'Total Kilometers': "N/A",
                        'Loaded/Total Ratio': "N/A"
                    })

            # Add breakdown by vehicle type
            for vehicle_type in sorted(stats_df['vehicle_type'].unique()):
                type_stats = stats_df[stats_df['vehicle_type'] == vehicle_type]

                total_revenue = type_stats['total_revenue'].sum()
                total_loads = type_stats['num_loads'].sum()
                total_km = type_stats['total_km'].sum()
                total_idle_days = type_stats['total_idle_days'].sum() if 'total_idle_days' in type_stats.columns else 0

                # Calculate avg revenue per vehicle and track number of vehicles
                total_vehicle_days = 0
                num_vehicles_type = 0

                for month in type_stats['month'].unique():
                    month_data = type_stats[type_stats['month'] == month]
                    month_vehicles = month_data['num_vehicles_used'].sum()
                    num_vehicles_type = max(num_vehicles_type, month_vehicles)

                    # Count actual active vehicle-days for this month and vehicle type from active_vehicles.csv
                    if active_vehicles_for_stats is not None:
                        try:
                            month_dt = pd.to_datetime(month)
                            month_start = month_dt.replace(day=1)
                            month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                            month_active_type = active_vehicles_for_stats[
                                (active_vehicles_for_stats['active_date'] >= month_start) &
                                (active_vehicles_for_stats['active_date'] <= month_end) &
                                (active_vehicles_for_stats['vehicle_type'] == vehicle_type)
                            ]
                            total_vehicle_days += len(month_active_type)
                        except:
                            month_dt = pd.to_datetime(month)
                            days_in_month = month_dt.days_in_month
                            total_vehicle_days += month_vehicles * days_in_month
                    else:
                        try:
                            month_dt = pd.to_datetime(month)
                            days_in_month = month_dt.days_in_month
                            total_vehicle_days += month_vehicles * days_in_month
                        except:
                            total_vehicle_days += month_vehicles * 30

                revenue_per_vehicle_per_day = total_revenue / total_vehicle_days if total_vehicle_days > 0 else 0
                loaded_ratio = (type_stats['total_loaded_km'].sum() / type_stats['total_km'].sum() * 100) if type_stats['total_km'].sum() > 0 else 0
                avg_revenue_per_load = total_revenue / total_loads if total_loads > 0 else 0

                # Add vehicle type row (Simulated)
                all_metrics_data.append({
                    'Vehicle Type': f"{vehicle_type} - Simulated",
                    'Total Revenue': f"SAR {total_revenue:,.0f}",
                    'Num Vehicles': f"{num_vehicles_type:.0f}",
                    'Total Loads': f"{total_loads:.0f}",
                    'Total Idle Days': f"{total_idle_days:.0f}",
                    'Avg Revenue/Load': f"SAR {avg_revenue_per_load:,.0f}",
                    'Revenue/Vehicle/Day': f"SAR {revenue_per_vehicle_per_day:,.0f}",
                    'Total Kilometers': f"{total_km:,.0f}",
                    'Loaded/Total Ratio': f"{loaded_ratio:.1f}%"
                })

                # Add Actual and Difference rows for this vehicle type if actuals data available
                if actuals_data is not None:
                    actuals_type = actuals_data[
                        (actuals_data['vehicle_type'] == vehicle_type) &
                        (actuals_data['month_name'].isin(type_stats['month'].unique()))
                    ]
                    if not actuals_type.empty:
                        actual_revenue_type = actuals_type['gb'].sum()
                        actual_loads_type = actuals_type['loads'].sum()
                        actual_vehicles_type = actuals_type['vehicles'].sum()
                        actual_avg_revenue_per_load_type = actual_revenue_type / actual_loads_type if actual_loads_type > 0 else 0

                        # Calculate actual vehicle-days for this type using actual active days from active_vehicles.csv
                        actual_vehicle_days_type = 0
                        for month in type_stats['month'].unique():
                            # Count actual active vehicle-days for this month and vehicle type from active_vehicles.csv
                            if active_vehicles_for_stats is not None:
                                try:
                                    month_dt = pd.to_datetime(month)
                                    month_start = month_dt.replace(day=1)
                                    month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                                    month_active_type = active_vehicles_for_stats[
                                        (active_vehicles_for_stats['active_date'] >= month_start) &
                                        (active_vehicles_for_stats['active_date'] <= month_end) &
                                        (active_vehicles_for_stats['vehicle_type'] == vehicle_type)
                                    ]
                                    actual_vehicle_days_type += len(month_active_type)
                                except:
                                    # Fallback to old calculation
                                    month_actuals = actuals_type[actuals_type['month_name'] == month]
                                    if not month_actuals.empty:
                                        month_dt = pd.to_datetime(month)
                                        days_in_month = month_dt.days_in_month
                                        actual_vehicle_days_type += month_actuals['vehicles'].sum() * days_in_month
                            else:
                                # Fallback if active_vehicles.csv not available
                                month_actuals = actuals_type[actuals_type['month_name'] == month]
                                if not month_actuals.empty:
                                    try:
                                        month_dt = pd.to_datetime(month)
                                        days_in_month = month_dt.days_in_month
                                        actual_vehicle_days_type += month_actuals['vehicles'].sum() * days_in_month
                                    except:
                                        actual_vehicle_days_type += month_actuals['vehicles'].sum() * 30

                        actual_rev_per_vehicle_per_day_type = actual_revenue_type / actual_vehicle_days_type if actual_vehicle_days_type > 0 else 0

                        # Add Actual row
                        all_metrics_data.append({
                            'Vehicle Type': f"{vehicle_type} - Actual",
                            'Total Revenue': f"SAR {actual_revenue_type:,.0f}",
                            'Num Vehicles': f"{actual_vehicles_type:.0f}",
                            'Total Loads': f"{actual_loads_type:.0f}",
                            'Total Idle Days': "N/A",
                            'Avg Revenue/Load': f"SAR {actual_avg_revenue_per_load_type:,.0f}",
                            'Revenue/Vehicle/Day': f"SAR {actual_rev_per_vehicle_per_day_type:,.0f}",
                            'Total Kilometers': "N/A",
                            'Loaded/Total Ratio': "N/A"
                        })

                        # Add Difference row (Simulated - Actual)
                        diff_revenue_type = total_revenue - actual_revenue_type
                        diff_vehicles_type = num_vehicles_type - actual_vehicles_type
                        diff_loads_type = total_loads - actual_loads_type
                        diff_avg_rev_per_load_type = avg_revenue_per_load - actual_avg_revenue_per_load_type
                        diff_rev_per_veh_per_day_type = revenue_per_vehicle_per_day - actual_rev_per_vehicle_per_day_type

                        all_metrics_data.append({
                            'Vehicle Type': f"{vehicle_type} - Difference",
                            'Total Revenue': f"SAR {diff_revenue_type:+,.0f}",
                            'Num Vehicles': f"{diff_vehicles_type:+.0f}",
                            'Total Loads': f"{diff_loads_type:+.0f}",
                            'Total Idle Days': "N/A",
                            'Avg Revenue/Load': f"SAR {diff_avg_rev_per_load_type:+,.0f}",
                            'Revenue/Vehicle/Day': f"SAR {diff_rev_per_veh_per_day_type:+,.0f}",
                            'Total Kilometers': "N/A",
                            'Loaded/Total Ratio': "N/A"
                        })

            # Display combined metrics table
            combined_metrics_df = pd.DataFrame(all_metrics_data)

            # Apply styling to shade difference rows
            def highlight_difference_rows(row):
                if 'Difference' in str(row['Vehicle Type']):
                    return ['background-color: #f0f0f0'] * len(row)
                return [''] * len(row)

            styled_df = combined_metrics_df.style.apply(highlight_difference_rows, axis=1)
            st.dataframe(styled_df, width='stretch', hide_index=True)

            st.markdown("---")

        if 'all_schedules' in st.session_state:
            # Month filter - at the top
            def parse_month_key(month_str):
                try:
                    return pd.to_datetime(month_str, format='%B %Y')
                except:
                    try:
                        return pd.to_datetime(month_str, format='%B')
                    except:
                        try:
                            return pd.to_datetime(month_str)
                        except:
                            return pd.to_datetime('2099-12-31')

            month_options = sorted(
                st.session_state['all_schedules'].keys(),
                key=parse_month_key
            )

            # Month selection
            col1, col2 = st.columns([1, 3])
            with col1:
                month_mode = st.radio(
                    "View",
                    ["All Months", "Specific Month"],
                    help="Choose to view all months or a specific month"
                )

            with col2:
                if month_mode == "Specific Month":
                    selected_month = st.selectbox(
                        "Select Month",
                        options=month_options,
                        help="Choose which month to visualize"
                    )
                    months_to_display = [selected_month]
                else:
                    months_to_display = month_options

            st.markdown("---")

            # Calculate per-vehicle statistics across selected months
            st.subheader("🚛 Vehicle Statistics")

            # Get KM tolerance factor from session state
            km_tolerance = st.session_state.get('km_tolerance', 1.25)

            # Load active vehicles data for idle days calculation
            try:
                active_vehicles_df = pd.read_csv('inputs/active_vehicles.csv')
                active_vehicles_df['active_date'] = pd.to_datetime(active_vehicles_df['active_date'], format='%m/%d/%y')
            except Exception as e:
                st.warning(f"Could not load active vehicles for idle days calculation: {e}")
                active_vehicles_df = None

            all_vehicles_stats = []
            for month in months_to_display:
                schedule_df_month = st.session_state['all_schedules'][month]
                if not schedule_df_month.empty:
                    # Group by vehicle
                    vehicle_stats = schedule_df_month.groupby('vehicle_id').agg({
                        'revenue': 'sum',
                        'load_id': 'count',
                        'vehicle_key': 'first',
                        'license_plate': 'first',
                        'vehicle_type': 'first'
                    }).reset_index()

                    # Create space-separated load IDs per vehicle
                    load_ids_by_vehicle = schedule_df_month.groupby('vehicle_id')['load_id'].apply(
                        lambda x: ' '.join(map(str, x))
                    ).reset_index()
                    load_ids_by_vehicle.columns = ['vehicle_id', 'load_ids']

                    # Merge load IDs into vehicle stats
                    vehicle_stats = vehicle_stats.merge(load_ids_by_vehicle, on='vehicle_id', how='left')

                    # Calculate total kilometers (loaded + deadmiles) and idle days per vehicle
                    for idx, row in vehicle_stats.iterrows():
                        vehicle_id = row['vehicle_id']
                        vehicle_loads = schedule_df_month[schedule_df_month['vehicle_id'] == vehicle_id].sort_values('load_sequence')

                        total_km = 0
                        loaded_km = 0
                        deadmile_km = 0
                        prev_dropoff_lat = None
                        prev_dropoff_lng = None

                        for _, load in vehicle_loads.iterrows():
                            # Add travel distance if not first load (deadmiles)
                            if prev_dropoff_lat is not None:
                                travel_km = haversine_distance(
                                    prev_dropoff_lat, prev_dropoff_lng,
                                    load['pickup_lat'], load['pickup_lng']
                                )
                                deadmile_km += travel_km
                                total_km += travel_km

                            # Add load distance (loaded kilometers)
                            load_km = haversine_distance(
                                load['pickup_lat'], load['pickup_lng'],
                                load['dropoff_lat'], load['dropoff_lng']
                            )
                            loaded_km += load_km
                            total_km += load_km

                            prev_dropoff_lat = load['dropoff_lat']
                            prev_dropoff_lng = load['dropoff_lng']

                        # Apply KM tolerance factor to account for actual routes vs direct distance
                        vehicle_stats.at[idx, 'total_km'] = total_km * km_tolerance
                        vehicle_stats.at[idx, 'loaded_km'] = loaded_km * km_tolerance
                        vehicle_stats.at[idx, 'deadmile_km'] = deadmile_km * km_tolerance

                        # Calculate idle days for this vehicle in this month
                        if active_vehicles_df is not None:
                            # Get the month date range
                            month_dt = pd.to_datetime(month)
                            month_start = month_dt.replace(day=1)
                            month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

                            # Get active dates for this vehicle in this month
                            vehicle_active = active_vehicles_df[
                                (active_vehicles_df['VehicleKey'] == vehicle_id) &
                                (active_vehicles_df['active_date'] >= month_start) &
                                (active_vehicles_df['active_date'] <= month_end)
                            ]
                            active_dates = set(vehicle_active['active_date'].dt.date)

                            # Get dates when vehicle has loads
                            dates_with_loads = set()
                            for _, load in vehicle_loads.iterrows():
                                # Add all dates from pickup to dropoff
                                load_start = pd.to_datetime(load['pickup_date']).date()
                                load_end = pd.to_datetime(load['dropoff_date']).date()
                                current_date = load_start
                                while current_date <= load_end:
                                    dates_with_loads.add(current_date)
                                    current_date += timedelta(days=1)

                            # Idle days = active days without loads
                            idle_dates = active_dates - dates_with_loads
                            vehicle_stats.at[idx, 'idle_days'] = len(idle_dates)
                        else:
                            vehicle_stats.at[idx, 'idle_days'] = 0

                    vehicle_stats['month'] = month
                    all_vehicles_stats.append(vehicle_stats)

            if all_vehicles_stats:
                combined_vehicle_stats = pd.concat(all_vehicles_stats, ignore_index=True)

                # Aggregate across months if multiple months selected
                agg_dict = {
                    'vehicle_key': 'first',
                    'license_plate': 'first',
                    'vehicle_type': 'first',
                    'revenue': 'sum',
                    'load_id': 'sum',
                    'total_km': 'sum',
                    'load_ids': lambda x: ' '.join(x)  # Combine load IDs across months
                }
                if 'idle_days' in combined_vehicle_stats.columns:
                    agg_dict['idle_days'] = 'sum'
                if 'loaded_km' in combined_vehicle_stats.columns:
                    agg_dict['loaded_km'] = 'sum'
                if 'deadmile_km' in combined_vehicle_stats.columns:
                    agg_dict['deadmile_km'] = 'sum'

                final_vehicle_stats = combined_vehicle_stats.groupby('vehicle_id').agg(agg_dict).reset_index()

                # Display separate table for each vehicle type
                for vehicle_type in sorted(final_vehicle_stats['vehicle_type'].unique()):
                    st.markdown(f"### {vehicle_type}")

                    type_stats = final_vehicle_stats[final_vehicle_stats['vehicle_type'] == vehicle_type].copy()

                    # Reorder and rename columns
                    columns_order = ['license_plate', 'vehicle_id', 'vehicle_key', 'revenue', 'load_id', 'load_ids']
                    column_names = ['Vehicle Plate', 'Vehicle Key', 'Vehicle ID', 'Total Revenue (SAR)', 'Number of Loads', 'Load IDs']

                    if 'idle_days' in type_stats.columns:
                        columns_order.append('idle_days')
                        column_names.append('Idle Days')
                    if 'loaded_km' in type_stats.columns:
                        columns_order.append('loaded_km')
                        column_names.append('Loaded KM')
                    if 'deadmile_km' in type_stats.columns:
                        columns_order.append('deadmile_km')
                        column_names.append('Deadmile KM')
                    columns_order.append('total_km')
                    column_names.append('Total KM')

                    type_stats = type_stats[columns_order]
                    type_stats.columns = column_names

                    # Display as table
                    format_dict = {
                        'Total Revenue (SAR)': '{:,.0f}',
                        'Number of Loads': '{:.0f}',
                        'Total KM': '{:,.1f}'
                    }
                    if 'Idle Days' in type_stats.columns:
                        format_dict['Idle Days'] = '{:.0f}'
                    if 'Loaded KM' in type_stats.columns:
                        format_dict['Loaded KM'] = '{:,.1f}'
                    if 'Deadmile KM' in type_stats.columns:
                        format_dict['Deadmile KM'] = '{:,.1f}'

                    st.dataframe(
                        type_stats.style.format(format_dict),
                        width='stretch'
                    )

                    st.markdown("---")

            st.markdown("---")

            # Display breakdown by shipper/entity
            st.subheader("📦 Assignments Breakdown by Shipper (Entity)")

            all_assignments_for_breakdown = []
            for month in months_to_display:
                schedule_df_month = st.session_state['all_schedules'][month]
                if not schedule_df_month.empty:
                    all_assignments_for_breakdown.append(schedule_df_month)

            if all_assignments_for_breakdown:
                combined_for_breakdown = pd.concat(all_assignments_for_breakdown, ignore_index=True)

                # Group by entity
                entity_stats = combined_for_breakdown.groupby('entity').agg({
                    'load_id': 'count',
                    'revenue': 'sum',
                    'vehicle_id': 'nunique',
                    'vehicle_type': lambda x: ', '.join(sorted(x.unique())),
                    'duration_hours': 'sum'
                }).reset_index()

                # Calculate average revenue per load
                entity_stats['avg_revenue_per_load'] = entity_stats['revenue'] / entity_stats['load_id']

                # Rename columns
                entity_stats.columns = [
                    'Shipper/Entity',
                    'Total Loads',
                    'Total Revenue (SAR)',
                    'Vehicles Used',
                    'Vehicle Types',
                    'Total Duration (hrs)',
                    'Avg Revenue/Load (SAR)'
                ]

                # Sort by total revenue descending
                entity_stats = entity_stats.sort_values('Total Revenue (SAR)', ascending=False)

                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Shippers", len(entity_stats))
                with col2:
                    st.metric("Total Loads", entity_stats['Total Loads'].sum())
                with col3:
                    st.metric("Total Revenue", f"SAR {entity_stats['Total Revenue (SAR)'].sum():,.0f}")
                with col4:
                    st.metric("Avg Revenue/Load", f"SAR {entity_stats['Total Revenue (SAR)'].sum() / entity_stats['Total Loads'].sum():,.0f}")

                # Display table
                st.dataframe(
                    entity_stats.style.format({
                        'Total Loads': '{:.0f}',
                        'Total Revenue (SAR)': '{:,.0f}',
                        'Vehicles Used': '{:.0f}',
                        'Total Duration (hrs)': '{:,.1f}',
                        'Avg Revenue/Load (SAR)': '{:,.0f}'
                    }),
                    width='stretch',
                    height=400
                )

                # Download button
                csv_buffer = io.StringIO()
                entity_stats.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Shipper Breakdown as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="shipper_breakdown.csv",
                    mime="text/csv",
                    key="download_shipper_breakdown"
                )

            st.markdown("---")

            # Display detailed assignments table
            st.subheader("📋 All Assignments")

            all_assignments = []
            for month in months_to_display:
                schedule_df_month = st.session_state['all_schedules'][month]
                if not schedule_df_month.empty:
                    all_assignments.append(schedule_df_month)

            if all_assignments:
                combined_assignments = pd.concat(all_assignments, ignore_index=True)

                # Select and reorder columns for display
                display_columns = [
                    'month', 'vehicle_type', 'license_plate', 'vehicle_key', 'vehicle_id',
                    'load_sequence', 'load_id', 'load_key', 'entity',
                    'pickup_city', 'destination_city',
                    'pickup_date', 'dropoff_date', 'duration_hours',
                    'revenue', 'gb_per_day_median', 'status', 'rental',
                    'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng'
                ]

                # Only include columns that exist
                available_columns = [col for col in display_columns if col in combined_assignments.columns]
                assignments_display = combined_assignments[available_columns].copy()

                # Convert rental boolean to True/False string
                if 'rental' in assignments_display.columns:
                    assignments_display['rental'] = assignments_display['rental'].apply(
                        lambda x: 'True' if x else 'False'
                    )

                # Sort by month, vehicle type, vehicle, and load sequence
                sort_columns = ['month', 'vehicle_type']
                if 'license_plate' in assignments_display.columns:
                    sort_columns.append('license_plate')
                sort_columns.append('load_sequence')
                assignments_display = assignments_display.sort_values(sort_columns)

                # Rename columns for better readability
                column_rename = {
                    'month': 'Month',
                    'vehicle_type': 'Vehicle Type',
                    'license_plate': 'Vehicle Plate',
                    'vehicle_key': 'Vehicle Key',
                    'vehicle_id': 'Vehicle ID',
                    'load_sequence': 'Sequence',
                    'load_id': 'Load ID',
                    'load_key': 'Load Key',
                    'entity': 'Entity',
                    'pickup_city': 'Pickup City',
                    'destination_city': 'Destination City',
                    'pickup_date': 'Pickup Date',
                    'dropoff_date': 'Dropoff Date',
                    'duration_hours': 'Duration (hrs)',
                    'revenue': 'Selling Price (SAR)',
                    'gb_per_day_median': 'GB/Day (SAR)',
                    'status': 'Status',
                    'rental': 'Rental',
                    'pickup_lat': 'Pickup Lat',
                    'pickup_lng': 'Pickup Lng',
                    'dropoff_lat': 'Dropoff Lat',
                    'dropoff_lng': 'Dropoff Lng'
                }
                assignments_display = assignments_display.rename(columns=column_rename)

                # Format the table
                format_dict = {
                    'Selling Price (SAR)': '{:,.0f}',
                    'GB/Day (SAR)': '{:,.0f}',
                    'Duration (hrs)': '{:.1f}',
                    'Sequence': '{:.0f}',
                    'Pickup Lat': '{:.6f}',
                    'Pickup Lng': '{:.6f}',
                    'Dropoff Lat': '{:.6f}',
                    'Dropoff Lng': '{:.6f}'
                }

                st.info(f"Total Assignments: {len(assignments_display)}")
                st.dataframe(
                    assignments_display.style.format(format_dict),
                    width='stretch',
                    height=600
                )

                # Download button for assignments
                csv_buffer = io.StringIO()
                assignments_display.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download All Assignments as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="all_assignments.csv",
                    mime="text/csv",
                    key="download_all_assignments"
                )

            st.markdown("---")

            # Display schedules for selected months
            for month in months_to_display:
                schedule_df = st.session_state['all_schedules'][month]

                if not schedule_df.empty:
                    # Create and display Gantt charts (split by vehicle type)
                    avg_speed = st.session_state.get('avg_speed', 60)
                    figures = create_gantt_chart(schedule_df, month, avg_speed)
                    if figures:
                        for vehicle_type, fig in figures.items():
                            st.plotly_chart(fig, width='stretch')

                    # Status breakdown chart by vehicle type
                    st.subheader(f"📊 Loads by Status - {month}")

                    # Create rental category column
                    schedule_df['rental_category'] = schedule_df['rental'].apply(
                        lambda x: 'Rental' if x else 'Non-Rental'
                    )

                    # Create display status: keep COMPLETED as is, show others individually
                    schedule_df['display_status'] = schedule_df['status'].apply(
                        lambda x: 'COMPLETED' if x == 'COMPLETED' else x
                    )

                    # Sort: COMPLETED first, then other statuses alphabetically
                    def status_sort_key(status):
                        if status == 'COMPLETED':
                            return (0, status)
                        else:
                            return (1, status)

                    # Get unique vehicle types
                    vehicle_types = sorted(schedule_df['vehicle_type'].dropna().unique())

                    # Create a chart for each vehicle type
                    for vehicle_type in vehicle_types:
                        st.markdown(f"#### {vehicle_type}")

                        type_df = schedule_df[schedule_df['vehicle_type'] == vehicle_type]

                        # Count loads by status and rental
                        status_rental_counts = type_df.groupby(['display_status', 'rental_category']).size().reset_index(name='Count')

                        status_rental_counts['sort_key'] = status_rental_counts['display_status'].apply(status_sort_key)
                        status_rental_counts = status_rental_counts.sort_values('sort_key')

                        # Create stacked bar chart
                        fig_status = px.bar(
                            status_rental_counts,
                            x='display_status',
                            y='Count',
                            color='rental_category',
                            title=f'{vehicle_type} - Loads by Status',
                            text='Count',
                            color_discrete_map={'Rental': '#3498db', 'Non-Rental': '#ffcdd2'},  # Dark blue for Rental, Light red for Non-Rental
                            barmode='stack'
                        )

                        fig_status.update_traces(textposition='inside', textfont_size=12)
                        fig_status.update_layout(
                            xaxis_title='Status',
                            yaxis_title='Number of Loads',
                            legend_title='Type',
                            height=400
                        )

                        st.plotly_chart(fig_status, use_container_width=True)

                    # Show overall status summary metrics
                    st.markdown("---")
                    st.markdown("#### Overall Summary")
                    col1, col2, col3 = st.columns(3)
                    total_loads = len(schedule_df)

                    with col1:
                        completed = len(schedule_df[schedule_df['status'] == 'COMPLETED'])
                        st.metric("Completed Loads", completed, f"{completed/total_loads*100:.1f}%" if total_loads > 0 else "0%")

                    with col2:
                        st.metric("Total Loads", total_loads)

                    with col3:
                        unique_statuses = schedule_df['status'].nunique()
                        st.metric("Unique Statuses", unique_statuses)

                    # Download button
                    csv_buffer = io.StringIO()
                    schedule_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label=f"📥 Download {month} Schedule",
                        data=csv_buffer.getvalue(),
                        file_name=f"schedule_{month.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_{month}"
                    )

                    if month_mode == "All Months":
                        st.markdown("---")
                else:
                    st.info(f"No loads scheduled for {month}")
        else:
            st.info("👆 Run the scheduler first to see visualizations")

    with tab3:
        st.header("Actual vs Simulated Comparison")

        if 'all_stats' in st.session_state:
            try:
                # Load actuals data
                actuals_df = load_actuals_data()

                if actuals_df is None:
                    raise FileNotFoundError("No actuals data available")

                # Drop the original 'month' column if it exists (to avoid conflicts)
                if 'month' in actuals_df.columns and 'month_name' in actuals_df.columns:
                    actuals_df = actuals_df.drop(columns=['month'])

                # Rename columns for compatibility with comparison chart
                actuals_df = actuals_df.rename(columns={
                    'month_name': 'month',
                    'gb_per_vehicle': 'actual_gb_per_vehicle'
                })

                # Create comparison charts
                vehicle_type_figs, total_fig = create_comparison_chart(actuals_df, st.session_state['all_stats'])

                # Display total chart first
                st.subheader("📊 Total (All Vehicle Types)")
                st.plotly_chart(total_fig, width='stretch')

                st.markdown("---")

                # Display individual vehicle type charts
                st.subheader("📊 By Vehicle Type")
                for fig in vehicle_type_figs:
                    st.plotly_chart(fig, width='stretch')

                # Comparison table
                st.subheader("📋 Detailed Comparison by Vehicle Type")
                merged = pd.merge(
                    actuals_df[['month', 'vehicle_type', 'actual_gb_per_vehicle']],
                    st.session_state['all_stats'][['month', 'vehicle_type', 'total_revenue', 'num_vehicles']],
                    on=['month', 'vehicle_type'],
                    how='outer'
                )
                merged['simulated_gb_per_vehicle'] = merged['total_revenue'] / merged['num_vehicles']
                merged['difference'] = merged['simulated_gb_per_vehicle'] - merged['actual_gb_per_vehicle']
                merged['difference_pct'] = (merged['difference'] / merged['actual_gb_per_vehicle'] * 100)

                # Sort by month and vehicle type
                merged = merged.sort_values(['month', 'vehicle_type'])

                st.dataframe(
                    merged[['month', 'vehicle_type', 'actual_gb_per_vehicle', 'simulated_gb_per_vehicle', 'difference', 'difference_pct']].style.format({
                        'actual_gb_per_vehicle': 'SAR {:,.0f}',
                        'simulated_gb_per_vehicle': 'SAR {:,.0f}',
                        'difference': 'SAR {:,.0f}',
                        'difference_pct': '{:.1f}%'
                    }),
                    width='stretch'
                )
            except FileNotFoundError:
                st.info("📤 Add inputs/actuals_per_vehicle.csv or inputs/actuals.csv file to see comparisons")
            except Exception as e:
                st.error(f"Error loading actuals: {e}")
        else:
            st.info("👆 Run the scheduler first to see comparisons")

    with tab4:
        st.header("Data Preview")

        # Loads Data with filters
        st.subheader("📦 Loads Data")

        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                search_text = st.text_input("Search (Load ID, Entity, Cities)", "", key="loads_search")

            with col2:
                if 'month' in loads_df.columns:
                    month_filter_loads = st.multiselect(
                        "Filter by Month",
                        options=sorted(loads_df['month'].dropna().unique()),
                        default=[],
                        key="loads_month"
                    )
                else:
                    month_filter_loads = []

            with col3:
                if 'status' in loads_df.columns:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=sorted(loads_df['status'].dropna().unique()),
                        default=[],
                        key="loads_status"
                    )
                else:
                    status_filter = []

            with col4:
                # Date range filter
                if 'pickup_date' in loads_df.columns:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(loads_df['pickup_date']):
                        loads_df['pickup_date'] = pd.to_datetime(loads_df['pickup_date'])

                    min_date = loads_df['pickup_date'].min().date()
                    max_date = loads_df['pickup_date'].max().date()

                    date_range = st.date_input(
                        "Pickup Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="loads_date_range"
                    )
                else:
                    date_range = None

        # Apply filters
        filtered_loads = loads_df.copy()

        if search_text:
            mask = (
                filtered_loads['id'].astype(str).str.contains(search_text, case=False, na=False) |
                filtered_loads.get('entity', pd.Series()).astype(str).str.contains(search_text, case=False, na=False) |
                filtered_loads.get('pickup_city', pd.Series()).astype(str).str.contains(search_text, case=False, na=False) |
                filtered_loads.get('destination_city', pd.Series()).astype(str).str.contains(search_text, case=False, na=False)
            )
            filtered_loads = filtered_loads[mask]

        if month_filter_loads:
            filtered_loads = filtered_loads[filtered_loads['month'].isin(month_filter_loads)]

        if status_filter:
            filtered_loads = filtered_loads[filtered_loads['status'].isin(status_filter)]

        # Apply date range filter
        if date_range and 'pickup_date' in filtered_loads.columns:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_loads = filtered_loads[
                    (filtered_loads['pickup_date'].dt.date >= start_date) &
                    (filtered_loads['pickup_date'].dt.date <= end_date)
                ]
            elif isinstance(date_range, date):
                # Single date selected
                filtered_loads = filtered_loads[filtered_loads['pickup_date'].dt.date == date_range]

        st.info(f"Showing {len(filtered_loads)} of {len(loads_df)} loads")
        st.dataframe(filtered_loads, width='stretch', height=400)

        st.markdown("---")

        # Active Vehicles Data with filters
        st.subheader("🚛 Active Vehicles Data")

        try:
            vehicles_df = pd.read_csv(vehicles_path)

            with st.expander("🔍 Filters", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    search_vehicle = st.text_input("Search (Plate, Key, City)", "", key="vehicles_search")

                with col2:
                    if 'vehicle_type' in vehicles_df.columns:
                        type_filter = st.multiselect(
                            "Filter by Vehicle Type",
                            options=sorted(vehicles_df['vehicle_type'].dropna().unique()),
                            default=[],
                            key="vehicles_type"
                        )
                    else:
                        type_filter = []

                with col3:
                    # Date range filter
                    if 'active_date' in vehicles_df.columns:
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(vehicles_df['active_date']):
                            vehicles_df['active_date'] = pd.to_datetime(vehicles_df['active_date'], format='%m/%d/%y')

                        min_date_veh = vehicles_df['active_date'].min().date()
                        max_date_veh = vehicles_df['active_date'].max().date()

                        date_range_veh = st.date_input(
                            "Active Date Range",
                            value=(min_date_veh, max_date_veh),
                            min_value=min_date_veh,
                            max_value=max_date_veh,
                            key="vehicles_date_range"
                        )
                    else:
                        date_range_veh = None

            # Apply filters
            filtered_vehicles = vehicles_df.copy()

            if search_vehicle:
                mask = (
                    filtered_vehicles.get('vehicle_plate', pd.Series()).astype(str).str.contains(search_vehicle, case=False, na=False) |
                    filtered_vehicles.get('VehicleKey', pd.Series()).astype(str).str.contains(search_vehicle, case=False, na=False) |
                    filtered_vehicles.get('destination_city', pd.Series()).astype(str).str.contains(search_vehicle, case=False, na=False)
                )
                filtered_vehicles = filtered_vehicles[mask]

            if type_filter:
                filtered_vehicles = filtered_vehicles[filtered_vehicles['vehicle_type'].isin(type_filter)]

            # Apply date range filter
            if date_range_veh and 'active_date' in filtered_vehicles.columns:
                if isinstance(date_range_veh, tuple) and len(date_range_veh) == 2:
                    start_date_veh, end_date_veh = date_range_veh
                    filtered_vehicles = filtered_vehicles[
                        (filtered_vehicles['active_date'].dt.date >= start_date_veh) &
                        (filtered_vehicles['active_date'].dt.date <= end_date_veh)
                    ]
                elif isinstance(date_range_veh, date):
                    # Single date selected
                    filtered_vehicles = filtered_vehicles[filtered_vehicles['active_date'].dt.date == date_range_veh]

            st.info(f"Showing {len(filtered_vehicles)} of {len(vehicles_df)} vehicle records")
            st.dataframe(filtered_vehicles, width='stretch', height=400)

        except Exception as e:
            st.warning(f"Could not load active vehicles: {e}")


if __name__ == "__main__":
    main()
