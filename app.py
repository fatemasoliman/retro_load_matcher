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
from datetime import datetime, timedelta
import numpy as np

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
</style>
""", unsafe_allow_html=True)


def create_gantt_chart(schedule_df, month_name, avg_speed_kmh=60):
    """Create an interactive Plotly Gantt chart."""
    if schedule_df.empty:
        return None

    # Sort by vehicle and sequence
    schedule_df = schedule_df.sort_values(['vehicle_id', 'load_sequence'])

    # Create mapping of vehicle_id to (vehicle_key, origin_city)
    vehicle_info = {}
    for vehicle_id in schedule_df['vehicle_id'].unique():
        vehicle_data_df = schedule_df[schedule_df['vehicle_id'] == vehicle_id]
        if not vehicle_data_df.empty:
            vehicle_data = vehicle_data_df.iloc[0]
            vehicle_key = vehicle_data.get('vehicle_key', '')
            origin_city = vehicle_data.get('pickup_city', 'N/A')  # First pickup city as origin
            vehicle_info[vehicle_id] = (vehicle_key, origin_city)
        else:
            vehicle_info[vehicle_id] = ('', 'N/A')

    # Prepare data for Gantt chart
    gantt_data = []

    for vehicle_id in schedule_df['vehicle_id'].unique():
        vehicle_loads = schedule_df[schedule_df['vehicle_id'] == vehicle_id].sort_values('load_sequence')

        # Create vehicle label with plate, key, and origin city
        vehicle_key, origin_city = vehicle_info[vehicle_id]
        vehicle_label = f"{vehicle_id} - {vehicle_key} - {origin_city}" if vehicle_key else str(vehicle_id)

        prev_dropoff_date = None
        prev_dropoff_lat = None
        prev_dropoff_lng = None

        for idx, load in vehicle_loads.iterrows():
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
                    'Pickup_City': '',
                    'Destination_City': '',
                    'Status': '',
                    'Rental': ''
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
                'Pickup_City': load.get('pickup_city', 'N/A'),
                'Destination_City': load.get('destination_city', 'N/A'),
                'Status': load.get('status', 'N/A'),
                'Rental': 'Yes' if load.get('rental') else 'No'
            })

            prev_dropoff_date = load['dropoff_date']
            prev_dropoff_lat = load['dropoff_lat']
            prev_dropoff_lng = load['dropoff_lng']

    gantt_df = pd.DataFrame(gantt_data)

    # Build custom hover text that only shows non-empty fields
    hover_text = []
    for idx, row in gantt_df.iterrows():
        if row['Type'] == 'Travel':
            hover_text.append('<b>Travel</b>')
        else:
            # Build hover text dynamically based on available data
            parts = []

            # Load ID - always show if available
            if row['Load_ID'] and str(row['Load_ID']).strip() and str(row['Load_ID']) != 'nan':
                parts.append(f"<b>Load: {row['Load_ID']}</b>")
            else:
                parts.append("<b>Load</b>")

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

            # Revenue - always show
            if row['Revenue'] and row['Revenue'] > 0:
                parts.append(f"Revenue: SAR {row['Revenue']:,.0f}")

            # Status
            if row['Status'] and str(row['Status']).strip() and str(row['Status']) not in ['nan', 'N/A', 'None']:
                parts.append(f"Status: {row['Status']}")

            # Rental
            if row['Rental'] and str(row['Rental']).strip() and str(row['Rental']) not in ['nan', 'N/A', 'None', 'No']:
                parts.append(f"Rental: {row['Rental']}")

            hover_text.append('<br>'.join(parts))

    gantt_df['hover_text'] = hover_text

    # Create Gantt chart
    fig = px.timeline(
        gantt_df,
        x_start='Start',
        x_end='Finish',
        y='Vehicle',
        color='Type',
        title=f'Load Schedule - {month_name}',
        color_discrete_map={'Load': '#3498db', 'Travel': '#95a5a6'},
        hover_data={'hover_text': True}
    )

    # Use custom hover text
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        customdata=gantt_df[['hover_text']].values
    )

    fig.update_layout(
        height=max(400, len(schedule_df['vehicle_id'].unique()) * 40),
        xaxis_title='Date',
        yaxis_title='Vehicle',
        showlegend=True,
        hovermode='closest'
    )

    return fig


def create_comparison_chart(actual_df, simulated_df):
    """Create interactive comparison chart."""
    # Merge data
    merged = pd.merge(
        actual_df,
        simulated_df[['month', 'total_revenue', 'num_vehicles']],
        on='month',
        how='outer'
    )

    # Calculate simulated revenue per vehicle
    if 'actual_vehicles' in merged.columns:
        merged['simulated_revenue_per_vehicle'] = merged['total_revenue'] / merged['actual_vehicles']
    else:
        merged['simulated_revenue_per_vehicle'] = merged['total_revenue'] / merged['num_vehicles']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add actual revenue per vehicle
    fig.add_trace(
        go.Scatter(
            x=merged['month'],
            y=merged['actual_gb_per_vehicle'],
            name='Actual',
            mode='lines+markers',
            marker=dict(size=10, color='#2ecc71'),
            line=dict(width=3)
        ),
        secondary_y=False
    )

    # Add simulated revenue per vehicle
    fig.add_trace(
        go.Scatter(
            x=merged['month'],
            y=merged['simulated_revenue_per_vehicle'],
            name='Simulated',
            mode='lines+markers',
            marker=dict(size=10, color='#3498db', symbol='square'),
            line=dict(width=3)
        ),
        secondary_y=False
    )

    # Add difference bars
    merged['difference'] = merged['simulated_revenue_per_vehicle'] - merged['actual_gb_per_vehicle']
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in merged['difference']]

    fig.add_trace(
        go.Bar(
            x=merged['month'],
            y=merged['difference'],
            name='Difference',
            marker_color=colors,
            opacity=0.3
        ),
        secondary_y=False
    )

    # Update layout
    fig.update_layout(
        title='Actual vs Simulated Revenue per Vehicle',
        xaxis_title='Month',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Revenue per Vehicle (SAR)", secondary_y=False)

    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🚚 Load Scheduler</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📁 Data Files")
        loads_file = st.file_uploader("Loads CSV", type=['csv'], help="Upload loads.csv file")
        vehicles_file = st.file_uploader("Active Vehicles CSV (optional)", type=['csv'])
        actuals_file = st.file_uploader("Actuals CSV (optional)", type=['csv'])

        st.subheader("🔧 Parameters")
        avg_speed = st.slider("Average Speed (km/h)", 30, 100, 60, 5)
        deadmile_weight = st.slider("Deadmile Weight", 0.0, 1.0, 0.3, 0.05,
                                   help="Higher values prioritize reducing empty miles")

        st.subheader("📊 Processing Options")

        # Get available months from data
        if loads_file is None:
            try:
                temp_df = pd.read_csv('inputs/loads.csv')
                if 'month' in temp_df.columns:
                    available_months = sorted(temp_df['month'].unique(),
                                             key=lambda x: pd.to_datetime(x, errors='coerce'))
                else:
                    temp_df['pickup_date'] = pd.to_datetime(temp_df['pickup_date'])
                    available_months = sorted(temp_df['pickup_date'].dt.strftime('%B %Y').unique(),
                                             key=lambda x: pd.to_datetime(x, format='%B %Y', errors='coerce'))
            except:
                available_months = []
        else:
            # Will be populated after file is loaded
            available_months = []

        month_mode = st.radio(
            "Month Selection",
            ["All Months", "Specific Months"],
            help="Choose to process all months or select specific ones"
        )

        if month_mode == "Specific Months" and len(available_months) > 0:
            selected_months = st.multiselect(
                "Select Months",
                options=available_months,
                default=[available_months[0]] if available_months else [],
                help="Select one or more months to process"
            )
            process_all_months = False
            month_filter = selected_months
        else:
            process_all_months = True
            month_filter = None

    # Use default files if none uploaded
    if loads_file is None:
        try:
            loads_df = load_csv('inputs/loads.csv', month_filter)
            st.info("📂 Using default loads.csv from inputs/")
        except:
            st.error("⚠️ Please upload a loads CSV file or ensure inputs/loads.csv exists")
            return
    else:
        loads_df = pd.read_csv(loads_file)
        loads_df['shipper_price'] = loads_df['shipper_price'].apply(
            lambda x: float(str(x).replace(',', ''))
        )
        loads_df['distance'] = loads_df['distance'].apply(
            lambda x: float(str(x).replace(',', ''))
        )
        loads_df['pickup_date'] = pd.to_datetime(loads_df['pickup_date'])
        loads_df['duration_hours'] = loads_df['median_duration_days'] * 24
        loads_df['dropoff_date'] = loads_df['pickup_date'] + pd.to_timedelta(
            loads_df['duration_hours'], unit='h'
        )

    # Save vehicles file temporarily if uploaded
    vehicles_path = 'inputs/active_vehicles.csv'
    if vehicles_file is not None:
        vehicles_df = pd.read_csv(vehicles_file)
        vehicles_df.to_csv('/tmp/active_vehicles.csv', index=False)
        vehicles_path = '/tmp/active_vehicles.csv'

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📅 Schedule", "📈 Comparison", "📋 Data"])

    with tab1:
        st.header("Dashboard")

        if st.button("🚀 Run Scheduler", type="primary", use_container_width=True):
            with st.spinner("Running scheduler..."):
                if process_all_months or (month_filter and len(month_filter) > 0):
                    # Process multiple months
                    if 'month' in loads_df.columns:
                        unique_months = loads_df['month'].unique()
                    else:
                        loads_df['month_temp'] = loads_df['pickup_date'].dt.strftime('%B %Y')
                        unique_months = loads_df['month_temp'].unique()

                    # Filter to selected months if specified
                    if not process_all_months and month_filter:
                        unique_months = [m for m in unique_months if m in month_filter]

                    if len(unique_months) == 0:
                        st.error("⚠️ No months selected. Please select at least one month.")
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

                    sorted_months = sorted(unique_months, key=sort_month_key)

                    progress_bar = st.progress(0)
                    for i, month in enumerate(sorted_months):
                        # Filter data for this month
                        if 'month' in loads_df.columns:
                            month_df = loads_df[loads_df['month'] == month].copy()
                        else:
                            month_df = loads_df[loads_df['month_temp'] == month].copy()

                        # Run scheduler
                        vehicles = schedule_loads(
                            month_df, None, avg_speed, deadmile_weight, vehicles_path
                        )

                        # Store schedule
                        schedule_rows = []
                        for vehicle in vehicles:
                            for seq, load in enumerate(vehicle.loads):
                                schedule_rows.append({
                                    'vehicle_id': vehicle.license_plate if vehicle.license_plate else vehicle.id,
                                    'vehicle_key': vehicle.id,
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
                                    'rental': load.rental
                                })

                        all_schedules[month] = pd.DataFrame(schedule_rows)

                        # Calculate stats
                        active_vehicles_by_date, _, _, _ = load_active_vehicles(vehicles_path)
                        stats_by_type = calculate_month_statistics_by_vehicle_type(
                            vehicles, avg_speed, active_vehicles_by_date, month_df
                        )
                        all_stats.extend(stats_by_type)

                        progress_bar.progress((i + 1) / len(unique_months))

                    # Store in session state
                    st.session_state['all_schedules'] = all_schedules
                    st.session_state['all_stats'] = pd.DataFrame(all_stats)

                    # Show summary of what was processed
                    if len(sorted_months) == 1:
                        st.success(f"✅ Scheduler completed successfully for {sorted_months[0]}!")
                    else:
                        st.success(f"✅ Scheduler completed successfully for {len(sorted_months)} months!")

        # Display metrics
        if 'all_stats' in st.session_state:
            st.subheader("📊 Overall Metrics")
            stats_df = st.session_state['all_stats']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"SAR {stats_df['total_revenue'].sum():,.0f}")
            with col2:
                st.metric("Total Loads", f"{stats_df['num_loads'].sum():,.0f}")
            with col3:
                avg_revenue_per_vehicle = stats_df['total_revenue'].sum() / stats_df['num_vehicles_used'].sum()
                st.metric("Avg Revenue/Vehicle", f"SAR {avg_revenue_per_vehicle:,.0f}")
            with col4:
                loaded_ratio = (stats_df['total_loaded_km'].sum() / stats_df['total_km'].sum() * 100)
                st.metric("Loaded/Total Ratio", f"{loaded_ratio:.1f}%")

            # Monthly breakdown
            st.subheader("📅 Monthly Breakdown")
            st.dataframe(
                stats_df[[
                    'month', 'vehicle_type', 'num_vehicles', 'num_loads',
                    'total_revenue', 'revenue_per_vehicle', 'loaded_ratio'
                ]].style.format({
                    'total_revenue': 'SAR {:,.0f}',
                    'revenue_per_vehicle': 'SAR {:,.0f}',
                    'loaded_ratio': '{:.1f}%'
                }),
                use_container_width=True
            )

    with tab2:
        st.header("Schedule Visualization")

        if 'all_schedules' in st.session_state:
            # Month selector - sort chronologically
            def parse_month_key(month_str):
                try:
                    # Try parsing with different formats
                    return pd.to_datetime(month_str, format='%B %Y')
                except:
                    try:
                        # Try with just month name (no year)
                        return pd.to_datetime(month_str, format='%B')
                    except:
                        try:
                            # Try flexible parsing
                            return pd.to_datetime(month_str)
                        except:
                            # If all else fails, return a far future date so it sorts to end
                            return pd.to_datetime('2099-12-31')

            month_options = sorted(
                st.session_state['all_schedules'].keys(),
                key=parse_month_key
            )
            selected_month = st.selectbox("Select Month", month_options)

            schedule_df = st.session_state['all_schedules'][selected_month]

            if not schedule_df.empty:
                # Create and display Gantt chart
                fig = create_gantt_chart(schedule_df, selected_month, avg_speed)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Download button
                csv_buffer = io.StringIO()
                schedule_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download {selected_month} Schedule",
                    data=csv_buffer.getvalue(),
                    file_name=f"schedule_{selected_month.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No loads scheduled for {selected_month}")
        else:
            st.info("👆 Run the scheduler first to see visualizations")

    with tab3:
        st.header("Actual vs Simulated Comparison")

        if actuals_file is not None and 'all_stats' in st.session_state:
            # Load actuals
            actuals_df = pd.read_csv(actuals_file)

            # Identify columns
            month_col = [c for c in actuals_df.columns if 'month' in c.lower()][0]
            revenue_col = [c for c in actuals_df.columns if 'gb_per_vehicle' in c.lower() or 'per_vehicle' in c.lower()][0]

            actuals_df = actuals_df.rename(columns={
                month_col: 'month',
                revenue_col: 'actual_gb_per_vehicle'
            })

            # Parse month
            try:
                actuals_df['month'] = pd.to_datetime(actuals_df['month']).dt.strftime('%B %Y')
            except:
                pass

            # Create comparison chart
            fig = create_comparison_chart(actuals_df, st.session_state['all_stats'])
            st.plotly_chart(fig, use_container_width=True)

            # Comparison table
            st.subheader("📋 Detailed Comparison")
            merged = pd.merge(
                actuals_df[['month', 'actual_gb_per_vehicle']],
                st.session_state['all_stats'][['month', 'total_revenue', 'num_vehicles']],
                on='month',
                how='outer'
            )
            merged['simulated_gb_per_vehicle'] = merged['total_revenue'] / merged['num_vehicles']
            merged['difference'] = merged['simulated_gb_per_vehicle'] - merged['actual_gb_per_vehicle']
            merged['difference_pct'] = (merged['difference'] / merged['actual_gb_per_vehicle'] * 100)

            st.dataframe(
                merged[['month', 'actual_gb_per_vehicle', 'simulated_gb_per_vehicle', 'difference', 'difference_pct']].style.format({
                    'actual_gb_per_vehicle': 'SAR {:,.0f}',
                    'simulated_gb_per_vehicle': 'SAR {:,.0f}',
                    'difference': 'SAR {:,.0f}',
                    'difference_pct': '{:.1f}%'
                }),
                use_container_width=True
            )
        else:
            st.info("📤 Upload an actuals CSV file and run the scheduler to see comparisons")

    with tab4:
        st.header("Data Preview")

        st.subheader("Loads Data")
        st.dataframe(loads_df.head(100), use_container_width=True)

        if vehicles_file is not None or st.session_state.get('vehicles_df') is not None:
            st.subheader("Active Vehicles Data")
            try:
                vehicles_df = pd.read_csv(vehicles_path)
                st.dataframe(vehicles_df.head(100), use_container_width=True)
            except:
                pass


if __name__ == "__main__":
    main()
