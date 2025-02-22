import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Scenario generation functions
def generate_slow_reference_curve():
    """Scenario 1: Reference curve is slow to rise with specific reference volumes"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(30)]
    
    # Define the cumulative reference volumes
    cumulative_reference = np.array([
        0, 0, 0, 0, 0,                    # Week-5
        0, 0, 0, 0, 0,                    # Week-4
        3, 3, 4, 7, 10,                   # Week-3
        19, 19, 22, 25, 26,               # Week-2
        32, 33, 33, 34, 33,               # Week-1
        34, 35, 37, 37, 37                # Departure Week
    ])
    
    # Calculate daily reference volumes (difference between consecutive days)
    daily_reference = np.zeros(30)
    daily_reference[0] = cumulative_reference[0]
    daily_reference[1:] = np.diff(cumulative_reference)
    
    # Generate spot volumes that consistently exceed reference volumes
    # We'll add a random increment of 20-40% to the cumulative reference at each point
    excess_factor = np.random.uniform(1.05, 1.1, 30)  # Random factor between 1.2 and 1.4
    cumulative_spot = cumulative_reference * excess_factor
    
    # Calculate daily spot volumes
    daily_volumes = np.zeros(30)
    daily_volumes[0] = cumulative_spot[0]
    daily_volumes[1:] = np.diff(cumulative_spot)
    
    # Ensure daily volumes are non-negative
    daily_volumes = np.maximum(daily_volumes, 0)
    
    # Initialize spot rates
    spot_rate = np.zeros(30)
    base_rate = 1000
    
    # Check each week if previous week's daily volumes were higher than reference
    for week in range(1, 5):  # Start from week 1 to check previous week
        week_start = (week-1) * 7
        week_end = week * 7
        
        # Calculate average daily difference in previous week
        prev_week_volume_higher = np.mean(cumulative_spot[week_start:week_end] > 
                                        cumulative_reference[week_start:week_end])
        
        # If volumes were higher for most of the previous week, increase rate
        if prev_week_volume_higher > 0.5:
            base_rate += 100  # Increase rate by $100
        
        # Set rate for current week
        current_week_start = week * 7
        current_week_end = min((week + 1) * 7, 30)  # Ensure we don't exceed array bounds
        spot_rate[current_week_start:current_week_end] = base_rate
    
    # Set first week's rate
    spot_rate[:7] = 1000
    
    # Calculate daily revenue
    daily_revenue = daily_volumes * spot_rate
    print("daily vol", daily_volumes)
    print("daily spot", spot_rate)
    print("daily revenue", daily_revenue)
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate, daily_revenue,
                         "Slow Reference Curve - Fast Actual Bookings")


def generate_slow_reference_curve_old():
    """Scenario 1: Reference curve is slow to rise"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate slower reference curve
    daily_reference = np.zeros(35)
    daily_reference[:21] = np.random.uniform(0.6, 0.8, 21)  # Slower initial pace
    daily_reference[21:] = np.random.uniform(0.8, 1.0, 14)  # Slight acceleration
    
    # Generate consistently faster actual bookings
    daily_volumes = np.zeros(35)
    daily_volumes[:] = np.random.uniform(1.2, 1.5, 35)  # Consistently higher bookings
    
    # Scale volumes to reasonable TEU ranges
    max_volume = max(np.max(np.cumsum(daily_volumes)), np.max(np.cumsum(daily_reference)))
    scale_factor = 35 / max_volume
    daily_volumes_scaled = daily_volumes * scale_factor
    
    # Initialize spot rates
    spot_rate = np.zeros(35)
    base_rate = 1000
    
    # Check each week if previous week's daily volumes were higher than reference
    for week in range(1, 5):  # Start from week 1 to check previous week
        week_start = (week-1) * 7
        week_end = week * 7
        
        # Calculate average daily difference in previous week
        prev_week_volume_higher = np.mean(daily_volumes[week_start:week_end] > 
                                        daily_reference[week_start:week_end])
        
        # If volumes were higher for most of the previous week, increase rate
        if prev_week_volume_higher > 0.5:
            base_rate += 100  # Increase rate by $100
        
        # Set rate for current week
        current_week_start = week * 7
        current_week_end = (week + 1) * 7
        spot_rate[current_week_start:current_week_end] = base_rate
    
    # Set first week's rate
    spot_rate[:7] = 1000
    
    # Calculate daily revenue
    daily_revenue = daily_volumes_scaled * spot_rate
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate, daily_revenue,
                         "Slow Reference Curve - Fast Actual Bookings")

def generate_slow_reference_curve_old_old():
    """Scenario 1: Reference curve is slow to rise"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate slower reference curve
    daily_reference = np.zeros(35)
    daily_reference[:21] = np.random.uniform(0.6, 0.8, 21)  # Slower initial pace
    daily_reference[21:] = np.random.uniform(0.8, 1.0, 14)  # Slight acceleration
    
    # Generate consistently faster actual bookings
    daily_volumes = np.zeros(35)
    daily_volumes[:] = np.random.uniform(1.2, 1.5, 35)  # Consistently higher bookings
    
    # Initialize spot rates
    spot_rate = np.zeros(35)
    base_rate = 1000
    
    # Check each week if previous week's daily volumes were higher than reference
    for week in range(1, 5):  # Start from week 1 to check previous week
        week_start = (week-1) * 7
        week_end = week * 7
        
        # Calculate average daily difference in previous week
        prev_week_volume_higher = np.mean(daily_volumes[week_start:week_end] > 
                                        daily_reference[week_start:week_end])
        
        # If volumes were higher for most of the previous week, increase rate
        if prev_week_volume_higher > 0.5:
            base_rate += 100  # Increase rate by $100
        
        # Set rate for current week
        current_week_start = week * 7
        current_week_end = (week + 1) * 7
        spot_rate[current_week_start:current_week_end] = base_rate
    
    # Set first week's rate
    spot_rate[:7] = 1000
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Slow Reference Curve - Fast Actual Bookings")

def generate_fast_reference_curve():
    """Scenario 2: Reference curve rises too quickly compared to actual bookings"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(30)]
    
    # Define an accelerated cumulative reference volume curve
    cumulative_reference = np.array([
        0, 0, 2, 4, 6,                     # Week-5 (faster start)
        8, 10, 13, 16, 19,                 # Week-4 (rapid increase)
        22, 24, 26, 28, 30,                # Week-3 (continued strong growth)
        31, 32, 33, 34, 35,                # Week-2 (reaching near target)
        36, 37, 37, 37, 37,                # Week-1 (at target)
        37, 37, 37, 37, 37                 # Departure Week (maintaining target)
    ])
    
    # Calculate daily reference volumes
    daily_reference = np.zeros(30)
    daily_reference[0] = cumulative_reference[0]
    daily_reference[1:] = np.diff(cumulative_reference)
    
    # Generate slower spot volumes (60-80% of reference)
    reduction_factor = np.random.uniform(0.6, 0.8, 30)  # Random factor between 0.6 and 0.8
    cumulative_spot = cumulative_reference * reduction_factor
    
    # Calculate daily spot volumes
    daily_volumes = np.zeros(30)
    daily_volumes[0] = cumulative_spot[0]
    daily_volumes[1:] = np.diff(cumulative_spot)
    
    # Ensure daily volumes are non-negative
    daily_volumes = np.maximum(daily_volumes, 0)
    
    # Initialize spot rates with higher initial values that drop over time
    spot_rate = np.zeros(30)
    base_rate = 1400  # Start with higher rate
    
    # Adjust rates weekly based on the gap between reference and actual volumes
    for week in range(5):
        week_start = week * 7
        week_end = min((week + 1) * 7, 30)
        
        # Calculate volume gap for the week
        volume_gap = np.mean(cumulative_reference[week_start:week_end] - 
                           cumulative_spot[week_start:week_end])
        
        # Reduce rate if volume gap is significant
        if volume_gap > 5:  # Threshold for rate reduction
            base_rate = max(1000, base_rate - 100)  # Don't go below 1000
            
        # Set rate for the week
        spot_rate[week_start:week_end] = base_rate
    
    # Calculate daily revenue
    daily_revenue = daily_volumes * spot_rate
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate, daily_revenue,
                         "Fast Reference Curve - Slow Actual Bookings")

def generate_fast_reference_curve_old():
    """Scenario 2: Reference curve is too fast to rise"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate faster reference curve
    daily_reference = np.zeros(35)
    daily_reference[:21] = np.random.uniform(1.2, 1.5, 21)
    daily_reference[21:] = np.random.uniform(0.6, 0.8, 14)
    
    # Generate slower actual bookings
    daily_volumes = np.zeros(35)
    daily_volumes[:21] = np.random.uniform(0.6, 0.8, 21)
    daily_volumes[21:] = np.random.uniform(0.4, 0.6, 14)
    
    # Generate high initial rates that drop midway
    spot_rate = np.zeros(35)
    spot_rate[:14] = 1400
    spot_rate[14:21] = 1200
    spot_rate[21:28] = 1000
    spot_rate[28:] = 800
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Fast Reference Curve - Slow Actual Bookings")

def generate_unrealistic_target():
    """Scenario 3: Unrealistic Target Volume Setting"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Set unrealistically high target
    csa_target_value = 45  # Much higher than normal
    
    daily_reference = np.random.uniform(1.0, 1.2, 35)
    daily_volumes = np.random.uniform(0.8, 1.0, 35)
    
    # Generate declining spot rates trying to chase volume
    spot_rate = np.zeros(35)
    for week in range(5):
        spot_rate[week*7:(week+1)*7] = 1400 - week * 150
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Unrealistic Target Volume", csa_target_value)

def generate_misestimated_elasticity():
    """Scenario 4: Misestimated Price Elasticity"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate reference volumes
    daily_reference = np.random.uniform(1.0, 1.2, 35)
    
    # Generate actual volumes that don't respond to price changes as expected
    daily_volumes = np.zeros(35)
    daily_volumes[:14] = np.random.uniform(0.8, 1.0, 14)
    daily_volumes[14:28] = np.random.uniform(0.9, 1.1, 14)  # Small increase despite big price drop
    daily_volumes[28:] = np.random.uniform(0.7, 0.9, 7)
    
    # Generate spot rates with significant drops
    spot_rate = np.zeros(35)
    spot_rate[:14] = 1300
    spot_rate[14:28] = 1000  # Big price drop
    spot_rate[28:] = 800   # Further drop
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Misestimated Price Elasticity")

def generate_low_spot_price():
    """Scenario 5: Spot Price Set Too Low"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate reference volumes
    daily_reference = np.random.uniform(1.0, 1.2, 35)
    
    # Generate rapid booking pace
    daily_volumes = np.zeros(35)
    daily_volumes[:14] = np.random.uniform(1.5, 1.8, 14)  # Very fast initial bookings
    daily_volumes[14:] = np.random.uniform(0.3, 0.5, 21)  # Sharp dropoff
    
    # Generate low spot rates
    spot_rate = np.zeros(35)
    spot_rate[:] = 900  # Consistently low rates
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Low Spot Price - Rapid Bookings")

def generate_high_spot_price():
    """Scenario 6: Spot Price Set Too High"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate reference volumes
    daily_reference = np.random.uniform(1.0, 1.2, 35)
    
    # Generate slow booking pace
    daily_volumes = np.zeros(35)
    daily_volumes[:28] = np.random.uniform(0.4, 0.6, 28)  # Very slow bookings
    daily_volumes[28:] = np.random.uniform(1.2, 1.5, 7)   # Late surge with discounts
    
    # Generate high initial rates with late drop
    spot_rate = np.zeros(35)
    spot_rate[:28] = 1500  # High initial rate
    spot_rate[28:] = 900   # Late discount
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "High Spot Price - Slow Bookings")

def generate_volatile_price():
    """Scenario 7: Overreactive Price Adjustments"""
    dates = [datetime.now() - timedelta(weeks=5) + timedelta(days=x) for x in range(35)]
    
    # Generate reference volumes
    daily_reference = np.random.uniform(1.0, 1.2, 35)
    
    # Generate erratic booking pace
    daily_volumes = np.zeros(35)
    for week in range(5):
        daily_volumes[week*7:(week+1)*7] = np.random.uniform(0.8 + 0.2 * np.sin(week), 
                                                            1.2 + 0.2 * np.sin(week), 7)
    
    # Generate volatile spot rates
    spot_rate = np.zeros(35)
    for week in range(5):
        spot_rate[week*7:(week+1)*7] = 1200 + 200 * np.sin(week * 2)
    
    return create_dataset(dates, daily_volumes, daily_reference, spot_rate,
                         "Volatile Price Adjustments")

def calculate_missed_opportunity(df, scenario_title):
    """Calculate missed revenue opportunities based on scenario"""
    
    # Special calculation for Slow Reference Curve scenario
    if scenario_title == "Slow Reference Curve":
        max_spot_rate = df['SpotRate'].max()  # Maximum potential spot rate for this scenario
        actual_volume = df['VolumesSpot'].iloc[-1]
        actual_revenue = df['Revenue'].iloc[-1]
        optimal_revenue = actual_volume * max_spot_rate
        total_missed = optimal_revenue - actual_revenue
        
        return {
            'total_missed': total_missed,
            'volume_inefficiency': 0,  # No volume inefficiency in this scenario
            'price_inefficiency': total_missed,  # All missed opportunity is price-related
            'optimal_revenue': optimal_revenue,
            'actual_revenue': actual_revenue
        }
    
    # Default calculation for other scenarios
    optimal_volume = df['CSATarget'].iloc[-1]
    max_spot_rate = df['SpotRate'].max()
    optimal_revenue = optimal_volume * max_spot_rate
    actual_revenue = df['Revenue'].iloc[-1]
    
    volume_inefficiency = max(0, (optimal_volume - df['VolumesSpot'].iloc[-1]) * df['SpotRate'].iloc[-1])
    price_inefficiency = max(0, df['VolumesSpot'].iloc[-1] * (max_spot_rate - df['SpotRate'].iloc[-1]))
    
    total_missed = optimal_revenue - actual_revenue
    
    return {
        'total_missed': total_missed,
        'volume_inefficiency': volume_inefficiency,
        'price_inefficiency': price_inefficiency,
        'optimal_revenue': optimal_revenue,
        'actual_revenue': actual_revenue
    }


def create_dataset(dates, daily_volumes, daily_reference, spot_rate, daily_revenue, title, csa_target_value=37):
    """
    Helper function to create consistent dataset structure
    
    Parameters:
    dates: list of datetime objects
    daily_volumes: array of daily booking volumes
    daily_reference: array of daily reference volumes
    spot_rate: array of daily spot rates
    daily_revenue: array of daily revenue values
    title: string describing the scenario
    csa_target_value: target value for CSA (default=37)
    
    Returns:
    DataFrame with booking and revenue data, and scenario title
    """
    # Create week labels
    week_labels = []
    start_date = dates[0]
    for d in dates:
        week_num = 5 - (d - start_date).days // 7
        day_name = d.strftime('%a')
        if week_num == 0:
            week_labels.append(f"{day_name}\nDeparture Week")
        else:
            week_labels.append(f"{day_name}\nWeek-{week_num}")
    
    # Calculate cumulative volumes
    volumes_spot = np.cumsum(daily_volumes)
    reference_volumes = np.cumsum(daily_reference)
    
    # Scale volumes to reasonable TEU ranges
    max_volume = max(np.max(volumes_spot), np.max(reference_volumes))
    scale_factor = 35 / max_volume
    
    # Scale daily and cumulative volumes
    daily_volumes_scaled = daily_volumes #* scale_factor
    volumes_spot_scaled = volumes_spot #* scale_factor
    reference_volumes_scaled = reference_volumes #* scale_factor
    
    # Calculate cumulative revenue
    cumulative_revenue = np.cumsum(daily_revenue)
    
    # Generate recommended rate (less volatile than spot)
    recommended_rate = np.ones(len(dates)) * 1050
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'WeekLabel': week_labels,
        'DailyVolumes': daily_volumes_scaled,
        'VolumesSpot': volumes_spot_scaled,
        'ReferenceVolumes': reference_volumes_scaled,
        'CSATarget': csa_target_value,
        'SpotRate': spot_rate,
        'RecommendedRate': recommended_rate,
        'DailyRevenue': daily_revenue,
        'Revenue': cumulative_revenue
    })
    
    return df, title


def create_dataset1(dates, daily_volumes, daily_reference, spot_rate, title, csa_target_value=37):
    """Helper function to create consistent dataset structure"""
    # Create week labels
    week_labels = []
    start_date = dates[0]
    for d in dates:
        week_num = 5 - (d - start_date).days // 7
        day_name = d.strftime('%a')
        if week_num == 0:
            week_labels.append(f"{day_name}\nDeparture Week")
        else:
            week_labels.append(f"{day_name}\nWeek-{week_num}")
    
    # Calculate cumulative volumes
    volumes_spot = np.cumsum(daily_volumes)
    reference_volumes = np.cumsum(daily_reference)
    
    # Scale volumes to reasonable TEU ranges
    max_volume = max(np.max(volumes_spot), np.max(reference_volumes))
    volumes_spot = volumes_spot * (35 / max_volume)
    reference_volumes = reference_volumes * (35 / max_volume)
    
    # Generate recommended rate (less volatile than spot)
    recommended_rate = np.ones(len(dates)) * 1050
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'WeekLabel': week_labels,
        'VolumesSpot': volumes_spot,
        'DailyVolumes': daily_volumes,
        'ReferenceVolumes': reference_volumes,
        'CSATarget': csa_target_value,
        'SpotRate': spot_rate,
        'RecommendedRate': recommended_rate,
        'Revenue': volumes_spot * spot_rate
    })
    
    return df, title

def create_visualization(df, selected_scenario,current_day=None):
    if current_day is not None:
        df = df.iloc[:current_day+1]
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df['WeekLabel'],
            y=df['VolumesSpot'],
            name='Volumes Spot',
            fill='tozeroy',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(144, 238, 144, 0.3)'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['WeekLabel'],
            y=df['ReferenceVolumes'],
            name='Reference Volumes Spot',
            mode='lines',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['WeekLabel'],
            y=df['CSATarget'].repeat(len(df)),
            name='Spot CSA Target',
            mode='lines',
            line=dict(color='red', width=2, dash='dot')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['WeekLabel'],
            y=df['SpotRate'],
            name='Spot Rate',
            mode='lines',
            line=dict(color='gold', width=2),
            yaxis='y2'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['WeekLabel'],
            y=df['RecommendedRate'],
            name='Recommended Neutral Spot Rate',
            mode='lines',
            line=dict(color='gold', width=2, dash='dot'),
            yaxis='y2'
        )
    )
    
    # Add revenue metric
    total_revenue = df['Revenue'].iloc[-1]
    
    # Calculate missed opportunities
    opportunities = calculate_missed_opportunity(df, st.session_state.selected_scenario)
    
    fig.update_layout(
        title=f'Shipping Container Bookings Dashboard<br>Revenue: ${total_revenue:,.0f} | ' +
              f'Missed Opportunity: ${opportunities["total_missed"]:,.0f}<br>' +
              f'Volume Inefficiency: ${opportunities["volume_inefficiency"]:,.0f} | ' +
              f'Price Inefficiency: ${opportunities["price_inefficiency"]:,.0f}',
        xaxis=dict(title='', showgrid=False, gridcolor='lightgray'),
        yaxis=dict(title='TEUs', showgrid=False, gridcolor='lightgray', range=[0, 40]),
        yaxis2=dict(title='USD', overlaying='y', side='right', range=[0, 1600]),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray')
    
    return fig

# Streamlit app
st.set_page_config(page_title="Shipping Container Bookings", layout="wide")

st.title("Shipping Container Bookings Dashboard")

# Scenario selection
scenarios = {
    "Slow Reference Curve": generate_slow_reference_curve,
    "Fast Reference Curve": generate_fast_reference_curve,
    "Unrealistic Target": generate_unrealistic_target,
    "Misestimated Price Elasticity": generate_misestimated_elasticity,
    "Low Spot Price": generate_low_spot_price,
    "High Spot Price": generate_high_spot_price,
    "Volatile Price": generate_volatile_price
}

# Initialize session state
if 'running_simulation' not in st.session_state:
    st.session_state.running_simulation = False
if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = list(scenarios.keys())[0]
if 'df' not in st.session_state:
    st.session_state.df, _ = scenarios[st.session_state.selected_scenario]()

# Convert current_day to integer explicitly
st.session_state.current_day = int(st.session_state.current_day)

# Controls
col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1])
with col1:
    new_scenario = st.selectbox('Select Scenario', list(scenarios.keys()))
    if new_scenario != st.session_state.selected_scenario:
        st.session_state.selected_scenario = new_scenario
        st.session_state.df, _ = scenarios[new_scenario]()  # We don't need to store title separately
        st.session_state.running_simulation = False
        st.session_state.current_day = 0

with col2:
    if st.button('Start Simulation'):
        st.session_state.running_simulation = True
        st.session_state.current_day = 0
with col3:
    if st.button('Reset Simulation'):
        st.session_state.running_simulation = False
        st.session_state.current_day = 0
        st.session_state.df, st.session_state.title = scenarios[st.session_state.selected_scenario]()
with col4:
    simulation_speed = st.slider('Speed', 1, 10, 5)

# Create placeholder for the chart
chart_placeholder = st.empty()

# Description placeholder
desc_placeholder = st.empty()

# Simulation loop
if st.session_state.running_simulation:
    while st.session_state.current_day < len(st.session_state.df):
        # Update visualization
        fig = create_visualization(st.session_state.df, st.session_state.selected_scenario, st.session_state.current_day)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Update simulation state
        st.session_state.current_day += 1
        
        # Control simulation speed
        time.sleep(1/simulation_speed)
        
        if st.session_state.current_day >= len(st.session_state.df):
            st.session_state.running_simulation = False
            break
else:
    # Show full visualization when not simulating
    fig = create_visualization(st.session_state.df, st.session_state.selected_scenario)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

# Scenario descriptions
scenario_descriptions = {
    "Slow Reference Curve": """
    **Issue:** The reference curve rises too slowly compared to actual booking velocity.
    **Impact:** Spot prices are kept artificially low due to the misaligned reference curve.
    **Result:** Lower total revenue despite good booking volume due to suboptimal pricing.
    """,
    
    "Fast Reference Curve": """
    **Issue:** The reference curve rises too quickly during a low season.
    **Impact:** Spot prices remain too high initially, leading to slow bookings.
    **Result:** Late price corrections can't fully recover the booking volume.
    """,
    
    "Unrealistic Target": """
    **Issue:** Target volume is set too high relative to market demand.
    **Impact:** System forces aggressive discounting to chase unrealistic goals.
    **Result:** Lower revenue and potential vessel under-utilization despite price cuts.
    """,
    
    "Misestimated Price Elasticity": """
    **Issue:** Incorrect assumptions about how booking volume responds to price changes.
    **Impact:** Price adjustments don't generate expected volume changes.
    **Result:** Suboptimal balance between price and volume, leading to revenue loss.
    """,
    
    "Low Spot Price": """
    **Issue:** Initial spot price set too low relative to market demand.
    **Impact:** Rapid early bookings at suboptimal rates.
    **Result:** Missed opportunity to capture higher-value bookings later.
    """,
    
    "High Spot Price": """
    **Issue:** Initial spot price set too high for current market conditions.
    **Impact:** Extremely slow booking pace requiring steep late discounts.
    **Result:** Lower total revenue and potential vessel under-utilization.
    """,
    
    "Volatile Price": """
    **Issue:** Overreactive price adjustments to short-term booking variations.
    **Impact:** Unstable pricing creates confusion and erratic booking patterns.
    **Result:** Suboptimal revenue and damaged customer confidence.
    """
}

# Display metrics
if not st.session_state.running_simulation:
    opportunities = calculate_missed_opportunity(st.session_state.df, st.session_state.selected_scenario)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"${opportunities['actual_revenue']:,.0f}")
    with col2:
        st.metric("Total Missed Opportunity", f"${opportunities['total_missed']:,.0f}")
    with col3:
        efficiency_ratio = (opportunities['actual_revenue'] / opportunities['optimal_revenue']) * 100
        st.metric("Revenue Efficiency", f"{efficiency_ratio:.1f}%")

# Display scenario description
st.markdown("### Scenario Description")
st.markdown(scenario_descriptions[st.session_state.selected_scenario])

# Optional data display
if st.checkbox("Show raw data"):
    st.dataframe(st.session_state.df)