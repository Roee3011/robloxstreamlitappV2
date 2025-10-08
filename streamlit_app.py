import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
import difflib

# Set page config
st.set_page_config(
    page_title="Revenue Analysis & Investment Calculator",
    page_icon="📊",
    layout="wide"
)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@st.cache_data
def load_games_catalog():
    """Load the top 500 earning games catalog"""
    df = pd.read_csv('top_500_earning_games.csv')
    return df

def search_games(games_df, search_term):
    """Search games by name with fuzzy matching suggestions"""
    if not search_term:
        return games_df
    
    # Case-insensitive exact/partial matches first
    exact_mask = games_df['name'].str.contains(search_term, case=False, na=False)
    exact_matches = games_df[exact_mask]
    
    # If we have good matches, return them
    if len(exact_matches) >= 5:
        return exact_matches.head(20)
    
    # If few or no exact matches, add fuzzy matching
    # Filter out NaN values and ensure all names are strings
    valid_games = games_df.dropna(subset=['name'])
    all_game_names = valid_games['name'].astype(str).tolist()
    
    # Get fuzzy matches using difflib
    fuzzy_matches = difflib.get_close_matches(
        str(search_term), 
        all_game_names, 
        n=10,  # Get up to 10 suggestions
        cutoff=0.4  # Lower cutoff for more suggestions
    )
    
    # Combine exact and fuzzy matches
    fuzzy_mask = games_df['name'].isin(fuzzy_matches)
    fuzzy_games = games_df[fuzzy_mask]
    
    # Combine and remove duplicates
    combined = pd.concat([exact_matches, fuzzy_games]).drop_duplicates(subset=['universe_id'])
    
    return combined.head(20)

def get_search_suggestions(games_df, search_term, max_suggestions=5):
    """Get search suggestions based on fuzzy matching"""
    if not search_term or len(search_term) < 2:
        return []
    
    # Filter out NaN values and ensure all names are strings
    valid_games = games_df.dropna(subset=['name'])
    all_game_names = valid_games['name'].astype(str).tolist()
    
    # Get fuzzy matches
    suggestions = difflib.get_close_matches(
        str(search_term),
        all_game_names,
        n=max_suggestions,
        cutoff=0.3
    )
    
    return suggestions

def find_game_data_file(universe_id):
    """Find the daily revenue JSON file for a given universe ID"""
    # Look in Top500v2 folder first
    folder_path = f'Top500v2/universe_{universe_id}'
    if os.path.exists(folder_path):
        # Find daily_revenue_*.json file (but exclude per_visit files)
        pattern = os.path.join(folder_path, 'daily_revenue_*.json')
        files = glob.glob(pattern)
        # Filter out per_visit files
        revenue_files = [f for f in files if 'per_visit' not in f]
        if revenue_files:
            return revenue_files[0]  # Return the first (should be only) match
    
    # If not found in Top500v2, check other folders (sportGames, etc.)
    # This maintains backward compatibility
    sport_games_patterns = [
        f'sportGames/BasketballZero_universe_{universe_id}/daily_revenue_*.json',
        f'sportGames/BlueLockRivals_universe_{universe_id}/daily_revenue_*.json'
    ]
    
    for pattern in sport_games_patterns:
        files = glob.glob(pattern)
        # Filter out per_visit files
        revenue_files = [f for f in files if 'per_visit' not in f]
        if revenue_files:
            return revenue_files[0]
    
    # Check the root universe folder (for the default game)
    if universe_id == '6931042565':
        pattern = f'universe_{universe_id}/daily_revenue_*.json'
        files = glob.glob(pattern)
        # Filter out per_visit files
        revenue_files = [f for f in files if 'per_visit' not in f]
        if revenue_files:
            return revenue_files[0]
    
    return None

@st.cache_data
def load_data(universe_id='6931042565'):
    """Load the daily revenue data for a specific universe ID"""
    file_path = find_game_data_file(universe_id)
    
    if not file_path or not os.path.exists(file_path):
        st.error(f"Data file not found for universe ID: {universe_id}")
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if kpiData exists and is not empty
        if 'kpiData' not in data:
            st.error(f"No kpiData found in file for universe ID: {universe_id}")
            return pd.DataFrame()
        
        if not data['kpiData']:
            st.error(f"Empty kpiData in file for universe ID: {universe_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['kpiData'])
        
        # Check if required columns exist
        if 'Estimated Revenue' not in df.columns:
            st.error(f"'Estimated Revenue' column not found in data for universe ID: {universe_id}")
            st.write("Available columns:", df.columns.tolist())
            return pd.DataFrame()
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            st.error(f"'date' column not found in data for universe ID: {universe_id}")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        st.error(f"Error loading data for universe ID {universe_id}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_additional_games_data():
    """Load additional games data for comparison - now deprecated in favor of dynamic loading"""
    # This function is kept for backward compatibility but returns empty dict
    # The new system allows users to select any game from the top 500 list
    return {}

def calculate_additional_games_weekly_averages(additional_games, robux_to_usd):
    """Calculate weekly averages for additional games"""
    weekly_data = {}
    
    for game_name, game_df in additional_games.items():
        # Convert to USD
        game_df['Estimated Revenue USD'] = game_df['Estimated Revenue'] * robux_to_usd
        
        # Create week column (starting from Monday)
        game_df['week'] = game_df['date'].dt.to_period('W-MON')
        
        # Calculate weekly averages
        weekly_avg = game_df.groupby('week').agg({
            'Estimated Revenue USD': 'mean'
        }).reset_index()
        weekly_avg['week_start'] = weekly_avg['week'].dt.start_time
        
        weekly_data[game_name] = weekly_avg
    
    return weekly_data

def calculate_weekly_averages(df, robux_to_usd):
    """Calculate weekly averages from daily data"""
    # Check if DataFrame is empty or missing required columns
    if df.empty:
        st.error("Cannot calculate weekly averages: DataFrame is empty")
        return df, pd.DataFrame()
    
    if 'Estimated Revenue' not in df.columns:
        st.error("Cannot calculate weekly averages: 'Estimated Revenue' column not found")
        return df, pd.DataFrame()
    
    if 'date' not in df.columns:
        st.error("Cannot calculate weekly averages: 'date' column not found")
        return df, pd.DataFrame()
    
    try:
        # Convert Robux to USD
        df['Estimated Revenue USD'] = df['Estimated Revenue'] * robux_to_usd
        
        # Create a week column (starting from Monday)
        df['week'] = df['date'].dt.to_period('W-MON')
        
        # Calculate weekly averages for both Robux and USD
        weekly_avg = df.groupby('week').agg({
            'Estimated Revenue': 'mean',
            'Estimated Revenue USD': 'mean'
        }).reset_index()
        weekly_avg['week_start'] = weekly_avg['week'].dt.start_time
        
        return df, weekly_avg
    except Exception as e:
        st.error(f"Error calculating weekly averages: {str(e)}")
        return df, pd.DataFrame()

def generate_weekly_projection_with_deviations(start_date, lifetime_mean, december_scale, summer_scale,
                                            normal_week_deviations, 
                                            historical_peak_weekly, last_observed_weekly, decay_pct=0.0, summer_peaks=True, december_peaks=True, peak_summer_month=6, months=18, use_polynomial_growth=False, use_current_average_peaks=False, polynomial_peak_multiplier=1000, use_standard_growth=False, standard_growth_percentage=10, protected_growth_months=1.0):
    """Generate weekly projection using average as base and applying weekly deviation patterns"""
    projection_weekly_dates = []
    projection_weekly_revenues = []
    
    current_date = start_date
    end_date = start_date + pd.DateOffset(months=18)  # Exactly 18 months
    
    # Month definitions
    may_month = 5
    december_month = 12
    
    # Calculate decay factor (convert percentage to decimal)
    decay_factor = 1.0 - (decay_pct / 100.0)
    week_counter = 0  # Track weeks for decay calculation
    
    # Generate weekly data for exactly 18 months
    while current_date < end_date:
        month = current_date.month
        
        # Determine base scaling factor for the month based on toggle settings
        # Note: When use_current_average_peaks is True, the actual peak scaling 
        # will be handled later in the weekly calculation loop
        if not use_current_average_peaks:
            # Traditional mode: use month_scale for peaks
            if month == peak_summer_month and summer_peaks:
                month_scale = summer_scale  # Apply peak only to the specific summer month
            elif month == december_month and december_peaks:
                month_scale = december_scale  # December uses peak multiple when enabled
            else:
                month_scale = 1.0  # All other months use normal scaling
        else:
            # Current average mode: month_scale is not used, peaks handled separately
            month_scale = 1.0
        
        # Get weeks in this month
        if month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1)
        
        # Don't go beyond 18 months
        if next_month > end_date:
            next_month = end_date
        
        # Generate weekly data for this month
        weeks_in_month = []
        temp_date = current_date
        
        while temp_date < next_month:
            # Find Monday of the week
            days_since_monday = temp_date.weekday()
            week_start = temp_date - pd.DateOffset(days=days_since_monday)
            
            if week_start not in weeks_in_month and week_start < end_date:
                weeks_in_month.append(week_start)
            
            temp_date += pd.DateOffset(days=7)
        
        # Apply weekly deviation pattern based on month type and toggle settings
        for week_idx, week_start in enumerate(weeks_in_month):
            if month == peak_summer_month and summer_peaks:
                # Create gradual peak pattern: ramp up to center, then ramp down
                num_weeks = len(weeks_in_month)
                if num_weeks == 1:
                    # Single week - use full peak
                    weekly_deviation = 1.0
                else:
                    # Multi-week month - create symmetric bell curve
                    center_week = (num_weeks - 1) / 2.0
                    distance_from_center = abs(week_idx - center_week)
                    max_distance = num_weeks / 2.0
                    
                    # Bell curve formula: peak at center (1.0), decline to edges (0.6)
                    peak_intensity = 1.0 - (distance_from_center / max_distance) * 0.4
                    weekly_deviation = max(0.6, peak_intensity)
            elif month == december_month and december_peaks:
                # Create gradual peak pattern for December
                num_weeks = len(weeks_in_month)
                if num_weeks == 1:
                    weekly_deviation = 1.0
                else:
                    center_week = (num_weeks - 1) / 2.0
                    distance_from_center = abs(week_idx - center_week)
                    max_distance = num_weeks / 2.0
                    
                    # Bell curve formula: peak at center (1.0), decline to edges (0.6)
                    peak_intensity = 1.0 - (distance_from_center / max_distance) * 0.4
                    weekly_deviation = max(0.6, peak_intensity)
            else:
                # Use small deviations for other months or when peaks are disabled
                deviation_idx = week_idx % len(normal_week_deviations)
                weekly_deviation = normal_week_deviations[deviation_idx]
            
            # For the first projected week, use the last observed weekly value as the baseline
            if week_counter == 0:
                # First week starts from the last observed value
                weekly_revenue = last_observed_weekly
            else:
                # Calculate base weekly revenue using the passed baseline (which is current average when that mode is enabled)
                # When standard growth is enabled, don't apply decay for the protected growth period
                protected_weeks = int(protected_growth_months * 4.33)  # Convert months to weeks (approximately)
                if use_standard_growth and week_counter <= protected_weeks:
                    # No decay for protected period when using standard growth
                    base_weekly_revenue = lifetime_mean * weekly_deviation
                else:
                    # Apply decay starting from week 0 after the protected period ends
                    if use_standard_growth:
                        # Decay starts counting from 0 after protected period
                        decay_weeks = week_counter - protected_weeks
                    else:
                        # Normal decay from beginning
                        decay_weeks = week_counter
                    base_weekly_revenue = lifetime_mean * weekly_deviation * (decay_factor ** decay_weeks)
                
                # Apply peak scaling based on the mode
                if (month == peak_summer_month and summer_peaks) or (month == december_month and december_peaks):
                    if use_current_average_peaks:
                        # Current average mode: create symmetrical peak at 150% of the current week's running average
                        # Use the weekly_deviation as the scaling factor for the symmetrical curve
                        # weekly_deviation ranges from 0.6 to 1.0, we need to map this to create a 150% peak
                        # Transform weekly_deviation (0.6-1.0) to peak scaling (1.0-1.50)
                        peak_scaling = 1.0 + (weekly_deviation - 0.6) / (1.0 - 0.6) * 0.50
                        # Apply decay factor to peaks as well (but not for protected period if standard growth is on)
                        if use_standard_growth and week_counter <= protected_weeks:
                            weekly_revenue = last_observed_weekly * peak_scaling
                        else:
                            # Apply decay starting from week 0 after the protected period ends
                            if use_standard_growth:
                                # Decay starts counting from 0 after protected period
                                decay_weeks = week_counter - protected_weeks
                            else:
                                # Normal decay from beginning
                                decay_weeks = week_counter
                            weekly_revenue = last_observed_weekly * peak_scaling * (decay_factor ** decay_weeks)
                    else:
                        # Historical mode: use the traditional month_scale multiplier
                        weekly_revenue = base_weekly_revenue * month_scale
                else:
                    # Non-peak months: use base calculation
                    weekly_revenue = base_weekly_revenue
                
                # Apply polynomial growth scenario if enabled
                if use_polynomial_growth:
                    # 5th degree polynomial with peaks at 0.5 months and 2 months (very early growth)
                    # Normalize week_counter to months (approximately)
                    month_position = week_counter / 4.33  # Convert weeks to months
                    
                    # Convert percentage to multiplier (e.g., 1000% = 10x, 500% = 5x)
                    peak_multiplier_factor = polynomial_peak_multiplier / 100.0
                    first_peak_multiplier = peak_multiplier_factor  # Use full multiplier for first peak
                    second_peak_multiplier = peak_multiplier_factor * 0.5  # Half of first peak for second peak
                    
                    # Define the polynomial multiplier
                    # Create polynomial coefficients for the desired shape
                    if month_position <= 18:  # Only apply within 18 months
                        # Polynomial function: designed to peak at months 0.5 and 2
                        # Using a scaled 5th degree polynomial
                        x = month_position
                        polynomial_multiplier = (
                            1.0 +  # Base multiplier
                            ((first_peak_multiplier - 1.0) * x / 0.5) * (1 - abs(x - 0.5) / 0.5) * max(0, 1 - abs(x - 0.5) / 0.5) +  # First peak at 0.5 months
                            ((second_peak_multiplier - 1.0) * x / 2.0) * (1 - abs(x - 2) / 2.0) * max(0, 1 - abs(x - 2) / 2.0)    # Second peak at 2 months
                        )
                        
                        # Ensure polynomial doesn't go below 1.0 and smooth the curve
                        polynomial_multiplier = max(1.0, polynomial_multiplier)
                        
                        # Apply smooth decay after peaks
                        if x > 2:
                            decay_after_peak = max(0.5, 1.0 - (x - 2) * 0.08)
                            polynomial_multiplier *= decay_after_peak
                        
                        weekly_revenue *= polynomial_multiplier
                
                # Apply standard growth scenario if enabled (gradual ramp up affecting entire future projection)
                if use_standard_growth and week_counter > 0:
                    # Gradual ramp-up over first 4 weeks, then maintain the full growth rate
                    ramp_weeks = 4
                    if week_counter <= ramp_weeks:
                        # Gradual increase: week 1 = 25%, week 2 = 50%, week 3 = 75%, week 4 = 100%
                        growth_factor = (week_counter / ramp_weeks) * (standard_growth_percentage / 100.0)
                    else:
                        # Full growth rate for all subsequent weeks
                        growth_factor = standard_growth_percentage / 100.0
                    
                    # Apply the growth factor (1.0 + growth_factor gives us the multiplier)
                    weekly_revenue *= (1.0 + growth_factor)
            
            projection_weekly_dates.append(week_start)
            projection_weekly_revenues.append(weekly_revenue)
            
            # Increment week counter for decay calculation
            week_counter += 1
        
        # Move to next month
        current_date = next_month
    
    return projection_weekly_dates, projection_weekly_revenues

def create_revenue_analysis_chart(df, weekly_avg, projection_df, additional_games_weekly=None, show_additional_games=True, current_week_info=None):
    """Create the main revenue analysis chart"""
    fig = go.Figure()
    
    # Convert dates to strings to avoid timestamp arithmetic issues
    df_dates_str = df['date'].dt.strftime('%Y-%m-%d')
    weekly_dates_str = weekly_avg['week_start'].dt.strftime('%Y-%m-%d')
    projection_dates_str = projection_df['date'].dt.strftime('%Y-%m-%d')
    
    # Add additional games data as dim lines (weekly averages)
    if show_additional_games and additional_games_weekly:
        for game_name, game_weekly in additional_games_weekly.items():
            game_dates_str = game_weekly['week_start'].dt.strftime('%Y-%m-%d')
            
            # Set colors for specific games
            if game_name == 'Basketball Zero':
                color = 'orange'
            elif game_name == 'Blue Lock Rivals':
                color = 'pink'
            else:
                color = 'lightgray'
            
            fig.add_trace(go.Scatter(
                x=game_dates_str,
                y=game_weekly['Estimated Revenue USD'],
                mode='lines+markers',
                name=f'{game_name} (Historical)',
                line=dict(color=color, width=0.8),
                marker=dict(size=4, color=color),
                opacity=0.4,
                showlegend=True
            ))
    
    # Plot historical daily revenue
    fig.add_trace(go.Scatter(
        x=df_dates_str,
        y=df['Estimated Revenue USD'],
        mode='lines',
        name='Historical Daily Revenue (USD)',
        line=dict(color='lightblue', width=1),
        opacity=0.6
    ))
    
    # Plot historical weekly averages
    fig.add_trace(go.Scatter(
        x=weekly_dates_str,
        y=weekly_avg['Estimated Revenue USD'],
        mode='markers+lines',
        name='Historical Weekly Average (USD)',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8, color='blue'),
        opacity=0.8
    ))
    
    # Add current running week marker if provided
    if current_week_info and current_week_info.get('show_marker', False):
        current_week_date = current_week_info['date']
        current_week_value = current_week_info['value']
        current_week_days = current_week_info.get('days_count', 0)
        
        fig.add_trace(go.Scatter(
            x=[current_week_date.strftime('%Y-%m-%d')],
            y=[current_week_value],
            mode='markers',
            name=f'Current Running Week ({current_week_days} days)',
            marker=dict(size=12, color='cyan', symbol='circle', line=dict(color='darkblue', width=2)),
            opacity=1.0
        ))
    
    # Plot projected weekly revenue
    fig.add_trace(go.Scatter(
        x=projection_dates_str,
        y=projection_df['Estimated Revenue USD'],
        mode='markers+lines',
        name='Projected Weekly Revenue (USD)',
        line=dict(color='red', width=1.5),
        marker=dict(size=6, color='red'),
        opacity=0.8
    ))
    
    # Highlight special months
    summer_proj_weeks = projection_df[projection_df['date'].dt.month.isin([6, 7, 8])]
    december_proj_weeks = projection_df[projection_df['date'].dt.month == 12]
    may_proj_weeks = projection_df[projection_df['date'].dt.month == 5]
    
    if len(summer_proj_weeks) > 0:
        fig.add_trace(go.Scatter(
            x=summer_proj_weeks['date'].dt.strftime('%Y-%m-%d'),
            y=summer_proj_weeks['Estimated Revenue USD'],
            mode='markers',
            name='Summer Weeks (Jun-Aug)',
            marker=dict(size=10, color='orange', symbol='triangle-up'),
            opacity=0.9
        ))
    
    if len(december_proj_weeks) > 0:
        fig.add_trace(go.Scatter(
            x=december_proj_weeks['date'].dt.strftime('%Y-%m-%d'),
            y=december_proj_weeks['Estimated Revenue USD'],
            mode='markers',
            name='December Weeks (May pattern)',
            marker=dict(size=10, color='green', symbol='diamond'),
            opacity=0.9
        ))
    
    if len(may_proj_weeks) > 0:
        fig.add_trace(go.Scatter(
            x=may_proj_weeks['date'].dt.strftime('%Y-%m-%d'),
            y=may_proj_weeks['Estimated Revenue USD'],
            mode='markers',
            name='May Weeks',
            marker=dict(size=10, color='purple', symbol='circle'),
            opacity=0.9
        ))
    
    # Add vertical line to separate historical from projection
    split_date_str = df['date'].max().strftime('%Y-%m-%d')
    fig.add_shape(
        type="line",
        x0=split_date_str,
        x1=split_date_str,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(
            color="black",
            width=2,
            dash="dash"
        )
    )
    
    # Add annotation for the split line
    fig.add_annotation(
        x=split_date_str,
        y=0.95,
        yref="paper",
        text="Historical/Projection Split",
        showarrow=True,
        arrowhead=2,
        arrowcolor="black",
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Revenue Analysis: Historical Data + 18-Month Weekly Projection',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Daily Revenue (USD)',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        height=600,
        showlegend=True
    )
    
    # Calculate dynamic y-axis maximum based on peak values in the data
    max_values = []
    
    # Add historical daily revenue max
    if not df.empty and 'Estimated Revenue USD' in df.columns:
        max_values.append(df['Estimated Revenue USD'].max())
    
    # Add weekly average max
    if not weekly_avg.empty and 'Estimated Revenue USD' in weekly_avg.columns:
        max_values.append(weekly_avg['Estimated Revenue USD'].max())
    
    # Add projection max
    if not projection_df.empty and 'Estimated Revenue USD' in projection_df.columns:
        max_values.append(projection_df['Estimated Revenue USD'].max())
    
    # Add additional games max if present
    if show_additional_games and additional_games_weekly:
        for game_weekly in additional_games_weekly.values():
            if not game_weekly.empty and 'Estimated Revenue USD' in game_weekly.columns:
                max_values.append(game_weekly['Estimated Revenue USD'].max())
    
    # Calculate y-axis maximum (20% higher than peak, with a minimum of 10000)
    if max_values:
        peak_value = max(max_values)
        y_max = peak_value * 1.2  # 20% higher than peak
        y_max = max(y_max, 10000)  # Minimum of 10,000 for readability
    else:
        y_max = 100000  # Fallback to original value if no data
    
    # Format y-axis to show currency and set dynamic range
    fig.update_yaxes(tickformat='$,.0f', range=[0, y_max])
    
    return fig

def calculate_investment_analysis(df, weekly_avg, projection_df, upfront, earnout, equity_pct, robux_to_usd, earnout_period_months=0, earnout_pct=0):
    """Calculate investment analysis metrics"""
    # Calculate lifetime mean from historical data
    lifetime_mean = df['Estimated Revenue USD'].mean()
    
    # Calculate investor's share
    investor_upfront = upfront * (equity_pct / 100)
    investor_earnout = 0  # Earnout goes to developers, not investors
    investor_total = investor_upfront + investor_earnout
    
    # Calculate projected revenues for different periods
    # Note: projection_df contains weekly averages, so multiply by 7 to get weekly totals
    projection_start = projection_df['date'].min()
    projection_6m = projection_df[projection_df['date'] <= projection_start + pd.DateOffset(months=6)]
    projection_12m = projection_df[projection_df['date'] <= projection_start + pd.DateOffset(months=12)]
    
    # Convert weekly averages to weekly totals (multiply by 7 days)
    next_6_months = (projection_6m['Estimated Revenue USD'] * 7).sum()
    next_12_months = (projection_12m['Estimated Revenue USD'] * 7).sum()
    next_18_months = (projection_df['Estimated Revenue USD'] * 7).sum()
    
    # Calculate previous 6 months from historical data
    historical_end = df['date'].max()
    historical_start_6m = historical_end - pd.DateOffset(months=6)
    previous_6_months = df[df['date'] >= historical_start_6m]['Estimated Revenue USD'].sum()
    
    # Calculate developer earnout (only for 100% equity deals)
    if equity_pct == 100.0 and earnout > 0 and earnout_period_months > 0:
        # Calculate earnout based on revenue during the earnout period
        earnout_period_revenue = projection_df[projection_df['date'] <= projection_start + pd.DateOffset(months=earnout_period_months)]['Estimated Revenue USD'] * 7
        total_earnout_period_revenue = earnout_period_revenue.sum()
        
        # Developer earnout is a percentage of revenue during earnout period (capped at max earnout)
        developer_earnout_rate = earnout_pct / 100.0  # User-defined percentage of revenue during earnout period
        developer_earnout = min(total_earnout_period_revenue * developer_earnout_rate, earnout)
    else:
        developer_earnout = 0
        total_earnout_period_revenue = 0
    
    # Calculate investor's share of revenue (reduced by developer earnout)
    investor_6m = (next_6_months * (equity_pct / 100)) - (developer_earnout if earnout_period_months <= 6 else 0)
    investor_12m = (next_12_months * (equity_pct / 100)) - (developer_earnout if earnout_period_months <= 12 else 0)
    investor_18m = (next_18_months * (equity_pct / 100)) - developer_earnout
    
    # Calculate return multiples
    return_6m = investor_6m / investor_total if investor_total > 0 else 0
    return_12m = investor_12m / investor_total if investor_total > 0 else 0
    return_18m = investor_18m / investor_total if investor_total > 0 else 0
    
    return {
        'lifetime_mean': lifetime_mean,
        'investor_upfront': investor_upfront,
        'investor_earnout': investor_earnout,
        'investor_total': investor_total,
        'previous_6_months': previous_6_months,
        'next_6_months': next_6_months,
        'next_12_months': next_12_months,
        'next_18_months': next_18_months,
        'investor_6m': investor_6m,
        'investor_12m': investor_12m,
        'investor_18m': investor_18m,
        'return_6m': return_6m,
        'return_12m': return_12m,
        'return_18m': return_18m,
        'developer_earnout': developer_earnout,
        'total_earnout_period_revenue': total_earnout_period_revenue,
        'earnout_period_months': earnout_period_months
    }

def main():
    st.title("📊 Revenue Analysis & Investment Calculator")
    st.markdown("---")
    
    # Game Selection Interface
    st.header("🎮 Select Game to Analyze")
    
    # Load games catalog
    games_df = load_games_catalog()
    
    # Create two columns for game selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize session state for search term and selected game
        if 'search_term' not in st.session_state:
            st.session_state.search_term = ""
        if 'selected_game_name' not in st.session_state:
            # Set default selected game to Punch Wall (universe ID: 8574921891)
            default_game = games_df[games_df['universe_id'] == 8574921891]
            if not default_game.empty:
                st.session_state.selected_game_name = default_game.iloc[0]['name']
            else:
                st.session_state.selected_game_name = ""
        
        # Search bar
        search_term = st.text_input(
            "Search for a game:",
            value=st.session_state.search_term,
            placeholder="Type game name to search...",
            help="Search through 500+ top earning games",
            key="game_search_input"
        )
        
        # Update session state
        st.session_state.search_term = search_term
        
        # Show clickable search suggestions if user is typing
        if search_term and len(search_term) >= 2:
            suggestions = get_search_suggestions(games_df, search_term)
            if suggestions:
                st.caption("💡 **Quick Select:**")
                # Create clickable buttons for suggestions
                suggestion_cols = st.columns(min(len(suggestions), 3))  # Max 3 columns
                
                for i, suggestion in enumerate(suggestions[:3]):  # Show max 3 suggestions
                    with suggestion_cols[i]:
                        if st.button(
                            suggestion,
                            key=f"suggestion_{i}_{hash(suggestion)}",
                            help=f"Click to select '{suggestion}'",
                            use_container_width=True
                        ):
                            # Update both search term and selected game, then rerun
                            st.session_state.search_term = suggestion
                            st.session_state.selected_game_name = suggestion
                            st.rerun()
                
                # Show additional suggestions as smaller buttons if any
                if len(suggestions) > 3:
                    with st.expander(f"Show {len(suggestions) - 3} more suggestions"):
                        for j, suggestion in enumerate(suggestions[3:], 3):
                            if st.button(
                                suggestion,
                                key=f"suggestion_{j}_{hash(suggestion)}",
                                help=f"Click to select '{suggestion}'"
                            ):
                                # Update both search term and selected game, then rerun
                                st.session_state.search_term = suggestion
                                st.session_state.selected_game_name = suggestion
                                st.rerun()
        
        # Filter games based on search
        if search_term:
            filtered_games = search_games(games_df, search_term)
            if len(filtered_games) == 0:
                st.warning("No games found matching your search. Try checking spelling or using fewer words.")
                # Show some popular games as fallback
                st.info("Here are some popular games to explore:")
                filtered_games = games_df.head(10)  # Show top 10 as fallback
            else:
                # Show count of results
                st.caption(f"Found {len(filtered_games)} game(s) matching '{search_term}'")
        else:
            filtered_games = games_df.head(20)  # Show top 20 by default
            st.caption("Showing top 20 games by revenue. Use search to find specific games.")
        
        # Game selection dropdown
        game_options = filtered_games['name'].tolist()
        
        # Only add Punch Wall as default option when there's no search term
        punch_wall_name = "Punch Wall"
        if not search_term and punch_wall_name not in game_options:
            # Add Punch Wall to the beginning of the options only when showing default list
            game_options.insert(0, punch_wall_name)
        
        # Determine the default index for the selectbox
        default_index = 0
        
        # First priority: if we have a previously selected game that's in the current options, select it
        if st.session_state.selected_game_name and st.session_state.selected_game_name in game_options:
            default_index = game_options.index(st.session_state.selected_game_name)
        # Second priority: if no search term and Punch Wall is available, select it as default (only for initial load)
        elif not search_term and punch_wall_name in game_options:
            default_index = game_options.index(punch_wall_name)
            if 'selected_game_name' not in st.session_state or st.session_state.selected_game_name == "":
                st.session_state.selected_game_name = punch_wall_name
        # Third priority: default to first option
        elif game_options:
            default_index = 0
            if game_options:
                st.session_state.selected_game_name = game_options[0]
        else:
            default_index = None
        
        selected_game_name = st.selectbox(
            "Choose a game:",
            options=game_options,
            index=default_index,
            help="Select from filtered results"
        )
        
        # Update session state with the current selection
        if selected_game_name:
            st.session_state.selected_game_name = selected_game_name
    
    with col2:
        if selected_game_name:
            # Get selected game info
            selected_game = games_df[games_df['name'] == selected_game_name].iloc[0]
            
            # Display game icon
            if pd.notna(selected_game['iconUrl']) and selected_game['iconUrl']:
                st.image(
                    selected_game['iconUrl'], 
                    width=100, 
                    caption=f"Universe ID: {selected_game['universe_id']}"
                )
            else:
                st.info("No icon available")
    
    st.markdown("---")
    
    # Load data for selected game
    selected_universe_id = None
    selected_game = None
    
    if selected_game_name:
        selected_game = games_df[games_df['name'] == selected_game_name].iloc[0]
        selected_universe_id = str(selected_game['universe_id'])
        df = load_data(selected_universe_id)
        
        # Display selected game info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Game", selected_game_name)
        with col2:
            st.metric("Universe ID", selected_universe_id)
        with col3:
            genre_text = f"{selected_game['genre_l1']} - {selected_game['genre_l2']}" if pd.notna(selected_game['genre_l2']) else selected_game['genre_l1']
            st.metric("Genre", genre_text)
    else:
        # Fallback to default game
        df = load_data()
    
    # Check if data was loaded successfully
    if df.empty:
        st.error("Failed to load game data. Please try selecting a different game.")
        return
    
    # Verify required columns exist
    if 'Estimated Revenue' not in df.columns:
        st.error("Data is missing the required 'Estimated Revenue' column. Please try selecting a different game.")
        return
    
    if 'date' not in df.columns:
        st.error("Data is missing the required 'date' column. Please try selecting a different game.")
        return
    
    additional_games = load_additional_games_data()
    
    # Sidebar for investment parameters
    st.sidebar.header("💰 Investment Parameters")
    
    # Calculate annual revenue for multiple calculation (temporary calculation)
    try:
        temp_annual_revenue = df['Estimated Revenue'].sum() * 0.0038 * (365 / len(df))  # Rough estimate
    except Exception as e:
        st.error(f"Error calculating annual revenue: {str(e)}")
        return
    
    # Toggle between input modes
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Evaluation Multiple", "Evaluation Amount", "Price & Equity"],
        index=2,  # Default to "Price & Equity"
        help="Choose whether to input multiple, amount directly, or price with equity percentage"
    )
    
    # Initialize session state for values
    if 'evaluation_multiple' not in st.session_state:
        st.session_state.evaluation_multiple = 1.0
    if 'evaluation_amount' not in st.session_state:
        st.session_state.evaluation_amount = int(temp_annual_revenue * 1.0)
    if 'investment_price' not in st.session_state:
        st.session_state.investment_price = 250000
    if 'equity_percentage' not in st.session_state:
        st.session_state.equity_percentage = 80.0
    
    if input_mode == "Evaluation Multiple":
        # Input multiple, calculate amount
        evaluation_multiple = st.sidebar.slider(
            "Evaluation Multiple (Annual Revenue)",
            min_value=0.1,
            max_value=3.0,
            value=st.session_state.evaluation_multiple,
            step=0.05,
            help="Company valuation as multiple of annual revenue"
        )
        
        # Calculate amount from multiple
        upfront = int(temp_annual_revenue * evaluation_multiple)
        
        # Update session state
        st.session_state.evaluation_multiple = evaluation_multiple
        st.session_state.evaluation_amount = upfront
        
        # Display calculated amount
        st.sidebar.metric(
            "Evaluation Amount",
            f"${upfront:,}",
            help="Calculated from annual revenue × multiple"
        )
        
    elif input_mode == "Price & Equity":
        # Input price and equity percentage, calculate evaluation
        investment_price = st.sidebar.number_input(
            "Investment Price ($)",
            min_value=1000,
            value=st.session_state.investment_price,
            step=5000,
            help="Amount you're willing to invest"
        )
        
        equity_percentage = st.sidebar.slider(
            "Equity Percentage (%)",
            min_value=1.0,
            max_value=100.0,
            value=st.session_state.equity_percentage,
            step=0.5,
            help="Percentage of equity you want for your investment"
        )
        
        # Calculate evaluation from price and equity percentage
        # If investing X for Y%, then company valuation = X / (Y/100)
        upfront = int(investment_price / (equity_percentage / 100))
        evaluation_multiple = upfront / temp_annual_revenue
        
        # Update session state
        st.session_state.investment_price = investment_price
        st.session_state.equity_percentage = equity_percentage
        st.session_state.evaluation_amount = upfront
        st.session_state.evaluation_multiple = evaluation_multiple
        
        # Display calculated values
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                "Company Valuation",
                f"${upfront:,}",
                help="Calculated from investment ÷ equity percentage"
            )
        with col2:
            st.metric(
                "Valuation Multiple",
                f"{evaluation_multiple:.2f}x",
                help="Company valuation ÷ annual revenue"
            )
        
    else:  # input_mode == "Evaluation Amount"
        # Input amount, calculate multiple
        upfront = st.sidebar.number_input(
            "Evaluation Amount ($)",
            min_value=1000,
            value=st.session_state.evaluation_amount,
            step=10000,
            help="Total company valuation amount"
        )
        
        # Calculate multiple from amount
        evaluation_multiple = upfront / temp_annual_revenue
        
        # Update session state
        st.session_state.evaluation_amount = upfront
        st.session_state.evaluation_multiple = evaluation_multiple
        
        # Display calculated multiple
        st.sidebar.metric(
            "Evaluation Multiple (Annual Revenue)",
            f"{evaluation_multiple:.2f}x",
            help="Calculated from evaluation amount ÷ annual revenue"
        )
    
    # Investor configuration (outside of input mode conditional)
    if input_mode == "Price & Equity":
        # Use the equity percentage from the Price & Equity input mode
        equity_pct = st.session_state.equity_percentage
    else:
        # Show equity slider for other input modes
        equity_pct = st.sidebar.slider(
            "Investor Equity (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.1
        )
    
    # Earnout only available for 100% equity deals
    if equity_pct == 100.0:
        earnout_pct = st.sidebar.slider(
            "Earnout Percentage (%)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=0.5,
            help="Percentage of revenue during earnout period paid to developers"
        )
        
        earnout = st.sidebar.number_input(
            "Total Earnout (max) ($)",
            min_value=0,
            value=1000000,
            step=100000,
            format="%d",
            help="Maximum earnout payment to developers (reduces investor returns)"
        )
        
        earnout_period_months = st.sidebar.slider(
            "Earnout Period (months)",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
            help="Period over which earnout is calculated and paid to developers"
        )
    else:
        earnout_pct = 0
        earnout = 0
        earnout_period_months = 0
    
    
    robux_to_usd = st.sidebar.number_input(
        "Robux to USD Conversion Rate",
        min_value=0.0001,
        max_value=1.0,
        value=0.0038,
        step=0.0001,
        format="%.4f"
    )
    
    decay_pct = st.sidebar.slider(
        "Weekly Growth/Decay Rate (%)",
        min_value=-5.0,
        max_value=100.0,
        value=0.0,
        step=0.05,
        help="Week-on-week exponential growth (negative) or decay (positive) percentage applied to projected revenue"
    )
    
    # Toggle for polynomial growth scenario
    use_polynomial_growth = st.sidebar.toggle(
        "Polynomial Growth Scenario",
        value=False,
        help="Use a 5th degree polynomial growth model with peaks at 0.5 months and 2 months"
    )
    
    # Polynomial growth peak multiplier slider (only shown when polynomial growth is enabled)
    if use_polynomial_growth:
        polynomial_peak_multiplier = st.sidebar.slider(
            "Polynomial Peak Multiplier (%)",
            min_value=100,
            max_value=2000,
            value=250,
            step=50,
            help="Peak multiplier percentage for polynomial growth. 1000% = 10x revenue at first peak, 500% = 5x at second peak"
        )
    else:
        polynomial_peak_multiplier = 1000  # Default value when not using polynomial growth
    
    # Toggle for standard growth scenario
    use_standard_growth = st.sidebar.toggle(
        "Standard Growth Scenario",
        value=False,
        help="Apply 10% growth multiplier over the first 2-4 prediction weeks"
    )
    
    # Standard growth parameters (only shown when standard growth is enabled)
    if use_standard_growth:
        standard_growth_percentage = st.sidebar.slider(
            "Growth Percentage (%)",
            min_value=5,
            max_value=1000,
            value=10,
            step=1,
            help="Percentage growth to gradually ramp up to over the first few weeks, then maintain for the entire projection"
        )
        
        protected_growth_months = st.sidebar.slider(
            "Protected Growth Period (months)",
            min_value=0.5,
            max_value=18.0,
            value=1.0,
            step=0.5,
            help="Period during which decay factor is not applied when standard growth is enabled"
        )
    else:
        standard_growth_percentage = 10  # Default 10% growth
        protected_growth_months = 1.0  # Default 1 month
    
    # Toggle for using current average
    use_current_average = st.sidebar.toggle(
        "Use Current Average",
        value=True,
        help="When enabled, use the most recent week's revenue as the prediction baseline instead of the last 3 months average"
    )
    
    # Calculate data with user parameters (needed for intelligent peak detection)
    df_with_usd, weekly_avg = calculate_weekly_averages(df.copy(), robux_to_usd)
    
    # Calculate lifetime mean for peak detection
    lifetime_mean = df_with_usd['Estimated Revenue USD'].mean()
    
    # Calculate last 3 months average for projection baseline
    three_months_ago = df_with_usd['date'].max() - pd.DateOffset(months=3)
    # Ensure datetime compatibility by converting to same timezone-naive format
    three_months_ago = pd.to_datetime(three_months_ago).tz_localize(None)
    weekly_start_normalized = pd.to_datetime(weekly_avg['week_start']).dt.tz_localize(None)
    last_three_months_data = weekly_avg[weekly_start_normalized >= three_months_ago]
    
    if use_current_average:
        # Use the most recent week's revenue as baseline
        if len(weekly_avg) > 0:
            last_three_months_mean = weekly_avg['Estimated Revenue USD'].iloc[-1]
        else:
            last_three_months_mean = lifetime_mean
    else:
        # Use the standard last 3 months average
        if len(last_three_months_data) > 0:
            last_three_months_mean = last_three_months_data['Estimated Revenue USD'].mean()
        else:
            # Fallback to lifetime mean if not enough recent data
            last_three_months_mean = lifetime_mean
    
    # Analyze historical data to determine specific summer peak month
    summer_months = [5, 6, 7, 8]  # May, June, July, August
    summer_month_avgs = {}
    peak_summer_month = 6  # Default to June
    summer_peak_detected = False
    
    if use_current_average:
        # When using current average, enable peaks by default as 150% of current baseline
        summer_peak_detected = True
        # Still find the best historical summer month for timing preference
        for month in summer_months:
            month_data = weekly_avg[weekly_avg['week_start'].dt.month == month]
            if len(month_data) > 0:
                month_avg = month_data['Estimated Revenue USD'].mean()
                summer_month_avgs[month] = month_avg
        
        if summer_month_avgs:
            peak_summer_month = max(summer_month_avgs.keys(), key=lambda k: summer_month_avgs[k])
        else:
            peak_summer_month = 6  # Default to June
    else:
        # Use historical comparison logic when not using current average
        # Calculate baseline from non-summer months to avoid skewing by recent trends
        non_summer_data = weekly_avg[~weekly_avg['week_start'].dt.month.isin(summer_months)]
        if len(non_summer_data) > 0:
            baseline_mean = non_summer_data['Estimated Revenue USD'].mean()
        else:
            # Fallback to lifetime mean if no non-summer data
            baseline_mean = lifetime_mean
        
        # Detect explosive growth pattern
        # If non-summer baseline is significantly higher than summer data,
        # it indicates explosive growth where summer data is from early lifecycle
        summer_data = weekly_avg[weekly_avg['week_start'].dt.month.isin(summer_months)]
        if len(summer_data) > 0:
            summer_mean = summer_data['Estimated Revenue USD'].mean()
            # If non-summer average is more than 3x summer average, we have explosive growth
            explosive_growth = baseline_mean > summer_mean * 3.0
        else:
            explosive_growth = False
            summer_mean = lifetime_mean
        
        for month in summer_months:
            month_data = weekly_avg[weekly_avg['week_start'].dt.month == month]
            if len(month_data) > 0:
                month_avg = month_data['Estimated Revenue USD'].mean()
                summer_month_avgs[month] = month_avg
        
        # Smart peak detection logic
        if explosive_growth:
            # For explosive growth games, default to summer peaks being 150% of recent baseline
            # This assumes summer peaks are beneficial and should be enabled
            summer_peak_detected = True
            # Still find the best historical summer month for timing
            if summer_month_avgs:
                peak_summer_month = max(summer_month_avgs.keys(), key=lambda k: summer_month_avgs[k])
            else:
                peak_summer_month = 6  # Default to June
        else:
            # For stable games, use traditional comparison method
            if summer_month_avgs:
                best_month = max(summer_month_avgs.keys(), key=lambda k: summer_month_avgs[k])
                best_avg = summer_month_avgs[best_month]
                
                if best_avg > baseline_mean * 1.25:  # 25% greater than non-summer average
                    peak_summer_month = best_month
                    summer_peak_detected = True
                else:
                    # No significant peak detected, default to June
                    peak_summer_month = 6
                    summer_peak_detected = False
    
    # December analysis
    december_data = weekly_avg[weekly_avg['week_start'].dt.month == 12]
    december_explosive_growth = False
    if use_current_average:
        # When using current average, enable December peaks by default
        december_peak_detected = True
    else:
        # Use historical comparison logic when not using current average
        if len(december_data) > 0:
            december_avg = december_data['Estimated Revenue USD'].mean()
            # Use non-December months as baseline for fair comparison
            non_december_data = weekly_avg[weekly_avg['week_start'].dt.month != 12]
            if len(non_december_data) > 0:
                december_baseline = non_december_data['Estimated Revenue USD'].mean()
            else:
                december_baseline = lifetime_mean
            
            # Check for explosive growth pattern for December too
            if december_baseline > december_avg * 3.0:
                # Explosive growth: December data is from early lifecycle, enable peaks by default
                december_peak_detected = True
                december_explosive_growth = True
            else:
                # Normal growth: use traditional comparison
                december_peak_detected = december_avg > december_baseline * 1.25  # 25% greater than non-December average
        else:
            december_peak_detected = False
    
    # Month names for display
    month_names = {5: 'May', 6: 'June', 7: 'July', 8: 'August'}
    peak_month_name = month_names.get(peak_summer_month, 'June')
    
    # Seasonal peak toggles
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Seasonal Peaks**")
    
    # Generate help text based on detection method
    if use_current_average:
        summer_help_text = f"Enable higher revenue during {peak_month_name}. Auto-detected: Peak enabled (using 150% of current average)"
        december_help_text = f"Enable higher revenue during December. Auto-detected: Peak enabled (using 150% of current average)"
    else:
        # In historical mode, we may have explosive growth detection
        summer_help_text = f"Enable higher revenue during {peak_month_name}. Auto-detected: {'Peak found in ' + peak_month_name if summer_peak_detected else 'No significant peak, defaulting to June'}"
        december_help_text = f"Enable higher revenue during December. Auto-detected: {'Peak enabled due to explosive growth pattern' if december_explosive_growth else ('Peak found' if december_peak_detected else 'No significant peak')}"
    
    summer_peaks = st.sidebar.toggle(
        f"Summer Peak ({peak_month_name})",
        value=summer_peak_detected,
        help=summer_help_text
    )
    
    december_peaks = st.sidebar.toggle(
        "December Peaks",
        value=december_peak_detected,
        help=december_help_text
    )
    
    # Additional games toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Comparison Games**")
    
    show_additional_games = st.sidebar.toggle(
        "Show Comparison Games",
        value=False,
        help="Display Basketball Zero and Blue Lock Rivals for comparison"
    )
    
    # Calculate weekly averages for additional games
    additional_games_weekly = calculate_additional_games_weekly_averages(additional_games, robux_to_usd)
    
    # Calculate projection parameters
    historical_peak_weekly = weekly_avg['Estimated Revenue USD'].max()
    
    # Calculate peak multiple - will be recalculated after last_observed_weekly is determined
    if use_current_average:
        # When using current average, peaks should be 150% of current baseline
        peak_multiple = 1.50
    else:
        # When using historical average, calculate peak multiple based on highest observed peak
        if historical_peak_weekly > 0 and last_three_months_mean > 0:
            peak_multiple = historical_peak_weekly / last_three_months_mean
        else:
            peak_multiple = 1.5  # Conservative fallback
    
    # All peaks use the same peak multiple
    december_scale = peak_multiple
    summer_scale = peak_multiple
    


    normal_week_deviations = [0.9, 1.0, 1.1, 0.95, 1.05]  # Small variations around average
    
    # Calculate the current running weekly average (including incomplete current week)
    if len(weekly_avg) > 0:
        # Check if there's an incomplete current week by comparing the last week in weekly_avg 
        # with the actual last date in the data
        last_data_date = df_with_usd['date'].max()
        last_week_start = weekly_avg['week_start'].iloc[-1]
        last_week_end = last_week_start + pd.DateOffset(days=6)
        
        # Normalize timezone information to avoid comparison issues
        if hasattr(last_data_date, 'tz') and last_data_date.tz is not None:
            last_data_date = last_data_date.tz_localize(None)
        if hasattr(last_week_end, 'tz') and last_week_end.tz is not None:
            last_week_end = last_week_end.tz_localize(None)
        if hasattr(last_week_start, 'tz') and last_week_start.tz is not None:
            last_week_start = last_week_start.tz_localize(None)
        
        # Convert to pandas Timestamp for safe comparison
        last_data_date = pd.to_datetime(last_data_date).tz_localize(None)
        last_week_end = pd.to_datetime(last_week_end).tz_localize(None)
        last_week_start = pd.to_datetime(last_week_start).tz_localize(None)
        
        # If the last week is incomplete (current week is still ongoing)
        if last_data_date < last_week_end:
            # Calculate running average for the current incomplete week
            # Normalize dates in DataFrame for comparison
            df_dates_normalized = pd.to_datetime(df_with_usd['date']).dt.tz_localize(None)
            current_week_data = df_with_usd[df_dates_normalized >= last_week_start]
            if len(current_week_data) > 0:
                current_week_running_avg = current_week_data['Estimated Revenue USD'].mean()
                last_observed_weekly = current_week_running_avg
            else:
                # Fallback to last complete week if current week has no data
                last_observed_weekly = weekly_avg['Estimated Revenue USD'].iloc[-1]
        else:
            # Use the last complete week's average
            last_observed_weekly = weekly_avg['Estimated Revenue USD'].iloc[-1]
    else:
        # Fallback if no weekly data available
        last_observed_weekly = last_three_months_mean
    
    # Generate projection using the appropriate baseline (current week or last 3 months average)
    baseline_for_projection = last_observed_weekly if use_current_average else last_three_months_mean
    start_date = df_with_usd['date'].max() + pd.DateOffset(days=1)
    projection_dates, projection_revenues = generate_weekly_projection_with_deviations(
        start_date, baseline_for_projection, december_scale, summer_scale,
        normal_week_deviations, 
        historical_peak_weekly, last_observed_weekly, decay_pct, summer_peaks, december_peaks, peak_summer_month, 18, use_polynomial_growth, use_current_average, polynomial_peak_multiplier, use_standard_growth, standard_growth_percentage, protected_growth_months
    )
    
    projection_df = pd.DataFrame({
        'date': projection_dates,
        'Estimated Revenue USD': projection_revenues
    })
    
    # Prepare current week information for chart visualization
    current_week_info = None
    if use_current_average and len(weekly_avg) > 0:
        last_data_date = df_with_usd['date'].max()
        last_week_start = weekly_avg['week_start'].iloc[-1]
        last_week_end = last_week_start + pd.DateOffset(days=6)
        
        # Normalize timezone for comparison
        last_data_date_norm = pd.to_datetime(last_data_date).tz_localize(None)
        last_week_end_norm = pd.to_datetime(last_week_end).tz_localize(None)
        last_week_start_norm = pd.to_datetime(last_week_start).tz_localize(None)
        
        # Check if current week is incomplete
        if last_data_date_norm < last_week_end_norm:
            # Get current week data for display
            df_dates_normalized = pd.to_datetime(df_with_usd['date']).dt.tz_localize(None)
            current_week_data = df_with_usd[df_dates_normalized >= last_week_start_norm]
            if len(current_week_data) > 0:
                current_week_info = {
                    'show_marker': True,
                    'date': last_week_start,
                    'value': current_week_data['Estimated Revenue USD'].mean(),
                    'days_count': len(current_week_data)
                }
    
    # Create the main chart
    col1, col2 = st.columns([1, 4])

    
    fig = create_revenue_analysis_chart(df_with_usd, weekly_avg, projection_df, additional_games_weekly, show_additional_games, current_week_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Investment Analysis
    st.header("💼 Investment Analysis")
    
    analysis = calculate_investment_analysis(df_with_usd, weekly_avg, projection_df, 
                                          upfront, earnout, equity_pct, robux_to_usd, earnout_period_months, earnout_pct)
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Investor's Total Investment",
            f"${analysis['investor_total']:,.0f}",
            help="Upfront + Earnout based on equity percentage"
        )
    
    with col2:
        st.metric(
            "18-Month Projected Revenue",
            f"${analysis['next_18_months']:,.0f}",
            help="Total projected revenue over 18 months"
        )
    
    with col3:
        st.metric(
            "Investor's 18-Month Share",
            f"${analysis['investor_18m']:,.0f}",
            help="Investor's share of 18-month projected revenue"
        )
    
    with col4:
        st.metric(
            "18-Month Return Multiple",
            f"{analysis['return_18m']:.2f}x",
            help="Return multiple on investment over 18 months"
        )
    
    # Add 6-month investor share metric
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric(
            "Investor's 6-Month Share",
            f"${analysis['investor_6m']:,.0f}",
            help="Investor's share of 6-month projected revenue (after earnout deduction)"
        )
    
    with col6:
        st.metric(
            "6-Month Return Multiple",
            f"{analysis['return_6m']:.2f}x",
            help="Return multiple on investment over 6 months"
        )
    
    # Detailed analysis
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Revenue Projections:**")
        st.write(f"• Previous 6 months: ${analysis['previous_6_months']:,.0f}")
        st.write(f"• Next 6 months: ${analysis['next_6_months']:,.0f}")
        st.write(f"• Next 12 months: ${analysis['next_12_months']:,.0f}")
        st.write(f"• Next 18 months: ${analysis['next_18_months']:,.0f}")
        
        st.markdown("**Investor's Share:**")
        st.write(f"• Upfront Investment: ${analysis['investor_upfront']:,.0f}")
        st.write(f"• Earnout Investment: ${analysis['investor_earnout']:,.0f}")
        st.write(f"• Total Investment: ${analysis['investor_total']:,.0f}")
    
    with col2:
        st.markdown("**Return Analysis:**")
        st.write(f"• 6-month return: {analysis['return_6m']:.2f}x")
        st.write(f"• 12-month return: {analysis['return_12m']:.2f}x")
        st.write(f"• 18-month return: {analysis['return_18m']:.2f}x")
        
        st.markdown("**Key Metrics:**")
        st.write(f"• Lifetime average (USD): ${analysis['lifetime_mean']:,.0f}")
        st.write(f"• Historical peak weekly: ${historical_peak_weekly:,.0f}")
        st.write(f"• Equity percentage: {equity_pct}%")
        
        # Show developer earnout information if applicable
        if equity_pct == 100.0 and analysis['developer_earnout'] > 0:
            st.markdown("**Developer Earnout:**")
            st.write(f"• Earnout period: {analysis['earnout_period_months']} months")
            st.write(f"• Earnout percentage: {earnout_pct:.1f}% of revenue")
            st.write(f"• Revenue during earnout period: ${analysis['total_earnout_period_revenue']:,.0f}")
            st.write(f"• Developer earnout ({earnout_pct:.1f}% of revenue, capped): ${analysis['developer_earnout']:,.0f}")
            st.write(f"• Note: Developer earnout reduces investor returns")
    
    # Data summary
    st.subheader("📋 Data Summary")
    st.write(f"**Historical Data:** {len(df_with_usd)} days from {df_with_usd['date'].min().strftime('%Y-%m-%d')} to {df_with_usd['date'].max().strftime('%Y-%m-%d')}")
    st.write(f"**Projection Period:** {len(projection_df)} weeks from {projection_df['date'].min().strftime('%Y-%m-%d')} to {projection_df['date'].max().strftime('%Y-%m-%d')}")
    st.write(f"**Robux to USD Rate:** {robux_to_usd:.4f}")
    
    # Calculate and display current evaluation multiple
    actual_annual_revenue = df_with_usd['Estimated Revenue USD'].sum() * (365 / len(df_with_usd))
    current_multiple = upfront / actual_annual_revenue if actual_annual_revenue > 0 else 0
    st.write(f"**Annual Revenue (extrapolated):** ${actual_annual_revenue:,.0f}")
    st.write(f"**Current Evaluation Multiple:** {current_multiple:.1f}x")
    baseline_type = "Current Week" if use_current_average else "Last 3 Months Average"
    baseline_value = baseline_for_projection
    st.write(f"**Projection Baseline ({baseline_type}):** ${baseline_value:,.0f}")
    st.write(f"**Lifetime Average:** ${lifetime_mean:,.0f}")
    if decay_pct < 0:
        st.write(f"**Weekly Growth Rate:** {abs(decay_pct):.1f}% per week")
    elif decay_pct > 0:
        st.write(f"**Weekly Decay Rate:** {decay_pct:.1f}% per week")
    else:
        st.write(f"**Weekly Growth/Decay Rate:** 0.0% per week (no change)")
    
    st.write(f"**Summer Peaks (May, Jun-Aug):** {'Enabled' if summer_peaks else 'Disabled'}")
    st.write(f"**December Peaks:** {'Enabled' if december_peaks else 'Disabled'}")
    if use_polynomial_growth:
        first_peak_value = polynomial_peak_multiplier / 100.0
        second_peak_value = (polynomial_peak_multiplier / 100.0) * 0.5
        st.write(f"**Polynomial Growth Scenario:** Enabled ({first_peak_value:.1f}x at 0.5 months, {second_peak_value:.1f}x at 2 months)")
    else:
        st.write(f"**Polynomial Growth Scenario:** Disabled")
    if use_standard_growth:
        st.write(f"**Standard Growth Scenario:** Enabled ({standard_growth_percentage}% growth ramped up over 4 weeks, then maintained)")
    else:
        st.write(f"**Standard Growth Scenario:** Disabled")
    st.write(f"**Use Current Average:** {'Enabled (using most recent week)' if use_current_average else 'Disabled (using last 3 months average)'}")
    st.write(f"**Comparison Games:** {'Shown' if show_additional_games else 'Hidden'}")
    
    # Additional games info
    if additional_games and show_additional_games:
        st.markdown("**Additional Games (Weekly Averages):**")
        for game_name, game_weekly in additional_games_weekly.items():
            st.write(f"• {game_name}: {len(game_weekly)} weeks from {game_weekly['week_start'].min().strftime('%Y-%m-%d')} to {game_weekly['week_start'].max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
