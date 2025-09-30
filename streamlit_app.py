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
def load_data():
    """Load the daily revenue data"""
    with open('universe_6931042565/daily_revenue_20250921_011820.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['kpiData'])
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

@st.cache_data
def load_additional_games_data():
    """Load additional games data for comparison"""
    additional_games = {}
    
    # Basketball Zero
    try:
        with open('sportGames/BasketballZero_universe_7028566528/daily_revenue_20250921_011912.json', 'r') as f:
            data = json.load(f)
        df_basketball = pd.DataFrame(data['kpiData'])
        df_basketball['date'] = pd.to_datetime(df_basketball['date'])
        additional_games['Basketball Zero'] = df_basketball
    except FileNotFoundError:
        pass
    
    # Blue Lock Rivals
    try:
        with open('sportGames/BlueLockRivals_universe_6325068386/daily_revenue_20250921_011832.json', 'r') as f:
            data = json.load(f)
        df_bluelock = pd.DataFrame(data['kpiData'])
        df_bluelock['date'] = pd.to_datetime(df_bluelock['date'])
        additional_games['Blue Lock Rivals'] = df_bluelock
    except FileNotFoundError:
        pass
    
    return additional_games

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

def generate_weekly_projection_with_deviations(start_date, lifetime_mean, may_scale, december_scale, 
                                            may_deviation_pattern, normal_week_deviations, 
                                            historical_peak_weekly, decay_pct=0.0, summer_peaks=True, december_peaks=True, months=18):
    """Generate weekly projection using average as base and applying weekly deviation patterns"""
    projection_weekly_dates = []
    projection_weekly_revenues = []
    
    current_date = start_date
    end_date = start_date + pd.DateOffset(months=18)  # Exactly 18 months
    
    # Month definitions
    summer_months = [6, 7, 8]  # June, July, August
    may_month = 5
    december_month = 12
    
    # Calculate decay factor (convert percentage to decimal)
    decay_factor = 1.0 - (decay_pct / 100.0)
    week_counter = 0  # Track weeks for decay calculation
    
    # Generate weekly data for exactly 18 months
    while current_date < end_date:
        month = current_date.month
        
        # Determine base scaling factor for the month based on toggle settings
        if month in summer_months:
            month_scale = 1.2 if summer_peaks else 1.0  # Summer months with moderate peak if enabled
        elif month == may_month:
            month_scale = may_scale if summer_peaks else 1.0  # May uses its historical pattern if summer peaks enabled
        elif month == december_month:
            month_scale = may_scale if december_peaks else 1.0  # Use May's scale for December if enabled
        else:
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
            if month == may_month and summer_peaks:
                # Use May's deviation pattern for May if summer peaks enabled
                deviation_idx = week_idx % len(may_deviation_pattern)
                weekly_deviation = may_deviation_pattern[deviation_idx]
            elif month == december_month and december_peaks:
                # Use May's deviation pattern for December if December peaks enabled
                deviation_idx = week_idx % len(may_deviation_pattern)
                weekly_deviation = may_deviation_pattern[deviation_idx]
            else:
                # Use small deviations for other months or when peaks are disabled
                deviation_idx = week_idx % len(normal_week_deviations)
                weekly_deviation = normal_week_deviations[deviation_idx]
            
            # Calculate weekly revenue: base * month_scale * weekly_deviation * decay_factor^week_counter
            weekly_revenue = lifetime_mean * month_scale * weekly_deviation * (decay_factor ** week_counter)
            
            # Cap the revenue to not exceed historical weekly peak
            weekly_revenue = min(weekly_revenue, historical_peak_weekly)
            
            projection_weekly_dates.append(week_start)
            projection_weekly_revenues.append(weekly_revenue)
            
            # Increment week counter for decay calculation
            week_counter += 1
        
        # Move to next month
        current_date = next_month
    
    return projection_weekly_dates, projection_weekly_revenues

def create_revenue_analysis_chart(df, weekly_avg, projection_df, additional_games_weekly=None, show_additional_games=True):
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
    
    # Format y-axis to show currency and set range
    fig.update_yaxes(tickformat='$,.0f', range=[0, 100000])
    
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
    
    # Load data
    df = load_data()
    additional_games = load_additional_games_data()
    
    # Sidebar for investment parameters
    st.sidebar.header("💰 Investment Parameters")
    
    # Calculate annual revenue for multiple calculation (temporary calculation)
    temp_annual_revenue = df['Estimated Revenue'].sum() * 0.0038 * (365 / len(df))  # Rough estimate
    
    # Toggle between input modes
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Evaluation Multiple", "Evaluation Amount"],
        help="Choose whether to input multiple or amount directly"
    )
    
    # Initialize session state for values
    if 'evaluation_multiple' not in st.session_state:
        st.session_state.evaluation_multiple = 1.0
    if 'evaluation_amount' not in st.session_state:
        st.session_state.evaluation_amount = int(temp_annual_revenue * 1.0)
    
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
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Week-on-week exponential growth (negative) or decay (positive) percentage applied to projected revenue"
    )
    
    # Seasonal peak toggles
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Seasonal Peaks**")
    
    summer_peaks = st.sidebar.toggle(
        "Summer Peaks (May, Jun-Aug)",
        value=True,
        help="Enable higher revenue during peak months (May, June, July, August)"
    )
    
    december_peaks = st.sidebar.toggle(
        "December Peaks",
        value=True,
        help="Enable higher revenue during December (using May's pattern)"
    )
    
    # Additional games toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Comparison Games**")
    
    show_additional_games = st.sidebar.toggle(
        "Show Comparison Games",
        value=False,
        help="Display Basketball Zero and Blue Lock Rivals for comparison"
    )
    
    # Calculate data with user parameters
    df_with_usd, weekly_avg = calculate_weekly_averages(df.copy(), robux_to_usd)
    
    # Calculate weekly averages for additional games
    additional_games_weekly = calculate_additional_games_weekly_averages(additional_games, robux_to_usd)
    
    # Calculate projection parameters
    lifetime_mean = df_with_usd['Estimated Revenue USD'].mean()
    historical_peak_weekly = weekly_avg['Estimated Revenue USD'].max()
    
    # May scaling and deviation patterns (from historical data)
    may_weeks = weekly_avg[weekly_avg['week_start'].dt.month == 5]
    if len(may_weeks) > 0:
        may_weekly_avg = may_weeks['Estimated Revenue USD'].mean()
        may_scale = may_weekly_avg / lifetime_mean
        may_weekly_deviations = may_weeks['Estimated Revenue USD'] / lifetime_mean
        may_deviation_pattern = may_weekly_deviations.tolist()
    else:
        may_scale = 1.0
        may_deviation_pattern = [1.0]
    
    december_scale = may_scale  # Use May's scale for December
    normal_week_deviations = [0.9, 1.0, 1.1, 0.95, 1.05]  # Small variations around average
    
    # Generate projection
    start_date = df_with_usd['date'].max() + pd.DateOffset(days=1)
    projection_dates, projection_revenues = generate_weekly_projection_with_deviations(
        start_date, lifetime_mean, may_scale, december_scale, 
        may_deviation_pattern, normal_week_deviations, 
        historical_peak_weekly, decay_pct, summer_peaks, december_peaks
    )
    
    projection_df = pd.DataFrame({
        'date': projection_dates,
        'Estimated Revenue USD': projection_revenues
    })
    
    # Create the main chart
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://tr.rbxcdn.com/180DAY-a1260693b7f075c5e8482b83e0531ad7/512/512/Image/Webp/noFilter", width=100)
    with col2:
        st.header("Game Investment Analysis")
    
    fig = create_revenue_analysis_chart(df_with_usd, weekly_avg, projection_df, additional_games_weekly, show_additional_games)
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
    if decay_pct < 0:
        st.write(f"**Weekly Growth Rate:** {abs(decay_pct):.1f}% per week")
    elif decay_pct > 0:
        st.write(f"**Weekly Decay Rate:** {decay_pct:.1f}% per week")
    else:
        st.write(f"**Weekly Growth/Decay Rate:** 0.0% per week (no change)")
    
    st.write(f"**Summer Peaks (May, Jun-Aug):** {'Enabled' if summer_peaks else 'Disabled'}")
    st.write(f"**December Peaks:** {'Enabled' if december_peaks else 'Disabled'}")
    st.write(f"**Comparison Games:** {'Shown' if show_additional_games else 'Hidden'}")
    
    # Additional games info
    if additional_games and show_additional_games:
        st.markdown("**Additional Games (Weekly Averages):**")
        for game_name, game_weekly in additional_games_weekly.items():
            st.write(f"• {game_name}: {len(game_weekly)} weeks from {game_weekly['week_start'].min().strftime('%Y-%m-%d')} to {game_weekly['week_start'].max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
