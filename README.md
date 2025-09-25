# Revenue Analysis & Investment Calculator

A Streamlit application that analyzes historical revenue data and provides 18-month projections with investment analysis.

## Features

- **Interactive Revenue Analysis**: Historical daily revenue data with weekly averages
- **18-Month Projection**: Weekly revenue projections with seasonal patterns
- **Investment Calculator**: Configurable investment parameters with return analysis
- **Interactive Charts**: Plotly-powered visualizations with zoom and hover capabilities

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Adjust Investment Parameters** in the sidebar:
   - Total Upfront Investment
   - Total Earnout (maximum)
   - Investor Equity Percentage
   - Company Valuation
   - Robux to USD Conversion Rate

2. **View the Revenue Analysis Chart** showing:
   - Historical daily revenue (light blue line)
   - Historical weekly averages (blue dashed line with markers)
   - Projected weekly revenue (red line with markers)
   - Seasonal highlights (Summer, May, December)

3. **Review Investment Analysis** including:
   - Return multiples for 6, 12, and 18 months
   - Investor's share of projected revenue
   - Key financial metrics

## Data Source

The application uses historical revenue data from `universe_6931042565/daily_revenue_20250921_011820.json`.

## Projection Logic

- **Base Revenue**: Uses historical average as the baseline
- **Seasonal Patterns**: 
  - May and December show higher revenue (using May's historical pattern)
  - Summer months (June-August) show moderate increases
  - Other months stay close to average with small variations
- **Weekly Deviations**: Applies realistic weekly variation patterns
- **Peak Capping**: All projections are capped at historical weekly peak
