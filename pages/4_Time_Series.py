import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path to import from data_cleaner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import safe_display_dataframe
except ImportError:
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Time Series Analysis",
    page_icon="📈",
    layout="wide"
)

st.markdown("# Time Series Analysis")
st.sidebar.header("Time Series Analysis")
st.write(
    """
    This page provides tools for analyzing time series data and generating forecasts.
    Select your time series column and configure analysis options.
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

# Get data from session state
df = get_data()

if df is None:
    st.info("No data loaded. Please upload data on the Home page.")
else:
    st.write("### Data Preview")
    st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        safe_display_dataframe(df.head(10))
    
    # Check for datetime columns or allow user to convert a column to datetime
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
    
    if not date_cols:
        st.warning("No datetime columns detected. Please select a column to convert to datetime.")
        date_col_options = df.columns.tolist()
        selected_date_col = st.selectbox("Select column to convert to datetime", date_col_options)
        
        if st.button("Convert to Datetime"):
            try:
                df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                st.success(f"Successfully converted {selected_date_col} to datetime format.")
                date_cols = [selected_date_col]
                # Update session state
                st.session_state.data = df
            except Exception as e:
                st.error(f"Error converting to datetime: {str(e)}")
    
    # Time series analysis options
    if date_cols:
        st.write("### Time Series Analysis")
        
        # Select datetime column
        date_col = st.selectbox("Select Date/Time Column", date_cols)
        
        # Select value column
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            value_col = st.selectbox("Select Value Column", numeric_cols)
            
            # Prepare time series data
            ts_data = df[[date_col, value_col]].copy()
            ts_data = ts_data.sort_values(by=date_col)
            
            # Check for duplicate dates and allow resampling
            has_duplicates = ts_data[date_col].duplicated().any()
            if has_duplicates:
                st.warning("Duplicate timestamps detected. Consider resampling the data.")
                
                # Offer resampling options
                resample = st.checkbox("Resample data")
                if resample:
                    freq_options = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y"
                    }
                    freq = st.selectbox("Select resampling frequency", list(freq_options.keys()))
                    agg_method = st.selectbox("Select aggregation method", ["Mean", "Sum", "Min", "Max", "Median"])
                    
                    # Map aggregation method to pandas function
                    agg_map = {
                        "Mean": "mean",
                        "Sum": "sum",
                        "Min": "min",
                        "Max": "max",
                        "Median": "median"
                    }
                    
                    # Resample data
                    ts_data = ts_data.set_index(date_col)
                    ts_data = ts_data.resample(freq_options[freq]).agg(agg_map[agg_method])
                    ts_data = ts_data.reset_index()
            
            # Display time series plot
            st.write("#### Time Series Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ts_data[date_col], ts_data[value_col], marker='o', linestyle='-', alpha=0.7)
            ax.set_title(f"Time Series: {value_col} over time")
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Time series decomposition
            st.write("### Time Series Decomposition")
            st.write("Decompose the time series into trend, seasonal, and residual components.")
            
            # Need to check if we have enough data and regular frequency for decomposition
            min_periods = 2  # Minimum periods needed for decomposition
            if len(ts_data) >= min_periods:
                try:
                    # Set datetime as index for decomposition
                    ts = ts_data.set_index(date_col)[value_col]
                    
                    # Check if ts is regular and if not, make it regular
                    if not ts.index.is_monotonic_increasing:
                        st.warning("Time series index is not monotonic increasing. Sorting by date.")
                        ts = ts.sort_index()
                    
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Determine appropriate period for decomposition
                    period_options = {
                        "None (Detect automatically)": None,
                        "Daily (7 days)": 7,
                        "Weekly (52 weeks/year)": 52,
                        "Monthly (12 months/year)": 12,
                        "Quarterly (4 quarters/year)": 4
                    }
                    
                    period_selection = st.selectbox("Select seasonal period", list(period_options.keys()))
                    period = period_options[period_selection]
                    
                    # If None, try to detect
                    if period is None:
                        # Simple heuristic - if we have years of data, use 12 (monthly seasonality)
                        # If we have months of data, use 4 (weekly)
                        # If days, use 7 (daily)
                        date_range = (ts.index.max() - ts.index.min()).days
                        if date_range > 365*2:  # More than 2 years
                            period = 12
                        elif date_range > 120:  # More than ~4 months
                            period = 4
                        else:
                            period = 7
                    
                    # Select decomposition model
                    model = st.selectbox("Select decomposition model", ["Additive", "Multiplicative"])
                    
                    # Perform decomposition with selected parameters
                    decomposition = seasonal_decompose(
                        ts,
                        model=model.lower(),
                        period=period
                    )
                    
                    # Plot decomposition
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
                    
                    # Original
                    ax1.plot(decomposition.observed)
                    ax1.set_title('Original Time Series')
                    
                    # Trend
                    ax2.plot(decomposition.trend)
                    ax2.set_title('Trend')
                    
                    # Seasonal
                    ax3.plot(decomposition.seasonal)
                    ax3.set_title('Seasonality')
                    
                    # Residual
                    ax4.plot(decomposition.resid)
                    ax4.set_title('Residuals')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error in decomposition: {str(e)}")
                    st.info("Time series decomposition requires regular time intervals. Try resampling your data first.")
            else:
                st.warning(f"Need at least {min_periods} data points for decomposition. Current data has {len(ts_data)} points.")
            
            # Simple Forecasting
            st.write("### Time Series Forecasting")
            st.write("Generate forecasts using different models.")
            
            forecast_method = st.selectbox(
                "Select Forecasting Method",
                ["Simple Moving Average", "Exponential Smoothing", "Linear Trend"]
            )
            
            forecast_periods = st.slider("Forecast Periods", 1, 30, 10)
            
            if st.button("Generate Forecast"):
                # Set up figure for forecast plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot original data
                ax.plot(ts_data[date_col], ts_data[value_col], label='Original Data', marker='o', alpha=0.7)
                
                # Generate forecast based on selected method
                if forecast_method == "Simple Moving Average":
                    # Set window size
                    window_size = st.slider("Window Size", 2, min(10, len(ts_data)), 3)
                    
                    # Calculate moving average
                    ma = ts_data[value_col].rolling(window=window_size).mean()
                    
                    # Generate forecast dates
                    last_date = ts_data[date_col].max()
                    forecast_dates = []
                    
                    # Determine frequency of data
                    if len(ts_data) > 1:
                        # Check if we have regular frequency
                        date_diffs = pd.Series(ts_data[date_col].diff().dropna())
                        most_common_diff = date_diffs.value_counts().index[0]
                        
                        # Generate future dates based on most common difference
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + i * most_common_diff)
                    else:
                        # Default to daily if can't determine
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + timedelta(days=i))
                    
                    # Use the last moving average value for forecast
                    forecast_values = [ma.iloc[-1]] * forecast_periods
                    
                    # Plot moving average
                    ax.plot(ts_data[date_col], ma, label='Moving Average', linestyle='--')
                    # Plot forecast
                    ax.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--', color='red')
                    
                elif forecast_method == "Exponential Smoothing":
                    alpha = st.slider("Smoothing Factor (Alpha)", 0.0, 1.0, 0.3, 0.1)
                    
                    # Apply simple exponential smoothing
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    model = SimpleExpSmoothing(ts_data[value_col]).fit(smoothing_level=alpha)
                    
                    # Generate forecast
                    forecast = model.forecast(forecast_periods)
                    
                    # Generate forecast dates
                    last_date = ts_data[date_col].max()
                    forecast_dates = []
                    
                    # Determine frequency
                    if len(ts_data) > 1:
                        # Check if we have regular frequency
                        date_diffs = pd.Series(ts_data[date_col].diff().dropna())
                        most_common_diff = date_diffs.value_counts().index[0]
                        
                        # Generate future dates
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + i * most_common_diff)
                    else:
                        # Default to daily
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + timedelta(days=i))
                    
                    # Plot fitted values
                    ax.plot(ts_data[date_col], model.fittedvalues, label='Fitted', linestyle='--')
                    # Plot forecast
                    ax.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='red')
                    
                elif forecast_method == "Linear Trend":
                    # Simple linear regression
                    from sklearn.linear_model import LinearRegression
                    
                    # Convert dates to numeric for linear regression
                    # Create a feature based on days since the first date
                    ts_data['days_from_start'] = (ts_data[date_col] - ts_data[date_col].min()).dt.total_seconds() / (24*60*60)
                    X = ts_data[['days_from_start']]
                    y = ts_data[value_col]
                    
                    # Fit linear model
                    model = LinearRegression().fit(X, y)
                    
                    # Generate forecast dates
                    last_date = ts_data[date_col].max()
                    last_days = ts_data['days_from_start'].max()
                    forecast_dates = []
                    forecast_days = []
                    
                    # Determine frequency
                    if len(ts_data) > 1:
                        # Check if we have regular frequency
                        date_diffs = pd.Series(ts_data[date_col].diff().dropna())
                        most_common_diff = date_diffs.value_counts().index[0]
                        day_diff = most_common_diff.total_seconds() / (24*60*60)
                        
                        # Generate future dates and days
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + i * most_common_diff)
                            forecast_days.append(last_days + i * day_diff)
                    else:
                        # Default to daily
                        for i in range(1, forecast_periods + 1):
                            forecast_dates.append(last_date + timedelta(days=i))
                            forecast_days.append(last_days + i)
                    
                    # Generate predictions for both original and forecast data
                    ts_data['prediction'] = model.predict(X)
                    forecast_values = model.predict(np.array(forecast_days).reshape(-1, 1))
                    
                    # Plot fitted values
                    ax.plot(ts_data[date_col], ts_data['prediction'], label='Fitted Trend', linestyle='--')
                    # Plot forecast
                    ax.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--', color='red')
                
                # Finalize the plot
                ax.set_title(f"{forecast_method} Forecast for {value_col}")
                ax.set_xlabel(date_col)
                ax.set_ylabel(value_col)
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display forecast data
                st.write("#### Forecast Data")
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    f'Forecast {value_col}': forecast_values if forecast_method == "Linear Trend" else forecast
                })
                safe_display_dataframe(forecast_df)
                
                # Show forecast metrics if we have them
                if forecast_method == "Exponential Smoothing":
                    st.write("#### Model Summary")
                    st.write(f"Alpha (smoothing factor): {alpha}")
                    st.write(f"Initial value: {model.params['initial_level']:.4f}")
                elif forecast_method == "Linear Trend":
                    st.write("#### Linear Model Summary")
                    st.write(f"Slope: {model.coef_[0]:.4f}")
                    st.write(f"Intercept: {model.intercept_:.4f}")
                    st.write(f"Equation: {value_col} = {model.coef_[0]:.4f} * days + {model.intercept_:.4f}")
        
        else:
            st.error("No numeric columns found for time series analysis.")
    else:
        st.warning("Please convert a column to datetime format to continue with time series analysis.") 