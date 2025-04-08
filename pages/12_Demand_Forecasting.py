import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Add parent directory to path to import DataCleaner class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import DataCleaner, safe_display_dataframe
    # Check if plotly is available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
except ImportError:
    st.error("Unable to import DataCleaner class. Please ensure data_cleaner.py is in the project root.")
    
    # Define a basic version for demo purposes if import fails
    class DataCleaner:
        def __init__(self, df):
            self.df = df.copy()
            self.original_df = df.copy()
            
        def forecast_demand(self, date_column, demand_column, forecast_periods=30, algorithm='prophet', seasonality='auto'):
            st.warning("Full forecasting functionality not available. Please ensure the DataCleaner module is properly imported.")
            return None, None
            
        def simulate_forecast_aggregation(self, num_stores=8, num_weeks=1, store_df=None):
            st.warning("Forecast aggregation simulation not available. Please ensure the DataCleaner module is properly imported.")
            return None, None, None, None
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)
    
    PLOTLY_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.markdown("# Demand Forecasting")
st.sidebar.header("Demand Forecasting")
st.write(
    """
    This page provides demand forecasting tools to predict future values based on historical time series data.
    Choose from different forecasting algorithms and customize parameters to get the most accurate predictions.
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

# Initialize session state for the cleaner
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None

# Get data from session state
df = get_data()

if df is None:
    st.info("No data loaded. Please upload data on the Home page.")
else:
    # Initialize DataCleaner if not already done
    if st.session_state.cleaner is None:
        st.session_state.cleaner = DataCleaner(df)
    
    cleaner = st.session_state.cleaner
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        st.write("### Data Sample")
        safe_display_dataframe(df.head(5), cleaner)
    
    # Check if Plotly is available
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Some visualizations may be limited. Install with 'pip install plotly' for interactive plots.")
    
    # Demand Forecasting
    with st.expander("Generate Demand Forecast", expanded=True):
        st.write("### Demand Forecasting")
        st.write("""
        Generate forecasts for future demand based on historical time series data using various algorithms.
        """)
        
        # Find potential date columns
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                date_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
        
        if not date_cols:
            st.warning("No date/time columns detected in the dataset. Please convert at least one column to datetime format.")
        else:
            # Get numeric columns for demand
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found for demand values.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    date_column = st.selectbox("Select date/time column", date_cols)
                
                with col2:
                    demand_column = st.selectbox("Select demand column", numeric_cols)
                
                # Forecasting parameters
                st.write("#### Forecasting Parameters")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    forecast_periods = st.slider(
                        "Number of periods to forecast",
                        min_value=1,
                        max_value=365,
                        value=30
                    )
                
                with col2:
                    algorithm = st.selectbox(
                        "Forecasting algorithm",
                        ["prophet", "arima", "exponential"]
                    )
                
                with col3:
                    seasonality = st.selectbox(
                        "Seasonality type",
                        ["auto", "daily", "weekly", "monthly", "yearly"]
                    )
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating demand forecast..."):
                        try:
                            # Run forecasting
                            forecast_results, forecast_fig = cleaner.forecast_demand(
                                date_column=date_column,
                                demand_column=demand_column,
                                forecast_periods=forecast_periods,
                                algorithm=algorithm,
                                seasonality=seasonality
                            )
                            
                            if forecast_results is not None:
                                st.success("Forecast generated successfully!")
                                
                                # Display forecast plot
                                st.write("#### Forecast Visualization")
                                if PLOTLY_AVAILABLE and forecast_fig is not None:
                                    st.plotly_chart(forecast_fig, use_container_width=True)
                                else:
                                    # Fallback to matplotlib
                                    forecast_dates = forecast_results['date']
                                    forecast_values = forecast_results['demand']
                                    
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.plot(forecast_dates, forecast_values, color='blue', label='Forecast')
                                    
                                    # Add confidence intervals if available
                                    if 'demand_lower' in forecast_results.columns and 'demand_upper' in forecast_results.columns:
                                        ax.fill_between(
                                            forecast_dates,
                                            forecast_results['demand_lower'],
                                            forecast_results['demand_upper'],
                                            alpha=0.2,
                                            color='blue',
                                            label='95% Confidence Interval'
                                        )
                                    
                                    ax.set_title(f'Demand Forecast for {demand_column} ({forecast_periods} periods)')
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Demand')
                                    ax.legend()
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Display forecast results table
                                st.write("#### Forecast Results")
                                
                                # Format date column for display
                                display_results = forecast_results.copy()
                                display_results['date'] = display_results['date'].dt.strftime('%Y-%m-%d')
                                
                                safe_display_dataframe(display_results, cleaner)
                                
                                # Provide download button for forecast results
                                csv = forecast_results.to_csv(index=False)
                                st.download_button(
                                    label="Download Forecast CSV",
                                    data=csv,
                                    file_name=f"forecast_{demand_column}_{algorithm}_{forecast_periods}periods.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Failed to generate forecast. Please check your data and parameters.")
                        except Exception as e:
                            st.error(f"Error during forecasting: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
    
    # Forecast Accuracy Evaluation
    with st.expander("Forecast Accuracy Evaluation", expanded=False):
        st.write("### Evaluate Forecast Accuracy")
        st.write("""
        If you have actual values and forecasts, evaluate the accuracy of your forecasts using various metrics.
        """)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for forecast evaluation (actual and forecast values).")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                actual_column = st.selectbox("Select actual values column", numeric_cols, key="actual_col")
            
            with col2:
                forecast_column = st.selectbox("Select forecast values column", 
                                             [col for col in numeric_cols if col != actual_column], 
                                             key="forecast_col")
            
            # Optional product/category column for grouped evaluation
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            product_column = None
            
            if categorical_cols:
                product_column = st.selectbox(
                    "Select product/category column (optional)",
                    [None] + categorical_cols
                )
            
            if st.button("Evaluate Forecast Accuracy"):
                with st.spinner("Calculating forecast accuracy metrics..."):
                    # Check for missing or invalid values
                    invalid_mask = df[actual_column].isna() | df[forecast_column].isna() | (df[forecast_column] == 0)
                    
                    if invalid_mask.any():
                        st.warning(f"Found {invalid_mask.sum()} rows with missing values or zero forecasts. These will be excluded from the analysis.")
                    
                    valid_df = df[~invalid_mask].copy()
                    
                    # Check if we have data to analyze
                    if len(valid_df) == 0:
                        st.error("No valid data available for accuracy evaluation after removing missing values and zero forecasts.")
                    else:
                        try:
                            # Calculate basic accuracy metrics
                            accuracy_df = valid_df.copy()
                            
                            # Calculate metrics
                            accuracy_df['Error'] = accuracy_df[actual_column] - accuracy_df[forecast_column]
                            accuracy_df['Absolute Error'] = abs(accuracy_df['Error'])
                            accuracy_df['Percentage Error'] = (accuracy_df['Error'] / accuracy_df[forecast_column]) * 100
                            accuracy_df['Absolute Percentage Error'] = abs(accuracy_df['Percentage Error'])
                            accuracy_df['A/F Ratio'] = accuracy_df[actual_column] / accuracy_df[forecast_column]
                            
                            # Group by product if specified
                            if product_column:
                                result = accuracy_df.groupby(product_column).agg({
                                    actual_column: 'sum',
                                    forecast_column: 'sum',
                                    'A/F Ratio': 'mean',
                                    'Error': 'sum',
                                    'Absolute Error': 'mean',
                                    'Absolute Percentage Error': 'mean'
                                }).round(2)
                                
                                # Rename columns for clarity
                                result = result.rename(columns={
                                    actual_column: 'Total Actual',
                                    forecast_column: 'Total Forecast',
                                    'Absolute Error': 'MAE',
                                    'Absolute Percentage Error': 'MAPE (%)'
                                })
                                
                                # Display results by product/category
                                st.write("#### Accuracy Metrics by Product/Category")
                                st.dataframe(result)
                                
                                # Overall metrics
                                total_actual = accuracy_df[actual_column].sum()
                                total_forecast = accuracy_df[forecast_column].sum()
                                
                                overall = pd.DataFrame({
                                    'Metric': [
                                        'Total Actual', 'Total Forecast', 
                                        'Overall A/F Ratio', 'Overall Bias (%)',
                                        'MAE', 'MAPE (%)', 'Count'
                                    ],
                                    'Value': [
                                        total_actual,
                                        total_forecast,
                                        total_actual / total_forecast,
                                        ((total_forecast - total_actual) / total_actual) * 100,
                                        accuracy_df['Absolute Error'].mean(),
                                        accuracy_df['Absolute Percentage Error'].mean(),
                                        len(accuracy_df)
                                    ]
                                })
                                
                                st.write("#### Overall Accuracy Metrics")
                                st.dataframe(overall)
                                
                                # Visualization of actual vs forecast by product
                                st.write("#### Actual vs Forecast by Product/Category")
                                
                                product_summary = accuracy_df.groupby(product_column).agg({
                                    actual_column: 'sum',
                                    forecast_column: 'sum'
                                }).reset_index()
                                
                                if PLOTLY_AVAILABLE:
                                    fig = px.bar(
                                        product_summary,
                                        x=product_column,
                                        y=[actual_column, forecast_column],
                                        barmode='group',
                                        title=f'Actual vs Forecast by {product_column}'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    product_summary.plot(
                                        x=product_column,
                                        y=[actual_column, forecast_column],
                                        kind='bar',
                                        ax=ax
                                    )
                                    ax.set_title(f'Actual vs Forecast by {product_column}')
                                    ax.set_ylabel('Value')
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            else:
                                # Overall metrics without grouping
                                total_actual = accuracy_df[actual_column].sum()
                                total_forecast = accuracy_df[forecast_column].sum()
                                
                                overall = pd.DataFrame({
                                    'Metric': [
                                        'Total Actual', 'Total Forecast', 
                                        'Overall A/F Ratio', 'Overall Bias (%)',
                                        'MAE', 'MAPE (%)', 'RMSE', 'Count'
                                    ],
                                    'Value': [
                                        total_actual,
                                        total_forecast,
                                        total_actual / total_forecast,
                                        ((total_forecast - total_actual) / total_actual) * 100,
                                        accuracy_df['Absolute Error'].mean(),
                                        accuracy_df['Absolute Percentage Error'].mean(),
                                        np.sqrt((accuracy_df['Error'] ** 2).mean()),
                                        len(accuracy_df)
                                    ]
                                })
                                
                                st.write("#### Overall Accuracy Metrics")
                                st.dataframe(overall)
                                
                                # Visualization of actual vs forecast
                                st.write("#### Actual vs Forecast Comparison")
                                
                                if PLOTLY_AVAILABLE:
                                    fig = px.scatter(
                                        accuracy_df,
                                        x=forecast_column,
                                        y=actual_column,
                                        title='Actual vs Forecast Scatter Plot'
                                    )
                                    
                                    # Add diagonal line (perfect forecast)
                                    max_val = max(accuracy_df[actual_column].max(), accuracy_df[forecast_column].max())
                                    min_val = min(accuracy_df[actual_column].min(), accuracy_df[forecast_column].min())
                                    
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[min_val, max_val],
                                            y=[min_val, max_val],
                                            mode='lines',
                                            name='Perfect Forecast',
                                            line=dict(color='red', dash='dash')
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    ax.scatter(accuracy_df[forecast_column], accuracy_df[actual_column], alpha=0.5)
                                    
                                    # Add diagonal line (perfect forecast)
                                    max_val = max(accuracy_df[actual_column].max(), accuracy_df[forecast_column].max())
                                    min_val = min(accuracy_df[actual_column].min(), accuracy_df[forecast_column].min())
                                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Forecast')
                                    
                                    ax.set_title('Actual vs Forecast Scatter Plot')
                                    ax.set_xlabel(forecast_column)
                                    ax.set_ylabel(actual_column)
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Histogram of errors
                                st.write("#### Error Distribution")
                                
                                if PLOTLY_AVAILABLE:
                                    fig = px.histogram(
                                        accuracy_df,
                                        x='Error',
                                        nbins=30,
                                        title='Distribution of Forecast Errors'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.hist(accuracy_df['Error'], bins=30)
                                    ax.set_title('Distribution of Forecast Errors')
                                    ax.set_xlabel('Error (Actual - Forecast)')
                                    ax.set_ylabel('Frequency')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error evaluating forecast accuracy: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
    
    # Forecast Aggregation Simulation
    with st.expander("Forecast Aggregation Simulation", expanded=False):
        st.write("### Forecast Aggregation Simulation")
        st.write("""
        Simulate how forecasting at different levels affects forecast accuracy due to aggregation effects.
        This tool helps understand the impact of aggregation in hierarchical forecasting.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_stores = st.slider(
                "Number of stores to simulate",
                min_value=2,
                max_value=20,
                value=8
            )
        
        with col2:
            num_weeks = st.slider(
                "Number of weeks to simulate",
                min_value=1,
                max_value=12,
                value=4
            )
        
        if st.button("Run Simulation"):
            with st.spinner("Running forecast aggregation simulation..."):
                try:
                    # Run simulation
                    store_df, simulation_results, comparison_data, run_history = cleaner.simulate_forecast_aggregation(
                        num_stores=num_stores,
                        num_weeks=num_weeks
                    )
                    
                    if store_df is not None and simulation_results is not None:
                        st.success("Simulation completed successfully!")
                        
                        # Display store data
                        st.write("#### Store Characteristics")
                        safe_display_dataframe(store_df, cleaner)
                        
                        # Display simulation results
                        st.write("#### Simulation Results")
                        st.write("""
                        This table shows the forecasts and actuals at different aggregation levels.
                        Compare the accuracy of forecasts created at the item level vs. aggregated levels.
                        """)
                        safe_display_dataframe(simulation_results, cleaner)
                        
                        # Display comparison data with visualization
                        st.write("#### Accuracy Comparison")
                        st.write("""
                        This shows how forecast accuracy changes as you aggregate to higher levels.
                        Typically, higher aggregation levels have better accuracy due to demand pooling effects.
                        """)
                        
                        if comparison_data is not None:
                            safe_display_dataframe(comparison_data, cleaner)
                            
                            # Visualization of MAPE by aggregation level
                            if PLOTLY_AVAILABLE:
                                fig = px.bar(
                                    comparison_data,
                                    x='Aggregation Level',
                                    y='MAPE (%)',
                                    title='Forecast Accuracy by Aggregation Level'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(comparison_data['Aggregation Level'], comparison_data['MAPE (%)'])
                                ax.set_title('Forecast Accuracy by Aggregation Level')
                                ax.set_xlabel('Aggregation Level')
                                ax.set_ylabel('MAPE (%)')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        # Display run history
                        if run_history is not None:
                            st.write("#### Simulation Run History")
                            safe_display_dataframe(run_history, cleaner)
                            
                            # Visualization of run history
                            if 'Run' in run_history.columns and 'Overall MAPE' in run_history.columns:
                                if PLOTLY_AVAILABLE:
                                    fig = px.line(
                                        run_history,
                                        x='Run',
                                        y='Overall MAPE',
                                        title='MAPE by Simulation Run'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(run_history['Run'], run_history['Overall MAPE'])
                                    ax.set_title('MAPE by Simulation Run')
                                    ax.set_xlabel('Run')
                                    ax.set_ylabel('Overall MAPE')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                    else:
                        st.error("Simulation failed. Please check parameters and try again.")
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc()) 