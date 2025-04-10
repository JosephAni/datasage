import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="AI-Powered Data Cleaning Agent",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Include custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B70E2;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .feature-card h3 {
        color: #2C3E50;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    .feature-card p {
        color: #34495E;
        font-size: 0.95rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Define a function to load shared data
@st.cache_data
def load_sample_data():
    """Load a sample dataset for demonstration"""
    try:
        # Try loading sample data
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df
    except:
        # Return an empty dataframe if loading fails
        return pd.DataFrame()

def safe_display_dataframe(df, cleaner=None, **kwargs):
    """
    Safely display a dataframe in Streamlit by ensuring it's compatible with PyArrow.
    This avoids common Arrow serialization errors.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        cleaner (DataCleaner, optional): DataCleaner instance, created if None
        **kwargs: Additional arguments to pass to st.dataframe
        
    Returns:
        The result of st.dataframe
    """
    if df is None or df.empty:
        return st.dataframe(pd.DataFrame(), **kwargs)
    
    try:
        # Create a copy to avoid modifying the original
        display_df = df.copy()
        
        # Handle specific types that cause PyArrow issues
        for col in display_df.columns:
            # Convert any object columns to string
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
                
            # Handle any columns with mixed types
            elif 'mixed' in str(display_df[col].dtype).lower():
                display_df[col] = display_df[col].astype(str)
            
            # Special handling for datetime and complex types
            elif pd.api.types.is_datetime64_dtype(display_df[col]):
                # Keep datetime as is
                pass
            elif display_df[col].dtype == complex:
                display_df[col] = display_df[col].apply(str)
                
        # Display the safe dataframe
        return st.dataframe(display_df, **kwargs)
    except Exception as e:
        # Fallback to displaying as strings if all else fails
        st.warning(f"Error preparing dataframe for display: {str(e)}. Showing text representation.")
        return st.dataframe(df.astype(str), **kwargs)

def main():
    # Main app header
    st.markdown("<h1 class='main-header'>AI-Powered Data Cleaning Agent 🖨️💎🛁📊🤖</h1>", unsafe_allow_html=True)
    
    # Create a custom container with dark background and light text for better visibility
    st.markdown("""
    <div style="background-color: #1E3A8A; color: white; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
        <h2 style="color: #FFFFFF; font-size: 1.8rem;">Transform Messy Data into Actionable Insights</h2>
        <p style="font-size: 1.1rem; margin: 15px 0; color: #E5E7EB;">
            Our AI-powered data cleaning agent helps businesses and analysts tackle challenging data preparation tasks with ease.
            Stop spending 80% of your time cleaning data and focus on extracting valuable insights and making data-driven decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use columns with colorful cards for better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #4B70E2; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: white; font-size: 1.3rem;">Why Use This Tool?</h3>
            <ul style="color: white; padding-left: 20px;">
                <li>Handles missing data, outliers, and incorrect formats automatically</li>
                <li>Transforms raw data into ML-ready datasets with feature engineering</li>
                <li>Provides advanced forecasting and business analytics capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #2C3E50; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: white; font-size: 1.3rem;">Ideal For</h3>
            <ul style="color: white; padding-left: 20px;">
                <li>Business analysts preparing data for reports and dashboards</li>
                <li>Data scientists building predictive models and ML pipelines</li>
                <li>Retailers analyzing inventory, sales, and customer data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Get started section with contrasting background
    st.markdown("""
    <div style="background-color: #4F46E5; color: white; padding: 15px; border-radius: 8px; margin: 10px 0 30px 0;">
        <h3 style="color: white; font-size: 1.3rem; margin-bottom: 10px;">Get Started in 3 Simple Steps:</h3>
        <ol style="color: white; padding-left: 25px;">
            <li><strong>Upload your data</strong> via the sidebar (CSV, Excel, or JSON)</li>
            <li><strong>Choose a feature</strong> from the navigation menu that matches your need</li>
            <li><strong>Get instant results</strong> with visualizations and downloadable clean data</li>
        </ol>
        <p style="color: #E5E7EB; font-style: italic; margin-top: 10px; font-size: 0.9rem;">
            No data? No problem! Use our sample dataset to explore the application's capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="background-color: #2C3E50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: white; margin-bottom: 10px;">Welcome to the AI-Powered Data Cleaning Agent!</h3>
    <p style="color: white; margin-bottom: 15px;">This tool helps you clean, analyze, and visualize your data 
    with advanced features including:</p>
    
    <ul style="color: white; padding-left: 25px; margin-bottom: 15px;">
    <li>Data cleaning and preprocessing</li>
    <li>Exploratory data analysis</li>
    <li>Time series analysis and forecasting</li>
    <li>Customer Lifetime Value (CLV) calculation</li>
    <li>Inventory turnover simulation</li>
    <li>Advanced data cleaning and analysis</li>
    <li>And much more!</li>
    </ul>
    
    <p style="color: white; font-weight: 500;">Get started by uploading your data file using the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader in the sidebar
    with st.sidebar:
        st.markdown("### Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", 
                                         type=["csv", "xlsx", "xls", "json"])
        
        # Add a checkbox to use sample data
        use_sample_data = st.checkbox("Use sample data")
        
        # Add navigation links to all pages
        st.markdown("### Navigation")
        st.markdown("""
        - [Home](/)
        - [Data Cleaning](/1_Data_Cleaning)
        - [Inventory Turnover](/2_Inventory_Turnover)
        - [Data Visualization](/3_Data_Visualization)
        - [Time Series Analysis](/4_Time_Series)
        - [CLV Analysis](/5_CLV_Analysis) 
        - [Machine Learning](/6_Machine_Learning)
        - [Feature Engineering](/7_Feature_Engineering)
        - [Convert Data Types](/8_Convert_Data_Types)
        - [Data Interpretation](/9_Data_Interpretation)
        - [Advanced Data Cleaning](/10_Advanced_Data_Cleaning)
        - [Advanced Data Analysis](/11_Advanced_Data_Analysis)
        - [Demand Forecasting](/12_Demand_Forecasting)
        - [Cost of Inventory](/13_Cost_of_Inventory)
        - [EOQ Simulation](/14_EOQ_Simulation)
        - [Newsvendor Simulator](/15_Newsvendor_Simulator)
        """)
    
    # Data loading logic
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
                
            # Save to session state to share across pages
            st.session_state.data = df
            
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Use sample data if checked and no file is uploaded
    elif use_sample_data:
        df = load_sample_data()
        # Save to session state to share across pages
        st.session_state.data = df
        st.info("Using sample Titanic dataset. Upload your own file to process your data.")
    
    # If we have data either from upload or sample, ensure it's PyArrow compatible
    if df is not None:
        # Ensure data types are compatible with PyArrow
        for col in df.columns:
            # Convert any complex types to string to prevent PyArrow serialization issues
            if df[col].dtype == 'object' and not pd.api.types.is_string_dtype(df[col]):
                try:
                    df[col] = df[col].astype(str)
                except:
                    # If conversion fails, at least try to handle nulls
                    df[col] = df[col].fillna("").astype(str)
        
        # Update session state with compatible data
        st.session_state.data = df
    
    # Display available features with descriptions
    st.markdown("<h2 class='sub-header'>Available Features</h2>", unsafe_allow_html=True)
    
    # First row
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>Data Cleaning</h3>
        <p>Clean and preprocess your data with tools for handling missing values, outliers, and data transformations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>Customer Lifetime Value</h3>
        <p>Calculate and analyze customer lifetime value to understand customer behavior and identify valuable segments.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row1_col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>Cost of Inventory</h3>
        <p>Calculate and analyze inventory costs including holding costs, turnover metrics, and optimization recommendations based on industry benchmarks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row2_col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>Data Visualization</h3>
        <p>Create insightful visualizations to understand your data better with interactive charts and plots.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>Inventory Turnover Simulation</h3>
        <p>Visualize inventory turnover rates and compare different scenarios to improve inventory management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>Advanced Data Cleaning</h3>
        <p>Advanced techniques for data cleaning, including handling skewness, high cardinality, and advanced imputation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Third row
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    
    with row3_col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>Time Series Analysis</h3>
        <p>Analyze and forecast time series data with advanced statistical models and machine learning techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row3_col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>Feature Engineering</h3>
        <p>Create new features, transform variables, and prepare your data for machine learning models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row3_col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>Data Interpretation</h3>
        <p>Extract insights and understand the meaning of your data with statistical analysis and hypothesis testing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Fourth row for the last two cards
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    
    with row4_col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>Demand Forecasting 📈</h3>
        <p>Advanced demand forecasting tools using time series analysis and machine learning techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row4_col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>EOQ Simulation 📊</h3>
        <p>Interactive Economic Order Quantity simulator to optimize inventory ordering decisions. Visualize cost trade-offs between order costs and holding costs to find the optimal order quantity that minimizes total inventory costs.</p>
        </div>
        """, unsafe_allow_html=True)

    with row4_col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>Newsvendor Simulator📰</h3>
        <p>Optimize single-period inventory decisions under uncertainty. Find the best order quantity balancing overstocking and understocking costs using Normal or Uniform demand distributions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display basic data information if data is loaded
    if df is not None:
        st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
        safe_display_dataframe(df)
        
        # Basic data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        # Data types summary
        st.markdown("<h3>Data Types</h3>", unsafe_allow_html=True)
        dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
        dtypes = dtypes.reset_index().rename(columns={'index': 'Column'})
        safe_display_dataframe(dtypes)
    
    # Show instructions for navigating to other pages
    if df is None:
        st.markdown("""
        <div style="background-color: #2C3E50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin-bottom: 10px;">Getting Started</h3>
        <p style="color: white; margin-bottom: 15px;">To get started:</p>
        <ol style="color: white; padding-left: 25px; margin-bottom: 15px;">
            <li>Upload your data using the file uploader in the sidebar, or use the sample dataset.</li>
            <li>Navigate to different pages using the sidebar menu to access specific features.</li>
            <li>Each page provides specialized tools for different aspects of data analysis and cleaning.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 