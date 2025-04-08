import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import DataCleaner class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import DataCleaner, safe_display_dataframe
except ImportError:
    st.error("Unable to import DataCleaner class. Please ensure data_cleaner.py is in the project root.")
    
    # Define a basic version for demo purposes if import fails
    class DataCleaner:
        def __init__(self, df):
            self.df = df.copy()
            self.original_df = df.copy()
            
        def convert_datatypes(self, column_types):
            for column, dtype in column_types.items():
                try:
                    if dtype == 'int':
                        self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
                    elif dtype == 'float':
                        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                    elif dtype == 'category':
                        self.df[column] = self.df[column].astype('category')
                    elif dtype == 'str':
                        self.df[column] = self.df[column].astype(str)
                    elif dtype == 'datetime':
                        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                except Exception as e:
                    st.warning(f"Error converting column '{column}' to type '{dtype}': {str(e)}")
                    
        def convert_datetime_columns(self, df=None, columns=None):
            if df is None:
                df = self.df.copy()
            if columns is None:
                return df
            
            for col in columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Convert Data Types",
    page_icon="🔄",
    layout="wide"
)

st.markdown("# Convert Data Types")
st.sidebar.header("Convert Data Types")
st.write(
    """
    This page helps you convert data types in your dataset to ensure proper analysis.
    Choose columns and target data types to clean and standardize your data.
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
    
    st.write("### Data Preview")
    safe_display_dataframe(df.head(10), cleaner)
    
    # Display current data types
    st.write("### Current Data Types")
    dtypes_df = pd.DataFrame(df.dtypes, columns=['Current Type'])
    dtypes_df = dtypes_df.reset_index().rename(columns={'index': 'Column'})
    safe_display_dataframe(dtypes_df, cleaner)
    
    # Basic data type conversion
    with st.expander("Basic Data Type Conversion", expanded=True):
        st.write("#### Convert Column Data Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            col_to_convert = st.selectbox("Select column to convert", df.columns)
        
        with col2:
            target_type = st.selectbox(
                "Select target data type",
                ["float", "int", "str", "category", "datetime"]
            )
        
        if st.button("Convert Data Type"):
            with st.spinner(f"Converting {col_to_convert} to {target_type}..."):
                try:
                    column_types = {col_to_convert: target_type}
                    cleaner.convert_datatypes(column_types)
                    
                    # Update session state
                    st.session_state.data = cleaner.df
                    st.success(f"Converted {col_to_convert} to {target_type}.")
                    
                    # Show updated data types
                    st.write("#### Updated Data Types")
                    updated_dtypes = pd.DataFrame(cleaner.df.dtypes, columns=['Current Type'])
                    updated_dtypes = updated_dtypes.reset_index().rename(columns={'index': 'Column'})
                    safe_display_dataframe(updated_dtypes, cleaner)
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
    
    # Batch data type conversion
    with st.expander("Batch Data Type Conversion", expanded=False):
        st.write("#### Convert Multiple Columns at Once")
        
        # Create a dataframe to hold conversion selections
        col_selections = {}
        
        for col in df.columns:
            col_type = st.selectbox(
                f"Convert '{col}' to:", 
                [None, 'int', 'float', 'str', 'category', 'datetime'],
                key=f"batch_{col}"
            )
            if col_type:
                col_selections[col] = col_type
        
        if st.button("Apply Batch Conversion"):
            if col_selections:
                with st.spinner("Converting data types..."):
                    cleaner.convert_datatypes(col_selections)
                    
                    # Update session state
                    st.session_state.data = cleaner.df
                    st.success(f"Converted {len(col_selections)} columns.")
                    
                    # Show updated data types
                    st.write("#### Updated Data Types")
                    updated_dtypes = pd.DataFrame(cleaner.df.dtypes, columns=['Current Type'])
                    updated_dtypes = updated_dtypes.reset_index().rename(columns={'index': 'Column'})
                    safe_display_dataframe(updated_dtypes, cleaner)
            else:
                st.warning("No columns selected for conversion.")
    
    # Auto-detect and convert datetime columns
    with st.expander("Auto-Detect Date/Time Columns", expanded=False):
        st.write("#### Auto-Detect and Convert Date/Time Columns")
        
        if st.button("Detect Date/Time Columns"):
            with st.spinner("Detecting date/time columns..."):
                # Try to auto-detect datetime columns
                date_columns = []
                for col in df.columns:
                    if pd.api.types.is_object_dtype(df[col]):
                        # Try to convert to datetime and check if it works
                        datetime_col = pd.to_datetime(df[col], errors='coerce')
                        if not datetime_col.isna().all() and datetime_col.isna().sum() / len(datetime_col) < 0.5:
                            date_columns.append(col)
                
                if date_columns:
                    st.success(f"Detected {len(date_columns)} potential date/time columns.")
                    
                    selected_date_cols = st.multiselect(
                        "Select columns to convert to datetime",
                        date_columns
                    )
                    
                    if selected_date_cols and st.button("Convert Selected to Datetime"):
                        updated_df = cleaner.convert_datetime_columns(columns=selected_date_cols)
                        
                        # Update the cleaner's dataframe and session state
                        cleaner.df = updated_df
                        st.session_state.data = cleaner.df
                        st.success(f"Converted {len(selected_date_cols)} columns to datetime format.")
                        
                        # Show updated data types
                        st.write("#### Updated Data Types")
                        updated_dtypes = pd.DataFrame(cleaner.df.dtypes, columns=['Current Type'])
                        updated_dtypes = updated_dtypes.reset_index().rename(columns={'index': 'Column'})
                        safe_display_dataframe(updated_dtypes, cleaner)
                else:
                    st.info("No potential date/time columns detected in the dataset.")
    
    # Standardize column names
    with st.expander("Standardize Column Names", expanded=False):
        st.write("#### Standardize Column Names")
        st.write("""
        This will convert all column names to lowercase and replace spaces with underscores,
        which is a recommended practice for data analysis.
        """)
        
        if st.button("Standardize Column Names"):
            with st.spinner("Standardizing column names..."):
                original_columns = list(cleaner.df.columns)
                cleaner.df.columns = [col.lower().replace(' ', '_') for col in cleaner.df.columns]
                
                # Update session state
                st.session_state.data = cleaner.df
                
                # Show before/after comparison
                comparison = pd.DataFrame({
                    'Original': original_columns,
                    'Standardized': list(cleaner.df.columns)
                })
                st.success("Column names have been standardized.")
                safe_display_dataframe(comparison, cleaner) 