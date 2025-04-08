import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import io

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
            
        def handle_missing_values(self, strategy='mean', columns=None):
            if columns is None:
                columns = self.df.columns[self.df.isnull().any()].tolist()
            
            for col in columns:
                if self.df[col].isnull().any():
                    if strategy in ['mean', 'median'] and pd.api.types.is_numeric_dtype(self.df[col]):
                        imputer = SimpleImputer(strategy=strategy)
                        self.df[col] = imputer.fit_transform(self.df[[col]])
                    elif strategy == 'mode':
                        imputer = SimpleImputer(strategy='most_frequent')
                        self.df[col] = imputer.fit_transform(self.df[[col]])
                    elif strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
                        
        def detect_outliers(self, columns=None, contamination=0.1):
            # Basic outlier detection
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(columns) == 0:
                st.warning("No numeric columns available for outlier detection.")
                return None
                
            X = self.df[columns].copy()
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(X)
            
            self.df['is_outlier'] = outliers == -1
            return self.df[self.df['is_outlier']]
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

st.markdown("# Data Cleaning")
st.sidebar.header("Data Cleaning")
st.write(
    """
    This page provides tools for cleaning and preprocessing your data, including
    handling missing values, detecting outliers, and transforming variables.
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
    
    st.write("### Data Overview")
    st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        safe_display_dataframe(df.head(10), cleaner)
    
    # Missing values analysis
    with st.expander("Missing Values Analysis", expanded=True):
        missing_cols = df.columns[df.isnull().any()].tolist()
        missing_counts = df[missing_cols].isnull().sum()
        
        if missing_counts.empty:
            st.write("No missing values found in the dataset! 🎉")
        else:
            st.write("#### Missing Values by Column")
            missing_df = pd.DataFrame({
                'Column': missing_counts.index,
                'Missing Count': missing_counts.values,
                'Missing Percentage': (missing_counts.values / len(df) * 100).round(2)
            }).sort_values(by='Missing Count', ascending=False)
            
            safe_display_dataframe(missing_df, cleaner)
            
            # Plot missing values
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(missing_df['Column'], missing_df['Missing Percentage'])
            ax.set_ylabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Column')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Handle missing values
            st.write("#### Handle Missing Values")
            
            imputation_strategy = st.selectbox(
                "Select imputation strategy",
                ["mean", "median", "mode", "drop", "knn"]
            )
            
            selected_cols = st.multiselect(
                "Select columns to apply imputation (or leave empty for all columns with missing values)",
                missing_cols
            )
            
            if st.button("Apply Imputation"):
                with st.spinner("Applying imputation..."):
                    if not selected_cols:
                        selected_cols = missing_cols
                    
                    if imputation_strategy == "knn":
                        # For KNN imputation, only select numeric columns
                        numeric_cols = df[selected_cols].select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            knn_imputer = KNNImputer(n_neighbors=5)
                            cleaner.df[numeric_cols] = pd.DataFrame(
                                knn_imputer.fit_transform(cleaner.df[numeric_cols]),
                                columns=numeric_cols,
                                index=cleaner.df.index
                            )
                            st.success(f"Applied KNN imputation to {len(numeric_cols)} numeric columns.")
                        else:
                            st.warning("No numeric columns selected for KNN imputation.")
                    else:
                        cleaner.handle_missing_values(imputation_strategy, selected_cols)
                        st.success(f"Applied {imputation_strategy} imputation to {len(selected_cols)} columns.")
                    
                    # Update session state
                    st.session_state.data = cleaner.df
    
    # Outlier detection
    with st.expander("Outlier Detection", expanded=False):
        st.write("#### Detect Outliers")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for outlier detection.")
        else:
            selected_cols = st.multiselect(
                "Select columns for outlier detection (or leave empty for all numeric columns)",
                numeric_cols
            )
            
            contamination = st.slider(
                "Contamination (expected proportion of outliers)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
            )
            
            if st.button("Detect Outliers"):
                with st.spinner("Detecting outliers..."):
                    outliers = cleaner.detect_outliers(
                        columns=selected_cols if selected_cols else None,
                        contamination=contamination
                    )
                    
                    if outliers is not None and not outliers.empty:
                        st.write(f"Found {len(outliers)} potential outliers out of {len(df)} rows.")
                        safe_display_dataframe(outliers, cleaner)
                        
                        # Plot distributions with outliers highlighted
                        if selected_cols:
                            cols_to_plot = selected_cols
                        else:
                            # Limit to a reasonable number of columns for plotting
                            cols_to_plot = numeric_cols[:5]
                        
                        for col in cols_to_plot:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.histplot(
                                data=df, x=col, color='blue', alpha=0.5, ax=ax, label='All Data'
                            )
                            sns.histplot(
                                data=outliers, x=col, color='red', alpha=0.5, ax=ax, label='Outliers'
                            )
                            ax.set_title(f'Distribution of {col} with Outliers Highlighted')
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Option to remove outliers
                        if st.button("Remove Outliers"):
                            cleaner.df = cleaner.df[~cleaner.df['is_outlier']].drop(columns=['is_outlier'])
                            st.session_state.data = cleaner.df
                            st.success(f"Removed {len(outliers)} outliers from the dataset.")
                    else:
                        st.success("No outliers detected with the current settings.")
    
    # Data type conversion
    with st.expander("Data Type Conversion", expanded=False):
        st.write("#### Convert Data Types")
        
        # Show current data types
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Current Type'])
        dtypes_df = dtypes_df.reset_index().rename(columns={'index': 'Column'})
        safe_display_dataframe(dtypes_df, cleaner)
        
        # Data type conversion
        col_to_convert = st.selectbox("Select column to convert", df.columns)
        target_type = st.selectbox(
            "Select target data type",
            ["float", "int", "str", "category", "datetime"]
        )
        
        if st.button("Convert Data Type"):
            with st.spinner(f"Converting {col_to_convert} to {target_type}..."):
                try:
                    if target_type == "float":
                        cleaner.df[col_to_convert] = cleaner.df[col_to_convert].astype(float)
                    elif target_type == "int":
                        cleaner.df[col_to_convert] = cleaner.df[col_to_convert].astype(int)
                    elif target_type == "str":
                        cleaner.df[col_to_convert] = cleaner.df[col_to_convert].astype(str)
                    elif target_type == "category":
                        cleaner.df[col_to_convert] = cleaner.df[col_to_convert].astype('category')
                    elif target_type == "datetime":
                        cleaner.df[col_to_convert] = pd.to_datetime(cleaner.df[col_to_convert], errors='coerce')
                    
                    st.session_state.data = cleaner.df
                    st.success(f"Converted {col_to_convert} to {target_type}.")
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
    
    # Save cleaned data
    with st.expander("Save Cleaned Data", expanded=False):
        st.write("#### Save Cleaned Data")
        
        if st.button("Update Session Data"):
            st.session_state.data = cleaner.df
            st.success("Session data updated with cleaned dataset.")
        
        # Download option
        output_format = st.selectbox("Select download format", ["CSV", "Excel", "JSON"])
        
        if st.button("Download Cleaned Data"):
            with st.spinner("Preparing download..."):
                if output_format == "CSV":
                    csv = cleaner.df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                elif output_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        cleaner.df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                elif output_format == "JSON":
                    json_str = cleaner.df.to_json(orient="records")
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="cleaned_data.json",
                        mime="application/json"
                    ) 