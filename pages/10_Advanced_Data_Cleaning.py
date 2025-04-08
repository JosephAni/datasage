import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer

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
            
        def transform_skewed_features(self, threshold=0.5, method='yeo-johnson'):
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            skewed_cols = []
            transformed_cols = []
            
            for col in numeric_cols:
                skewness = self.df[col].skew()
                if abs(skewness) > threshold:
                    skewed_cols.append((col, skewness))
                    
                    # Basic implementation of transformation
                    if method == 'log':
                        # Log transformation (avoid negative values)
                        min_val = self.df[col].min()
                        if min_val <= 0:
                            self.df[col] = np.log(self.df[col] - min_val + 1)
                        else:
                            self.df[col] = np.log(self.df[col])
                    elif method == 'sqrt':
                        # Square root transformation
                        min_val = self.df[col].min()
                        if min_val < 0:
                            self.df[col] = np.sqrt(self.df[col] - min_val)
                        else:
                            self.df[col] = np.sqrt(self.df[col])
                    else:
                        # Default to Yeo-Johnson transformation
                        try:
                            pt = PowerTransformer(method='yeo-johnson')
                            self.df[col] = pt.fit_transform(self.df[[col]])
                        except Exception as e:
                            st.warning(f"Error transforming {col}: {str(e)}")
                            continue
                        
                    transformed_cols.append(col)
            
            return transformed_cols, skewed_cols
            
        def handle_high_cardinality(self, max_categories=10, method='target_encoding', target_column=None):
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            high_card_cols = []
            
            for col in cat_cols:
                if self.df[col].nunique() > max_categories:
                    high_card_cols.append((col, self.df[col].nunique()))
                    
                    if method == 'group_small':
                        # Group less frequent categories into 'Other'
                        value_counts = self.df[col].value_counts()
                        top_cats = value_counts.nlargest(max_categories).index
                        self.df[col] = np.where(self.df[col].isin(top_cats), self.df[col], 'Other')
                        
            return high_card_cols
            
        def advanced_imputation(self, strategy='knn', columns=None, n_neighbors=5):
            if columns is None:
                columns = self.df.columns[self.df.isnull().any()].tolist()
                
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
            
            if strategy == 'knn' and numeric_cols:
                try:
                    # Simple KNN imputation for numeric columns
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    self.df[numeric_cols] = pd.DataFrame(
                        imputer.fit_transform(self.df[numeric_cols]),
                        columns=numeric_cols,
                        index=self.df.index
                    )
                except Exception as e:
                    st.warning(f"KNN imputation failed: {str(e)}")
                    return False
                return True
            
            return False
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Advanced Data Cleaning",
    page_icon="🧪",
    layout="wide"
)

st.markdown("# Advanced Data Cleaning")
st.sidebar.header("Advanced Data Cleaning")
st.write(
    """
    This page provides advanced tools for data cleaning and transformation, including:
    
    - Handling skewed distributions
    - Managing high cardinality categorical variables
    - Advanced missing value imputation
    - Outlier treatment and transformation
    
    These techniques help prepare your data for more accurate analysis and modeling.
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
    
    # Skewed Feature Transformation
    with st.expander("Transform Skewed Features", expanded=True):
        st.write("### Transform Skewed Distributions")
        st.write("""
        Skewed distributions can negatively affect statistical analyses and machine learning models.
        This tool helps identify and transform skewed numeric features to more normal distributions.
        """)
        
        # Analyze skewness in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns available for skewness analysis.")
        else:
            # Calculate skewness for all numeric columns
            skewness_data = {}
            for col in numeric_cols:
                skewness_data[col] = df[col].skew()
            
            skew_df = pd.DataFrame({
                'Column': list(skewness_data.keys()),
                'Skewness': list(skewness_data.values())
            }).sort_values('Skewness', key=abs, ascending=False)
            
            st.write("#### Skewness Analysis")
            st.write("""
            **Interpretation:**
            - |Skewness| < 0.5: Approximately symmetric
            - 0.5 < |Skewness| < 1: Moderately skewed
            - |Skewness| > 1: Highly skewed
            """)
            
            safe_display_dataframe(skew_df, cleaner)
            
            # Display skewed column distributions
            highly_skewed = skew_df[abs(skew_df['Skewness']) > 1]['Column'].tolist()
            
            if highly_skewed:
                st.write("#### Highly Skewed Columns")
                selected_col = st.selectbox("Select column to visualize", highly_skewed)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Original distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(df[selected_col], kde=True, ax=ax)
                    ax.set_title(f"Original Distribution of {selected_col}\nSkewness: {df[selected_col].skew():.2f}")
                    st.pyplot(fig)
            
            # Transform skewed features
            st.write("#### Transform Skewed Features")
            
            threshold = st.slider(
                "Skewness threshold (transform features with absolute skewness above this value)",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1
            )
            
            transform_method = st.selectbox(
                "Transformation method",
                ["yeo-johnson", "log", "sqrt", "box-cox"],
                help="Yeo-Johnson works on negative values, Box-Cox requires all positive values"
            )
            
            cols_to_transform = st.multiselect(
                "Select columns to transform (leave empty to use all above threshold)",
                numeric_cols
            )
            
            if st.button("Apply Transformation"):
                with st.spinner("Transforming skewed features..."):
                    # If specific columns selected, only transform those meeting threshold
                    if cols_to_transform:
                        # Filter selected columns by threshold
                        cols_to_transform = [col for col in cols_to_transform if abs(skewness_data[col]) > threshold]
                        
                        if not cols_to_transform:
                            st.warning("None of the selected columns meet the skewness threshold.")
                        else:
                            # Apply transformation
                            try:
                                # Custom implementation for specific columns
                                transformed_cols = []
                                for col in cols_to_transform:
                                    # Box-Cox requires positive values
                                    if transform_method == 'box-cox':
                                        if df[col].min() <= 0:
                                            shift = abs(df[col].min()) + 1
                                            df[col] = df[col] + shift
                                        
                                        from scipy import stats
                                        df[col], _ = stats.boxcox(df[col])
                                    elif transform_method == 'log':
                                        # Log transformation (avoid negative values)
                                        min_val = df[col].min()
                                        if min_val <= 0:
                                            df[col] = np.log(df[col] - min_val + 1)
                                        else:
                                            df[col] = np.log(df[col])
                                    elif transform_method == 'sqrt':
                                        # Square root transformation
                                        min_val = df[col].min()
                                        if min_val < 0:
                                            df[col] = np.sqrt(df[col] - min_val)
                                        else:
                                            df[col] = np.sqrt(df[col])
                                    else:
                                        # Yeo-Johnson transformation
                                        pt = PowerTransformer(method='yeo-johnson')
                                        df[col] = pt.fit_transform(df[[col]])
                                    
                                    transformed_cols.append(col)
                                
                                # Update session state
                                cleaner.df = df
                                st.session_state.data = df
                                
                                st.success(f"Successfully transformed {len(transformed_cols)} columns.")
                                
                                # Show before/after comparison for first transformed column
                                if transformed_cols:
                                    col = transformed_cols[0]
                                    after_skew = df[col].skew()
                                    
                                    with col2:
                                        # Transformed distribution
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        sns.histplot(df[col], kde=True, ax=ax)
                                        ax.set_title(f"Transformed Distribution of {col}\nSkewness: {after_skew:.2f}")
                                        st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Transformation error: {str(e)}")
                    else:
                        # Apply transformation to all columns above threshold
                        try:
                            transformed, skewed = cleaner.transform_skewed_features(threshold, transform_method)
                            
                            # Update session state
                            st.session_state.data = cleaner.df
                            
                            if transformed:
                                st.success(f"Successfully transformed {len(transformed)} columns.")
                                
                                # Show before/after comparison for first transformed column
                                if transformed:
                                    col = transformed[0]
                                    after_skew = cleaner.df[col].skew()
                                    
                                    with col2:
                                        # Transformed distribution
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        sns.histplot(cleaner.df[col], kde=True, ax=ax)
                                        ax.set_title(f"Transformed Distribution of {col}\nSkewness: {after_skew:.2f}")
                                        st.pyplot(fig)
                            else:
                                st.info("No columns needed transformation based on the threshold.")
                        except Exception as e:
                            st.error(f"Transformation error: {str(e)}")
    
    # Handle High Cardinality
    with st.expander("Handle High Cardinality Variables", expanded=False):
        st.write("### Handle High Cardinality Categorical Variables")
        st.write("""
        Categorical variables with many unique values (high cardinality) can cause problems in analysis and modeling.
        This tool provides methods to reduce cardinality while preserving information.
        """)
        
        # Get categorical columns and their counts
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            st.warning("No categorical columns found in the dataset.")
        else:
            cardinality_data = {}
            for col in cat_cols:
                cardinality_data[col] = df[col].nunique()
            
            card_df = pd.DataFrame({
                'Column': list(cardinality_data.keys()),
                'Unique Values': list(cardinality_data.values())
            }).sort_values('Unique Values', ascending=False)
            
            st.write("#### Cardinality Analysis")
            safe_display_dataframe(card_df, cleaner)
            
            # Handle high cardinality
            st.write("#### Reduce Cardinality")
            
            max_categories = st.slider(
                "Maximum number of categories to keep",
                min_value=2,
                max_value=50,
                value=10
            )
            
            reduction_method = st.selectbox(
                "Reduction method",
                ["group_small", "target_encoding", "frequency_encoding", "binary_encoding"],
                help="Group Small: merge less frequent categories into 'Other'"
            )
            
            target_col = None
            if reduction_method == "target_encoding":
                target_options = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_options:
                    target_col = st.selectbox(
                        "Select target column for encoding (required for target encoding)",
                        target_options
                    )
                else:
                    st.warning("Target encoding requires a numeric target column.")
            
            cols_to_reduce = st.multiselect(
                "Select columns to reduce cardinality (leave empty to use all above max categories)",
                cat_cols
            )
            
            if st.button("Apply Cardinality Reduction"):
                with st.spinner("Reducing cardinality..."):
                    # If specific columns selected, only reduce those
                    if cols_to_reduce:
                        # Only process columns with high cardinality
                        high_card_selected = [col for col in cols_to_reduce if cardinality_data[col] > max_categories]
                        
                        if not high_card_selected:
                            st.info("None of the selected columns exceed the maximum category threshold.")
                        else:
                            # Apply reduction
                            modified_cols = {}
                            
                            for col in high_card_selected:
                                original_nunique = df[col].nunique()
                                
                                if reduction_method == "group_small":
                                    # Group less frequent categories into 'Other'
                                    value_counts = df[col].value_counts()
                                    top_cats = value_counts.nlargest(max_categories).index
                                    df[col] = np.where(df[col].isin(top_cats), df[col], 'Other')
                                elif reduction_method == "frequency_encoding":
                                    # Replace categories with their frequency
                                    freq_map = df[col].value_counts(normalize=True).to_dict()
                                    df[col] = df[col].map(freq_map)
                                elif reduction_method == "binary_encoding":
                                    # Create binary encoded variables
                                    from category_encoders import BinaryEncoder
                                    encoder = BinaryEncoder(cols=[col])
                                    encoded = encoder.fit_transform(df[col])
                                    
                                    # Add encoded columns to dataframe
                                    for enc_col in encoded.columns:
                                        new_col_name = f"{col}_bin_{enc_col}"
                                        df[new_col_name] = encoded[enc_col]
                                    
                                    # Optionally drop original column
                                    # df.drop(col, axis=1, inplace=True)
                                elif reduction_method == "target_encoding" and target_col:
                                    try:
                                        from category_encoders import TargetEncoder
                                        encoder = TargetEncoder(cols=[col])
                                        df[f"{col}_encoded"] = encoder.fit_transform(df[col], df[target_col])
                                    except Exception as e:
                                        st.error(f"Target encoding failed: {str(e)}")
                                        continue
                                
                                new_nunique = df[col].nunique()
                                modified_cols[col] = (original_nunique, new_nunique)
                            
                            # Update session state
                            cleaner.df = df
                            st.session_state.data = df
                            
                            # Show results
                            st.success(f"Cardinality reduction applied to {len(modified_cols)} columns.")
                            
                            if modified_cols:
                                results_df = pd.DataFrame({
                                    'Column': list(modified_cols.keys()),
                                    'Original Categories': [v[0] for v in modified_cols.values()],
                                    'New Categories': [v[1] for v in modified_cols.values()]
                                })
                                
                                st.write("#### Cardinality Reduction Results")
                                safe_display_dataframe(results_df, cleaner)
                    else:
                        # Apply to all high cardinality columns
                        try:
                            high_card_cols = cleaner.handle_high_cardinality(max_categories, reduction_method, target_col)
                            
                            # Update session state
                            st.session_state.data = cleaner.df
                            
                            if high_card_cols:
                                st.success(f"Successfully reduced cardinality in {len(high_card_cols)} columns.")
                                
                                results_df = pd.DataFrame({
                                    'Column': [col for col, _ in high_card_cols],
                                    'Original Categories': [count for _, count in high_card_cols],
                                    'New Categories': [cleaner.df[col].nunique() for col, _ in high_card_cols]
                                })
                                
                                st.write("#### Cardinality Reduction Results")
                                safe_display_dataframe(results_df, cleaner)
                            else:
                                st.info("No columns needed cardinality reduction based on the threshold.")
                        except Exception as e:
                            st.error(f"Cardinality reduction error: {str(e)}")
    
    # Advanced Imputation
    with st.expander("Advanced Missing Value Imputation", expanded=False):
        st.write("### Advanced Missing Value Imputation")
        st.write("""
        Standard imputation methods like mean or median don't consider relationships between variables.
        Advanced methods like KNN imputation use information from similar records to fill missing values more intelligently.
        """)
        
        # Get columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.info("No missing values found in the dataset.")
        else:
            # Display missing value statistics
            missing_stats = pd.DataFrame({
                'Column': missing_cols,
                'Missing Count': [df[col].isnull().sum() for col in missing_cols],
                'Missing Percentage': [(df[col].isnull().sum() / len(df) * 100).round(2) for col in missing_cols]
            }).sort_values('Missing Count', ascending=False)
            
            st.write("#### Missing Value Statistics")
            safe_display_dataframe(missing_stats, cleaner)
            
            # Imputation options
            st.write("#### Advanced Imputation Options")
            
            imputation_method = st.selectbox(
                "Imputation method",
                ["knn", "iterative", "regression", "mice"],
                help="KNN: K-Nearest Neighbors, MICE: Multiple Imputation by Chained Equations"
            )
            
            if imputation_method == "knn":
                n_neighbors = st.slider(
                    "Number of neighbors (k)",
                    min_value=1,
                    max_value=20,
                    value=5
                )
            
            cols_to_impute = st.multiselect(
                "Select columns to impute (leave empty to use all with missing values)",
                missing_cols
            )
            
            if st.button("Apply Advanced Imputation"):
                with st.spinner("Applying advanced imputation..."):
                    # If specific columns selected, only impute those
                    if cols_to_impute:
                        if imputation_method == "knn":
                            numeric_cols = [col for col in cols_to_impute if pd.api.types.is_numeric_dtype(df[col])]
                            
                            if not numeric_cols:
                                st.warning("KNN imputation requires numeric columns. None of the selected columns are numeric.")
                            else:
                                try:
                                    # Apply KNN imputation
                                    imputer = KNNImputer(n_neighbors=n_neighbors)
                                    df[numeric_cols] = pd.DataFrame(
                                        imputer.fit_transform(df[numeric_cols]),
                                        columns=numeric_cols,
                                        index=df.index
                                    )
                                    
                                    # Update session state
                                    cleaner.df = df
                                    st.session_state.data = df
                                    
                                    st.success(f"Successfully applied KNN imputation to {len(numeric_cols)} columns.")
                                except Exception as e:
                                    st.error(f"KNN imputation failed: {str(e)}")
                        elif imputation_method == "iterative":
                            try:
                                from sklearn.experimental import enable_iterative_imputer
                                from sklearn.impute import IterativeImputer
                                
                                numeric_cols = [col for col in cols_to_impute if pd.api.types.is_numeric_dtype(df[col])]
                                
                                if numeric_cols:
                                    # Apply iterative imputation
                                    imputer = IterativeImputer(max_iter=10, random_state=0)
                                    df[numeric_cols] = pd.DataFrame(
                                        imputer.fit_transform(df[numeric_cols]),
                                        columns=numeric_cols,
                                        index=df.index
                                    )
                                    
                                    # Update session state
                                    cleaner.df = df
                                    st.session_state.data = df
                                    
                                    st.success(f"Successfully applied iterative imputation to {len(numeric_cols)} columns.")
                                else:
                                    st.warning("Iterative imputation requires numeric columns. None of the selected columns are numeric.")
                            except Exception as e:
                                st.error(f"Iterative imputation failed: {str(e)}")
                        elif imputation_method == "regression":
                            st.warning("Regression imputation is not implemented yet. Please choose another method.")
                        elif imputation_method == "mice":
                            st.warning("MICE imputation is not implemented yet. Please choose another method.")
                    else:
                        # Apply to all columns with missing values
                        if imputation_method == "knn":
                            success = cleaner.advanced_imputation('knn', None, n_neighbors)
                            
                            # Update session state
                            st.session_state.data = cleaner.df
                            
                            if success:
                                st.success("Successfully applied KNN imputation to all applicable columns.")
                            else:
                                st.warning("KNN imputation failed or no suitable columns found.")
                        else:
                            st.warning(f"{imputation_method.upper()} imputation is only available when specific columns are selected.")
    
    # Outlier Treatment
    with st.expander("Outlier Treatment", expanded=False):
        st.write("### Outlier Treatment")
        st.write("""
        Outliers can significantly impact statistical analyses and models. This tool helps identify and treat outliers using
        various methods like capping, transformation, or removal.
        """)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for outlier treatment.")
        else:
            # Select a column to visualize
            selected_col = st.selectbox("Select column to visualize", numeric_cols, key="outlier_viz_col")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot to visualize outliers
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(y=df[selected_col], ax=ax)
                ax.set_title(f"Box Plot of {selected_col}")
                st.pyplot(fig)
            
            with col2:
                # Calculate outlier statistics
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)][selected_col]
                
                st.write("#### Outlier Statistics")
                st.write(f"- Q1 (25th percentile): {Q1:.4f}")
                st.write(f"- Median: {df[selected_col].median():.4f}")
                st.write(f"- Q3 (75th percentile): {Q3:.4f}")
                st.write(f"- IQR (Interquartile Range): {IQR:.4f}")
                st.write(f"- Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
                st.write(f"- Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")
                st.write(f"- Number of outliers: {len(outliers)}")
                st.write(f"- Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")
            
            # Outlier treatment options
            st.write("#### Outlier Treatment Options")
            
            treatment_method = st.selectbox(
                "Treatment method",
                ["capping", "removal", "iqr_based", "z_score"]
            )
            
            cols_to_treat = st.multiselect(
                "Select columns for outlier treatment",
                numeric_cols
            )
            
            if treatment_method == "capping":
                cap_method = st.radio(
                    "Capping method",
                    ["IQR-based", "Percentile-based"]
                )
                
                if cap_method == "Percentile-based":
                    lower_percentile = st.slider("Lower percentile", 0.0, 10.0, 1.0, 0.1)
                    upper_percentile = st.slider("Upper percentile", 90.0, 100.0, 99.0, 0.1)
            
            if treatment_method == "z_score":
                z_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
            
            if st.button("Apply Outlier Treatment"):
                if not cols_to_treat:
                    st.warning("Please select at least one column for outlier treatment.")
                else:
                    with st.spinner("Treating outliers..."):
                        # Store original values for comparison
                        original_stats = {}
                        for col in cols_to_treat:
                            original_stats[col] = {
                                'min': df[col].min(),
                                'max': df[col].max(),
                                'mean': df[col].mean(),
                                'std': df[col].std()
                            }
                        
                        # Apply treatment
                        for col in cols_to_treat:
                            if treatment_method == "capping":
                                if cap_method == "IQR-based":
                                    Q1 = df[col].quantile(0.25)
                                    Q3 = df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower = Q1 - 1.5 * IQR
                                    upper = Q3 + 1.5 * IQR
                                else:  # Percentile-based
                                    lower = df[col].quantile(lower_percentile / 100)
                                    upper = df[col].quantile(upper_percentile / 100)
                                
                                # Apply capping
                                df[col] = df[col].clip(lower=lower, upper=upper)
                            
                            elif treatment_method == "removal":
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - 1.5 * IQR
                                upper = Q3 + 1.5 * IQR
                                
                                # Mark outliers
                                outlier_mask = (df[col] < lower) | (df[col] > upper)
                                
                                # Store count of outliers
                                outlier_count = outlier_mask.sum()
                                
                                # Remove outliers
                                df = df[~outlier_mask]
                            
                            elif treatment_method == "iqr_based":
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - 1.5 * IQR
                                upper = Q3 + 1.5 * IQR
                                
                                # Replace outliers with bounds
                                df[col] = np.where(df[col] < lower, lower, df[col])
                                df[col] = np.where(df[col] > upper, upper, df[col])
                            
                            elif treatment_method == "z_score":
                                # Calculate z-scores
                                z_scores = (df[col] - df[col].mean()) / df[col].std()
                                
                                # Replace outliers (|z| > threshold) with NaN
                                df[col] = df[col].mask(abs(z_scores) > z_threshold, np.nan)
                                
                                # Then fill NaN with median
                                df[col] = df[col].fillna(df[col].median())
                        
                        # Update session state
                        cleaner.df = df
                        st.session_state.data = df
                        
                        # Show results
                        st.success(f"Successfully treated outliers in {len(cols_to_treat)} columns.")
                        
                        # Calculate new statistics
                        new_stats = {}
                        for col in cols_to_treat:
                            new_stats[col] = {
                                'min': df[col].min(),
                                'max': df[col].max(),
                                'mean': df[col].mean(),
                                'std': df[col].std()
                            }
                        
                        # Show before/after comparison
                        st.write("#### Before/After Statistics")
                        
                        for col in cols_to_treat:
                            st.write(f"**{col}**")
                            
                            comparison = pd.DataFrame({
                                'Statistic': ['Min', 'Max', 'Mean', 'Std Dev'],
                                'Before': [
                                    original_stats[col]['min'],
                                    original_stats[col]['max'],
                                    original_stats[col]['mean'],
                                    original_stats[col]['std']
                                ],
                                'After': [
                                    new_stats[col]['min'],
                                    new_stats[col]['max'],
                                    new_stats[col]['mean'],
                                    new_stats[col]['std']
                                ]
                            })
                            
                            safe_display_dataframe(comparison, cleaner)
                            
                            # Plot before/after for first treated column
                            if col == cols_to_treat[0]:
                                # Plot updated boxplot
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.boxplot(y=df[col], ax=ax)
                                ax.set_title(f"Box Plot of {col} After Treatment")
                                st.pyplot(fig) 