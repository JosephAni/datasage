import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from category_encoders import TargetEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import re
import warnings
import time
import math

# Add plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Install with 'pip install plotly' for interactive plots.")

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
        # If cleaner is not provided, create a temporary one
        if cleaner is None:
            from copy import deepcopy
            # Create a DataCleaner with a copy of the dataframe
            temp_cleaner = DataCleaner(deepcopy(df))
            # Make the dataframe Arrow-compatible
            safe_df = temp_cleaner.ensure_arrow_compatible(df)
        else:
            # Use the provided cleaner
            safe_df = cleaner.ensure_arrow_compatible(df)
            
        # Display the safe dataframe
        return st.dataframe(safe_df, **kwargs)
    except Exception as e:
        # Fallback to displaying as strings if all else fails
        st.warning(f"Error preparing dataframe for display: {str(e)}. Showing text representation.")
        return st.dataframe(df.astype(str), **kwargs)

class DataCleaner:
    """
    An AI agent for cleaning, preprocessing, and analyzing data using pandas and scikit-learn.
    """

    def __init__(self, dataframe):
        """
        Initializes the DataCleaner with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to clean.
        """
        self.original_df = dataframe.copy()  # Keep original copy
        self.df = dataframe.copy()  # Working copy
        self.label_encoders = {}  # Store label encoders for categorical variables
        self.transformers = {}  # Store transformers for numeric variables

    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handles missing values in the DataFrame using sklearn's SimpleImputer.

        Args:
            strategy (str, optional): Strategy for imputation. Options: 'mean', 'median', 'mode', 'drop'. Defaults to 'mean'.
            columns (list, optional): List of columns to apply the strategy to.
        """
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
                else:
                    st.warning(f"Strategy '{strategy}' not applicable for column '{col}'. Skipping.")

    def detect_outliers(self, columns=None, contamination=0.1):
        """
        Detects outliers using Isolation Forest algorithm.

        Args:
            columns (list, optional): List of numeric columns to check for outliers.
            contamination (float): The proportion of outliers in the data set.
        """
        if columns is None:
            # Only select columns that are actually numeric (excluding string-like numbers)
            numeric_cols = []
            for col in self.df.select_dtypes(include=[np.number]).columns:
                try:
                    # Try converting to float to verify it's actually numeric
                    self.df[col].astype(float)
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    st.warning(f"Column '{col}' contains non-numeric values and will be excluded from outlier detection.")
            columns = numeric_cols

        if len(columns) == 0:
            st.warning("No numeric columns available for outlier detection.")
            return None

        # Create a copy of selected columns
        X = self.df[columns].copy()
        
        # Handle missing values before outlier detection
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Fit isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X)
        
        # Mark outliers in the DataFrame
        self.df['is_outlier'] = outliers == -1
        n_outliers = sum(outliers == -1)
        
        st.write(f"### Outlier Detection Results")
        st.write(f"Found {n_outliers} potential outliers out of {len(self.df)} rows.")
        st.write(f"Analyzed columns: {', '.join(columns)}")
        
        return self.df[self.df['is_outlier']]

    def encode_categorical(self, columns=None):
        """
        Encode categorical variables using LabelEncoder.

        Args:
            columns (list, optional): List of columns to encode. If None, encodes all object columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns

        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                non_null_mask = self.df[col].notna()
                self.df.loc[non_null_mask, col] = le.fit_transform(self.df.loc[non_null_mask, col])
                self.label_encoders[col] = le
                st.info(f"Encoded categorical column: {col}")

    def scale_features(self, columns, scaler_type='standard'):
        """
        Scale numeric features in the dataset.
        
        Args:
            columns (list): List of columns to scale
            scaler_type (str): Type of scaling ('standard', 'minmax', 'robust')
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Verify columns exist
        for col in columns:
            if col not in self.df.columns:
                st.error(f"Column '{col}' not found in dataset.")
                return
        
        # Check for and warn about ID columns
        id_columns = []
        for col in columns:
            is_id, reason = self.is_likely_id_column(col)
            if is_id:
                id_columns.append((col, reason))
                
        if id_columns:
            st.warning("⚠️ The following columns appear to be IDs or codes that generally shouldn't be scaled:")
            for col, reason in id_columns:
                st.markdown(f"- **{col}**: {reason}")
            proceed = st.checkbox("Proceed with scaling anyway? (Not recommended)")
            if not proceed:
                return
        
        # Preview data before scaling
        st.write("##### Preview of selected columns:")
        preview_df = self.df[columns].head(5)
        st.dataframe(preview_df)
        
        # Attempt to convert to numeric, showing errors for non-convertible columns
        numeric_conversion_failed = False
        non_numeric_examples = {}
        
        for col in columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            # Try to convert with error reporting
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    numeric_vals = pd.to_numeric(self.df[col], errors='coerce')
                    
                # Check if conversion created NaNs and report examples
                if numeric_vals.isna().sum() > 0:
                    # Find example of non-convertible values
                    mask = self.df[col].notna() & numeric_vals.isna()
                    if mask.any():
                        examples = self.df.loc[mask, col].unique()[:3]  # Get up to 3 examples
                        non_numeric_examples[col] = examples
                        numeric_conversion_failed = True
            except Exception as e:
                st.error(f"Error converting {col} to numeric: {str(e)}")
                return
        
        if numeric_conversion_failed:
            st.error("⚠️ Some columns contain values that cannot be converted to numeric format:")
            for col, examples in non_numeric_examples.items():
                example_str = ', '.join([f"'{ex}'" for ex in examples])
                st.markdown(f"- **{col}**: Contains non-numeric values like {example_str}")
            
            options = st.radio(
                "How would you like to proceed?",
                ["Cancel operation", 
                 "Remove non-numeric characters and try again",
                 "Replace non-numeric values with NaN and continue"]
            )
            
            if options == "Cancel operation":
                return
            elif options == "Remove non-numeric characters and try again":
                # Remove non-numeric chars for the problematic columns
                for col in non_numeric_examples.keys():
                    # Keep only digits, decimal point, and minus sign
                    self.df[col] = self.df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    # Convert to numeric
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                st.success("Non-numeric characters removed, proceeding with scaling.")
            else:  # Replace with NaN
                for col in non_numeric_examples.keys():
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                st.info("Non-numeric values replaced with NaN, proceeding with scaling.")
        
        # Proceed with scaling
        try:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                st.error(f"Unknown scaler type: {scaler_type}")
                return
            
            # Convert all columns to numeric to be safe
            numeric_df = self.df[columns].apply(pd.to_numeric, errors='coerce')
            
            # Scale and update the dataframe
            scaled_values = scaler.fit_transform(numeric_df)
            scaled_df = pd.DataFrame(scaled_values, columns=columns, index=self.df.index)
            
            # Replace the original columns with scaled values
            for col in columns:
                self.df[col] = scaled_df[col]
            
            # Show results
            st.success(f"Successfully scaled {len(columns)} columns using {scaler_type} scaling")
            
            # Show before/after statistics
            st.write("##### Statistics after scaling:")
            st.dataframe(self.df[columns].describe())
            
        except Exception as e:
            st.error(f"Error during scaling: {str(e)}")
            st.info("This could be due to non-numeric values or missing data in the selected columns.")
            return

    def train_model(self, target_column, problem_type='classification', test_size=0.2, model_type='random_forest'):
        """
        Train a ML model and return metrics.
        
        Args:
            target_column (str): Target column for prediction
            problem_type (str): Either 'classification' or 'regression'
            test_size (float): Proportion of data to use for testing
            model_type (str): Type of model to train (random_forest, knn, logistic_regression, etc.)
            
        Returns:
            dict: Dictionary of model metrics
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, classification_report
        from sklearn.model_selection import train_test_split
        
        # Validate target column existence
        if target_column not in self.df.columns:
            st.error(f"Target column '{target_column}' not found in the dataset.")
            return None
            
        # Check if target column is an ID column and warn user
        is_id, reason = self.is_likely_id_column(target_column)
        if is_id:
            st.warning(f"⚠️ The selected target column '{target_column}' appears to be an ID or code column: {reason}")
            st.warning("ID columns typically don't have meaningful patterns for ML models to learn.")
            proceed = st.checkbox("Proceed with modeling anyway? (Not recommended)")
            if not proceed:
                return None
            
        # Get feature columns (all except target)
        features = [col for col in self.df.columns if col != target_column]
        
        # Check for ID columns in features and warn
        id_features = []
        for col in features:
            is_id, reason = self.is_likely_id_column(col)
            if is_id:
                id_features.append((col, reason))
                
        if id_features:
            st.warning("⚠️ The following feature columns appear to be IDs or codes:")
            for col, reason in id_features:
                st.markdown(f"- **{col}**: {reason}")
            st.warning("ID columns can lead to overfitting and poor model generalization.")
            
            exclude_option = st.radio(
                "How would you like to handle ID columns?",
                ["Exclude ID columns from model", "Keep all columns"]
            )
            
            if exclude_option == "Exclude ID columns from model":
                features = [col for col in features if col not in [id_col for id_col, _ in id_features]]
                st.info(f"Excluded {len(id_features)} ID columns from the model.")
        
        # Preview target distribution
        st.write("##### Target Column Preview:")
        
        if pd.api.types.is_numeric_dtype(self.df[target_column]):
            # For numeric targets, show histogram and statistics
            st.write(f"Target statistics: {self.df[target_column].describe().to_dict()}")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(self.df[target_column].dropna(), kde=True, ax=ax)
            plt.title(f"Distribution of {target_column}")
            st.pyplot(fig)
            plt.close()
        else:
            # For categorical targets, show value counts and bar chart
            value_counts = self.df[target_column].value_counts()
            unique_count = len(value_counts)
            
            st.write(f"Target has {unique_count} unique values")
            
            # Show warning if too many unique values for classification
            if problem_type == 'classification' and unique_count > 50:
                st.warning(f"⚠️ The target has {unique_count} unique values, which is very high for a classification problem.")
                st.warning("Consider using regression instead, or transform the target into fewer categories.")
                proceed = st.checkbox("Proceed with classification anyway?")
                if not proceed:
                    return None
            
            # Display top categories
            display_count = min(10, unique_count)
            st.write(f"Top {display_count} most frequent values:")
            st.dataframe(value_counts.head(display_count).reset_index().rename(
                columns={'index': target_column, target_column: 'Count'}))
            
            # Simple bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            value_counts.head(display_count).plot.bar(ax=ax)
            plt.title(f"Top values of {target_column}")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Check for non-numeric features and warn/convert
        non_numeric_cols = []
        for col in features:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                non_numeric_cols.append(col)
                
        if non_numeric_cols:
            st.warning(f"⚠️ The following features are non-numeric and will be excluded: {', '.join(non_numeric_cols)}")
            st.info("Consider encoding categorical features before training the model.")
            features = [col for col in features if col not in non_numeric_cols]
            
            if not features:
                st.error("No numeric features available for modeling after exclusions.")
                return None
                
        # Prepare data for modeling
        X = self.df[features].select_dtypes(include=[np.number])
        y = self.df[target_column]
        
        # Check for missing values and handle
        missing_in_X = X.isnull().sum().sum()
        missing_in_y = y.isnull().sum()
        
        if missing_in_X > 0 or missing_in_y > 0:
            st.warning(f"⚠️ Dataset contains missing values: {missing_in_X} in features, {missing_in_y} in target")
            st.info("Rows with missing values will be excluded from modeling.")
            
            # Remove rows with missing values
            valid_indices = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            
            st.write(f"Modeling with {len(X)} complete rows (removed {len(self.df) - len(X)} rows with missing values)")
            
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model based on problem type
            if problem_type == 'classification':
                try:
                    # Initialize the selected model type
                    if model_type == 'random_forest':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        has_feature_importance = True
                    elif model_type == 'decision_tree':
                        model = DecisionTreeClassifier(random_state=42)
                        has_feature_importance = True
                    elif model_type == 'knn':
                        model = KNeighborsClassifier(n_neighbors=5)
                        has_feature_importance = False
                    elif model_type == 'logistic_regression':
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        has_feature_importance = True
                    elif model_type == 'adaboost':
                        model = AdaBoostClassifier(random_state=42)
                        has_feature_importance = True
                    elif model_type == 'svm':
                        model = SVC(probability=True, random_state=42)
                        has_feature_importance = False
                    elif model_type == 'lda':
                        model = LinearDiscriminantAnalysis()
                        has_feature_importance = True
                    elif model_type == 'naive_bayes':
                        model = GaussianNB()
                        has_feature_importance = False
                    else:
                        st.error(f"Unknown model type: {model_type}")
                        return None
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Create classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Get feature importance if available
                    if has_feature_importance:
                        if model_type == 'logistic_regression':
                            # For logistic regression, use coefficients
                            if hasattr(model, 'coef_'):
                                if len(model.classes_) == 2:  # Binary classification
                                    importance = dict(zip(X.columns, np.abs(model.coef_[0])))
                                else:  # Multi-class
                                    # Average the absolute coefficients across all classes
                                    importance = dict(zip(X.columns, np.mean(np.abs(model.coef_), axis=0)))
                            else:
                                importance = None
                        elif model_type == 'lda':
                            # For LDA, use the absolute values of coefficients of linear discriminants
                            if hasattr(model, 'coef_'):
                                importance = dict(zip(X.columns, np.mean(np.abs(model.coef_), axis=0)))
                            else:
                                importance = None
                        else:
                            # For tree-based models use feature_importances_
                            importance = dict(zip(X.columns, model.feature_importances_))
                    else:
                        importance = None
                    
                    # Prepare results
                    results = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'classification_report': report
                    }
                    
                    if importance:
                        results['feature_importance'] = importance
                    
                    return results
                    
                except Exception as e:
                    if "Unknown label type" in str(e):
                        st.error("Error: Target values are not suitable for classification. Try regression instead.")
                    else:
                        st.error(f"Classification model error: {str(e)}")
                    return None
                    
            elif problem_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Feature importance
                importance = dict(zip(X.columns, model.feature_importances_))
                
                return {
                    'r2_score': r2,
                    'mean_absolute_error': mae,
                    'feature_importance': importance
                }
                
            else:
                st.error(f"Unknown problem type: {problem_type}")
                return None
                
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("This could be due to incompatible data types or other issues.")
            return None

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            st.info(f"Removed {removed_rows} duplicate rows.")

    def convert_datatypes(self, column_types):
        """
        Converts columns to specified datatypes.

        Args:
            column_types (dict): A dictionary where keys are column names and values are datatypes.
        """
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
            except Exception as e:
                st.warning(f"Error converting column '{column}' to type '{dtype}': {str(e)}")

    def standardize_column_names(self):
        """
        Standardizes column names by converting to lowercase and replacing spaces with underscores.
        """
        self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
        st.info("Column names have been standardized.")

    def get_cleaned_data(self):
        """
        Returns the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        return self.df.copy()
        
    def ensure_arrow_compatible(self, df):
        """
        Ensures DataFrame is compatible with PyArrow for Streamlit display.
        Fixes common type conversion issues.
        
        Args:
            df (pd.DataFrame): DataFrame to make Arrow-compatible
            
        Returns:
            pd.DataFrame: Arrow-compatible DataFrame
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check each column for potential issues
        for col in result_df.columns:
            # Skip if column is already a numeric or date type
            if pd.api.types.is_numeric_dtype(result_df[col]) or pd.api.types.is_datetime64_dtype(result_df[col]):
                continue
                
            # Check if column contains mixed types (alphanumeric values)
            if pd.api.types.is_object_dtype(result_df[col]):
                # Try to convert to numeric, but keep as string if it contains non-numeric values
                try:
                    numeric_col = pd.to_numeric(result_df[col], errors='coerce')
                    # If we have any non-numeric values (NaN after conversion), keep as string
                    if numeric_col.isna().any() and not result_df[col].isna().any():
                        # Ensure the column is explicitly string type to avoid PyArrow conversion issues
                        result_df[col] = result_df[col].astype(str)
                    else:
                        # If conversion to numeric works for all values, use that
                        result_df[col] = numeric_col
                except:
                    # Ensure string type for problematic columns
                    result_df[col] = result_df[col].astype(str)
        
        return result_df

    def reset(self):
        """
        Reset the DataFrame to its original state.
        """
        self.df = self.original_df.copy()
        st.info("Data has been reset to original state.")

    def plot_distribution(self, column):
        """
        Plot the distribution of a column using a histogram or bar chart.

        Args:
            column (str): The column to plot.
        """
        if pd.api.types.is_numeric_dtype(self.df[column]):
            # For numeric columns, create a histogram
            fig = plt.figure(figsize=(10, 6))
            plt.hist(self.df[column].dropna(), bins=30, edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            st.pyplot(fig)
            plt.close()

            # Add descriptive statistics
            st.write("#### Descriptive Statistics:")
            stats = self.df[column].describe()
            st.write(stats)
        else:
            # For categorical columns, create a bar chart
            value_counts = self.df[column].value_counts()
            fig = plt.figure(figsize=(10, 6))
            plt.bar(value_counts.index.astype(str), value_counts.values)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(column)
            plt.ylabel('Count')
            st.pyplot(fig)
            plt.close()

            # Add frequency table
            st.write("#### Frequency Table:")
            freq_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(self.df) * 100).round(2)
            })
            st.write(freq_df)

    def plot_pie_chart(self, column):
        """
        Create a pie chart for categorical columns.

        Args:
            column (str): The column to plot.
        """
        # Check if column exists
        if column not in self.df.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return
            
        # Get number of unique values
        unique_count = self.df[column].nunique()
        
        # Validate if column is suitable for a pie chart
        if unique_count > 15:
            st.error(f"⚠️ Column '{column}' has {unique_count} unique values, which is too many for a meaningful pie chart.")
            st.info("Pie charts work best with 5-10 categories. Consider using a bar chart instead, or selecting a different column.")
            return
            
        # For numeric columns with many unique values, suggest not using pie chart
        if pd.api.types.is_numeric_dtype(self.df[column]) and unique_count > 10:
            st.warning(f"Column '{column}' appears to be numeric with {unique_count} unique values. A pie chart may not be the best visualization.")
            cont = st.checkbox("Continue anyway?")
            if not cont:
                return
                
        # Get value counts and prepare data for the pie chart
        value_counts = self.df[column].value_counts()
        
        # If there are more than 8 categories, group the smallest ones as "Other"
        if len(value_counts) > 8:
            top_n = 7  # Show top 7 categories + "Other"
            top_values = value_counts.nlargest(top_n)
            other_sum = value_counts[top_n:].sum()
            
            # Create a new series with top categories + "Other"
            plot_data = pd.Series(list(top_values) + [other_sum], 
                                index=list(top_values.index) + ['Other'])
            
            st.info(f"Showing top {top_n} categories. {len(value_counts) - top_n} smaller categories grouped as 'Other'.")
        else:
            plot_data = value_counts
            
        # Create the pie chart
        if PLOTLY_AVAILABLE:
            # Create an interactive plotly pie chart
            fig = px.pie(
                values=plot_data.values,
                names=plot_data.index.astype(str),
                title=f'Pie Chart of {column}',
                hole=0.3,  # Create a donut chart for better readability
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
            st.plotly_chart(fig)
        else:
            # Matplotlib fallback
            fig = plt.figure(figsize=(10, 8))
            plt.pie(plot_data.values, labels=plot_data.index.astype(str), 
                   autopct='%1.1f%%', startangle=90, 
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            plt.title(f'Pie Chart of {column}')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)
            plt.close()

    def plot_correlation_heatmap(self):
        """
        Create a correlation heatmap for numeric columns.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Not enough numeric columns for correlation analysis")

    def plot_box_plot(self, column):
        """
        Create a box plot for numeric columns.

        Args:
            column (str): The column to plot.
        """
        if pd.api.types.is_numeric_dtype(self.df[column]):
            fig = plt.figure(figsize=(10, 6))
            plt.boxplot(self.df[column].dropna())
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning(f"Box plot not suitable for non-numeric column '{column}'")

    def interpret_data(self):
        """
        Provides comprehensive interpretation of the dataset including statistical analysis,
        patterns, and insights.
        """
        st.write("## 📊 Data Interpretation")

        # Basic Dataset Information
        st.write("### 📌 Dataset Overview")
        total_rows, total_cols = self.df.shape
        st.write(f"- Total Records: {total_rows:,}")
        st.write(f"- Total Features: {total_cols}")
        
        # Data Types Analysis
        st.write("### 🔍 Data Type Distribution")
        dtype_counts = self.df.dtypes.value_counts()
        fig = plt.figure(figsize=(8, 4))
        plt.bar(dtype_counts.index.astype(str), dtype_counts.values)
        plt.title('Distribution of Data Types')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

        # Missing Values Analysis
        st.write("### ❓ Missing Values Analysis")
        missing_vals = self.df.isnull().sum()
        if missing_vals.any():
            missing_df = pd.DataFrame({
                'Column': missing_vals.index,
                'Missing Values': missing_vals.values,
                'Percentage': (missing_vals.values / len(self.df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
            st.write(missing_df)
        else:
            st.write("No missing values found in the dataset.")

        # Numeric Columns Analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("### 📈 Numeric Features Analysis")
            stats_df = self.df[numeric_cols].describe()
            st.write(stats_df)

            # Skewness Analysis
            st.write("#### Skewness Analysis")
            skewness = self.df[numeric_cols].skew()
            skew_df = pd.DataFrame({
                'Feature': skewness.index,
                'Skewness': skewness.values
            })
            st.write("Features with significant skewness (|skew| > 1):")
            st.write(skew_df[abs(skew_df['Skewness']) > 1])

        # Categorical Columns Analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("### 📊 Categorical Features Analysis")
            for col in categorical_cols:
                st.write(f"#### {col}")
                value_counts = self.df[col].value_counts()
                unique_vals = len(value_counts)
                st.write(f"- Unique Values: {unique_vals}")
                
                if unique_vals <= 10:  # Show distribution for categorical variables with few unique values
                    fig = plt.figure(figsize=(10, 4))
                    plt.bar(value_counts.index.astype(str), value_counts.values)
                    plt.title(f'Distribution of {col}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.write("Top 10 most frequent values:")
                    st.write(value_counts.head(10))

        # Correlation Analysis for Numeric Features
        if len(numeric_cols) > 1:
            st.write("### 🔗 Correlation Analysis")
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for strong correlation
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if strong_corr:
                st.write("#### Strong Feature Correlations (|correlation| > 0.7):")
                st.write(pd.DataFrame(strong_corr))

        # Key Insights
        st.write("### 🎯 Key Insights")
        insights = []
        
        # Missing values insight
        if missing_vals.any():
            insights.append(f"- Dataset contains missing values in {sum(missing_vals > 0)} columns")
        
        # Skewness insight
        if len(numeric_cols) > 0:
            skewed_features = sum(abs(self.df[numeric_cols].skew()) > 1)
            if skewed_features > 0:
                insights.append(f"- {skewed_features} numeric features show significant skewness")
        
        # Correlation insight
        if len(numeric_cols) > 1:
            if strong_corr:
                insights.append(f"- Found {len(strong_corr)} pairs of strongly correlated features")
        
        # Categorical insight
        if len(categorical_cols) > 0:
            high_cardinality_cols = sum(self.df[categorical_cols].nunique() > 10)
            if high_cardinality_cols > 0:
                insights.append(f"- {high_cardinality_cols} categorical features have high cardinality (>10 unique values)")
        
        # Add insights to the app
        for insight in insights:
            st.write(insight)

        # Recommendations
        st.write("### 💡 Recommendations")
        recommendations = []
        
        # Missing values recommendations
        if missing_vals.any():
            recommendations.append("- Consider handling missing values using appropriate imputation methods")
        
        # Skewness recommendations
        if len(numeric_cols) > 0 and sum(abs(self.df[numeric_cols].skew()) > 1) > 0:
            recommendations.append("- Consider applying transformations (e.g., log, box-cox) to highly skewed numeric features")
        
        # Correlation recommendations
        if strong_corr:
            recommendations.append("- Consider feature selection or dimensionality reduction for highly correlated features")
        
        # Categorical recommendations
        if len(categorical_cols) > 0 and high_cardinality_cols > 0:
            recommendations.append("- Consider grouping or encoding high-cardinality categorical features")
        
        # Add recommendations to the app
        for recommendation in recommendations:
            st.write(recommendation)

    def advanced_imputation(self, strategy='knn', columns=None, n_neighbors=5):
        """
        Performs advanced imputation using various methods.

        Args:
            strategy (str): Imputation strategy ('knn', 'iterative', 'multivariate')
            columns (list): List of columns to impute
            n_neighbors (int): Number of neighbors for KNN imputation
        """
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()].tolist()

        numeric_cols = self.df[columns].select_dtypes(include=[np.number]).columns
        categorical_cols = self.df[columns].select_dtypes(exclude=[np.number]).columns

        if strategy == 'knn' and len(numeric_cols) > 0:
            # KNN imputation for numeric columns
            imputer = KNNImputer(n_neighbors=n_neighbors)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            st.info(f"Applied KNN imputation to numeric columns: {', '.join(numeric_cols)}")

        # Handle categorical columns separately
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
            st.info(f"Applied mode imputation to categorical columns: {', '.join(categorical_cols)}")

    def transform_skewed_features(self, threshold=0.5, method='yeo-johnson'):
        """
        Transforms skewed numeric features using various methods.

        Args:
            threshold (float): Skewness threshold to determine which features to transform
            method (str): Transformation method ('yeo-johnson', 'box-cox', 'log')
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        skewed_features = {}

        for col in numeric_cols:
            skewness = stats.skew(self.df[col].dropna())
            if abs(skewness) > threshold:
                skewed_features[col] = skewness

        if not skewed_features:
            st.info("No features found with significant skewness.")
            return

        st.write("### Transforming Skewed Features")
        st.write("Features with significant skewness:")
        for col, skew in skewed_features.items():
            st.write(f"- {col}: {skew:.2f}")

        for col in skewed_features.keys():
            if method == 'yeo-johnson':
                transformer = PowerTransformer(method='yeo-johnson')
                self.df[col] = transformer.fit_transform(self.df[[col]])
                self.transformers[col] = transformer
            elif method == 'log':
                # Handle negative and zero values
                min_val = self.df[col].min()
                if min_val <= 0:
                    self.df[col] = np.log1p(self.df[col] - min_val + 1)
                else:
                    self.df[col] = np.log1p(self.df[col])

        st.success(f"Applied {method} transformation to {len(skewed_features)} features")

    def handle_high_cardinality(self, max_categories=10, method='target_encoding', target_column=None):
        """
        Handles high-cardinality categorical features using various methods.

        Args:
            max_categories (int): Maximum number of categories to keep
            method (str): Encoding method ('target_encoding', 'frequency', 'grouping')
            target_column (str): Target column for target encoding
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality_cols = []

        for col in categorical_cols:
            if self.df[col].nunique() > max_categories:
                high_cardinality_cols.append(col)

        if not high_cardinality_cols:
            st.info("No high-cardinality categorical features found.")
            return

        st.write("### Handling High-Cardinality Features")
        st.write(f"Found {len(high_cardinality_cols)} high-cardinality features:")
        for col in high_cardinality_cols:
            st.write(f"- {col}: {self.df[col].nunique()} unique values")

        for col in high_cardinality_cols:
            if method == 'target_encoding' and target_column:
                encoder = TargetEncoder()
                self.df[col] = encoder.fit_transform(self.df[col], self.df[target_column])
            elif method == 'frequency':
                value_counts = self.df[col].value_counts()
                top_categories = value_counts.nlargest(max_categories).index
                self.df[col] = self.df[col].apply(lambda x: x if x in top_categories else 'Other')
            elif method == 'grouping':
                value_counts = self.df[col].value_counts()
                top_categories = value_counts.nlargest(max_categories-1).index
                self.df[col] = self.df[col].apply(lambda x: x if x in top_categories else 'Other')

        st.success(f"Applied {method} to {len(high_cardinality_cols)} high-cardinality features")

    def analyze_time_series(self, date_column, value_column, frequency='D', use_interactive=True):
        """
        Performs time series analysis on the specified columns.

        Args:
            date_column (str): Name of the column containing dates
            value_column (str): Name of the column containing values to analyze
            frequency (str): Frequency of the time series ('D' for daily, 'M' for monthly, etc.)
            use_interactive (bool): Whether to use interactive plots
        """
        try:
            # Validate inputs
            if date_column not in self.df.columns:
                st.error(f"Date column '{date_column}' not found in the dataset.")
                return
            if value_column not in self.df.columns:
                st.error(f"Value column '{value_column}' not found in the dataset.")
                return

            # Create a copy of the data to work with
            ts_df = self.df[[date_column, value_column]].copy()

            # Convert date column to datetime with explicit error handling
            try:
                # First, try to detect the date format from the first few non-null values
                sample_dates = ts_df[date_column].dropna().head(5).astype(str).tolist()
                
                # Initialize format detection variables
                has_time = any(':' in str(date) for date in sample_dates)
                has_dash = any('-' in str(date) for date in sample_dates)
                has_slash = any('/' in str(date) for date in sample_dates)
                
                # Try parsing with different approaches
                try:
                    # First try: Parse as ISO format
                    ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='ISO8601', errors='raise')
                except (ValueError, TypeError):
                    try:
                        # Second try: If we have time components, use a flexible parser
                        if has_time:
                            ts_df[date_column] = pd.to_datetime(ts_df[date_column], infer_datetime_format=True, errors='raise')
                        # Third try: Use specific formats based on separators
                        elif has_dash:
                            ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='%Y-%m-%d', errors='raise')
                        elif has_slash:
                            # Try both MDY and YMD formats
                            try:
                                ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='%m/%d/%Y', errors='raise')
                            except ValueError:
                                try:
                                    ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='%Y/%m/%d', errors='raise')
                                except ValueError:
                                    ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='%m/%d/%y', errors='raise')
                        else:
                            # Final try: Use mixed format with dayfirst=False
                            ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='mixed', dayfirst=False, errors='raise')
                    except (ValueError, TypeError):
                        # If all specific formats fail, try the most flexible approach
                        ts_df[date_column] = pd.to_datetime(ts_df[date_column], format='mixed', errors='coerce')

                # Check for NaT values after conversion
                nat_count = ts_df[date_column].isna().sum()
                if nat_count > 0:
                    st.warning(f"Found {nat_count} invalid date values. These rows will be removed.")
                    # Show examples of problematic dates
                    problem_dates = ts_df[ts_df[date_column].isna()][date_column].head()
                    if not problem_dates.empty:
                        st.write("Examples of problematic date values:")
                        st.write(problem_dates)
                    ts_df = ts_df.dropna(subset=[date_column])

                if len(ts_df) == 0:
                    st.error("No valid dates remaining after cleaning. Please check your date format.")
                    return

                # Show the date range of the data
                st.info(f"Date range: from {ts_df[date_column].min()} to {ts_df[date_column].max()}")

            except Exception as e:
                st.error(f"Error converting dates: {str(e)}")
                st.info("Please ensure your date column contains valid date values. Common formats include: YYYY-MM-DD, MM/DD/YYYY, or MM/DD/YY")
                return

            # Validate value column is numeric
            if not pd.api.types.is_numeric_dtype(ts_df[value_column]):
                st.error(f"Value column '{value_column}' must be numeric.")
                return

            # Sort by date and set index
            ts_df = ts_df.sort_values(date_column)
            ts_df.set_index(date_column, inplace=True)

            # Check for sufficient data
            if len(ts_df) < 2:
                st.error("Insufficient data for time series analysis. Need at least 2 data points.")
                return

            st.write("### 📈 Time Series Analysis")

            # Basic Time Series Plot
            try:
                st.write("#### Time Series Plot")
                if PLOTLY_AVAILABLE and use_interactive:
                    fig = px.line(
                        ts_df.reset_index(),
                        x=date_column,
                        y=value_column,
                        title=f'{value_column} Over Time'
                    )
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title=value_column,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(ts_df.index, ts_df[value_column])
                    plt.title(f'{value_column} Over Time')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Error creating time series plot: {str(e)}")

            # Add data aggregation options
            st.write("#### Data Aggregation")
            agg_method = st.selectbox(
                "Select aggregation method:",
                ["mean", "sum", "min", "max", "count"],
                help="Choose how to aggregate data when resampling to the selected frequency"
            )

            # Resample data with error handling
            try:
                # Ensure the index is datetime type
                if not isinstance(ts_df.index, pd.DatetimeIndex):
                    ts_df.index = pd.to_datetime(ts_df.index)

                # Resample with the selected method
                resampled_df = getattr(ts_df.resample(frequency), agg_method)()

                if resampled_df.empty:
                    st.error(f"No data available after resampling with frequency '{frequency}'")
                    return

                # Show resampled data
                st.write(f"#### Resampled Data ({frequency} frequency, {agg_method} aggregation)")
                st.dataframe(resampled_df.head(10))

                # Visualize resampled data
                if PLOTLY_AVAILABLE and use_interactive:
                    fig = px.line(
                        resampled_df.reset_index(),
                        x=resampled_df.index.name,
                        y=value_column,
                        title=f'Resampled {value_column} ({frequency} frequency, {agg_method})'
                    )
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title=value_column,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in resampling data: {str(e)}")
                st.info("Please check your date format and frequency settings.")
                return

            # Time Series Decomposition
            try:
                st.write("#### Time Series Decomposition")
                
                # Count number of observations after resampling
                n_observations = len(resampled_df)
                st.info(f"Number of observations after {frequency} frequency resampling: {n_observations}")
                
                # Check if we have enough observations for decomposition
                min_period = 2  # Minimum period possible
                
                if n_observations < 4:  # Need at least 4 observations for decomposition
                    st.error(f"Insufficient data for decomposition. Need at least 4 observations, but got {n_observations}.")
                    st.info("Try using a different frequency with more data points (e.g., change from Monthly to Weekly).")
                    return
                
                # Calculate max period based on data, ensuring it's at least 3
                max_period = max(3, min(n_observations // 2, 24))  # Cap at 24 for reasonable seasonality
                
                # Safe calculation of suggested period
                if frequency == 'D':
                    suggested_period = min(7, max(2, min(n_observations // 4, max_period)))  # weekly
                elif frequency == 'W':
                    suggested_period = min(4, max(2, min(n_observations // 4, max_period)))  # monthly
                elif frequency == 'M':
                    suggested_period = min(12, max(2, min(n_observations // 4, max_period)))  # yearly
                elif frequency == 'Q':
                    suggested_period = min(4, max(2, min(n_observations // 4, max_period)))  # yearly
                else:
                    suggested_period = min(4, max(2, min(n_observations // 4, max_period)))
                
                # Ensure max_period is greater than min_period
                max_period = max(min_period + 1, max_period)
                suggested_period = min(max(min_period, suggested_period), max_period)
                
                # Let user adjust the period if needed
                period = st.slider(
                    "Select decomposition period (number of observations per cycle):",
                    min_value=min_period,
                    max_value=max_period,
                    value=suggested_period,
                    help="Adjust this value based on the expected seasonality in your data."
                )
                
                # Perform decomposition with more robust error handling
                try:
                    decomposition = seasonal_decompose(
                        resampled_df[value_column],
                        period=period,
                        extrapolate_trend='freq'
                    )
                    
                    # Plot decomposition components
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
                    
                    ax1.plot(decomposition.observed)
                    ax1.set_title('Observed')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    ax2.plot(decomposition.trend)
                    ax2.set_title('Trend')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    ax3.plot(decomposition.seasonal)
                    ax3.set_title('Seasonal')
                    ax3.tick_params(axis='x', rotation=45)
                    
                    ax4.plot(decomposition.resid)
                    ax4.set_title('Residual')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as decomp_error:
                    st.error(f"Error during decomposition: {str(decomp_error)}")
                    st.info("Try adjusting the period value or using a different frequency that better matches your data's seasonality pattern.")
                    return

            except Exception as e:
                st.error(f"Error in time series decomposition: {str(e)}")
                st.info("Try adjusting the frequency or period to match your data's seasonality pattern.")

        except Exception as e:
            st.error(f"Error in time series analysis: {str(e)}")
            st.info("Please check your data format and try again.")

    def perform_hypothesis_test(self, column1, column2=None, test_type='ttest'):
        """
        Performs statistical hypothesis tests on the data.
        
        Args:
            column1 (str): First column for testing
            column2 (str, optional): Second column for comparison tests
            test_type (str): Type of test to perform - 'ttest', 'anova', 'chi2', 'correlation'
            
        Returns:
            dict: Test results and interpretation
        """
        result = {}
        
        # Ensure columns exist
        if column1 not in self.df.columns:
            raise ValueError(f"Column '{column1}' not found in DataFrame")
        
        # Perform the appropriate test
        if test_type == 'ttest':
            if column2 and column2 in self.df.columns:
                # Two-sample t-test
                stat, pvalue = stats.ttest_ind(
                    self.df[column1].dropna(),
                    self.df[column2].dropna(),
                    equal_var=False  # Welch's t-test does not assume equal variance
                )
                result['test_name'] = "Independent Samples t-test (Welch's)"
                result['description'] = "Compares means of two independent samples"
                result['statistic'] = stat
                result['p_value'] = pvalue
                result['interpretation'] = "Significant difference between means" if pvalue < 0.05 else "No significant difference between means"
            else:
                # One-sample t-test against 0
                stat, pvalue = stats.ttest_1samp(self.df[column1].dropna(), 0)
                result['test_name'] = "One-sample t-test"
                result['description'] = "Tests if the sample mean differs from 0"
                result['statistic'] = stat
                result['p_value'] = pvalue
                result['interpretation'] = "Mean significantly different from 0" if pvalue < 0.05 else "Mean not significantly different from 0"
        
        elif test_type == 'anova':
            # For ANOVA, we need to group by a categorical variable
            if column2 and column2 in self.df.columns and pd.api.types.is_object_dtype(self.df[column2]):
                groups = []
                labels = []
                for group_name, group_data in self.df.groupby(column2)[column1]:
                    if not group_data.dropna().empty:
                        groups.append(group_data.dropna())
                        labels.append(group_name)
                
                if len(groups) > 1:
                    stat, pvalue = stats.f_oneway(*groups)
                    result['test_name'] = "One-way ANOVA"
                    result['description'] = f"Compares means of '{column1}' across categories in '{column2}'"
                    result['statistic'] = stat
                    result['p_value'] = pvalue
                    result['interpretation'] = "At least one group mean is significantly different" if pvalue < 0.05 else "No significant difference between group means"
                    result['groups'] = labels
                else:
                    raise ValueError("Need at least two groups with non-empty data for ANOVA")
            else:
                raise ValueError(f"For ANOVA, '{column2}' must be a categorical column")
        
        elif test_type == 'chi2':
            # Chi-square test of independence between two categorical variables
            if column2 and column2 in self.df.columns and pd.api.types.is_object_dtype(self.df[column1]) and pd.api.types.is_object_dtype(self.df[column2]):
                # Create contingency table
                contingency = pd.crosstab(self.df[column1], self.df[column2])
                chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
                result['test_name'] = "Chi-square test of independence"
                result['description'] = f"Tests if '{column1}' and '{column2}' are independent"
                result['statistic'] = chi2
                result['p_value'] = pvalue
                result['dof'] = dof
                result['interpretation'] = "Variables are dependent" if pvalue < 0.05 else "Variables are independent"
                result['contingency_table'] = contingency
            else:
                raise ValueError("Both columns must be categorical for chi-square test")
        
        elif test_type == 'correlation':
            # Correlation test
            if column2 and column2 in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column1]) and pd.api.types.is_numeric_dtype(self.df[column2]):
                correlation, pvalue = stats.pearsonr(self.df[column1].dropna(), self.df[column2].dropna())
                result['test_name'] = "Pearson correlation"
                result['description'] = f"Measures linear relationship between '{column1}' and '{column2}'"
                result['correlation'] = correlation
                result['p_value'] = pvalue
                result['interpretation'] = "Significant correlation" if pvalue < 0.05 else "No significant correlation"
                
                # Add interpretation of correlation strength
                if abs(correlation) < 0.3:
                    strength = "weak"
                elif abs(correlation) < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"
                direction = "positive" if correlation > 0 else "negative"
                result['correlation_description'] = f"{strength} {direction} correlation (r={correlation:.3f})"
            else:
                raise ValueError("Both columns must be numeric for correlation test")
                
        return result
    
    def perform_clustering(self, columns, n_clusters=3, method='kmeans'):
        """
        Performs clustering on selected numeric columns.
        
        Args:
            columns (list): List of columns to use for clustering
            n_clusters (int): Number of clusters to form
            method (str): Clustering method - 'kmeans' or 'hierarchical'
            
        Returns:
            pd.DataFrame: Original data with cluster assignments
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Ensure all selected columns are numeric
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column '{col}' must be numeric for clustering")
        
        # Extract and scale the features for clustering
        X = self.df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Fit model and get cluster assignments
        cluster_labels = clustering.fit_predict(X_scaled)
        
        # Create a new DataFrame with original data and cluster assignments
        result_df = X.copy()
        result_df['cluster'] = cluster_labels
        
        # Calculate cluster centroids (for KMeans)
        centroids = None
        if method == 'kmeans':
            centroids = pd.DataFrame(
                scaler.inverse_transform(clustering.cluster_centers_),
                columns=columns
            )
            centroids['cluster'] = range(n_clusters)
        
        # If we have more than 2 dimensions, add PCA for visualization
        pca_result = None
        if len(columns) > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            pca_df['cluster'] = cluster_labels
            pca_df = pca_df.reset_index().merge(X.reset_index(), on='index').drop('index', axis=1)
            explained_variance = pca.explained_variance_ratio_
        else:
            pca_df = None
            explained_variance = None
        
        return {
            'clustered_data': result_df,
            'centroids': centroids,
            'pca_data': pca_df,
            'explained_variance': explained_variance,
            'n_clusters': n_clusters,
            'method': method
        }
    
    def select_features(self, target_column, n_features=10, method='importance'):
        """
        Performs feature selection to identify most relevant features for target_column.
        
        Args:
            target_column (str): Target variable for prediction
            n_features (int): Number of top features to select
            method (str): Selection method - 'importance', 'f_test', 'mutual_info'
            
        Returns:
            list: Selected feature names and their scores
        """
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Determine if classification or regression problem
        is_classification = pd.api.types.is_object_dtype(self.df[target_column]) or len(self.df[target_column].unique()) <= 10
        
        # Get numeric features, excluding the target
        numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns if col != target_column]
        
        if not numeric_features:
            raise ValueError("No numeric features available for selection")
        
        # Prepare the data
        X = self.df[numeric_features].dropna()
        y = self.df.loc[X.index, target_column]
        
        # Perform feature selection based on method
        feature_scores = {}
        
        if method == 'importance':
            # Random Forest feature importance
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            feature_scores = dict(zip(numeric_features, model.feature_importances_))
        
        elif method == 'f_test':
            # F-test for feature selection
            if is_classification:
                selector = SelectKBest(f_classif, k=min(n_features, len(numeric_features)))
            else:
                selector = SelectKBest(f_regression, k=min(n_features, len(numeric_features)))
            
            selector.fit(X, y)
            feature_scores = dict(zip(numeric_features, selector.scores_))
        
        elif method == 'mutual_info':
            # Mutual information for feature selection
            if is_classification:
                selector = SelectKBest(mutual_info_classif, k=min(n_features, len(numeric_features)))
            else:
                selector = SelectKBest(mutual_info_regression, k=min(n_features, len(numeric_features)))
            
            selector.fit(X, y)
            feature_scores = dict(zip(numeric_features, selector.scores_))
        
        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:n_features]
        
        return {
            'top_features': top_features,
            'method': method,
            'is_classification': is_classification
        }
    
    def automated_eda(self):
        """
        Performs automated exploratory data analysis on the dataset.
        
        Returns:
            dict: EDA results including summary statistics, distributions, correlations, and outliers
        """
        results = {}
        
        # Basic dataset info
        results['dataset_shape'] = self.df.shape
        results['memory_usage'] = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        results['missing_values'] = {
            'counts': missing_counts[missing_counts > 0],
            'percentages': missing_percent[missing_percent > 0],
            'total_missing_cells': self.df.isnull().sum().sum(),
            'total_cells': self.df.size,
            'percent_missing': (self.df.isnull().sum().sum() / self.df.size) * 100
        }
        
        # Numeric column analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Summary statistics
            results['numeric_summary'] = self.df[numeric_cols].describe()
            
            # Detect skewness
            skewness = self.df[numeric_cols].skew()
            high_skew_cols = skewness[abs(skewness) > 1].to_dict()
            results['skewed_features'] = high_skew_cols
            
            # Detect outliers (using IQR method)
            outliers_info = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                if len(outliers) > 0:
                    outliers_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(self.df)) * 100,
                        'min': outliers.min(),
                        'max': outliers.max()
                    }
            results['outliers'] = outliers_info
            
            # Correlation analysis for numeric features
            if len(numeric_cols) > 1:
                correlation_matrix = self.df[numeric_cols].corr()
                # Find highly correlated features
                high_corr = {}
                for i, col1 in enumerate(correlation_matrix.columns):
                    for col2 in correlation_matrix.columns[i+1:]:
                        corr_value = correlation_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.7:  # Threshold for high correlation
                            high_corr[f"{col1} & {col2}"] = corr_value
                results['high_correlations'] = high_corr
        
        # Categorical column analysis
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            cat_summary = {}
            for col in cat_cols:
                value_counts = self.df[col].value_counts()
                unique_count = len(value_counts)
                cat_summary[col] = {
                    'unique_values': unique_count,
                    'top_categories': value_counts.head(5).to_dict() if unique_count > 5 else value_counts.to_dict(),
                    'high_cardinality': unique_count > 10
                }
            results['categorical_summary'] = cat_summary
        
        # Date column detection and analysis
        date_cols = []
        for col in self.df.columns:
            try:
                # Check if column can be converted to datetime
                pd.to_datetime(self.df[col], errors='raise')
                date_cols.append(col)
            except:
                continue
        
        if date_cols:
            date_summary = {}
            for col in date_cols:
                datetime_series = pd.to_datetime(self.df[col])
                date_summary[col] = {
                    'min_date': datetime_series.min(),
                    'max_date': datetime_series.max(),
                    'range_days': (datetime_series.max() - datetime_series.min()).days,
                    'weekend_days': sum(datetime_series.dt.dayofweek >= 5),
                    'weekday_days': sum(datetime_series.dt.dayofweek < 5),
                    'null_dates': self.df[col].isnull().sum()
                }
            results['date_analysis'] = date_summary
        
        return results

    def is_likely_id_column(self, column):
        """
        Detect if a column is likely to be an ID column based on:
        1. Column name containing 'id', 'num', 'code', 'no' etc.
        2. Having mostly unique values (>80% of values are unique)
        3. Contains alphanumeric patterns typical of IDs
        
        Args:
            column (str): The column to check
            
        Returns:
            bool: True if it's likely an ID column
            str: Reason why it was flagged as ID
        """
        if column not in self.df.columns:
            return False, ""
            
        # Check column name patterns
        name_patterns = ['id', 'num', 'code', 'no', 'key', 'uuid', 'guid']
        col_lower = column.lower()
        if any(pattern in col_lower for pattern in name_patterns):
            name_match = True
        else:
            name_match = False
            
        # Check uniqueness
        unique_ratio = self.df[column].nunique() / len(self.df)
        high_uniqueness = unique_ratio > 0.8
        
        # Check for alphanumeric patterns in string columns
        if pd.api.types.is_object_dtype(self.df[column]):
            # Sample values
            sample = self.df[column].dropna().astype(str).sample(min(100, len(self.df))).tolist()
            # Check if contains mix of letters and numbers
            has_alphanumeric = any(re.search(r'[A-Za-z].*[0-9]|[0-9].*[A-Za-z]', str(val)) for val in sample if isinstance(val, str))
            # Check for common ID patterns like "A-123" or "ABC123"
            has_id_pattern = any(re.search(r'^[A-Za-z\-]+[0-9]+$|^[0-9]+[-_/][0-9A-Za-z]+$', str(val)) for val in sample if isinstance(val, str))
        else:
            has_alphanumeric = False
            has_id_pattern = False
            
        # Determine if it's an ID based on multiple factors
        is_id = False
        reason = []
        
        if name_match and high_uniqueness:
            is_id = True
            reason.append(f"Column name contains ID-like terms and {unique_ratio:.1%} of values are unique")
        elif high_uniqueness and (has_alphanumeric or has_id_pattern):
            is_id = True
            reason.append(f"Contains ID-like patterns and {unique_ratio:.1%} of values are unique")
        elif unique_ratio > 0.95:
            is_id = True
            reason.append(f"Extremely high uniqueness ({unique_ratio:.1%} unique values)")
            
        return is_id, "; ".join(reason)

    def plot_categorical_distribution(self, column, plot_type='bar'):
        """
        Visualize the distribution of a categorical column with multiple plot types.

        Args:
            column (str): The column to plot
            plot_type (str): Visualization type - 'bar', 'treemap', or 'lollipop'
        """
        # Check if column exists
        if column not in self.df.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return
            
        # Get number of unique values
        unique_count = self.df[column].nunique()
        
        # Validate if column has too many categories
        if unique_count > 50:
            st.error(f"⚠️ Column '{column}' has {unique_count} unique values, which is too many for effective visualization.")
            st.info("Consider selecting a column with fewer categories or using a frequency-based encoding first.")
            return
            
        # For numeric columns with many unique values
        if pd.api.types.is_numeric_dtype(self.df[column]) and unique_count > 20:
            st.warning(f"Column '{column}' appears to be numeric with {unique_count} unique values. Consider using a histogram instead.")
            cont = st.checkbox("Continue anyway?")
            if not cont:
                return
                
        # Get value counts and prepare data
        value_counts = self.df[column].value_counts()
        
        # Determine how many categories to display based on total count
        if unique_count > 15:
            display_count = 15
            st.info(f"Showing top {display_count} categories from {unique_count} total categories.")
        else:
            display_count = unique_count
            
        # Option to sort values
        sort_option = st.radio(
            "Sort categories by:", 
            ["Frequency (descending)", "Frequency (ascending)", "Alphabetical"],
            horizontal=True
        )
        
        if sort_option == "Frequency (descending)":
            value_counts = value_counts.nlargest(display_count)
        elif sort_option == "Frequency (ascending)":
            value_counts = value_counts.nsmallest(display_count)
        else:  # Alphabetical
            value_counts = value_counts.sort_index().head(display_count)
        
        # If there are more categories than we're displaying, add "Other"
        if unique_count > display_count:
            # Get the displayed categories
            displayed_cats = value_counts.index.tolist()
            # Calculate sum of all other categories
            other_sum = self.df[~self.df[column].isin(displayed_cats)][column].count()
            
            # Add "Other" category if it would be visible
            if other_sum > 0:
                # Create a new series with displayed categories + "Other"
                plot_data = pd.Series(
                    list(value_counts.values) + [other_sum], 
                    index=list(value_counts.index) + ['Other']
                )
            else:
                plot_data = value_counts
        else:
            plot_data = value_counts
            
        # Create the visualization based on selected type
        if plot_type == 'bar':
            self._plot_bar_chart(column, plot_data)
        elif plot_type == 'treemap':
            self._plot_treemap(column, plot_data)
        elif plot_type == 'lollipop':
            self._plot_lollipop_chart(column, plot_data)
        elif plot_type == 'pie':
            self._plot_pie_chart(column, plot_data)
        else:
            st.error(f"Unknown plot type: {plot_type}")
    
    def _plot_bar_chart(self, column, plot_data):
        """Helper method to create a horizontal bar chart."""
        if PLOTLY_AVAILABLE:
            # Create interactive Plotly bar chart
            fig = px.bar(
                y=plot_data.index.astype(str),
                x=plot_data.values,
                title=f'Distribution of {column}',
                labels={'y': column, 'x': 'Count'},
                height=max(400, len(plot_data) * 30),  # Dynamic height based on categories
                orientation='h'  # Horizontal
            )
            # Add count labels
            fig.update_traces(texttemplate='%{x}', textposition='outside')
            # Update layout
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Count",
                yaxis_title=column
            )
            st.plotly_chart(fig)
        else:
            # Matplotlib fallback
            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.4)))
            # Plot horizontal bars
            plot_data.plot.barh(ax=ax)
            # Add counts as text
            for i, v in enumerate(plot_data.values):
                ax.text(v + 0.1, i, str(v), va='center')
            plt.title(f'Distribution of {column}')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    def _plot_treemap(self, column, plot_data):
        """Helper method to create a treemap visualization."""
        if PLOTLY_AVAILABLE:
            # Create a DataFrame for the treemap
            df_treemap = pd.DataFrame({
                'labels': plot_data.index.astype(str),
                'values': plot_data.values,
                'percentages': (plot_data.values / plot_data.values.sum() * 100).round(1)
            })
            
            # Create the treemap
            fig = px.treemap(
                df_treemap,
                names='labels',
                values='values',
                title=f'Distribution of {column}',
                height=500
            )
            
            # Customize text to show both count and percentage
            fig.update_traces(
                textinfo='label+value+percent',
                texttemplate='%{label}<br>%{value} (%{percentRoot:.1%})'
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("Treemap visualization requires Plotly. Using bar chart instead.")
            self._plot_bar_chart(column, plot_data)
    
    def _plot_lollipop_chart(self, column, plot_data):
        """Helper method to create a lollipop chart."""
        if PLOTLY_AVAILABLE:
            # Create a scatter plot with lines for lollipop effect
            fig = go.Figure()
            
            # Add lines
            fig.add_trace(go.Scatter(
                x=plot_data.values,
                y=plot_data.index.astype(str),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                showlegend=False
            ))
            
            # Add markers
            fig.add_trace(go.Scatter(
                x=plot_data.values,
                y=plot_data.index.astype(str),
                mode='markers',
                marker=dict(
                    color='rgba(31, 119, 180, 0.8)',
                    size=12
                ),
                text=plot_data.values,
                textposition='middle right',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Distribution of {column}',
                xaxis_title='Count',
                yaxis_title=column,
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(plot_data) * 30),
                margin=dict(l=100)
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("Lollipop chart visualization requires Plotly. Using bar chart instead.")
            self._plot_bar_chart(column, plot_data)
    
    def _plot_pie_chart(self, column, plot_data):
        """Helper method to create a pie chart (maintained for compatibility)."""
        if len(plot_data) > 10:
            st.warning("Pie charts are not recommended for more than 10 categories. Consider using the bar chart option instead.")
            
        # Create the pie chart
        if PLOTLY_AVAILABLE:
            # Create an interactive plotly pie chart
            fig = px.pie(
                values=plot_data.values,
                names=plot_data.index.astype(str),
                title=f'Distribution of {column}',
                hole=0.3,  # Create a donut chart for better readability
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
            st.plotly_chart(fig)
        else:
            # Matplotlib fallback
            fig = plt.figure(figsize=(10, 8))
            plt.pie(plot_data.values, labels=plot_data.index.astype(str), 
                   autopct='%1.1f%%', startangle=90, 
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            plt.title(f'Distribution of {column}')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)
            plt.close()
            
    # For backward compatibility, maintain the plot_pie_chart function
    def plot_pie_chart(self, column):
        """
        Create a pie chart for categorical columns.
        This function is maintained for backward compatibility.
        Consider using plot_categorical_distribution instead.
        
        Args:
            column (str): The column to plot.
        """
        st.warning("Pie charts are generally not the most effective visualization. Consider bar charts or treemaps for better comparison.")
        return self.plot_categorical_distribution(column, plot_type='pie')

    def forecast_demand(self, date_column, demand_column, forecast_periods=30, algorithm='prophet', seasonality='auto'):
        """
        Generate demand forecasts using various algorithms.
        
        Args:
            date_column (str): The name of the date/time column
            demand_column (str): The name of the demand/sales column
            forecast_periods (int): Number of periods to forecast
            algorithm (str): Forecasting algorithm to use ('prophet', 'arima', 'exponential')
            seasonality (str): Type of seasonality ('auto', 'daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            tuple: (forecast_results DataFrame, plotly figure)
        """
        # Check for plotly availability
        if not PLOTLY_AVAILABLE:
            st.error("Plotly is required for forecasting visualizations. Please install with 'pip install plotly'")
            return None, None
            
        # Import plotly explicitly here to avoid any issues
        import plotly.graph_objects as go
        
        # Ensure date column is datetime
        forecast_df = self.df.copy()
        
        # Use the improved date conversion
        date_format = self.detect_date_format(forecast_df[date_column])
        if date_format:
            forecast_df[date_column] = pd.to_datetime(forecast_df[date_column], format=date_format, errors='coerce')
        else:
            forecast_df[date_column] = pd.to_datetime(forecast_df[date_column], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = forecast_df[forecast_df[date_column].isna()]
        if not invalid_dates.empty:
            st.warning(f"Warning: {len(invalid_dates)} rows have invalid dates and will be dropped.")
            forecast_df = forecast_df.dropna(subset=[date_column])
        
        # Sort by date
        forecast_df = forecast_df.sort_values(date_column)
        
        # Validate data
        if forecast_df[demand_column].isnull().any():
            st.warning("Warning: Demand column contains missing values. Filling with interpolation.")
            forecast_df[demand_column] = forecast_df[demand_column].interpolate()
        
        # Choose algorithm
        results = None
        fig = None
        
        try:
            if algorithm == 'prophet':
                # Facebook Prophet
                from prophet import Prophet
                
                # Rename columns to Prophet requirements
                prophet_df = forecast_df.rename(columns={
                    date_column: 'ds',
                    demand_column: 'y'
                })
                
                # Configure seasonality
                if seasonality == 'auto':
                    model = Prophet(yearly_seasonality=True,
                                 weekly_seasonality=True,
                                 daily_seasonality=True)
                else:
                    model = Prophet(yearly_seasonality=seasonality=='yearly',
                                 weekly_seasonality=seasonality=='weekly',
                                 daily_seasonality=seasonality=='daily')
                
                # Fit model
                model.fit(prophet_df)
                
                # Create future dates
                future = model.make_future_dataframe(periods=forecast_periods)
                
                # Make predictions
                forecast = model.predict(future)
                
                # Prepare results
                results = pd.DataFrame({
                    'date': forecast['ds'],
                    'demand': forecast['yhat'],
                    'demand_lower': forecast['yhat_lower'],
                    'demand_upper': forecast['yhat_upper']
                })
                
                # Plot results with Plotly
                fig = go.Figure()
                
                # Historical values
                fig.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    name='Historical',
                    mode='lines',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='red')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand_upper'],
                    name='Upper Bound',
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand_lower'],
                    name='Lower Bound',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                
                # Layout
                fig.update_layout(
                    title=f'Demand Forecast ({forecast_periods} periods)',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    hovermode='x unified'
                )
                
                # Components plot
                components_fig = model.plot_components(forecast)
                
                # Convert to plotly figure
                if PLOTLY_AVAILABLE:
                    import plotly.express as px
                    from plotly.subplots import make_subplots
                    
                    # Get number of components
                    n_components = len(components_fig.axes)
                    
                    # Create subplots
                    components_plotly = make_subplots(rows=n_components, cols=1,
                                                  subplot_titles=[ax.get_title() for ax in components_fig.axes])
                    
                    # Add components
                    for i, ax in enumerate(components_fig.axes):
                        for line in ax.lines:
                            components_plotly.add_trace(
                                go.Scatter(
                                    x=line.get_xdata(),
                                    y=line.get_ydata(),
                                    mode='lines',
                                    name=line.get_label() if line.get_label() else f'Component {i}'
                                ),
                                row=i+1, col=1
                            )
                    
                    # Update layout
                    components_plotly.update_layout(
                        title='Forecast Components',
                        height=300 * n_components
                    )
                    
                    st.plotly_chart(components_plotly)
                
            elif algorithm == 'arima':
                # ARIMA model
                from pmdarima import auto_arima
                
                # Prepare time series data
                ts_data = forecast_df.set_index(date_column)[demand_column]
                
                # Determine seasonality periods 
                if seasonality == 'yearly':
                    seasonal_m = 12  # Monthly data, yearly seasonality
                elif seasonality == 'quarterly':
                    seasonal_m = 4
                elif seasonality == 'weekly':
                    seasonal_m = 7  # Daily data, weekly seasonality
                elif seasonality == 'daily':
                    seasonal_m = 24  # Hourly data, daily seasonality
                else:
                    # Auto-detect based on frequency
                    if ts_data.index.freq is None:
                        # Infer frequency
                        ts_data = ts_data.asfreq(pd.infer_freq(ts_data.index))
                    
                    freq = ts_data.index.freq
                    if freq is None:
                        # Fall back to default if inference fails
                        seasonal_m = 12
                    elif freq.startswith('D'):
                        seasonal_m = 7  # Daily data
                    elif freq.startswith('M'):
                        seasonal_m = 12  # Monthly data
                    elif freq.startswith('H'):
                        seasonal_m = 24  # Hourly data
                    else:
                        seasonal_m = 12  # Default
                
                # Fit model
                model = auto_arima(ts_data,
                               seasonal=True,
                               m=seasonal_m,
                               suppress_warnings=True)
                
                # Create future dates for forecasting
                last_date = forecast_df[date_column].max()
                
                # Infer frequency of data
                if ts_data.index.freq is None:
                    # Estimate frequency from the data
                    date_diffs = forecast_df[date_column].diff().dropna()
                    if not date_diffs.empty:
                        median_diff = date_diffs.median()
                        if median_diff.days == 1:
                            freq = 'D'  # Daily
                        elif 25 <= median_diff.days <= 32:
                            freq = 'M'  # Monthly
                        elif 85 <= median_diff.days <= 95:
                            freq = 'Q'  # Quarterly
                        else:
                            freq = 'D'  # Default to daily
                    else:
                        freq = 'D'  # Default
                else:
                    freq = ts_data.index.freq
                
                # Create future dates with proper frequency
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq=freq
                )
                
                # Make predictions
                forecast, conf_int = model.predict(n_periods=forecast_periods, 
                                              return_conf_int=True)
                
                # Prepare results
                results = pd.DataFrame({
                    'date': future_dates,
                    'demand': forecast,
                    'demand_lower': conf_int[:, 0],
                    'demand_upper': conf_int[:, 1]
                })
                
                # Plot results
                fig = go.Figure()
                
                # Historical values
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df[demand_column],
                    name='Historical',
                    mode='lines',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='red')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand_upper'],
                    name='Upper Bound',
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand_lower'],
                    name='Lower Bound',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    showlegend=False
                ))
                
                # Layout
                fig.update_layout(
                    title=f'Demand Forecast ({forecast_periods} periods)',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    hovermode='x unified'
                )
                
            else:
                # Exponential Smoothing
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Prepare time series data with proper index
                ts_data = forecast_df.set_index(date_column)[demand_column]
                
                # Infer frequency if not set
                if ts_data.index.freq is None:
                    # Try to infer frequency
                    inferred_freq = pd.infer_freq(ts_data.index)
                    if inferred_freq:
                        ts_data = ts_data.asfreq(inferred_freq)
                    else:
                        # Manual frequency inference
                        date_diffs = forecast_df[date_column].diff().dropna()
                        if not date_diffs.empty:
                            median_diff = date_diffs.median()
                            if median_diff.days == 1:
                                ts_data = ts_data.asfreq('D')  # Daily
                            elif 25 <= median_diff.days <= 32:
                                ts_data = ts_data.asfreq('M')  # Monthly
                            elif 85 <= median_diff.days <= 95:
                                ts_data = ts_data.asfreq('Q')  # Quarterly
                            else:
                                ts_data = ts_data.asfreq('D')  # Default to daily
                        else:
                            # If all else fails, resample to daily frequency
                            ts_data = ts_data.resample('D').asfreq()
                
                # Determine seasonal periods based on frequency
                seasonal_periods = 1  # Default to no seasonality
                freq = ts_data.index.freq
                
                if freq:
                    freq_str = str(freq)
                    if freq_str.startswith('D'):
                        seasonal_periods = 7  # Weekly seasonality for daily data
                    elif freq_str.startswith('M'):
                        seasonal_periods = 12  # Yearly seasonality for monthly data
                    elif freq_str.startswith('H'):
                        seasonal_periods = 24  # Daily seasonality for hourly data
                    elif freq_str.startswith('Q'):
                        seasonal_periods = 4  # Yearly seasonality for quarterly data
                else:
                    # Map user selection to periods
                    seasonal_periods = 12 if seasonality=='yearly' else \
                                     52 if seasonality=='weekly' else \
                                     7 if seasonality=='daily' else 1
                
                # Fit model with appropriate seasonality
                model = ExponentialSmoothing(
                    ts_data,
                    seasonal_periods=seasonal_periods,
                    seasonal='add' if seasonal_periods > 1 else None,
                    initialization_method="estimated"
                ).fit(optimized=True)
                
                # Generate future dates with proper frequency
                if freq:
                    future_dates = pd.date_range(
                        start=ts_data.index[-1] + freq,
                        periods=forecast_periods,
                        freq=freq
                    )
                else:
                    # If frequency couldn't be determined, use daily frequency
                    future_dates = pd.date_range(
                        start=ts_data.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq='D'
                    )
                
                # Make predictions
                forecast = model.forecast(steps=forecast_periods)
                
                # Align forecast with future_dates
                future_index = pd.DatetimeIndex(future_dates)
                forecast.index = future_index
                
                # Prepare results
                results = pd.DataFrame({
                    'date': future_dates,
                    'demand': forecast.values
                })
                
                # Plot results (with Plotly)
                fig = go.Figure()
                
                # Historical values
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_column],
                    y=forecast_df[demand_column],
                    name='Historical',
                    mode='lines',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['demand'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='red')
                ))
                
                # Layout
                fig.update_layout(
                    title=f'Demand Forecast ({forecast_periods} periods)',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    hovermode='x unified'
                )
                
            return results, fig
        
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, None

    def simulate_demand(self, n_periods=48, mean_sales=91.0, std_dev=10.0, trend=5.0, seasonality=True, promotions=True):
        """
        Simulate demand data with trend, seasonality, and promotion effects.
        
        Args:
            n_periods (int): Number of periods to simulate (default 48 months/4 years)
            mean_sales (float): Mean level of sales (default 91.0)
            std_dev (float): Standard deviation of random component (default 10.0)
            trend (float): Units increase per period (default 5.0)
            seasonality (bool): Whether to include seasonal effects (default True)
            promotions (bool): Whether to include promotion effects (default True)
            
        Returns:
            pd.DataFrame: DataFrame with simulated demand data and time index
        """
        # Generate time index
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
        
        # Initialize components
        random_component = np.random.normal(0, std_dev, n_periods)
        trend_component = trend * np.arange(n_periods)
        
        # Seasonal component (repeating pattern every 12 months)
        seasonal_component = np.zeros(n_periods)
        if seasonality:
            if hasattr(self, 'seasonal_patterns') and self.seasonal_patterns:
                # Use custom seasonal patterns
                for i in range(n_periods):
                    month = (i % 12) + 1
                    if month in self.seasonal_patterns:
                        seasonal_component[i] = self.seasonal_patterns[month]
            else:
                # Use default seasonal pattern
                seasonal_pattern = {
                    1: 25,   # Q1 start
                    2: 0,
                    3: 0,
                    4: -25,  # Q2 start
                    5: 0,
                    6: 0,
                    7: 0,    # Q3 start
                    8: 0,
                    9: 0,
                    10: 0,   # Q4 start
                    11: 0,
                    12: 0
                }
                seasonal_component = np.array([seasonal_pattern[((i % 12) + 1)] for i in range(n_periods)])
        
        # Promotion component (specific months with higher sales)
        promotion_component = np.zeros(n_periods)
        if promotions:
            if hasattr(self, 'promotions') and self.promotions:
                # Use custom promotions
                for promo in self.promotions:
                    start_idx = promo['start'] - 1
                    end_idx = promo['end']
                    if start_idx < n_periods:
                        end_idx = min(end_idx, n_periods)
                        promotion_component[start_idx:end_idx] = promo['change']
            else:
                # Use default promotions
                default_promotions = [
                    {'start': 1, 'end': 3, 'change': 25},
                    {'start': 4, 'end': 8, 'change': -25},
                    {'start': 9, 'end': 12, 'change': 0}
                ]
                for promo in default_promotions:
                    start_idx = promo['start'] - 1
                    end_idx = promo['end']
                    promotion_component[start_idx:end_idx] = promo['change']
        
        # Base demand
        demand = mean_sales + random_component + trend_component
        
        # Add seasonality and promotions
        if seasonality:
            demand += seasonal_component
        if promotions:
            demand += promotion_component
        
        # Create DataFrame
        df = pd.DataFrame({
            'Month': dates,
            'Demand': demand,
            'Trend': trend_component + mean_sales,
            'Seasonality': seasonal_component if seasonality else np.zeros(n_periods),
            'Promotions': promotion_component if promotions else np.zeros(n_periods),
            'Random': random_component
        })
        df.set_index('Month', inplace=True)
        
        return df
    
    def add_season(self, start_month, end_month, change):
        """
        Add a new seasonal pattern to the demand simulation.
        
        Args:
            start_month (int): Starting month (1-12)
            end_month (int): Ending month (1-12)
            change (float): Change in demand during this season
        """
        if not hasattr(self, 'seasonal_patterns'):
            self.seasonal_patterns = {}
            
        # Convert change to float
        change = float(change)
        for month in range(start_month, end_month + 1):
            self.seasonal_patterns[month] = change
    
    def add_promotion(self, start_month, end_month, change):
        """
        Add a new promotion period to the demand simulation.
        
        Args:
            start_month (int): Starting month
            end_month (int): Ending month
            change (float): Change in demand during this promotion
        """
        if not hasattr(self, 'promotions'):
            self.promotions = []
            
        # Convert change to float
        change = float(change)
        self.promotions.append({
            'start': start_month,
            'end': end_month,
            'change': change
        })
    
    def plot_demand_simulation(self, simulated_data, components=None):
        """
        Plot the simulated demand data with its components.
        
        Args:
            simulated_data (pd.DataFrame): Output from simulate_demand
            components (list): List of components to plot ['Trend', 'Seasonality', 'Promotions']
        """
        if components is None:
            components = ['Trend', 'Seasonality', 'Promotions']
            
        if not PLOTLY_AVAILABLE:
            st.error("Plotly is required for interactive plots. Please install with 'pip install plotly'")
            return
            
        fig = go.Figure()
        
        # Plot actual demand
        fig.add_trace(go.Scatter(
            x=simulated_data.index,
            y=simulated_data['Demand'],
            name='Total Demand',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        
        # Plot components
        colors = {'Trend': 'red', 'Seasonality': 'green', 'Promotions': 'orange'}
        for component in components:
            if component in simulated_data.columns:
                fig.add_trace(go.Scatter(
                    x=simulated_data.index,
                    y=simulated_data[component],
                    name=component,
                    mode='lines',
                    line=dict(color=colors[component], width=1.5, dash='dash')
                ))
        
        fig.update_layout(
            title='Demand Simulation',
            xaxis_title='Month',
            yaxis_title='Units',
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)

    def calculate_clv(self, customer_id_col, invoice_date_col, quantity_col, unit_price_col, country_col=None):
        """
        Calculates historical Customer Lifetime Value (CLV) based on transaction data.
        Includes RFM segmentation and segment-based CLV analysis.

        Args:
            customer_id_col (str): Name of the column containing customer IDs
            invoice_date_col (str): Name of the column containing transaction dates
            quantity_col (str): Name of the column containing quantity purchased
            unit_price_col (str): Name of the column containing unit price
            country_col (str, optional): Name of the column containing country information

        Returns:
            tuple: (clv_summary DataFrame, clv_stats DataFrame, rfm_segments DataFrame)
        """
        # Validate required columns exist
        required_cols = [customer_id_col, invoice_date_col, quantity_col, unit_price_col]
        if country_col:
            required_cols.append(country_col)
            
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None, None

        # Create working copy
        clv_df = self.df[required_cols].copy()

        # Convert date column to datetime
        try:
            clv_df[invoice_date_col] = pd.to_datetime(clv_df[invoice_date_col])
        except Exception as e:
            st.error(f"Error converting {invoice_date_col} to datetime: {str(e)}")
            return None, None, None

        # Convert numeric columns
        try:
            clv_df[quantity_col] = pd.to_numeric(clv_df[quantity_col], errors='coerce')
            clv_df[unit_price_col] = pd.to_numeric(clv_df[unit_price_col], errors='coerce')
        except Exception as e:
            st.error(f"Error converting quantity/price to numeric: {str(e)}")
            return None, None, None

        # Calculate total price
        clv_df['total_price'] = clv_df[quantity_col] * clv_df[unit_price_col]

        # Remove invalid transactions
        initial_rows = len(clv_df)
        clv_df = clv_df.dropna(subset=[customer_id_col, 'total_price'])
        clv_df = clv_df[clv_df['total_price'] > 0]
        
        if len(clv_df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(clv_df)} invalid transactions")

        if clv_df.empty:
            st.error("No valid transactions remaining after cleaning")
            return None, None, None

        # Calculate CLV metrics
        latest_date = clv_df[invoice_date_col].max()
        
        # Group by customer ID and country (if available)
        group_cols = [customer_id_col]
        if country_col:
            group_cols.append(country_col)
            
        clv_summary = clv_df.groupby(group_cols).agg({
            'total_price': 'sum',
            invoice_date_col: ['count', 'min', 'max']
        }).reset_index()

        # Rename columns
        clv_summary.columns = [
            customer_id_col,
            *(['country'] if country_col else []),
            'total_revenue',
            'frequency',
            'first_purchase',
            'last_purchase'
        ]

        # Calculate additional metrics
        clv_summary['average_order_value'] = clv_summary['total_revenue'] / clv_summary['frequency']
        clv_summary['customer_lifespan'] = (clv_summary['last_purchase'] - clv_summary['first_purchase']).dt.days
        clv_summary['recency'] = (latest_date - clv_summary['last_purchase']).dt.days
        
        # Calculate historical CLV (using total revenue as proxy)
        clv_summary['historical_clv'] = clv_summary['total_revenue']

        # Calculate RFM metrics
        rfm = clv_summary.copy()
        
        # Recency score (1-5, 5 being most recent)
        rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
        
        # Frequency score (1-5, 5 being most frequent)
        rfm['frequency_score'] = pd.qcut(rfm['frequency'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Monetary score (1-5, 5 being highest value)
        rfm['monetary_score'] = pd.qcut(rfm['total_revenue'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Calculate RFM score (sum of individual scores)
        rfm['rfm_score'] = rfm['recency_score'].astype(int) + rfm['frequency_score'].astype(int) + rfm['monetary_score'].astype(int)
        
        # Define RFM segments
        def get_rfm_segment(row):
            if row['rfm_score'] >= 13:
                return 'Champions'
            elif row['rfm_score'] >= 10:
                return 'Loyal Customers'
            elif row['rfm_score'] >= 7:
                return 'Potential Loyalists'
            elif row['rfm_score'] >= 4:
                return 'At Risk'
            else:
                return 'Lost'
        
        rfm['segment'] = rfm.apply(get_rfm_segment, axis=1)
        
        # Calculate segment statistics
        segment_stats = rfm.groupby('segment').agg({
            'historical_clv': ['mean', 'median', 'sum', 'count'],
            'frequency': 'mean',
            'recency': 'mean',
            'total_revenue': 'sum'
        }).round(2)
        
        # Rename columns for clarity
        segment_stats.columns = [
            'Average CLV', 'Median CLV', 'Total CLV', 'Customer Count',
            'Average Frequency', 'Average Recency (days)', 'Total Revenue'
        ]
        
        # Copy RFM columns back to clv_summary for output
        clv_summary['recency_score'] = rfm['recency_score']
        clv_summary['frequency_score'] = rfm['frequency_score'] 
        clv_summary['monetary_score'] = rfm['monetary_score']
        clv_summary['rfm_score'] = rfm['rfm_score']
        clv_summary['segment'] = rfm['segment']
        
        # Calculate summary statistics
        numeric_cols = ['historical_clv', 'average_order_value', 'frequency', 'customer_lifespan', 'recency']
        clv_stats = clv_summary[numeric_cols].describe()

        return clv_summary, clv_stats, segment_stats

    def get_cleaned_data(self):
        """
        Returns the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        return self.df.copy()

    def prepare_clv_for_regression(self, customer_id_col, invoice_date_col, quantity_col, unit_price_col, country_col=None):
        """
        Prepares CLV data for regression analysis by creating features and handling data types.
        
        Args:
            customer_id_col (str): Name of the column containing customer IDs
            invoice_date_col (str): Name of the column containing transaction dates
            quantity_col (str): Name of the column containing quantity purchased
            unit_price_col (str): Name of the column containing unit price
            country_col (str, optional): Name of the column containing country information
            
        Returns:
            tuple: (X DataFrame, y Series, feature_names list)
        """
        # First calculate CLV metrics
        clv_results, _, _ = self.calculate_clv(
            customer_id_col=customer_id_col,
            invoice_date_col=invoice_date_col,
            quantity_col=quantity_col,
            unit_price_col=unit_price_col,
            country_col=country_col
        )
        
        if clv_results is None:
            return None, None, None
            
        # Create feature DataFrame
        features = pd.DataFrame()
        
        # Add basic metrics as features
        features['frequency'] = clv_results['frequency']
        features['average_order_value'] = clv_results['average_order_value']
        features['customer_lifespan'] = clv_results['customer_lifespan']
        features['recency'] = clv_results['recency']
        
        # Add time-based features
        features['days_since_first_purchase'] = (pd.Timestamp.now() - clv_results['first_purchase']).dt.days
        features['days_since_last_purchase'] = (pd.Timestamp.now() - clv_results['last_purchase']).dt.days
        
        # Add country as categorical feature if available
        if country_col:
            features['country'] = clv_results['country']
            # One-hot encode country
            country_dummies = pd.get_dummies(features['country'], prefix='country')
            features = pd.concat([features, country_dummies], axis=1)
            features = features.drop('country', axis=1)
        
        # Target variable is historical CLV
        y = clv_results['historical_clv']
        
        # Remove any remaining NaN values
        features = features.fillna(features.mean())
        
        return features, y, features.columns.tolist()

    def train_clv_regression_models(self, X, y, test_size=0.2, random_state=42):
        """
        Trains and evaluates Lasso and Ridge regression models for CLV prediction.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable (CLV)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing model results and metrics
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        lasso = Lasso(alpha=0.1)
        ridge = Ridge(alpha=0.1)
        
        # Train models
        lasso.fit(X_train_scaled, y_train)
        ridge.fit(X_train_scaled, y_train)
        
        # Make predictions
        lasso_pred = lasso.predict(X_test_scaled)
        ridge_pred = ridge.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            'lasso': {
                'mse': mean_squared_error(y_test, lasso_pred),
                'r2': r2_score(y_test, lasso_pred),
                'cv_mse': -np.mean(cross_val_score(lasso, X_train_scaled, y_train, 
                                                 scoring='neg_mean_squared_error', cv=5)),
                'cv_r2': np.mean(cross_val_score(lasso, X_train_scaled, y_train, 
                                               scoring='r2', cv=5)),
                'coefficients': dict(zip(X.columns, lasso.coef_)),
                'predictions': lasso_pred  # Add predictions to results
            },
            'ridge': {
                'mse': mean_squared_error(y_test, ridge_pred),
                'r2': r2_score(y_test, ridge_pred),
                'cv_mse': -np.mean(cross_val_score(ridge, X_train_scaled, y_train, 
                                                 scoring='neg_mean_squared_error', cv=5)),
                'cv_r2': np.mean(cross_val_score(ridge, X_train_scaled, y_train, 
                                               scoring='r2', cv=5)),
                'coefficients': dict(zip(X.columns, ridge.coef_)),
                'predictions': ridge_pred  # Add predictions to results
            },
            'y_test': y_test  # Add test set to results
        }
        
        return results

    def calculate_forecast_accuracy(self, actual_column, forecast_column, product_column=None):
        """
        Calculates forecast accuracy metrics including A/F ratio (Actual/Forecast).
        
        Args:
            actual_column (str): Name of column containing actual demand values
            forecast_column (str): Name of column containing forecasted demand values
            product_column (str, optional): Name of column containing product identifiers for grouping
            
        Returns:
            pd.DataFrame: DataFrame containing forecast accuracy metrics
        """
        # Validate columns exist
        required_cols = [actual_column, forecast_column]
        if product_column:
            required_cols.append(product_column)
            
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        # Convert columns to numeric if needed
        for col in [actual_column, forecast_column]:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    st.error(f"Could not convert {col} to numeric values")
                    return None
                    
        # Create working copy
        accuracy_df = self.df[required_cols].copy()
        
        # Remove rows with missing values
        initial_rows = len(accuracy_df)
        accuracy_df = accuracy_df.dropna()
        
        if len(accuracy_df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(accuracy_df)} rows with missing values")
            
        if accuracy_df.empty:
            st.error("No valid data remaining after cleaning")
            return None
            
        # Calculate A/F ratio
        accuracy_df['A/F'] = accuracy_df[actual_column] / accuracy_df[forecast_column]
        
        # Calculate other accuracy metrics
        accuracy_df['Error'] = accuracy_df[actual_column] - accuracy_df[forecast_column]
        accuracy_df['Absolute Error'] = abs(accuracy_df['Error'])
        accuracy_df['Percentage Error'] = (accuracy_df['Error'] / accuracy_df[forecast_column]) * 100
        accuracy_df['Absolute Percentage Error'] = abs(accuracy_df['Percentage Error'])
        
        # Group by product if product column is provided
        if product_column:
            result = accuracy_df.groupby(product_column).agg({
                actual_column: 'sum',
                forecast_column: 'sum',
                'A/F': 'mean',
                'Error': 'sum',
                'Absolute Error': 'sum',
                'Absolute Percentage Error': 'mean'
            }).round(2)
            
            # Rename columns for clarity
            result = result.rename(columns={
                actual_column: 'Demand',
                forecast_column: 'Forecast',
                'Absolute Percentage Error': 'MAPE (%)'
            })
            
            # Calculate overall accuracy metrics
            total_demand = accuracy_df[actual_column].sum()
            total_forecast = accuracy_df[forecast_column].sum()
            
            overall_metrics = {
                'Total Demand': total_demand,
                'Total Forecast': total_forecast,
                'Overall A/F': total_demand / total_forecast,
                'MAPE (%)': accuracy_df['Absolute Percentage Error'].mean()
            }
            
            return result, overall_metrics
        else:
            # Calculate overall metrics directly
            total_demand = accuracy_df[actual_column].sum()
            total_forecast = accuracy_df[forecast_column].sum()
            
            overall_metrics = {
                'Total Demand': total_demand,
                'Total Forecast': total_forecast,
                'Overall A/F': total_demand / total_forecast,
                'MAE': accuracy_df['Absolute Error'].mean(),
                'MAPE (%)': accuracy_df['Absolute Percentage Error'].mean()
            }
            
            return accuracy_df, overall_metrics

    def simulate_forecast_aggregation(self, num_stores=8, num_weeks=1, store_df=None):
        """
        Simulates forecast aggregation errors to demonstrate how forecasting at different levels affects accuracy.
        
        Args:
            num_stores (int): Number of stores to simulate
            num_weeks (int): Number of weeks to simulate
            store_df (pd.DataFrame, optional): Pre-defined DataFrame with Store, Expected Sales and Standard Deviation columns
            
        Returns:
            tuple: (store_df, simulation_results, comparison_data, run_history)
        """
        # If store_df is provided, use it for Expected Sales and Standard Deviation
        if store_df is not None:
            # Ensure store_df has the required columns
            if not all(col in store_df.columns for col in ['Store', 'Expected Sales', 'Standard Deviation']):
                raise ValueError("store_df must have 'Store', 'Expected Sales', and 'Standard Deviation' columns")
            
            # Make a copy to avoid modifying the original
            store_df = store_df.copy()
            
            # Ensure we have the right number of stores plus Total and Agg rows
            if len(store_df) != num_stores + 2:
                raise ValueError(f"store_df should have {num_stores + 2} rows (stores + Total and Agg)")
            
            # Extract the expected sales and standard deviations
            expected_sales = store_df['Expected Sales'].values
            std_devs = store_df['Standard Deviation'].values
        else:
            # Create a DataFrame for store simulations
            store_df = pd.DataFrame({
                'Store': list(range(1, num_stores + 1)) + ['Total', 'Agg']
            })

            # Generate expected sales (forecasts) for each store
            np.random.seed(42)  # For reproducibility
            expected_sales = []
            
            # Generate random expected sales for each store
            for _ in range(num_stores):
                expected_sales.append(np.random.randint(10, 65, 1)[0])
                
            # Add total and aggregated total (which are the same for expected)
            total_sales = sum(expected_sales)
            expected_sales.extend([total_sales, total_sales])
            store_df['Expected Sales'] = expected_sales
            
            # Calculate standard deviations for each store
            std_devs = []
            for i in range(num_stores):
                std_dev = np.random.choice([5, 8, 10, 15])
                std_devs.append(std_dev)
            
            # Add placeholder values for total and aggregated
            std_total = round(np.sqrt(sum([s**2 for s in std_devs])))
            std_agg = round(np.sqrt(sum([s**2 for s in std_devs])) / 2)  # Lower for aggregated
            std_devs.extend([std_total, std_agg])
            
            # Add to DataFrame
            store_df['Standard Deviation'] = std_devs

        # Generate actual sales with random variations based on expected sales and standard deviations
        actual_sales = []
        for i in range(num_stores):
            # Generate actual sales with noise using the provided standard deviation
            actual = max(0, np.random.normal(expected_sales[i], std_devs[i]))
            actual_sales.append(round(actual, 1))
            
        # Calculate total actual sales
        total_actual = sum(actual_sales)
        actual_sales.append(total_actual)
        actual_sales.append(total_actual)
        
        # Add to DataFrame
        store_df['Avg Weekly Sales'] = actual_sales
        
        # Calculate errors
        store_df['Avg Weekly Error'] = abs(store_df['Avg Weekly Sales'] - store_df['Expected Sales'])
        store_df['Avg Error %'] = round((store_df['Avg Weekly Error'] / store_df['Expected Sales']) * 100, 1)
        
        # Generate run history data
        run_history = []
        
        # Simulate multiple runs
        for run in range(1, 5):
            # Calculate total and aggregated error percentages
            total_avg_error = round(float(store_df.loc[store_df['Store'] == 'Total', 'Avg Error %']), 1)
            
            # For aggregated error, simulate how errors offset each other
            # Randomize whether some errors are positive or negative
            agg_avg_error = round(total_avg_error / np.random.uniform(3, 12), 1)
            
            run_history.append({
                'Run #': run,
                'Total Avg Error %': total_avg_error,
                'Agg Avg Error %': agg_avg_error
            })
        
        run_history_df = pd.DataFrame(run_history)
        
        return store_df, run_history_df

    def detect_date_format(self, series):
        """
        Attempts to detect the date format in a series by testing common formats.
        
        Args:
            series (pd.Series): The series to check for date format
            
        Returns:
            str or None: The detected date format string or None if not detected
        """
        # Skip if series is already datetime
        if pd.api.types.is_datetime64_dtype(series):
            return None
            
        # Common date formats to test
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',  # Standard formats
            '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y',  # Different separators
            '%d-%b-%Y', '%d-%B-%Y', '%b-%d-%Y', '%B-%d-%Y',  # With month names
            '%d-%b-%y', '%d-%B-%y', '%b-%d-%y', '%B-%d-%y',  # 2-digit years
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S',        # With time
            '%m/%d/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S'
        ]
        
        # Get a sample of non-null values
        sample = series.dropna().head(10).tolist()
        if not sample:
            return None
            
        # Try each format on the sample
        for date_format in date_formats:
            try:
                # Try to parse the whole sample with the format
                all_parsed = True
                for value in sample:
                    try:
                        pd.to_datetime(value, format=date_format)
                    except:
                        all_parsed = False
                        break
                
                if all_parsed:
                    return date_format
            except:
                continue
                
        return None
        
    def convert_datetime_columns(self, df=None, columns=None):
        """
        Intelligently converts columns to datetime format with proper format detection.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to process, uses self.df if None
            columns (list, optional): List of columns to convert, tries to detect if None
            
        Returns:
            pd.DataFrame: DataFrame with converted datetime columns
        """
        if df is None:
            df = self.df.copy()
        
        # If no columns specified, try to detect
        if columns is None:
            columns = []
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    # Try to convert to datetime and check if it works
                    datetime_col = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_col.isna().all() and datetime_col.isna().sum() / len(datetime_col) < 0.5:
                        columns.append(col)
        
        # Process each column
        for col in columns:
            if col in df.columns:
                # Detect the format
                date_format = self.detect_date_format(df[col])
                
                if date_format:
                    # Convert with the detected format
                    try:
                        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    except:
                        # Fallback to the default parser if format-specific parsing fails
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    # Use default parser if no format detected
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
        return df

    # Add Inventory Turnover Simulation section
    def inventory_turnover_simulation(self):
        """
        Visualize the rate at which inventory is sold based on turnover rates.
        
        Returns:
            None
        """
        # Initialize session state for simulation if not exists
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            st.session_state.inventory_company1 = 15  # Number of boxes in pyramid
            st.session_state.inventory_company2 = 15  # Number of boxes in pyramid
            st.session_state.last_update_time = None
        
        # Add How to Use instructions
        with st.expander("How to Use", expanded=True):
            st.markdown("""
            **What is Inventory Turnover?**
            Inventory turnover measures how many times a company sells and replaces its inventory in a given period. 
            A higher turnover rate indicates more efficient inventory management.
            
            **How to use this simulation:**
            1. **Adjust turnover rates** using the +/- buttons or input fields:
               - Higher turnover = faster inventory depletion
               - Default values: Company 1 (8), Company 2 (16)
            
            2. **View days inventory** values which show how many days inventory stays in stock:
               - Days Inventory = 365 / Turnover Rate
               - Lower days inventory = more efficient inventory management
            
            3. **Click START SELLING** to begin the simulation:
               - Watch as inventory boxes disappear in real-time
               - The company with higher turnover will deplete inventory faster
               - Click again to pause/reset the simulation
            
            4. **Compare companies** to see the impact of different turnover rates
               - Try making Company 1 twice as fast as Company 2
               - Observe how changing turnover affects inventory depletion speed
            """)
            
        # Create control inputs with increment/decrement buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Company 1")
            
            # Turnover rate for Company 1
            st.write("Turnover Company 1")
            turnover_col1_minus, turnover_col1_value, turnover_col1_plus = st.columns([1, 3, 1])
            with turnover_col1_minus:
                if st.button("-", key="turnover1_minus"):
                    if 'turnover_company1' in st.session_state and st.session_state.turnover_company1 > 1:
                        st.session_state.turnover_company1 -= 1
            
            with turnover_col1_value:
                if 'turnover_company1' not in st.session_state:
                    st.session_state.turnover_company1 = 8
                turnover_company1 = st.number_input("", 
                                                    min_value=1, 
                                                    max_value=100, 
                                                    value=st.session_state.turnover_company1,
                                                    key="turnover_company1_input",
                                                    label_visibility="collapsed")
                st.session_state.turnover_company1 = turnover_company1
            
            with turnover_col1_plus:
                if st.button("+", key="turnover1_plus"):
                    if 'turnover_company1' in st.session_state:
                        st.session_state.turnover_company1 += 1
            
            # Days Inventory for Company 1
            st.write("Days Inventory 1")
            days_col1_minus, days_col1_value, days_col1_plus = st.columns([1, 3, 1])
            with days_col1_minus:
                if st.button("-", key="days1_minus"):
                    if 'days_inventory1' in st.session_state and st.session_state.days_inventory1 > 1:
                        st.session_state.days_inventory1 -= 1
            
            with days_col1_value:
                if 'days_inventory1' not in st.session_state:
                    st.session_state.days_inventory1 = 46
                days_inventory1 = st.number_input("", 
                                                  min_value=1.0, 
                                                  max_value=365.0, 
                                                  value=float(st.session_state.days_inventory1),
                                                  key="days_inventory1_input",
                                                  label_visibility="collapsed")
                st.session_state.days_inventory1 = days_inventory1
            
            with days_col1_plus:
                if st.button("+", key="days1_plus"):
                    if 'days_inventory1' in st.session_state:
                        st.session_state.days_inventory1 += 1
        
        with col2:
            st.write("#### Company 2")
            
            # Turnover rate for Company 2
            st.write("Turnover Company 2")
            turnover_col2_minus, turnover_col2_value, turnover_col2_plus = st.columns([1, 3, 1])
            with turnover_col2_minus:
                if st.button("-", key="turnover2_minus"):
                    if 'turnover_company2' in st.session_state and st.session_state.turnover_company2 > 1:
                        st.session_state.turnover_company2 -= 1
            
            with turnover_col2_value:
                if 'turnover_company2' not in st.session_state:
                    st.session_state.turnover_company2 = 16
                turnover_company2 = st.number_input("", 
                                                    min_value=1, 
                                                    max_value=100, 
                                                    value=st.session_state.turnover_company2,
                                                    key="turnover_company2_input",
                                                    label_visibility="collapsed")
                st.session_state.turnover_company2 = turnover_company2
            
            with turnover_col2_plus:
                if st.button("+", key="turnover2_plus"):
                    if 'turnover_company2' in st.session_state:
                        st.session_state.turnover_company2 += 1
            
            # Days Inventory for Company 2
            st.write("Days Inventory 2")
            days_col2_minus, days_col2_value, days_col2_plus = st.columns([1, 3, 1])
            with days_col2_minus:
                if st.button("-", key="days2_minus"):
                    if 'days_inventory2' in st.session_state and st.session_state.days_inventory2 > 1:
                        st.session_state.days_inventory2 -= 1
            
            with days_col2_value:
                if 'days_inventory2' not in st.session_state:
                    st.session_state.days_inventory2 = 22.8
                days_inventory2 = st.number_input("", 
                                                  min_value=1.0, 
                                                  max_value=365.0, 
                                                  value=float(st.session_state.days_inventory2),
                                                  key="days_inventory2_input",
                                                  label_visibility="collapsed")
                st.session_state.days_inventory2 = days_inventory2
            
            with days_col2_plus:
                if st.button("+", key="days2_plus"):
                    if 'days_inventory2' in st.session_state:
                        st.session_state.days_inventory2 += 1
        
        # START SELLING button
        start_button_style = """
        <style>
        div.stButton > button {
            background-color: #4B70E2;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        </style>
        """
        st.markdown(start_button_style, unsafe_allow_html=True)
        
        if st.button("START SELLING", key="start_selling"):
            st.session_state.simulation_running = not st.session_state.simulation_running
            # Reset inventory if simulation was off and turning on
            if st.session_state.simulation_running:
                st.session_state.inventory_company1 = 15
                st.session_state.inventory_company2 = 15
                st.session_state.last_update_time = time.time()
        
        # Display the inventory visualization
        col1, col2 = st.columns(2)
        
        # Calculate inventory levels based on turnover rates if simulation is running
        if st.session_state.simulation_running and st.session_state.last_update_time is not None:
            current_time = time.time()
            elapsed_time = current_time - st.session_state.last_update_time
            
            # Scale elapsed time for faster visualization (1 second = 1 day)
            scaled_time = elapsed_time * 10
            
            # Calculate boxes to remove based on turnover rates
            company1_rate = st.session_state.turnover_company1 / 365  # Daily sales rate
            company2_rate = st.session_state.turnover_company2 / 365  # Daily sales rate
            
            # Calculate inventory to remove
            company1_remove = scaled_time * company1_rate 
            company2_remove = scaled_time * company2_rate
            
            # Update inventory levels
            st.session_state.inventory_company1 = max(0, st.session_state.inventory_company1 - company1_remove)
            st.session_state.inventory_company2 = max(0, st.session_state.inventory_company2 - company2_remove)
            
            # Update last update time
            st.session_state.last_update_time = current_time
        
        # Function to create pyramid visualization
        def create_pyramid(inventory_left, max_inventory=15, color="#3178c6"):
            # Calculate how many boxes to show based on inventory left
            boxes_to_show = math.ceil(inventory_left)
            boxes_to_show = min(boxes_to_show, max_inventory)  # Cap at max inventory
            
            # Create HTML for pyramid
            pyramid_html = "<div style='display: flex; flex-direction: column; align-items: center;'>"
            
            # Each row of the pyramid
            boxes_per_row = [1, 2, 3, 4, 5]  # 5 rows with increasing boxes
            boxes_used = 0
            
            for row_boxes in boxes_per_row:
                row_html = "<div style='display: flex; flex-direction: row;'>"
                for i in range(row_boxes):
                    if boxes_used < boxes_to_show:
                        # Box is visible
                        row_html += f"<div style='width: 30px; height: 30px; margin: 2px; background-color: {color};'></div>"
                    else:
                        # Box is invisible (removed)
                        row_html += "<div style='width: 30px; height: 30px; margin: 2px;'></div>"
                    boxes_used += 1
                row_html += "</div>"
                pyramid_html += row_html
            
            pyramid_html += "</div>"
            return pyramid_html
        
        # Display company names
        with col1:
            st.markdown("<h4 style='text-align: center;'>Company 1</h4>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4 style='text-align: center;'>Company 2</h4>", unsafe_allow_html=True)
        
        # Display pyramids
        with col1:
            company1_pyramid = create_pyramid(st.session_state.inventory_company1)
            st.markdown(company1_pyramid, unsafe_allow_html=True)
        
        with col2:
            company2_pyramid = create_pyramid(st.session_state.inventory_company2)
            st.markdown(company2_pyramid, unsafe_allow_html=True)
        
        # Add auto-refresh for animation (every 200ms)
        if st.session_state.simulation_running:
            st.empty()
            time.sleep(0.1)
            st.rerun()

def main():
    st.title("AI-Powered Data Cleaning Agent 🖨️💎🛁📊🤖")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                st.warning("Duplicate column names detected in the dataset. Renaming columns to make them unique.")
                
                # Find and rename duplicate columns
                cols = pd.Series(df.columns)
                for dup in cols[cols.duplicated()].unique(): 
                    cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup 
                                                                    for i in range(sum(cols == dup))]
                df.columns = cols
                
                st.info("Columns have been renamed. Duplicate columns now have suffixes (_1, _2, etc.)")

            # Create a DataCleaner instance
            cleaner = DataCleaner(df)
            
            # Convert date columns to proper datetime format
            df = cleaner.convert_datetime_columns(df)
            
            # Ensure data is Arrow-compatible for display
            df_display = cleaner.ensure_arrow_compatible(df)
            
            st.write("Original DataFrame:")
            safe_display_dataframe(df, cleaner)

            # Data Cleaning
            cleaner = DataCleaner(df)

            # Sidebar for cleaning options
            st.sidebar.title("Cleaning Options")

            # Basic Cleaning
            if st.sidebar.checkbox("Basic Cleaning"):
                st.write("### Basic Cleaning")
                
                if st.checkbox("Standardize Column Names (lowercase, spaces to underscores)?"):
                    cleaner.standardize_column_names()

                # Missing Values
                missing_cols = df.columns[df.isnull().any()].tolist()
                if missing_cols:
                    st.write("#### Handle Missing Values")
                    selected_cols = st.multiselect("Select columns with missing values to handle:", missing_cols)
                    if selected_cols:
                        missing_strategy = st.selectbox("Choose imputation strategy:", ['mean', 'median', 'mode', 'drop'])
                        cleaner.handle_missing_values(strategy=missing_strategy, columns=selected_cols)

                if st.checkbox("Remove Duplicate Rows?"):
                    cleaner.remove_duplicates()

            # AI/ML Features
            if st.sidebar.checkbox("AI/ML Features"):
                st.write("### AI/ML Features")

                # Outlier Detection
                if st.checkbox("Detect Outliers"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:  # Check if there are any numeric columns
                        contamination = st.slider("Contamination factor:", 0.01, 0.5, 0.1, 0.01)
                        outliers = cleaner.detect_outliers(contamination=contamination)
                        if outliers is not None and not outliers.empty:
                            st.write("#### Outlier Rows:")
                            st.dataframe(outliers)
                    else:
                        st.warning("No numeric columns available for outlier detection.")

                # Feature Engineering
                if st.checkbox("Encode Categorical Variables"):
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if len(categorical_cols) > 0:
                        selected_cats = st.multiselect("Select categorical columns to encode:", categorical_cols)
                        if selected_cats:
                            cleaner.encode_categorical(selected_cats)
                    else:
                        st.warning("No categorical columns available for encoding.")

                if st.checkbox("Scale Numeric Features"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:
                        st.write("##### Scale Numeric Features")
                        st.write("Scaling transforms values to be in similar ranges, which helps many ML algorithms perform better.")
                        
                        # Add scaler type selection
                        scaler_type = st.selectbox(
                            "Select scaling method:", 
                            ["standard", "minmax", "robust"],
                            format_func=lambda x: {
                                "standard": "Standard (Z-score scaling, centers around mean of 0 and std of 1)",
                                "minmax": "Min-Max (Scales to range 0-1, preserves distribution shape)",
                                "robust": "Robust (Uses median/IQR, robust to outliers)"
                            }[x],
                            help="Choose the scaling technique based on your data characteristics and analysis needs."
                        )

                        # Show explanation of selected method
                        if scaler_type == "standard":
                            st.info("Standard scaling subtracts mean and divides by standard deviation. Best for data with normal distribution.")
                        elif scaler_type == "minmax":
                            st.info("Min-Max scaling transforms values to 0-1 range. Good when you need bounded values.")
                        else:  # robust
                            st.info("Robust scaling uses median and IQR instead of mean/std. Ideal when data contains outliers.")
                        
                        # Get numeric columns with flags for potential ID columns
                        id_flags = {}
                        for col in numeric_cols:
                            is_id, reason = cleaner.is_likely_id_column(col)
                            if is_id:
                                id_flags[col] = reason
                        
                        # Display numeric columns with warnings for ID columns
                        st.write("#### Available Numeric Columns:")
                        
                        # Create a dataframe for better display
                        col_info = []
                        for col in numeric_cols:
                            # Check if it's an ID column
                            is_id = col in id_flags
                            # Get sample values and summary stats
                            sample_vals = df[col].dropna().head(3).tolist()
                            sample_str = ", ".join([str(val) for val in sample_vals])
                            
                            # Stats
                            try:
                                mean_val = df[col].mean()
                                min_val = df[col].min()
                                max_val = df[col].max()
                                stats = f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}"
                            except:
                                stats = "Stats unavailable"
                                
                            col_info.append({
                                "Column": col,
                                "Type": str(df[col].dtype),
                                "Is ID": "⚠️ Yes" if is_id else "No",
                                "Sample Values": sample_str,
                                "Statistics": stats
                            })
                            
                        # Display as a dataframe
                        col_df = pd.DataFrame(col_info)
                        st.dataframe(col_df)
                        
                        # Display ID column warnings
                        if id_flags:
                            st.warning("⚠️ Some columns may be identifiers (IDs) which typically shouldn't be scaled:")
                            for col, reason in id_flags.items():
                                st.markdown(f"- **{col}**: {reason}")
                        
                        # Column selection with ID columns highlighted
                        options = []
                        for col in numeric_cols:
                            if col in id_flags:
                                options.append(f"⚠️ {col} (ID)")
                            else:
                                options.append(col)
                        
                        # Let user select columns to scale
                        selected_opts = st.multiselect("Select numeric columns to scale:", options)
                        
                        # Convert option strings back to actual column names
                        selected_nums = [opt.split(" (ID)")[0].replace("⚠️ ", "") for opt in selected_opts]
                        
                        if selected_nums:
                            if st.button("Apply Scaling"):
                                cleaner.scale_features(selected_nums, scaler_type)
                    else:
                        st.warning("No numeric columns available for scaling.")

                # Basic ML Model
                if st.checkbox("Train Basic ML Model"):
                    target_col = st.selectbox("Select target column:", df.columns.tolist())
                    problem_type = st.selectbox("Select problem type:", ['classification', 'regression'])
                    
                    # Add model selection based on problem type
                    if problem_type == 'classification':
                        model_type = st.selectbox(
                            "Select classification model:",
                            [
                                'random_forest', 'decision_tree', 'knn', 'logistic_regression', 
                                'adaboost', 'svm', 'lda', 'naive_bayes'
                            ],
                            format_func=lambda x: {
                                'random_forest': 'Random Forest (ensemble of trees)',
                                'decision_tree': 'Decision Tree (simple, interpretable)',
                                'knn': 'K-Nearest Neighbors (distance-based)',
                                'logistic_regression': 'Logistic Regression (linear classifier)',
                                'adaboost': 'AdaBoost (boosting algorithm)',
                                'svm': 'Support Vector Machine (margin maximizer)',
                                'lda': 'Linear Discriminant Analysis (dimensionality reduction)',
                                'naive_bayes': 'Naive Bayes (probabilistic classifier)'
                            }[x]
                        )
                    else:
                        # For regression, keep it simple for now
                        model_type = 'random_forest'
                    
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            metrics = cleaner.train_model(target_col, problem_type, model_type=model_type)
                            if metrics:
                                st.write("#### Model Performance:")
                                for metric, value in metrics.items():
                                    if metric not in ['feature_importance', 'classification_report']:
                                        st.write(f"{metric}: {value:.4f}")
                                
                                # Display classification report if available
                                if 'classification_report' in metrics:
                                    st.write("#### Classification Report:")
                                    
                                    # Create a DataFrame from the classification report
                                    report = metrics['classification_report']
                                    report_df = pd.DataFrame(report).transpose()
                                    
                                    # Filter out unnecessary rows
                                    if 'accuracy' in report_df.index:
                                        report_df = report_df.drop('accuracy')
                                    
                                    st.dataframe(report_df)
                                
                                # Display feature importance if available
                                if 'feature_importance' in metrics:
                                    st.write("#### Feature Importance:")
                                    importance_df = pd.DataFrame.from_dict(
                                        metrics['feature_importance'], 
                                        orient='index', 
                                        columns=['importance']
                                    ).sort_values('importance', ascending=False)
                                    
                                    # Display as a bar chart
                                    if PLOTLY_AVAILABLE:
                                        fig = px.bar(
                                            importance_df, 
                                            x=importance_df.index, 
                                            y='importance',
                                            title="Feature Importance"
                                        )
                                        st.plotly_chart(fig)
                                    else:
                                        st.bar_chart(importance_df)

            # Datatype Conversion
            if st.sidebar.checkbox("Convert Data Types"):
                st.write("### Convert Data Types")
                column_types = {}
                for col in df.columns:
                    dtype = st.selectbox(
                        f"Convert '{col}' to:", 
                        [None, 'int', 'float', 'category', 'str'],
                        key=f"dtype_{col}"
                    )
                    if dtype:
                        column_types[col] = dtype
                if column_types:
                    cleaner.convert_datatypes(column_types)

            # Add Data Interpretation section
            if st.sidebar.checkbox("Data Interpretation"):
                cleaner.interpret_data()

            # Add Data Visualization section
            if st.sidebar.checkbox("Data Visualization"):
                st.write("### Data Visualization")
                
                # Visualization type selection
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Categorical Distribution", "Distribution Plot", "Box Plot", "Correlation", "Scatter Plot"]
                )
                
                if viz_type == "Categorical Distribution":
                    st.write("#### Categorical Distribution")
                    st.write("Visualize the distribution of values in a categorical column.")
                    
                    # Column selection based on data types with info
                    all_cols = df.columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Add columns with low cardinality from numeric columns
                    for col in numeric_cols:
                        if df[col].nunique() < 20:  # Low cardinality numeric columns can be treated as categorical
                            categorical_cols.append(col)
                    
                    # Remove duplicates and sort
                    categorical_cols = sorted(list(set(categorical_cols)))
                    
                    if not categorical_cols:
                        st.warning("No categorical columns detected. Please select any column to visualize:")
                        viz_col = st.selectbox("Select column:", all_cols)
                    else:
                        viz_col = st.selectbox("Select categorical column:", categorical_cols)
                    
                    # Visualization method selection with descriptions
                    plot_method = st.selectbox(
                        "Select visualization method:",
                        ["bar", "treemap", "lollipop", "pie"],
                        format_func=lambda x: {
                            "bar": "Bar Chart (best for precise comparison)",
                            "treemap": "Treemap (good for hierarchical proportions)",
                            "lollipop": "Lollipop Chart (elegant alternative to bars)",
                            "pie": "Pie Chart (shows parts of a whole, limited accuracy)"
                        }[x]
                    )
                    
                    # Preview of first few values
                    st.write(f"Preview of '{viz_col}' values:")
                    value_counts = df[viz_col].value_counts().head(5)
                    preview_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(preview_df)
                    
                    # Apply visualization
                    if st.button("Generate Visualization"):
                        cleaner.plot_categorical_distribution(viz_col, plot_method)
                
                elif viz_type == "Distribution Plot":
                    st.write("#### Distribution Plot")
                    st.write("Visualize the distribution of a numeric column.")
                    
                    # Only show numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        st.error("No numeric columns available for distribution plots.")
                    else:
                        viz_col = st.selectbox("Select numeric column:", numeric_cols)
                        cleaner.plot_distribution(viz_col)
                
                elif viz_type == "Box Plot":
                    st.write("#### Box Plot")
                    st.write("Visualize the distribution and identify outliers in numeric data.")
                    
                    # Only show numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        st.error("No numeric columns available for box plots.")
                    else:
                        viz_col = st.selectbox("Select numeric column:", numeric_cols)
                        
                        # Option to group by a categorical column
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if categorical_cols:
                            use_groups = st.checkbox("Group by categorical variable?")
                            if use_groups:
                                group_col = st.selectbox("Select grouping column:", categorical_cols)
                                
                                # Check if the grouping column has too many categories
                                n_categories = df[group_col].nunique()
                                if n_categories > 10:
                                    st.warning(f"Selected grouping column has {n_categories} categories, which may make the plot cluttered.")
                                    
                                    # Offer to show only top categories
                                    use_top = st.checkbox("Show only top categories?", value=True)
                                    if use_top:
                                        top_n = st.slider("Number of top categories:", 3, 10, 5)
                                        top_cats = df[group_col].value_counts().nlargest(top_n).index.tolist()
                                        filtered_df = df[df[group_col].isin(top_cats)]
                                        
                                        # Create boxplot with filtered data
                                        if PLOTLY_AVAILABLE:
                                            fig = px.box(filtered_df, x=group_col, y=viz_col, 
                                                       title=f"Box Plot of {viz_col} by {group_col} (Top {top_n} categories)")
                                            st.plotly_chart(fig)
                                        else:
                                            fig, ax = plt.subplots(figsize=(12, 6))
                                            sns.boxplot(x=group_col, y=viz_col, data=filtered_df, ax=ax)
                                            plt.title(f"Box Plot of {viz_col} by {group_col} (Top {top_n} categories)")
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close()
                                    else:
                                        cleaner.plot_box_plot(viz_col, group_col)
                                else:
                                    cleaner.plot_box_plot(viz_col, group_col)
                            else:
                                cleaner.plot_box_plot(viz_col)
                        else:
                            cleaner.plot_box_plot(viz_col)
                
                elif viz_type == "Correlation":
                    st.write("#### Correlation Analysis")
                    st.write("Visualize relationships between numeric variables.")
                    
                    # Option for correlation type
                    corr_type = st.radio(
                        "Select correlation visualization:",
                        ["Heatmap", "Pairplot"], 
                        horizontal=True
                    )
                    
                    if corr_type == "Heatmap":
                        # Option to filter variables
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(numeric_cols) > 10:
                            st.warning(f"Dataset has {len(numeric_cols)} numeric columns. Showing all may create a cluttered heatmap.")
                            use_filter = st.checkbox("Select specific columns?", value=True)
                            
                            if use_filter:
                                selected_cols = st.multiselect(
                                    "Choose columns for correlation analysis:",
                                    numeric_cols,
                                    default=numeric_cols[:6]
                                )
                                
                                if selected_cols:
                                    if PLOTLY_AVAILABLE:
                                        corr = df[selected_cols].corr()
                                        fig = px.imshow(
                                            corr,
                                            text_auto='.2f',
                                            color_continuous_scale='RdBu_r',
                                            zmin=-1, zmax=1,
                                            title="Correlation Heatmap"
                                        )
                                        st.plotly_chart(fig)
                                    else:
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        corr = df[selected_cols].corr()
                                        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                                        plt.title("Correlation Heatmap")
                                        st.pyplot(fig)
                                        plt.close()
                                else:
                                    st.info("Please select at least one column.")
                            else:
                                cleaner.plot_correlation_heatmap()
                        else:
                            cleaner.plot_correlation_heatmap()
                    
                    elif corr_type == "Pairplot":
                        # For pairplot, limit to fewer variables to avoid visual overload
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(numeric_cols) < 2:
                            st.error("Need at least 2 numeric columns for pairplot.")
                        else:
                            # Select a subset of columns
                            max_cols = min(6, len(numeric_cols))
                            selected_cols = st.multiselect(
                                "Choose columns for pairplot (recommended: 2-6 columns):",
                                numeric_cols,
                                default=numeric_cols[:min(4, max_cols)]
                            )
                            
                            # Option to color by a categorical variable
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            color_by = None
                            
                            if categorical_cols:
                                use_color = st.checkbox("Color by categorical variable?")
                                if use_color:
                                    color_by = st.selectbox("Select coloring variable:", categorical_cols)
                                    
                                    # Check for too many categories
                                    n_categories = df[color_by].nunique()
                                    if n_categories > 10:
                                        st.warning(f"Selected coloring variable has {n_categories} categories, which may make the plot unclear.")
                                        use_top = st.checkbox("Use only top categories for coloring?", value=True)
                                        
                                        if use_top:
                                            top_n = st.slider("Number of top categories:", 3, 10, 5)
                                            top_cats = df[color_by].value_counts().nlargest(top_n).index.tolist()
                                            
                                            # Create pairplot with filtered data
                                            if len(selected_cols) >= 2:
                                                filtered_df = df[df[color_by].isin(top_cats)]
                                                if len(filtered_df) > 0:
                                                    fig = sns.pairplot(
                                                        filtered_df,
                                                        vars=selected_cols,
                                                        hue=color_by,
                                                        diag_kind='kde',
                                                        plot_kws={'alpha': 0.6}
                                                    )
                                                    st.pyplot(fig)
                                                    plt.close()
                                                else:
                                                    st.error("No data available with the selected categories.")
                                            else:
                                                st.info("Please select at least 2 columns for the pairplot.")
                                        else:
                                            # Create regular pairplot with all categories
                                            if len(selected_cols) >= 2:
                                                fig = sns.pairplot(
                                                    df,
                                                    vars=selected_cols, 
                                                    hue=color_by,
                                                    diag_kind='kde',
                                                    plot_kws={'alpha': 0.6}
                                                )
                                                st.pyplot(fig)
                                                plt.close()
                                            else:
                                                st.info("Please select at least 2 columns for the pairplot.")
                                    else:
                                        # Create pairplot with manageable categories
                                        if len(selected_cols) >= 2:
                                            fig = sns.pairplot(
                                                df, 
                                                vars=selected_cols, 
                                                hue=color_by,
                                                diag_kind='kde',
                                                plot_kws={'alpha': 0.6}
                                            )
                                            st.pyplot(fig)
                                            plt.close()
                                        else:
                                            st.info("Please select at least 2 columns for the pairplot.")
                            else:
                                # Create pairplot without categorical coloring
                                if len(selected_cols) >= 2:
                                    fig = sns.pairplot(
                                        df, 
                                        vars=selected_cols,
                                        diag_kind='kde',
                                        plot_kws={'alpha': 0.6}
                                    )
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.info("Please select at least 2 columns for the pairplot.")
                                    
                elif viz_type == "Scatter Plot":
                    st.write("#### Scatter Plot")
                    st.write("Visualize relationship between two numeric variables.")
                    
                    # Only show numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for scatter plot.")
                    else:
                        x_col = st.selectbox("Select X-axis column:", numeric_cols)
                        y_col = st.selectbox("Select Y-axis column:", 
                                          [col for col in numeric_cols if col != x_col], 
                                          index=min(1, len(numeric_cols)-1))
                        
                        # Optional color by categorical variable
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        color_by = None
                        size_by = None
                        
                        # Advanced options
                        with st.expander("Advanced Options"):
                            if categorical_cols:
                                use_color = st.checkbox("Color points by category?")
                                if use_color:
                                    color_by = st.selectbox("Select coloring variable:", categorical_cols)
                            
                            # Size points by numeric value
                            use_size = st.checkbox("Size points by numeric value?")
                            if use_size:
                                size_options = [col for col in numeric_cols if col not in [x_col, y_col]]
                                if size_options:
                                    size_by = st.selectbox("Select sizing variable:", size_options)
                                else:
                                    st.info("No additional numeric columns available for sizing.")
                                    
                            # Add trendline option
                            add_trendline = st.checkbox("Add trendline?", value=True)
                            
                            # Add jitter for categorical x or y
                            add_jitter = False
                            if (df[x_col].nunique() < 20 or df[y_col].nunique() < 20):
                                add_jitter = st.checkbox("Add jitter to points?", 
                                                       help="Adds random noise to prevent overplotting with repeated values")
                                
                        # Create the scatter plot
                        if PLOTLY_AVAILABLE:
                            # Build scatter plot with Plotly
                            scatter_kwargs = {
                                'x': x_col,
                                'y': y_col,
                                'title': f'Scatter Plot of {y_col} vs {x_col}',
                                'labels': {x_col: x_col, y_col: y_col},
                                'hover_data': [x_col, y_col]
                            }
                            
                            if color_by:
                                scatter_kwargs['color'] = color_by
                            
                            if size_by:
                                scatter_kwargs['size'] = size_by
                                scatter_kwargs['size_max'] = 15
                            
                            # Create figure
                            fig = px.scatter(**scatter_kwargs, data_frame=df)
                            
                            # Add trendline if requested
                            if add_trendline:
                                if color_by:
                                    fig.update_layout(title=f'Scatter Plot with trendlines grouped by {color_by}')
                                    fig = px.scatter(
                                        df, x=x_col, y=y_col, color=color_by,
                                        trendline="ols", trendline_scope="overall" if len(df[color_by].unique()) > 5 else "trace",
                                        trendline_color_override="black"
                                    )
                                else:
                                    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols")
                                    
                            st.plotly_chart(fig)
                        else:
                            # Matplotlib fallback
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            scatter_kwargs = {'x': x_col, 'y': y_col, 'data': df, 'alpha': 0.7}
                            
                            if color_by:
                                scatter_kwargs['hue'] = color_by
                            
                            if size_by:
                                scatter_kwargs['size'] = size_by
                            
                            if add_jitter:
                                scatter_kwargs['x_jitter'] = True
                                scatter_kwargs['y_jitter'] = True
                                
                            # Create scatter plot
                            sns.scatterplot(**scatter_kwargs, ax=ax)
                            
                            # Add trendline if requested
                            if add_trendline:
                                if color_by and df[color_by].nunique() <= 5:
                                    # Add trendline for each category
                                    for category, group in df.groupby(color_by):
                                        sns.regplot(x=x_col, y=y_col, data=group, 
                                                  scatter=False, ax=ax, label=f"Trend: {category}")
                                else:
                                    # Add overall trendline
                                    sns.regplot(x=x_col, y=y_col, data=df, 
                                              scatter=False, ax=ax, line_kws={'color': 'black'})
                            
                            plt.title(f'Scatter Plot of {y_col} vs {x_col}')
                            plt.xlabel(x_col)
                            plt.ylabel(y_col)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                
                # Correlation heatmap for numeric columns
                if st.checkbox("Show Correlation Heatmap"):
                    cleaner.plot_correlation_heatmap()

            # Advanced Data Cleaning
            if st.sidebar.checkbox("Advanced Data Cleaning"):
                st.write("### Advanced Data Cleaning")
                
                # Advanced Imputation
                if st.checkbox("Advanced Missing Value Imputation"):
                    imputation_strategy = st.selectbox(
                        "Choose imputation strategy:",
                        ['knn', 'iterative', 'multivariate']
                    )
                    n_neighbors = st.slider("Number of neighbors (for KNN):", 1, 20, 5)
                    if st.button("Apply Advanced Imputation"):
                        cleaner.advanced_imputation(
                            strategy=imputation_strategy,
                            n_neighbors=n_neighbors
                        )

                # Feature Transformation
                if st.checkbox("Transform Skewed Features"):
                    skew_threshold = st.slider("Skewness threshold:", 0.0, 2.0, 0.5)
                    transform_method = st.selectbox(
                        "Choose transformation method:",
                        ['yeo-johnson', 'box-cox', 'log']
                    )
                    if st.button("Apply Transformation"):
                        cleaner.transform_skewed_features(
                            threshold=skew_threshold,
                            method=transform_method
                        )

                # High Cardinality Handling
                if st.checkbox("Handle High-Cardinality Features"):
                    max_cats = st.slider("Maximum categories to keep:", 5, 50, 10)
                    encoding_method = st.selectbox(
                        "Choose encoding method:",
                        ['frequency', 'grouping', 'target_encoding']
                    )
                    target_col = None
                    if encoding_method == 'target_encoding':
                        target_col = st.selectbox("Select target column:", df.columns)
                    if st.button("Apply Encoding"):
                        cleaner.handle_high_cardinality(
                            max_categories=max_cats,
                            method=encoding_method,
                            target_column=target_col
                        )

            # Add Time Series Analysis section
            if st.sidebar.checkbox("Time Series Analysis"):
                st.write("### Time Series Analysis")
                
                # Check for datetime columns with better format detection
                date_columns = []
                for col in df.columns:
                    # Sample the first few non-null values
                    sample_vals = df[col].dropna().head(5).tolist()
                    if len(sample_vals) == 0:
                        continue
                        
                    # Try to detect if this might be a date column
                    might_be_date = False
                    if all(isinstance(val, str) for val in sample_vals):
                        # Check for common date separators
                        if any('-' in str(val) or '/' in str(val) or ':' in str(val) for val in sample_vals):
                            might_be_date = True
                    
                    # Only try to convert columns that might be dates
                    if might_be_date or pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            # Try conversion without warnings
                            with pd.option_context('mode.chained_assignment', None):
                                pd.to_datetime(df[col], errors='coerce')
                            date_columns.append(col)
                        except:
                            pass
                
                if date_columns:
                    # Get numeric columns for value selection
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_cols:
                        st.error("No numeric columns available for time series analysis. Please ensure your dataset contains numeric values to analyze.")
                        return
                    
                    date_col = st.selectbox("Select date column:", date_columns)
                    
                    # Show description of what makes a good value column
                    st.info("Select a numeric column to analyze over time (e.g., sales amount, quantity, price, etc.)")
                    value_col = st.selectbox(
                        "Select value column:", 
                        numeric_cols,
                        help="Choose a numeric column that you want to analyze over time. This should be a measurable value like sales, quantity, or metrics."
                    )
                    
                    # Add interactive plot option if plotly is available
                    if PLOTLY_AVAILABLE:
                        use_interactive = st.checkbox("Use interactive plots", value=True, 
                                                        help="Enable interactive plots for better exploration")
                    else:
                        use_interactive = False
                    
                    # Add frequency selection with descriptions
                    st.write("#### Select Time Series Frequency")
                    st.info("Choose the time frequency that best matches your data and analysis needs:")
                    frequency = st.selectbox(
                        "Select frequency:", 
                        ['D', 'W', 'M', 'Q', 'YE'], 
                        format_func=lambda x: {
                            'D': 'Daily (for day-by-day analysis)',
                            'W': 'Weekly (for week-by-week patterns)',
                            'M': 'Monthly (for monthly trends)',
                            'Q': 'Quarterly (for quarterly performance)',
                            'YE': 'Yearly (for annual patterns)'
                        }[x]
                    )
                    
                    # Show data preview before analysis
                    st.write("#### Data Preview")
                    preview_df = df[[date_col, value_col]].head()
                    st.write("First few rows of selected data:")
                    st.dataframe(preview_df)
                    
                    if st.button("Analyze Time Series"):
                        if pd.api.types.is_numeric_dtype(df[value_col]):
                            cleaner.analyze_time_series(date_col, value_col, frequency, use_interactive)
                        else:
                            st.error(f"Selected value column '{value_col}' must contain numeric data. Please choose a different column.")
                else:
                    st.warning("No datetime columns detected in the dataset. Please ensure you have a column containing dates.")

            # Add Advanced Data Analysis section
            if st.sidebar.checkbox("Advanced Data Analysis"):
                st.write("### Advanced Data Analysis")
                
                analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Automated EDA", "Statistical Tests", "Clustering", "Feature Selection"]
                )
                
                if analysis_type == "Automated EDA":
                    if st.button("Generate Automated EDA"):
                        with st.spinner("Analyzing your data..."):
                            eda_results = cleaner.automated_eda()
                            
                            # Display dataset overview
                            st.write("#### Dataset Overview")
                            st.write(f"Shape: {eda_results['dataset_shape'][0]} rows, {eda_results['dataset_shape'][1]} columns")
                            st.write(f"Memory usage: {eda_results['memory_usage']:.2f} MB")
                            
                            # Display missing values info
                            st.write("#### Missing Values Analysis")
                            missing = eda_results['missing_values']
                            if missing['total_missing_cells'] > 0:
                                st.write(f"Total missing: {missing['total_missing_cells']} cells ({missing['percent_missing']:.2f}% of all data)")
                                st.write("Columns with missing values:")
                                missing_df = pd.DataFrame({
                                    'Count': missing['counts'],
                                    'Percentage': missing['percentages']
                                })
                                st.dataframe(missing_df)
                            else:
                                st.success("No missing values found in the dataset!")
                            
                            # Display numeric column analysis
                            if 'numeric_summary' in eda_results:
                                st.write("#### Numeric Columns Summary")
                                st.dataframe(eda_results['numeric_summary'])
                                
                                # Display skewed features
                                if eda_results['skewed_features']:
                                    st.write("#### Skewed Features")
                                    st.write("The following numeric features have high skewness (>1.0):")
                                    skew_df = pd.DataFrame.from_dict(eda_results['skewed_features'], orient='index', columns=['Skewness'])
                                    st.dataframe(skew_df)
                                    st.info("Consider applying transformations (log, sqrt, etc.) to these features.")
                                
                                # Display outliers info
                                if eda_results['outliers']:
                                    st.write("#### Outlier Detection")
                                    st.write("The following columns have potential outliers:")
                                    for col, info in eda_results['outliers'].items():
                                        st.write(f"- **{col}**: {info['count']} outliers ({info['percentage']:.2f}% of data)")
                                        st.write(f"  Range: {info['min']} to {info['max']}")
                                
                                # Display correlation info
                                if 'high_correlations' in eda_results and eda_results['high_correlations']:
                                    st.write("#### High Correlations")
                                    st.write("The following feature pairs have high correlation (>0.7):")
                                    corr_df = pd.DataFrame.from_dict(eda_results['high_correlations'], orient='index', columns=['Correlation'])
                                    st.dataframe(corr_df)
                                    st.info("Consider removing one feature from each highly correlated pair to reduce dimensionality.")
                            
                            # Display categorical column analysis
                            if 'categorical_summary' in eda_results:
                                st.write("#### Categorical Columns Analysis")
                                for col, info in eda_results['categorical_summary'].items():
                                    st.write(f"**{col}**: {info['unique_values']} unique values")
                                    if info['high_cardinality']:
                                        st.warning(f"High cardinality detected for '{col}'")
                                    st.write("Top categories:")
                                    cat_df = pd.DataFrame(list(info['top_categories'].items()), columns=['Value', 'Count'])
                                    st.dataframe(cat_df)
                            
                            # Display date column analysis
                            if 'date_analysis' in eda_results:
                                st.write("#### Date Columns Analysis")
                                for col, info in eda_results['date_analysis'].items():
                                    st.write(f"**{col}**:")
                                    st.write(f"- Range: {info['min_date']} to {info['max_date']} ({info['range_days']} days)")
                                    st.write(f"- Weekdays: {info['weekday_days']}, Weekends: {info['weekend_days']}")
                                    if info['null_dates'] > 0:
                                        st.write(f"- Missing dates: {info['null_dates']}")
                
                elif analysis_type == "Statistical Tests":
                    st.write("#### Statistical Hypothesis Testing")
                    
                    test_type = st.selectbox(
                        "Select test type:",
                        ["t-test", "ANOVA", "Chi-Square Test", "Correlation Test"],
                        format_func=lambda x: {
                            "t-test": "t-test (compare means)",
                            "ANOVA": "ANOVA (compare means across groups)",
                            "Chi-Square Test": "Chi-Square Test (categorical association)",
                            "Correlation Test": "Correlation Test (numeric relationship)"
                        }[x]
                    )
                    
                    # Column selection based on test type
                    if test_type == "t-test":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) == 0:
                            st.error("No numeric columns available for t-test")
                            return
                            
                        col1 = st.selectbox("Select first numeric column:", numeric_cols)
                        col2 = st.selectbox("Select second numeric column (optional):", ["None"] + numeric_cols)
                        col2 = None if col2 == "None" else col2
                        
                        if st.button("Run t-test"):
                            with st.spinner("Running test..."):
                                try:
                                    result = cleaner.perform_hypothesis_test(col1, col2, 'ttest')
                                    
                                    st.write(f"#### {result['test_name']}")
                                    st.write(result['description'])
                                    st.write(f"t-statistic: {result['statistic']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    
                                    if result['p_value'] < 0.05:
                                        st.success(f"Result: **{result['interpretation']}** (p < 0.05)")
                                    else:
                                        st.info(f"Result: **{result['interpretation']}** (p ≥ 0.05)")
                                        
                                except Exception as e:
                                    st.error(f"Error running t-test: {str(e)}")
                    
                    elif test_type == "ANOVA":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if len(numeric_cols) == 0 or len(cat_cols) == 0:
                            st.error("ANOVA requires at least one numeric column and one categorical column")
                            return
                            
                        target_col = st.selectbox("Select numeric column (dependent variable):", numeric_cols)
                        group_col = st.selectbox("Select categorical column (grouping variable):", cat_cols)
                        
                        if st.button("Run ANOVA"):
                            with st.spinner("Running ANOVA..."):
                                try:
                                    result = cleaner.perform_hypothesis_test(target_col, group_col, 'anova')
                                    
                                    st.write(f"#### {result['test_name']}")
                                    st.write(result['description'])
                                    st.write(f"F-statistic: {result['statistic']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    
                                    if result['p_value'] < 0.05:
                                        st.success(f"Result: **{result['interpretation']}** (p < 0.05)")
                                    else:
                                        st.info(f"Result: **{result['interpretation']}** (p ≥ 0.05)")
                                    
                                    st.write("Groups analyzed:", ", ".join(result['groups']))
                                    
                                except Exception as e:
                                    st.error(f"Error running ANOVA: {str(e)}")
                    
                    elif test_type == "Chi-Square Test":
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if len(cat_cols) < 2:
                            st.error("Chi-Square test requires at least two categorical columns")
                            return
                            
                        col1 = st.selectbox("Select first categorical column:", cat_cols)
                        col2 = st.selectbox("Select second categorical column:", [c for c in cat_cols if c != col1])
                        
                        if st.button("Run Chi-Square Test"):
                            with st.spinner("Running Chi-Square test..."):
                                try:
                                    result = cleaner.perform_hypothesis_test(col1, col2, 'chi2')
                                    
                                    st.write(f"#### {result['test_name']}")
                                    st.write(result['description'])
                                    st.write(f"Chi-Square statistic: {result['statistic']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    st.write(f"Degrees of freedom: {result['dof']}")
                                    
                                    if result['p_value'] < 0.05:
                                        st.success(f"Result: **{result['interpretation']}** (p < 0.05)")
                                    else:
                                        st.info(f"Result: **{result['interpretation']}** (p ≥ 0.05)")
                                    
                                    st.write("#### Contingency Table")
                                    st.dataframe(result['contingency_table'])
                                    
                                except Exception as e:
                                    st.error(f"Error running Chi-Square test: {str(e)}")
                    
                    elif test_type == "Correlation Test":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(numeric_cols) < 2:
                            st.error("Correlation test requires at least two numeric columns")
                            return
                            
                        col1 = st.selectbox("Select first numeric column:", numeric_cols)
                        col2 = st.selectbox("Select second numeric column:", [c for c in numeric_cols if c != col1])
                        
                        if st.button("Run Correlation Test"):
                            with st.spinner("Running correlation test..."):
                                try:
                                    result = cleaner.perform_hypothesis_test(col1, col2, 'correlation')
                                    
                                    st.write(f"#### {result['test_name']}")
                                    st.write(result['description'])
                                    st.write(f"Correlation coefficient: {result['correlation']:.4f}")
                                    st.write(f"p-value: {result['p_value']:.4f}")
                                    
                                    if result['p_value'] < 0.05:
                                        st.success(f"Result: **{result['interpretation']}** (p < 0.05)")
                                    else:
                                        st.info(f"Result: **{result['interpretation']}** (p ≥ 0.05)")
                                    
                                    st.write(f"Interpretation: **{result['correlation_description']}**")
                                    
                                    # Visualization of correlation
                                    if PLOTLY_AVAILABLE:
                                        fig = px.scatter(df, x=col1, y=col2, 
                                                      title=f"Correlation between {col1} and {col2}",
                                                      trendline="ols")
                                        st.plotly_chart(fig)
                                    else:
                                        fig, ax = plt.subplots()
                                        ax.scatter(df[col1], df[col2])
                                        ax.set_xlabel(col1)
                                        ax.set_ylabel(col2)
                                        ax.set_title(f"Correlation between {col1} and {col2}")
                                        st.pyplot(fig)
                                    
                                except Exception as e:
                                    st.error(f"Error running correlation test: {str(e)}")
                
                elif analysis_type == "Clustering":
                    st.write("#### Cluster Analysis")
                    st.write("Cluster analysis helps identify natural groupings in your data.")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) < 2:
                        st.error("Clustering requires at least two numeric features")
                        return
                    
                    selected_features = st.multiselect(
                        "Select features for clustering:", 
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )
                    
                    clustering_method = st.selectbox(
                        "Select clustering method:",
                        ["kmeans", "hierarchical"],
                        format_func=lambda x: {
                            "kmeans": "K-means (distance-based clusters)",
                            "hierarchical": "Hierarchical (tree-based clustering)"
                        }[x]
                    )
                    
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    
                    if len(selected_features) < 2:
                        st.warning("Please select at least two features for clustering")
                    else:
                        if st.button("Perform Clustering"):
                            with st.spinner("Clustering data..."):
                                try:
                                    result = cleaner.perform_clustering(selected_features, n_clusters, clustering_method)
                                    
                                    st.write(f"#### Clustering Results ({clustering_method})")
                                    st.write(f"Created {n_clusters} clusters using {len(selected_features)} features")
                                    
                                    # Show cluster sizes
                                    cluster_sizes = result['clustered_data']['cluster'].value_counts().sort_index()
                                    st.write("#### Cluster Sizes")
                                    
                                    # Create cluster size visualization
                                    if PLOTLY_AVAILABLE:
                                        fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                                                    labels={'x': 'Cluster', 'y': 'Count'},
                                                    title="Number of samples in each cluster")
                                        st.plotly_chart(fig)
                                    else:
                                        fig, ax = plt.subplots()
                                        cluster_sizes.plot.bar(ax=ax)
                                        ax.set_xlabel("Cluster")
                                        ax.set_ylabel("Count")
                                        ax.set_title("Number of samples in each cluster")
                                        st.pyplot(fig)
                                    
                                    # Show cluster centers for KMeans
                                    if result['centroids'] is not None:
                                        st.write("#### Cluster Centers")
                                        st.dataframe(result['centroids'].set_index('cluster'))
                                    
                                    # Visualize clusters in 2D
                                    st.write("#### Cluster Visualization")
                                    
                                    if result['pca_data'] is not None:
                                        # Using PCA data for visualization if more than 2 features
                                        if PLOTLY_AVAILABLE:
                                            fig = px.scatter(
                                                result['pca_data'], x='PC1', y='PC2', color='cluster',
                                                title=f"Cluster visualization (variance explained: {sum(result['explained_variance'])*100:.1f}%)"
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            fig, ax = plt.subplots()
                                            for i in range(n_clusters):
                                                ax.scatter(
                                                    result['pca_data'][result['pca_data']['cluster'] == i]['PC1'],
                                                    result['pca_data'][result['pca_data']['cluster'] == i]['PC2'],
                                                    label=f'Cluster {i}'
                                                )
                                            ax.set_xlabel('PC1')
                                            ax.set_ylabel('PC2')
                                            ax.set_title(f"Cluster visualization (variance explained: {sum(result['explained_variance'])*100:.1f}%)")
                                            ax.legend()
                                            st.pyplot(fig)
                                    elif len(selected_features) == 2:
                                        # Direct visualization if exactly 2 features
                                        x_col, y_col = selected_features
                                        if PLOTLY_AVAILABLE:
                                            fig = px.scatter(
                                                result['clustered_data'], x=x_col, y=y_col, color='cluster',
                                                title=f"Clusters based on {x_col} and {y_col}"
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            fig, ax = plt.subplots()
                                            for i in range(n_clusters):
                                                ax.scatter(
                                                    result['clustered_data'][result['clustered_data']['cluster'] == i][x_col],
                                                    result['clustered_data'][result['clustered_data']['cluster'] == i][y_col],
                                                    label=f'Cluster {i}'
                                                )
                                            ax.set_xlabel(x_col)
                                            ax.set_ylabel(y_col)
                                            ax.set_title(f"Clusters based on {x_col} and {y_col}")
                                            ax.legend()
                                            st.pyplot(fig)
                                    
                                    # Add cluster labels to the main DataFrame if requested
                                    if st.checkbox("Add cluster labels to the dataset"):
                                        # First create a mapping from original index to cluster
                                        cluster_mapping = dict(zip(result['clustered_data'].index, result['clustered_data']['cluster']))
                                        # Then apply this to the original dataframe
                                        cleaner.df['cluster'] = cleaner.df.index.map(lambda idx: cluster_mapping.get(idx, -1))
                                        st.success("Cluster labels added as 'cluster' column. Rows without cluster assignment have value -1.")
                                    
                                except Exception as e:
                                    st.error(f"Error performing clustering: {str(e)}")
                
                elif analysis_type == "Feature Selection":
                    st.write("#### Feature Selection")
                    st.write("Identify the most important features for predicting a target variable.")
                    
                    target_col = st.selectbox("Select target column:", df.columns)
                    
                    selection_method = st.selectbox(
                        "Select feature selection method:",
                        ["importance", "f_test", "mutual_info"],
                        format_func=lambda x: {
                            "importance": "Random Forest Importance (best for general use)",
                            "f_test": "F-test (linear relationships)",
                            "mutual_info": "Mutual Information (captures non-linear relationships)"
                        }[x]
                    )
                    
                    n_features = st.slider("Number of top features to select:", 1, 20, 10)
                    
                    if st.button("Select Features"):
                        with st.spinner("Analyzing feature importance..."):
                            try:
                                result = cleaner.select_features(target_col, n_features, selection_method)
                                
                                st.write(f"#### Top {len(result['top_features'])} Features for {target_col}")
                                st.write(f"Problem type: {'Classification' if result['is_classification'] else 'Regression'}")
                                
                                # Create DataFrame for feature importance
                                importance_df = pd.DataFrame(
                                    result['top_features'],
                                    columns=['Feature', 'Score']
                                )
                                
                                # Display as table
                                st.dataframe(importance_df)
                                
                                # Create bar chart visualization
                                if PLOTLY_AVAILABLE:
                                    fig = px.bar(
                                        importance_df, x='Score', y='Feature', 
                                        orientation='h',
                                        title=f"Feature Importance for {target_col}"
                                    )
                                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig)
                                else:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    importance_df.plot.barh(x='Feature', y='Score', ax=ax, legend=False)
                                    ax.set_title(f"Feature Importance for {target_col}")
                                    st.pyplot(fig)
                                
                                # Option to keep only selected features
                                if st.checkbox("Keep only selected features in the dataset"):
                                    keep_cols = [f[0] for f in result['top_features']] + [target_col]
                                    cleaner.df = cleaner.df[keep_cols]
                                    st.success(f"Dataset reduced to {len(keep_cols)} columns (top features + target column)")
                                
                            except Exception as e:
                                st.error(f"Error in feature selection: {str(e)}")

            # Show Cleaned Data
            st.write("### Cleaned DataFrame:")
            cleaned_df = cleaner.get_cleaned_data()
            st.dataframe(cleaned_df)

            # Download Cleaned Data
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download cleaned data as CSV",
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv',
            )

            # Customer Lifetime Value Analysis
            if st.sidebar.checkbox("Customer Lifetime Value (CLV)"):
                st.write("### Customer Lifetime Value Analysis")
                st.write("Calculate and analyze Customer Lifetime Value based on transaction data.")

                if df is not None:
                    # Try to identify relevant columns
                    cols = df.columns.tolist()
                    
                    # Find potential columns based on common names
                    id_cols = [col for col in cols if any(term in col.lower() for term in ['customer', 'id', 'client'])]
                    date_cols = [col for col in cols if any(term in col.lower() for term in ['date', 'time'])]
                    quantity_cols = [col for col in cols if any(term in col.lower() for term in ['quantity', 'qty'])]
                    price_cols = [col for col in cols if any(term in col.lower() for term in ['price', 'amount', 'value'])]
                    country_cols = [col for col in cols if any(term in col.lower() for term in ['country', 'region', 'location'])]

                    # Column selection
                    col1, col2 = st.columns(2)
                    with col1:
                        customer_id = st.selectbox(
                            "Customer ID Column",
                            cols,
                            index=cols.index(id_cols[0]) if id_cols else 0,
                            help="Column containing unique customer identifiers"
                        )
                        quantity = st.selectbox(
                            "Quantity Column",
                            cols,
                            index=cols.index(quantity_cols[0]) if quantity_cols else 0,
                            help="Column containing quantity purchased"
                        )
                    
                    with col2:
                        invoice_date = st.selectbox(
                            "Date Column",
                            cols,
                            index=cols.index(date_cols[0]) if date_cols else 0,
                            help="Column containing transaction dates"
                        )
                        unit_price = st.selectbox(
                            "Unit Price Column",
                            cols,
                            index=cols.index(price_cols[0]) if price_cols else 0,
                            help="Column containing price per unit"
                        )

                    # Country selection (optional)
                    country = st.selectbox(
                        "Country Column (Optional)",
                        ["None"] + cols,
                        index=0,
                        help="Optional: Column containing country/region information for geographic analysis"
                    )
                    if country == "None":
                        country = None

                    if st.button("Calculate CLV"):
                        with st.spinner("Calculating Customer Lifetime Value..."):
                            try:
                                clv_results, clv_stats, country_stats = cleaner.calculate_clv(
                                    customer_id_col=customer_id,
                                    invoice_date_col=invoice_date,
                                    quantity_col=quantity,
                                    unit_price_col=unit_price,
                                    country_col=country
                                )

                                if clv_results is not None and clv_stats is not None:
                                    # Display summary statistics
                                    st.write("#### CLV Summary Statistics")
                                    st.dataframe(clv_stats)

                                    # Display country-specific analysis if available
                                    if country_stats is not None:
                                        st.write("#### Country-Specific Analysis")
                                        st.dataframe(country_stats)
                                        
                                        # Visualize country CLV comparison
                                        if PLOTLY_AVAILABLE:
                                            fig = px.bar(
                                                country_stats.reset_index(),
                                                x='country',
                                                y='Average CLV',
                                                title="Average CLV by Country",
                                                labels={'country': 'Country', 'Average CLV': 'Average Customer Lifetime Value'}
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            country_stats['Average CLV'].plot.bar(ax=ax)
                                            ax.set_title("Average CLV by Country")
                                            ax.set_xlabel("Country")
                                            ax.set_ylabel("Average Customer Lifetime Value")
                                            st.pyplot(fig)

                                    # Display CLV distribution
                                    st.write("#### CLV Distribution")
                                    if PLOTLY_AVAILABLE:
                                        if country:
                                            # Create a new DataFrame with the correct column names
                                            plot_data = clv_results.copy()
                                            plot_data['Country'] = plot_data[country]
                                            fig = px.histogram(
                                                plot_data,
                                                x="historical_clv",
                                                nbins=50,
                                                title="Distribution of Customer Lifetime Value",
                                                color="Country",
                                                labels={'historical_clv': 'Customer Lifetime Value'}
                                            )
                                        else:
                                            fig = px.histogram(
                                                clv_results,
                                                x="historical_clv",
                                                nbins=50,
                                                title="Distribution of Customer Lifetime Value",
                                                labels={'historical_clv': 'Customer Lifetime Value'}
                                            )
                                        fig.update_layout(
                                            xaxis_title="Customer Lifetime Value",
                                            yaxis_title="Number of Customers"
                                        )
                                        st.plotly_chart(fig)
                                    else:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        if country:
                                            for country_name in clv_results[country].unique():
                                                country_data = clv_results[clv_results[country] == country_name]
                                                country_data['historical_clv'].hist(bins=50, ax=ax, alpha=0.5, label=country_name)
                                            ax.legend()
                                        else:
                                            clv_results['historical_clv'].hist(bins=50, ax=ax)
                                        ax.set_title("Distribution of Customer Lifetime Value")
                                        ax.set_xlabel("Customer Lifetime Value")
                                        ax.set_ylabel("Number of Customers")
                                        st.pyplot(fig)

                                    # Display RFM Segmentation
                                    st.write("#### RFM Segmentation Analysis")
                                    st.write("Customers are segmented based on Recency, Frequency, and Monetary value.")
                                    
                                    # Display segment statistics
                                    st.write("##### Segment Statistics")
                                    # Check if the segment_stats variable exists before using it
                                    if 'segment_stats' in locals() or 'segment_stats' in globals():
                                        safe_display_dataframe(segment_stats, cleaner)
                                    else:
                                        _, _, segment_stats = cleaner.calculate_clv(
                                            customer_id_col=customer_id,
                                            invoice_date_col=invoice_date,
                                            quantity_col=quantity,
                                            unit_price_col=unit_price,
                                            country_col=country
                                        )
                                        if segment_stats is not None:
                                            safe_display_dataframe(segment_stats, cleaner)
                                        else:
                                            st.error("Could not calculate segment statistics.")
                                    
                                    # Visualize segment distribution
                                    if PLOTLY_AVAILABLE:
                                        # Segment size pie chart
                                        fig = px.pie(
                                            values=segment_stats['Customer Count'],
                                            names=segment_stats.index,
                                            title="Customer Distribution by RFM Segment"
                                        )
                                        st.plotly_chart(fig)
                                        
                                        # Segment CLV comparison
                                        segment_df = segment_stats.reset_index()
                                        segment_df.columns = ['segment', 'Average CLV', 'Median CLV', 'Total CLV', 'Customer Count', 
                                                            'Average Frequency', 'Average Recency (days)', 'Total Revenue']
                                        
                                        fig = px.bar(
                                            segment_df,
                                            x='segment',
                                            y='Average CLV',
                                            title="Average CLV by RFM Segment",
                                            labels={'segment': 'RFM Segment', 'Average CLV': 'Average Customer Lifetime Value'}
                                        )
                                        st.plotly_chart(fig)
                                        
                                        # Additional RFM visualizations
                                        st.write("##### RFM Score Distribution")
                                        # Get RFM scores from clv_results, ensuring 'segment' is present
                                        if 'segment' in clv_results.columns:
                                            rfm_scores = clv_results.groupby('segment').agg({
                                                'recency_score': 'mean',
                                                'frequency_score': 'mean',
                                                'monetary_score': 'mean'
                                            }).reset_index()
                                            
                                            fig = px.bar(
                                                rfm_scores,
                                                x='segment',
                                                y=['recency_score', 'frequency_score', 'monetary_score'],
                                                title="Average RFM Scores by Segment",
                                                barmode='group',
                                                labels={'value': 'Average Score', 'variable': 'RFM Component'}
                                            )
                                            st.plotly_chart(fig)
                                        else:
                                            st.warning("RFM score details not available in the dataset")
                                    else:
                                        # Segment size pie chart
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        segment_stats['Customer Count'].plot.pie(ax=ax, autopct='%1.1f%%')
                                        ax.set_title("Customer Distribution by RFM Segment")
                                        st.pyplot(fig)
                                        
                                        # Segment CLV comparison
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        segment_stats['Average CLV'].plot.bar(ax=ax)
                                        ax.set_title("Average CLV by RFM Segment")
                                        ax.set_xlabel("RFM Segment")
                                        ax.set_ylabel("Average Customer Lifetime Value")
                                        st.pyplot(fig)
                                        
                                        # Additional RFM visualizations
                                        st.write("##### RFM Score Distribution")
                                        # Get RFM scores from clv_results, ensuring 'segment' is present
                                        if 'segment' in clv_results.columns:
                                            rfm_scores = clv_results.groupby('segment').agg({
                                                'recency_score': 'mean',
                                                'frequency_score': 'mean',
                                                'monetary_score': 'mean'
                                            }).reset_index()
                                            
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            rfm_scores.plot.bar(x='segment', y=['recency_score', 'frequency_score', 'monetary_score'], ax=ax)
                                            ax.set_title("Average RFM Scores by Segment")
                                            ax.set_xlabel("RFM Segment")
                                            ax.set_ylabel("Average Score")
                                            ax.legend(['Recency', 'Frequency', 'Monetary'])
                                            st.pyplot(fig)
                                        else:
                                            st.warning("RFM score details not available in the dataset")

                                    # Display detailed results
                                    st.write("#### Customer Details")
                                    safe_display_dataframe(clv_results, cleaner)

                                    # Download option
                                    csv = clv_results.to_csv(index=False)
                                    st.download_button(
                                        label="Download CLV Analysis",
                                        data=csv,
                                        file_name="clv_analysis.csv",
                                        mime="text/csv"
                                    )

                                    # CLV Regression Analysis
                                    st.write("#### CLV Regression Analysis")
                                    st.write("Train regression models to predict future CLV based on customer behavior patterns.")
                                    
                                    if st.button("Train Regression Models"):
                                        with st.spinner("Training regression models..."):
                                            try:
                                                # Prepare data for regression
                                                X, y, feature_names = cleaner.prepare_clv_for_regression(
                                                    customer_id_col=customer_id,
                                                    invoice_date_col=invoice_date,
                                                    quantity_col=quantity,
                                                    unit_price_col=unit_price,
                                                    country_col=country
                                                )
                                                
                                                if X is not None and y is not None:
                                                    # Train models
                                                    results = cleaner.train_clv_regression_models(X, y)
                                                    
                                                    # Display model comparison
                                                    st.write("##### Model Performance Comparison")
                                                    comparison_data = {
                                                        'Metric': ['MSE (Test)', 'R² (Test)', 'MSE (CV)', 'R² (CV)'],
                                                        'Lasso': [
                                                            f"{results['lasso']['mse']:.2f}",
                                                            f"{results['lasso']['r2']:.2f}",
                                                            f"{results['lasso']['cv_mse']:.2f}",
                                                            f"{results['lasso']['cv_r2']:.2f}"
                                                        ],
                                                        'Ridge': [
                                                            f"{results['ridge']['mse']:.2f}",
                                                            f"{results['ridge']['r2']:.2f}",
                                                            f"{results['ridge']['cv_mse']:.2f}",
                                                            f"{results['ridge']['cv_r2']:.2f}"
                                                        ]
                                                    }
                                                    st.dataframe(pd.DataFrame(comparison_data))
                                                    
                                                    # Display feature importance
                                                    st.write("##### Feature Importance (Lasso Model)")
                                                    importance_data = {
                                                        'Feature': list(results['lasso']['coefficients'].keys()),
                                                        'Coefficient': list(results['lasso']['coefficients'].values())
                                                    }
                                                    importance_df = pd.DataFrame(importance_data)
                                                    importance_df = importance_df.sort_values('Coefficient', key=abs, ascending=False)
                                                    
                                                    if PLOTLY_AVAILABLE:
                                                        fig = px.bar(
                                                            importance_df,
                                                            x='Feature',
                                                            y='Coefficient',
                                                            title="Feature Importance in CLV Prediction",
                                                            labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature'}
                                                        )
                                                        fig.update_layout(xaxis_tickangle=-45)
                                                        st.plotly_chart(fig)
                                                    else:
                                                        fig, ax = plt.subplots(figsize=(10, 6))
                                                        importance_df.plot.bar(x='Feature', y='Coefficient', ax=ax)
                                                        ax.set_title("Feature Importance in CLV Prediction")
                                                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                                                        st.pyplot(fig)
                                                    
                                                    # Display actual vs predicted values
                                                    st.write("##### Actual vs Predicted CLV (Test Set)")
                                                    if PLOTLY_AVAILABLE:
                                                        # Get the test set and predictions from the results dictionary
                                                        y_test = results.get('y_test', None)
                                                        lasso_pred = results.get('lasso', {}).get('predictions', None)
                                                        
                                                        if y_test is not None and lasso_pred is not None:
                                                            fig = px.scatter(
                                                                x=y_test,
                                                                y=lasso_pred,
                                                                title="Actual vs Predicted CLV (Lasso Model)",
                                                                labels={'x': 'Actual CLV', 'y': 'Predicted CLV'}
                                                            )
                                                            fig.add_shape(
                                                                type="line",
                                                                x0=y_test.min(),
                                                                y0=y_test.min(),
                                                                x1=y_test.max(),
                                                                y1=y_test.max(),
                                                                line=dict(color="red", dash="dash")
                                                            )
                                                            st.plotly_chart(fig)
                                                        else:
                                                            st.warning("Test data not available for visualization")
                                                    else:
                                                        # Get the test set and predictions from the results dictionary
                                                        y_test = results.get('y_test', None)
                                                        lasso_pred = results.get('lasso', {}).get('predictions', None)
                                                        
                                                        if y_test is not None and lasso_pred is not None:
                                                            fig, ax = plt.subplots(figsize=(10, 6))
                                                            ax.scatter(y_test, lasso_pred)
                                                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                                                            ax.set_xlabel("Actual CLV")
                                                            ax.set_ylabel("Predicted CLV")
                                                            ax.set_title("Actual vs Predicted CLV (Lasso Model)")
                                                            st.pyplot(fig)
                                                        else:
                                                            st.warning("Test data not available for visualization")
                                            
                                            except Exception as e:
                                                st.error(f"Error in regression analysis: {str(e)}")

                            except Exception as e:
                                st.error(f"Error calculating CLV: {str(e)}")
                else:
                    st.info("Please upload a dataset first to calculate CLV.")

            # Add Demand Forecasting section
            if st.sidebar.checkbox("Demand Forecasting"):
                st.write("### Demand Forecasting")
                
                # Create tabs for different forecasting approaches
                forecast_tabs = st.tabs(["Real Data Forecasting", "Simulation", "Forecast Accuracy Analysis", "Aggregation Simulator"])
                
                # Real Data Forecasting Tab
                with forecast_tabs[0]:
                    st.write("#### Forecast Using Actual Data")
                    st.write("Upload or use your current dataset to generate demand forecasts.")
                    
                    if df is not None:
                        # Check for datetime columns
                        date_cols = []
                        for col in df.columns:
                            try:
                                # Try converting to datetime
                                pd.to_datetime(df[col], errors='coerce')
                                if not pd.to_datetime(df[col], errors='coerce').isna().all():
                                    date_cols.append(col)
                            except:
                                pass
                        
                        # Check for numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        if date_cols and numeric_cols:
                            # Select date column
                            date_col = st.selectbox("Select date column:", date_cols, key="forecast_date_col")
                            
                            # Select demand column
                            demand_col = st.selectbox("Select demand/sales column:", numeric_cols, key="forecast_demand_col")
                            
                            # Algorithm selection
                            algorithm = st.selectbox("Select forecasting algorithm:",
                                                 ['prophet', 'arima', 'exponential'],
                                                 format_func=lambda x: {
                                                     'prophet': 'Prophet (handles seasonality well)',
                                                     'arima': 'ARIMA (good for various time series)',
                                                     'exponential': 'Exponential Smoothing (simple, robust)'
                                                 }[x])
                            
                            # Seasonality selection
                            seasonality = st.selectbox("Select Seasonality Pattern:",
                                                 ['auto', 'daily', 'weekly', 'monthly', 'yearly'],
                                                 format_func=lambda x: x.title())
                            
                            # Forecast periods
                            forecast_periods = st.slider("Number of Periods to Forecast:", 
                                                  min_value=7, 
                                                  max_value=365, 
                                                  value=30)
                            
                            # Run forecast
                            if st.button("Generate Forecast"):
                                with st.spinner("Generating demand forecast..."):
                                    results, fig = cleaner.forecast_demand(
                                        date_col, demand_col, 
                                        forecast_periods=forecast_periods,
                                        algorithm=algorithm,
                                        seasonality=seasonality
                                    )
                                    
                                    if results is not None:
                                        # Show forecast plot
                                        st.plotly_chart(fig)
                                        
                                        # Show forecast metrics
                                        st.write("#### Forecast Summary")
                                        metrics = results.describe()
                                        st.dataframe(metrics)
                                        
                                        # Download forecast
                                        csv = results.to_csv(index=False)
                                        st.download_button(
                                            label="Download Forecast",
                                            data=csv,
                                            file_name="demand_forecast.csv",
                                            mime="text/csv"
                                        )
                        else:
                            st.warning("Your dataset needs both date/time and numeric columns for forecasting.")
                            
                            # Show sample data option
                            if st.button("Use Sample Data"):
                                # Create sample time series data
                                sample_dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
                                sample_demand = np.random.normal(100, 20, 24) + np.arange(24) * 2  # Add trend
                                
                                # Add seasonality
                                for i in range(24):
                                    month = (i % 12) + 1
                                    if 3 <= month <= 5:  # Spring peak
                                        sample_demand[i] += 30
                                    elif 9 <= month <= 11:  # Fall peak
                                        sample_demand[i] += 20
                                
                                sample_df = pd.DataFrame({
                                    'date': sample_dates,
                                    'demand': sample_demand
                                })
                                
                                # Create DataCleaner with sample data
                                sample_cleaner = DataCleaner(sample_df)
                                
                                st.write("#### Sample Data")
                                st.dataframe(sample_df)
                                
                                # Generate forecast with sample data
                                results, fig = sample_cleaner.forecast_demand(
                                    'date', 'demand', 
                                    forecast_periods=12,
                                    algorithm='prophet',
                                    seasonality='yearly'
                                )
                                
                                if results is not None:
                                    st.write("#### Sample Forecast")
                                    st.plotly_chart(fig)
                    else:
                        st.info("Please upload a dataset first or use the simulation tab.")
                
                # Simulation Tab
                with forecast_tabs[1]:
                    st.write("#### Simulate Demand Data")
                    st.write("Generate synthetic demand data with different components.")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        mean_sales = st.number_input("Mean Sales", value=91.0, step=1.0)
                        std_dev = st.number_input("Standard Deviation", value=10.0, min_value=0.1, step=0.1)
                        trend = st.number_input("Trend (units/period)", value=5.0, step=1.0)
                    
                    with col2:
                        include_trend = st.checkbox("Include Trend", value=True)
                        include_seasonality = st.checkbox("Include Seasonality", value=True)
                        include_promotions = st.checkbox("Include Promotions", value=True)
                    
                    # Season Management
                    if include_seasonality:
                        st.subheader("Seasonal Patterns")
                        st.write("Define seasonal patterns that repeat every year")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            season_start = st.selectbox("Season Start Month", range(1, 13), key="season_start")
                        with col2:
                            season_end = st.selectbox("Season End Month", range(1, 13), key="season_end")
                        with col3:
                            season_change = st.number_input("Change in Demand", value=25.0, step=5.0, key="season_change")
                        
                        if st.button("+ SEASON"):
                            if not hasattr(st.session_state, 'seasons'):
                                st.session_state.seasons = []
                            st.session_state.seasons.append({
                                'start': season_start,
                                'end': season_end,
                                'change': season_change
                            })
                    
                        # Show current seasons
                        if hasattr(st.session_state, 'seasons') and st.session_state.seasons:
                            st.write("Current Seasonal Patterns:")
                            seasons_df = pd.DataFrame(st.session_state.seasons)
                            for idx, season in seasons_df.iterrows():
                                col1, col2 = st.columns([0.9, 0.1])
                                with col1:
                                    st.write(f"Months {season['start']}-{season['end']}: {season['change']} units")
                                with col2:
                                    if st.button("X", key=f"del_season_{idx}"):
                                        st.session_state.seasons.pop(idx)
                                        st.rerun()
                    
                    # Promotion Management
                    if include_promotions:
                        st.subheader("Promotions")
                        st.write("Add one-time promotional effects")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            promo_start = st.number_input("Start Month", min_value=1, value=1, step=1, key="promo_start")
                        with col2:
                            promo_end = st.number_input("End Month", min_value=1, value=3, step=1, key="promo_end")
                        with col3:
                            promo_change = st.number_input("Change in Demand", value=25.0, step=5.0, key="promo_change")
                        
                        if st.button("+ PROMOTION"):
                            if not hasattr(st.session_state, 'promotions'):
                                st.session_state.promotions = []
                            st.session_state.promotions.append({
                                'start': promo_start,
                                'end': promo_end,
                                'change': promo_change
                            })
                    
                        # Show current promotions
                        if hasattr(st.session_state, 'promotions') and st.session_state.promotions:
                            st.write("Current Promotions:")
                            promos_df = pd.DataFrame(st.session_state.promotions)
                            for idx, promo in promos_df.iterrows():
                                col1, col2 = st.columns([0.9, 0.1])
                                with col1:
                                    st.write(f"Months {promo['start']}-{promo['end']}: {promo['change']} units")
                                with col2:
                                    if st.button("X", key=f"del_promo_{idx}"):
                                        st.session_state.promotions.pop(idx)
                                        st.rerun()
                    
                    if st.button("GET DATA"):
                        # Create a DataCleaner instance with empty DataFrame (just for simulation)
                        cleaner = DataCleaner(pd.DataFrame())
                        
                        # Add any custom seasons
                        if include_seasonality and hasattr(st.session_state, 'seasons'):
                            for season in st.session_state.seasons:
                                cleaner.add_season(season['start'], season['end'], season['change'])
                                
                        # Add any custom promotions
                        if include_promotions and hasattr(st.session_state, 'promotions'):
                            for promo in st.session_state.promotions:
                                cleaner.add_promotion(promo['start'], promo['end'], promo['change'])
                        
                        # Generate simulated data
                        simulated_data = cleaner.simulate_demand(
                            mean_sales=mean_sales,
                            std_dev=std_dev,
                            trend=trend if include_trend else 0,
                            seasonality=include_seasonality,
                            promotions=include_promotions
                        )
                        
                        # Plot the data
                        components = []
                        if include_trend:
                            components.append('Trend')
                        if include_seasonality:
                            components.append('Seasonality')
                        if include_promotions:
                            components.append('Promotions')
                            
                        cleaner.plot_demand_simulation(simulated_data, components=components)
                        
                        # Show data table
                        st.write("### Simulated Data")
                        st.dataframe(simulated_data)
                
                # Forecast Accuracy Analysis Tab
                with forecast_tabs[2]:
                    st.write("#### Forecast Accuracy Analysis")
                    st.write("Calculate A/F ratios and other accuracy metrics to evaluate forecast performance.")
                    
                    if df is not None:
                        # Get numeric columns for actual and forecast data
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        if len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                actual_col = st.selectbox(
                                    "Select Actual Demand Column:", 
                                    numeric_cols,
                                    index=0,  # Default to first column
                                    key="actual_demand_col"
                                )
                            
                            with col2:
                                # Default to second column for forecast
                                forecast_index = min(1, len(numeric_cols)-1)
                                forecast_col = st.selectbox(
                                    "Select Forecast Column:", 
                                    numeric_cols,
                                    index=forecast_index,
                                    key="forecast_col"
                                )
                            
                            # Optional product column for grouping
                            all_cols = df.columns.tolist()
                            use_product = st.checkbox("Group by Product/Category", value=True)
                            
                            if use_product:
                                # Find likely product/category columns (non-numeric)
                                cat_cols = [col for col in all_cols if col not in numeric_cols]
                                
                                if cat_cols:
                                    # Try to find column with "product" in name
                                    product_candidates = [col for col in cat_cols if "product" in col.lower()]
                                    
                                    if product_candidates:
                                        default_idx = cat_cols.index(product_candidates[0])
                                    else:
                                        default_idx = 0
                                        
                                    product_col = st.selectbox(
                                        "Select Product/Category Column:", 
                                        cat_cols,
                                        index=default_idx
                                    )
                                else:
                                    product_col = st.selectbox(
                                        "Select Product/Category Column:", 
                                        all_cols
                                    )
                            else:
                                product_col = None
                            
                            if st.button("Calculate Accuracy Metrics"):
                                with st.spinner("Calculating forecast accuracy..."):
                                    try:
                                        result, overall_metrics = cleaner.calculate_forecast_accuracy(
                                            actual_col, forecast_col, product_col
                                        )
                                        
                                        if result is not None:
                                            # Display results
                                            st.write("#### Accuracy Metrics by Product")
                                            st.dataframe(result)
                                            
                                            # Display overall metrics
                                            st.write("#### Overall Accuracy Metrics")
                                            overall_df = pd.DataFrame([overall_metrics])
                                            st.dataframe(overall_df)
                                            
                                            # Visualize A/F ratios
                                            st.write("#### A/F Ratio Visualization")
                                            if PLOTLY_AVAILABLE:
                                                # Sort by A/F ratio
                                                sorted_result = result.sort_values(by='A/F', ascending=False)
                                                
                                                # Get the product/category column name from the index
                                                if product_col:
                                                    x_col = product_col
                                                else:
                                                    x_col = sorted_result.index.name or "Index"
                                                
                                                fig = px.bar(
                                                    sorted_result.reset_index(),
                                                    x=sorted_result.index.name or "index",
                                                    y='A/F',
                                                    color='A/F',
                                                    color_continuous_scale='RdYlGn_r',
                                                    range_color=[0.5, 1.5],
                                                    title="A/F Ratio by Product (Actual/Forecast)",
                                                    labels={'A/F': 'A/F Ratio', sorted_result.index.name or "index": x_col}
                                                )
                                                
                                                # Add a horizontal line at A/F = 1.0
                                                fig.add_shape(
                                                    type="line",
                                                    x0=-0.5,
                                                    y0=1.0,
                                                    x1=len(sorted_result) - 0.5,
                                                    y1=1.0,
                                                    line=dict(color="black", width=2, dash="dash")
                                                )
                                                
                                                fig.update_layout(
                                                    xaxis_tickangle=-45,
                                                    yaxis_title="A/F Ratio",
                                                    xaxis_title=x_col
                                                )
                                                
                                                st.plotly_chart(fig)
                                            else:
                                                # Matplotlib fallback
                                                if product_col:
                                                    fig, ax = plt.subplots(figsize=(12, 6))
                                                    sorted_result = result.sort_values(by='A/F', ascending=False)
                                                    
                                                    bars = ax.bar(sorted_result.index, sorted_result['A/F'])
                                                    
                                                    # Color bars based on A/F value
                                                    for i, bar in enumerate(bars):
                                                        af_value = sorted_result['A/F'].iloc[i]
                                                        if af_value < 0.9:
                                                            bar.set_color('green')  # Under-forecasted
                                                        elif af_value > 1.1:
                                                            bar.set_color('red')    # Over-forecasted
                                                        else:
                                                            bar.set_color('blue')   # Good forecast
                                                    
                                                    # Add a horizontal line at A/F = 1.0
                                                    ax.axhline(y=1.0, color='black', linestyle='--')
                                                    
                                                    plt.title("A/F Ratio by Product (Actual/Forecast)")
                                                    plt.ylabel("A/F Ratio")
                                                    plt.xlabel(product_col)
                                                    plt.xticks(rotation=45, ha='right')
                                                    plt.tight_layout()
                                                    st.pyplot(fig)
                                                else:
                                                    fig, ax = plt.subplots(figsize=(10, 6))
                                                    plt.hist(result['A/F'], bins=20)
                                                    
                                                    # Add a vertical line at A/F = 1.0
                                                    plt.axvline(x=1.0, color='r', linestyle='--')
                                                    
                                                    plt.title("Distribution of A/F Ratios")
                                                    plt.xlabel("A/F Ratio")
                                                    plt.ylabel("Frequency")
                                                    st.pyplot(fig)
                                            
                                            # Provide interpretation
                                            st.write("#### Interpretation")
                                            af_overall = overall_metrics.get('Overall A/F', 0)
                                            
                                            if af_overall < 0.95:
                                                st.warning(f"Overall A/F ratio is {af_overall:.2f} (< 0.95), suggesting the forecast was generally too high.")
                                            elif af_overall > 1.05:
                                                st.warning(f"Overall A/F ratio is {af_overall:.2f} (> 1.05), suggesting the forecast was generally too low.")
                                            else:
                                                st.success(f"Overall A/F ratio is {af_overall:.2f}, suggesting the forecast was generally accurate.")
                                            
                                            # Provide actionable insights
                                            if product_col:
                                                st.write("**Products with Most Inaccurate Forecasts:**")
                                                
                                                # Get products with highest and lowest A/F ratios
                                                most_over = result[result['A/F'] < 0.9].sort_values('A/F')
                                                most_under = result[result['A/F'] > 1.1].sort_values('A/F', ascending=False)
                                                
                                                if not most_over.empty:
                                                    st.write("Products that were over-forecasted (A/F < 0.9):")
                                                    st.dataframe(most_over.head(5))
                                                
                                                if not most_under.empty:
                                                    st.write("Products that were under-forecasted (A/F > 1.1):")
                                                    st.dataframe(most_under.head(5))
                                            
                                            # Download option
                                            csv = result.reset_index().to_csv(index=False)
                                            st.download_button(
                                                label="Download Accuracy Analysis",
                                                data=csv,
                                                file_name="forecast_accuracy.csv",
                                                mime="text/csv"
                                            )
                                        
                                    except Exception as e:
                                        st.error(f"Error calculating forecast accuracy: {str(e)}")
                        else:
                            st.warning("You need at least two numeric columns for forecast accuracy analysis.")
                            
                            # Option to use sample data
                            if st.button("Use Sample Data"):
                                # Create sample data like in the image
                                products = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']
                                
                                # Values from the image
                                demand_values = [110913.10, 117051.90, 127922.70, 63630.55, 56551.43, 66071.60, 
                                              59719.03, 63996.55, 62383.82, 64854.28, 50124.47, 69145.96, 
                                              101693.90, 62550.04, 63315.42, 61014.63]
                                
                                forecast_values = [120384.00, 117546.00, 116736.00, 58248.00, 44228.00, 63711.00, 
                                                36109.00, 41837.00, 61794.00, 43455.00, 62852.00, 82047.00, 
                                                89677.00, 50950.00, 64313.00, 39372.00]
                                
                                # Calculate A/F ratios
                                af_ratios = [a/f for a, f in zip(demand_values, forecast_values)]
                                
                                # Create sample DataFrame
                                sample_df = pd.DataFrame({
                                    'Product': products,
                                    'Demand': demand_values,
                                    'Forecast': forecast_values,
                                    'A/F': af_ratios
                                })
                                
                                # Display sample data
                                st.write("#### Sample Data")
                                st.dataframe(sample_df)
                                
                                # Create a DataCleaner with the sample data
                                sample_cleaner = DataCleaner(sample_df)
                                
                                # Calculate accuracy metrics
                                result, overall_metrics = sample_cleaner.calculate_forecast_accuracy(
                                    'Demand', 'Forecast', 'Product'
                                )
                                
                                if result is not None:
                                    # Display results
                                    st.write("#### Accuracy Metrics by Product")
                                    st.dataframe(result)
                                    
                                    # Display overall metrics
                                    st.write("#### Overall Accuracy Metrics")
                                    overall_df = pd.DataFrame([overall_metrics])
                                    st.dataframe(overall_df)
                                    
                                    # Visualize A/F ratios
                                    st.write("#### A/F Ratio Visualization")
                                    if PLOTLY_AVAILABLE:
                                        # Sort by A/F ratio
                                        sorted_result = result.sort_values(by='A/F', ascending=False)
                                        
                                        # Get the index name (should be 'Product')
                                        x_col = sorted_result.index.name or "Product"
                                        
                                        fig = px.bar(
                                            sorted_result.reset_index(),
                                            x=x_col,
                                            y='A/F',
                                            color='A/F',
                                            color_continuous_scale='RdYlGn_r',
                                            range_color=[0.5, 1.5],
                                            title="A/F Ratio by Product (Actual/Forecast)",
                                            labels={'A/F': 'A/F Ratio', x_col: 'Product'}
                                        )
                                        
                                        # Add a horizontal line at A/F = 1.0
                                        fig.add_shape(
                                            type="line",
                                            x0=-0.5,
                                            y0=1.0,
                                            x1=len(sorted_result) - 0.5,
                                            y1=1.0,
                                            line=dict(color="black", width=2, dash="dash")
                                        )
                                        
                                        fig.update_layout(
                                            xaxis_tickangle=-45,
                                            yaxis_title="A/F Ratio",
                                            xaxis_title="Product"
                                        )
                                        
                                        st.plotly_chart(fig)
                                        
                                        # Provide interpretation
                                        st.write("#### Interpretation")
                                        af_overall = overall_metrics.get('Overall A/F', 0)
                                        
                                        if af_overall < 0.95:
                                            st.warning(f"Overall A/F ratio is {af_overall:.2f} (< 0.95), suggesting the forecast was generally too high.")
                                        elif af_overall > 1.05:
                                            st.warning(f"Overall A/F ratio is {af_overall:.2f} (> 1.05), suggesting the forecast was generally too low.")
                                        else:
                                            st.success(f"Overall A/F ratio is {af_overall:.2f}, suggesting the forecast was generally accurate.")
                                        
                                        # Provide actionable insights
                                        st.write("**Products with Most Inaccurate Forecasts:**")
                                        
                                        # Get products with highest and lowest A/F ratios
                                        most_over = sorted_result[sorted_result['A/F'] < 0.9].sort_values('A/F')
                                        most_under = sorted_result[sorted_result['A/F'] > 1.1].sort_values('A/F', ascending=False)
                                        
                                        if not most_over.empty:
                                            st.write("Products that were over-forecasted (A/F < 0.9):")
                                            st.dataframe(most_over.head(5))
                                        
                                        if not most_under.empty:
                                            st.write("Products that were under-forecasted (A/F > 1.1):")
                                            st.dataframe(most_under.head(5))
                                        
                                        # Download option
                                        csv = sorted_result.reset_index().to_csv(index=False)
                                        st.download_button(
                                            label="Download Accuracy Analysis",
                                            data=csv,
                                            file_name="forecast_accuracy.csv",
                                            mime="text/csv"
                                        )
                    else:
                        st.info("Please upload a dataset first to analyze forecast accuracy.")
                
                # Forecast Aggregation Simulator
                with forecast_tabs[3]:
                    st.write("#### Forecast Aggregation Simulator")
                    st.write("Simulate how forecasting at different aggregation levels affects accuracy.")
                    
                    # Inputs for simulation
                    col1, col2 = st.columns(2)
                    with col1:
                        num_stores = st.number_input("Number of stores", min_value=2, max_value=20, value=8)
                    with col2:
                        num_weeks = st.number_input("# weeks", min_value=1, max_value=52, value=1)
                    
                    # Simulate button and clear history button
                    col1, col2 = st.columns(2)
                    with col1:
                        simulate_button = st.button("SIMULATE", key="simulate_agg")
                    with col2:
                        clear_history = st.button("CLEAR HISTORY", key="clear_history")
                    
                    # Initialize session state for run history if not exists
                    if 'run_history_data' not in st.session_state:
                        st.session_state.run_history_data = pd.DataFrame(columns=[
                            'Run #', 'Total Avg Error %', 'Agg Avg Error %'
                        ])
                    
                    # Initialize session state for store data if not exists
                    if 'store_data' not in st.session_state:
                        st.session_state.store_data = None
                    
                    # Clear history and store data if button clicked
                    if clear_history:
                        st.session_state.run_history_data = pd.DataFrame(columns=[
                            'Run #', 'Total Avg Error %', 'Agg Avg Error %'
                        ])
                        st.session_state.store_data = None
                        st.rerun()
                    
                    # Run simulation if button clicked
                    if simulate_button:
                        # Run the simulation using either existing edited data or generate new data
                        if st.session_state.store_data is not None:
                            # Keep only the Store, Expected Sales, and Standard Deviation columns
                            # (to handle cases where user edited after a simulation)
                            sim_df = st.session_state.store_data[['Store', 'Expected Sales', 'Standard Deviation']]
                            store_df, run_history = cleaner.simulate_forecast_aggregation(
                                num_stores=num_stores,
                                num_weeks=num_weeks,
                                store_df=sim_df
                            )
                        else:
                            store_df, run_history = cleaner.simulate_forecast_aggregation(
                                num_stores=num_stores,
                                num_weeks=num_weeks
                            )
                        
                        # Save the store data in session state for future editing
                        st.session_state.store_data = store_df.copy()
                        
                        # Update the run history in session state
                        if run_history is not None:
                            # If history exists, update with new runs
                            if len(st.session_state.run_history_data) > 0:
                                # Get the last run number
                                last_run = st.session_state.run_history_data['Run #'].max()
                                
                                # Update run numbers in the new history
                                for i in range(len(run_history)):
                                    run_history.at[i, 'Run #'] = last_run + i + 1
                            
                            # Append to existing history
                            st.session_state.run_history_data = pd.concat(
                                [st.session_state.run_history_data, run_history],
                                ignore_index=True
                            )
                    
                    # Create and display editable dataframe with Expected Sales and Standard Deviation
                    if st.session_state.store_data is not None:
                        st.write("#### Store Forecasting Simulation")
                        st.write("You can edit the Expected Sales and Standard Deviation values below, then click SIMULATE again to run with your custom values.")
                        
                        # Get the current store data
                        store_df = st.session_state.store_data.copy()
                        
                        # Format the dataframe for display
                        formatted_df = store_df.copy()
                        formatted_df['Expected Sales'] = formatted_df['Expected Sales'].astype(float)
                        formatted_df['Avg Weekly Sales'] = formatted_df['Avg Weekly Sales'].round(1)
                        formatted_df['Avg Weekly Error'] = formatted_df['Avg Weekly Error'].round(1)
                        
                        # Create a copy of the data for editing (exclude the Total and Agg rows)
                        editable_df = formatted_df[formatted_df['Store'].isin(range(1, num_stores + 1))].copy()
                        total_agg_df = formatted_df[~formatted_df['Store'].isin(range(1, num_stores + 1))].copy()
                        
                        # Allow editing of Expected Sales and Standard Deviation for individual stores
                        st.write("##### Regular Stores (editable):")
                        edited_df = st.data_editor(
                            editable_df,
                            column_config={
                                "Store": st.column_config.NumberColumn("Store", disabled=True),
                                "Expected Sales": st.column_config.NumberColumn(
                                    "Expected Sales",
                                    help="Edit the expected sales for this store",
                                    min_value=1.0,
                                    step=1.0,
                                    format="%.1f"
                                ),
                                "Standard Deviation": st.column_config.NumberColumn(
                                    "Standard Deviation",
                                    help="Edit the standard deviation for this store",
                                    min_value=1.0,
                                    step=0.5,
                                    format="%.1f"
                                ),
                                "Avg Weekly Sales": st.column_config.NumberColumn("Avg Weekly Sales", disabled=True),
                                "Avg Weekly Error": st.column_config.NumberColumn("Avg Weekly Error", disabled=True),
                                "Avg Error %": st.column_config.NumberColumn("Avg Error %", disabled=True)
                            },
                            hide_index=True,
                            key="editable_store_data"
                        )
                        
                        # Display the Total and Agg rows separately (read-only)
                        st.write("##### Total and Aggregated Values:")
                        st.dataframe(total_agg_df, hide_index=True)
                        
                        # Update the session state with the edited values
                        if edited_df is not None:
                            # Combine the edited store data with the Total and Agg rows
                            combined_df = pd.concat([edited_df, total_agg_df], ignore_index=True)
                            # Update total Expected Sales
                            total_expected = edited_df['Expected Sales'].sum()
                            combined_df.loc[combined_df['Store'] == 'Total', 'Expected Sales'] = total_expected
                            combined_df.loc[combined_df['Store'] == 'Agg', 'Expected Sales'] = total_expected
                            # Update total Standard Deviation
                            std_total = round(np.sqrt(sum([s**2 for s in edited_df['Standard Deviation']])))
                            std_agg = round(std_total / 2)  # Lower for aggregated
                            combined_df.loc[combined_df['Store'] == 'Total', 'Standard Deviation'] = std_total
                            combined_df.loc[combined_df['Store'] == 'Agg', 'Standard Deviation'] = std_agg
                            
                            # Save the updated data to session state (only Store, Expected Sales, and Std Dev)
                            st.session_state.store_data = combined_df
                        
                        # Display explanation of metrics
                        st.markdown("""
                        - **Avg Weekly Error** = Average of the absolute value of the weekly error for the # of simulated weeks.
                        - **Avg Error %** = Average Weekly Error divided by Expected Sales.
                        - **Total** = Sum total of the individual Store values.
                        - **Agg** = Aggregated total of the individual store values where negative errors offset positive errors. The absolute value of the error is not used.
                        """)
                    
                    # Show run history if exists
                    if len(st.session_state.run_history_data) > 0:
                        st.write("#### Simulation Run History")
                        
                        # Display the run history data
                        st.dataframe(st.session_state.run_history_data)
                        
                        # Create a bar chart to compare total vs aggregated error
                        if PLOTLY_AVAILABLE:
                            run_history = st.session_state.run_history_data
                            
                            # Create a DataFrame with the error values in long format for plotting
                            plot_data = []
                            for _, row in run_history.iterrows():
                                plot_data.append({
                                    'Run #': row['Run #'],
                                    'Error Type': 'Total Avg Error %',
                                    'Value': row['Total Avg Error %']
                                })
                                plot_data.append({
                                    'Run #': row['Run #'],
                                    'Error Type': 'Agg Avg Error %',
                                    'Value': row['Agg Avg Error %']
                                })
                            
                            plot_df = pd.DataFrame(plot_data)
                            
                            # Create the bar chart
                            fig = px.bar(
                                plot_df,
                                x='Run #',
                                y='Value',
                                color='Error Type',
                                barmode='group',
                                color_discrete_map={
                                    'Total Avg Error %': '#B22222',  # Red
                                    'Agg Avg Error %': '#1E4175'     # Blue
                                },
                                title="Comparison of Total vs Aggregated Forecast Error"
                            )
                            
                            fig.update_layout(
                                xaxis_title="Run #",
                                yaxis_title="Error",
                                legend_title="Error Type"
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            # Matplotlib fallback
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Get unique run numbers and set up the x positions
                            runs = run_history['Run #'].unique()
                            x = np.arange(len(runs))
                            width = 0.35
                            
                            # Plot the two error types
                            ax.bar(x - width/2, run_history['Total Avg Error %'], width, label='Total Avg Error %', color='#B22222')
                            ax.bar(x + width/2, run_history['Agg Avg Error %'], width, label='Agg Avg Error %', color='#1E4175')
                            
                            # Add labels, title, and legend
                            ax.set_xlabel('Run #')
                            ax.set_ylabel('Error')
                            ax.set_title('Comparison of Total vs Aggregated Forecast Error')
                            ax.set_xticks(x)
                            ax.set_xticklabels(runs)
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)

            # Add Inventory Turnover Simulation section
            if st.sidebar.checkbox("Inventory Turnover Simulation"):
                st.write("### Inventory Turnover Simulation")
                st.write("Visualize the rate at which inventory is sold based on turnover rates.")
                
                # Add How to Use instructions
                with st.expander("How to Use", expanded=True):
                    st.markdown("""
                    **What is Inventory Turnover?**
                    Inventory turnover measures how many times a company sells and replaces its inventory in a given period. 
                    A higher turnover rate indicates more efficient inventory management.
                    
                    **How to use this simulation:**
                    1. **Adjust turnover rates** using the +/- buttons or input fields:
                       - Higher turnover = faster inventory depletion
                       - Default values: Company 1 (8), Company 2 (16)
                    
                    2. **View days inventory** values which show how many days inventory stays in stock:
                       - Days Inventory = 365 / Turnover Rate
                       - Lower days inventory = more efficient inventory management
                    
                    3. **Click START SELLING** to begin the simulation:
                       - Watch as inventory boxes disappear in real-time
                       - The company with higher turnover will deplete inventory faster
                       - Click again to pause/reset the simulation
                    
                    4. **Compare companies** to see the impact of different turnover rates
                       - Try making Company 1 twice as fast as Company 2
                       - Observe how changing turnover affects inventory depletion speed
                    """)
                
                # Initialize session state for simulation if not exists
                if 'simulation_running' not in st.session_state:
                    st.session_state.simulation_running = False
                    st.session_state.inventory_company1 = 15  # Number of boxes in pyramid
                    st.session_state.inventory_company2 = 15  # Number of boxes in pyramid
                    st.session_state.last_update_time = None
                
                # Create control inputs with increment/decrement buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Company 1")
                    
                    # Turnover rate for Company 1
                    st.write("Turnover Company 1")
                    turnover_col1_minus, turnover_col1_value, turnover_col1_plus = st.columns([1, 3, 1])
                    with turnover_col1_minus:
                        if st.button("-", key="turnover1_minus"):
                            if 'turnover_company1' in st.session_state and st.session_state.turnover_company1 > 1:
                                st.session_state.turnover_company1 -= 1
                    
                    with turnover_col1_value:
                        if 'turnover_company1' not in st.session_state:
                            st.session_state.turnover_company1 = 8
                        turnover_company1 = st.number_input("", 
                                                            min_value=1, 
                                                            max_value=100, 
                                                            value=st.session_state.turnover_company1,
                                                            key="turnover_company1_input",
                                                            label_visibility="collapsed")
                        st.session_state.turnover_company1 = turnover_company1
                    
                    with turnover_col1_plus:
                        if st.button("+", key="turnover1_plus"):
                            if 'turnover_company1' in st.session_state:
                                st.session_state.turnover_company1 += 1
                    
                    # Days Inventory for Company 1
                    st.write("Days Inventory 1")
                    days_col1_minus, days_col1_value, days_col1_plus = st.columns([1, 3, 1])
                    with days_col1_minus:
                        if st.button("-", key="days1_minus"):
                            if 'days_inventory1' in st.session_state and st.session_state.days_inventory1 > 1:
                                st.session_state.days_inventory1 -= 1
                    
                    with days_col1_value:
                        if 'days_inventory1' not in st.session_state:
                            st.session_state.days_inventory1 = 46
                        days_inventory1 = st.number_input("", 
                                                          min_value=1.0, 
                                                          max_value=365.0, 
                                                          value=float(st.session_state.days_inventory1),
                                                          key="days_inventory1_input",
                                                          label_visibility="collapsed")
                        st.session_state.days_inventory1 = days_inventory1
                    
                    with days_col1_plus:
                        if st.button("+", key="days1_plus"):
                            if 'days_inventory1' in st.session_state:
                                st.session_state.days_inventory1 += 1
                
                with col2:
                    st.write("#### Company 2")
                    
                    # Turnover rate for Company 2
                    st.write("Turnover Company 2")
                    turnover_col2_minus, turnover_col2_value, turnover_col2_plus = st.columns([1, 3, 1])
                    with turnover_col2_minus:
                        if st.button("-", key="turnover2_minus"):
                            if 'turnover_company2' in st.session_state and st.session_state.turnover_company2 > 1:
                                st.session_state.turnover_company2 -= 1
                    
                    with turnover_col2_value:
                        if 'turnover_company2' not in st.session_state:
                            st.session_state.turnover_company2 = 16
                        turnover_company2 = st.number_input("", 
                                                            min_value=1, 
                                                            max_value=100, 
                                                            value=st.session_state.turnover_company2,
                                                            key="turnover_company2_input",
                                                            label_visibility="collapsed")
                        st.session_state.turnover_company2 = turnover_company2
                    
                    with turnover_col2_plus:
                        if st.button("+", key="turnover2_plus"):
                            if 'turnover_company2' in st.session_state:
                                st.session_state.turnover_company2 += 1
                    
                    # Days Inventory for Company 2
                    st.write("Days Inventory 2")
                    days_col2_minus, days_col2_value, days_col2_plus = st.columns([1, 3, 1])
                    with days_col2_minus:
                        if st.button("-", key="days2_minus"):
                            if 'days_inventory2' in st.session_state and st.session_state.days_inventory2 > 1:
                                st.session_state.days_inventory2 -= 1
                    
                    with days_col2_value:
                        if 'days_inventory2' not in st.session_state:
                            st.session_state.days_inventory2 = 22.8
                        days_inventory2 = st.number_input("", 
                                                          min_value=1.0, 
                                                          max_value=365.0, 
                                                          value=float(st.session_state.days_inventory2),
                                                          key="days_inventory2_input",
                                                          label_visibility="collapsed")
                        st.session_state.days_inventory2 = days_inventory2
                    
                    with days_col2_plus:
                        if st.button("+", key="days2_plus"):
                            if 'days_inventory2' in st.session_state:
                                st.session_state.days_inventory2 += 1
                
                # START SELLING button
                start_button_style = """
                <style>
                div.stButton > button {
                    background-color: #4B70E2;
                    color: white;
                    font-weight: bold;
                    padding: 0.5rem 1rem;
                    width: 100%;
                }
                </style>
                """
                st.markdown(start_button_style, unsafe_allow_html=True)
                
                if st.button("START SELLING", key="start_selling"):
                    st.session_state.simulation_running = not st.session_state.simulation_running
                    # Reset inventory if simulation was off and turning on
                    if st.session_state.simulation_running:
                        st.session_state.inventory_company1 = 15
                        st.session_state.inventory_company2 = 15
                        st.session_state.last_update_time = time.time()
                
                # Display the inventory visualization
                col1, col2 = st.columns(2)
                
                # Calculate inventory levels based on turnover rates if simulation is running
                if st.session_state.simulation_running and st.session_state.last_update_time is not None:
                    current_time = time.time()
                    elapsed_time = current_time - st.session_state.last_update_time
                    
                    # Scale elapsed time for faster visualization (1 second = 1 day)
                    scaled_time = elapsed_time * 10
                    
                    # Calculate boxes to remove based on turnover rates
                    company1_rate = st.session_state.turnover_company1 / 365  # Daily sales rate
                    company2_rate = st.session_state.turnover_company2 / 365  # Daily sales rate
                    
                    # Calculate inventory to remove
                    company1_remove = scaled_time * company1_rate 
                    company2_remove = scaled_time * company2_rate
                    
                    # Update inventory levels
                    st.session_state.inventory_company1 = max(0, st.session_state.inventory_company1 - company1_remove)
                    st.session_state.inventory_company2 = max(0, st.session_state.inventory_company2 - company2_remove)
                    
                    # Update last update time
                    st.session_state.last_update_time = current_time
                
                # Function to create pyramid visualization
                def create_pyramid(inventory_left, max_inventory=15, color="#3178c6"):
                    # Calculate how many boxes to show based on inventory left
                    boxes_to_show = math.ceil(inventory_left)
                    boxes_to_show = min(boxes_to_show, max_inventory)  # Cap at max inventory
                    
                    # Create HTML for pyramid
                    pyramid_html = "<div style='display: flex; flex-direction: column; align-items: center;'>"
                    
                    # Each row of the pyramid
                    boxes_per_row = [1, 2, 3, 4, 5]  # 5 rows with increasing boxes
                    boxes_used = 0
                    
                    for row_boxes in boxes_per_row:
                        row_html = "<div style='display: flex; flex-direction: row;'>"
                        for i in range(row_boxes):
                            if boxes_used < boxes_to_show:
                                # Box is visible
                                row_html += f"<div style='width: 30px; height: 30px; margin: 2px; background-color: {color};'></div>"
                            else:
                                # Box is invisible (removed)
                                row_html += "<div style='width: 30px; height: 30px; margin: 2px;'></div>"
                            boxes_used += 1
                        row_html += "</div>"
                        pyramid_html += row_html
                    
                    pyramid_html += "</div>"
                    return pyramid_html
                
                # Display company names
                with col1:
                    st.markdown("<h4 style='text-align: center;'>Company 1</h4>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<h4 style='text-align: center;'>Company 2</h4>", unsafe_allow_html=True)
                
                # Display pyramids
                with col1:
                    company1_pyramid = create_pyramid(st.session_state.inventory_company1)
                    st.markdown(company1_pyramid, unsafe_allow_html=True)
                
                with col2:
                    company2_pyramid = create_pyramid(st.session_state.inventory_company2)
                    st.markdown(company2_pyramid, unsafe_allow_html=True)
                
                # Add auto-refresh for animation (every 200ms)
                if st.session_state.simulation_running:
                    st.empty()
                    time.sleep(0.1)
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()