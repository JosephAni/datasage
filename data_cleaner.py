import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from category_encoders import TargetEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import re
import warnings

# Add plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Install with 'pip install plotly' for interactive plots.")

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

    def train_model(self, target_column, problem_type='classification', test_size=0.2):
        """
        Train a basic ML model and return metrics.
        
        Args:
            target_column (str): Target column for prediction
            problem_type (str): Either 'classification' or 'regression'
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Dictionary of model metrics
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
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
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Feature importance
                    importance = dict(zip(X.columns, model.feature_importances_))
                    
                    return {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'feature_importance': importance
                    }
                    
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
            pd.DataFrame: The cleaned DataFrame.
        """
        return self.df.copy()

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

            st.write("Original DataFrame:")
            st.dataframe(df)

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
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            metrics = cleaner.train_model(target_col, problem_type)
                            if metrics:
                                st.write("#### Model Performance:")
                                for metric, value in metrics.items():
                                    if metric != 'feature_importance':
                                        st.write(f"{metric}: {value:.4f}")
                                
                                st.write("#### Feature Importance:")
                                importance_df = pd.DataFrame.from_dict(
                                    metrics['feature_importance'], 
                                    orient='index', 
                                    columns=['importance']
                                ).sort_values('importance', ascending=False)
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
                
                # Column selection for visualizations
                viz_col = st.selectbox("Select column for visualization:", df.columns)
                
                # Visualization options
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Distribution Plot", "Pie Chart", "Box Plot"]
                )
                
                if viz_type == "Distribution Plot":
                    cleaner.plot_distribution(viz_col)
                elif viz_type == "Pie Chart":
                    cleaner.plot_pie_chart(viz_col)
                elif viz_type == "Box Plot":
                    cleaner.plot_box_plot(viz_col)
                
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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()