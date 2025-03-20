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

    def scale_features(self, columns=None):
        """
        Scale numeric features using StandardScaler.

        Args:
            columns (list, optional): List of columns to scale. If None, scales all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        if not columns:
            st.warning("No numeric columns available for scaling.")
            return

        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        st.info(f"Scaled numeric columns: {', '.join(columns)}")

    def train_model(self, target_column, problem_type='classification', test_size=0.2):
        """
        Train a basic ML model on the cleaned data.

        Args:
            target_column (str): The column to predict.
            problem_type (str): Either 'classification' or 'regression'.
            test_size (float): Proportion of data to use for testing.

        Returns:
            dict: Dictionary containing model performance metrics.
        """
        if target_column not in self.df.columns:
            st.error(f"Target column '{target_column}' not found in DataFrame.")
            return None

        # Prepare features and target
        X = self.df.drop([target_column, 'is_outlier'] if 'is_outlier' in self.df.columns else [target_column], axis=1)
        y = self.df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return {
                'accuracy': accuracy,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
        else:  # regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }

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
        if not pd.api.types.is_numeric_dtype(self.df[column]) or len(self.df[column].unique()) <= 10:
            value_counts = self.df[column].value_counts()
            fig = plt.figure(figsize=(10, 8))
            plt.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
            plt.title(f'Pie Chart of {column}')
            st.pyplot(fig)
            plt.close()
        else:
            st.warning(f"Pie chart not suitable for column '{column}' (too many unique values or numeric data)")

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
                        selected_nums = st.multiselect("Select numeric columns to scale:", numeric_cols)
                        if selected_nums:
                            cleaner.scale_features(selected_nums)
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