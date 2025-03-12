import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
            columns = self.df.select_dtypes(include=[np.number]).columns

        if not columns:
            st.warning("No numeric columns available for outlier detection.")
            return

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
                    contamination = st.slider("Contamination factor:", 0.01, 0.5, 0.1, 0.01)
                    outliers = cleaner.detect_outliers(contamination=contamination)
                    if outliers is not None and not outliers.empty:
                        st.write("#### Outlier Rows:")
                        st.dataframe(outliers)

                # Feature Engineering
                if st.checkbox("Encode Categorical Variables"):
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        selected_cats = st.multiselect("Select categorical columns to encode:", categorical_cols)
                        if selected_cats:
                            cleaner.encode_categorical(selected_cats)

                if st.checkbox("Scale Numeric Features"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_nums = st.multiselect("Select numeric columns to scale:", numeric_cols)
                        if selected_nums:
                            cleaner.scale_features(selected_nums)

                # Basic ML Model
                if st.checkbox("Train Basic ML Model"):
                    target_col = st.selectbox("Select target column:", df.columns)
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