import pandas as pd
import numpy as np
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
                    print(f"Warning: Strategy '{strategy}' not applicable for column '{col}'. Skipping.")

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
                    print(f"Warning: Column '{col}' contains non-numeric values and will be excluded from outlier detection.")
            columns = numeric_cols

        if len(columns) == 0:
            print("Warning: No numeric columns available for outlier detection.")
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
        
        print(f"Outlier Detection Results")
        print(f"Found {n_outliers} potential outliers out of {len(self.df)} rows.")
        print(f"Analyzed columns: {', '.join(columns)}")
        
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
                print(f"Encoded categorical column: {col}")

    def scale_features(self, columns, scaler_type='standard', proceed_with_ids=True):
        """
        Scale numeric features in the dataset.
        
        Args:
            columns (list): List of columns to scale
            scaler_type (str): Type of scaling ('standard', 'minmax', 'robust')
            proceed_with_ids (bool): Whether to proceed with scaling if ID-like columns are detected.
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Verify columns exist
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")
        
        # Check for and warn about ID columns
        id_columns = []
        for col in columns:
            is_id, reason = self.is_likely_id_column(col)
            if is_id:
                id_columns.append((col, reason))
                
        if id_columns:
            print("Warning: The following columns appear to be IDs or codes that generally shouldn't be scaled:")
            for col, reason in id_columns:
                print(f"- {col}: {reason}")
            if not proceed_with_ids:
                raise ValueError("Scaling aborted as ID-like columns were detected and proceed_with_ids is False.")
        
        # Preview data before scaling
        print("Preview of selected columns:")
        preview_df = self.df[columns].head(5)
        print(preview_df)
        
        # Attempt to convert to numeric, showing errors for non-convertible columns
        numeric_conversion_issues = {}
        
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
                if numeric_vals.isna().sum() > self.df[col].isna().sum(): # Check if *new* NaNs were created
                    mask = self.df[col].notna() & numeric_vals.isna()
                    if mask.any():
                        examples = self.df.loc[mask, col].unique()[:3]  # Get up to 3 examples
                        numeric_conversion_issues[col] = examples
            except Exception as e:
                 raise ValueError(f"Error converting column {col} to numeric during pre-scaling check: {str(e)}")
        
        if numeric_conversion_issues:
            error_messages = ["Error: One or more columns contain values that cannot be converted to numeric format for scaling:"]
            for col, examples in numeric_conversion_issues.items():
                example_str = ', '.join([f"'{ex}'" for ex in examples])
                error_messages.append(f"- Column '{col}': Contains non-numeric values like {example_str}. Please preprocess these columns before scaling.")
            raise ValueError("\n".join(error_messages))
        
        # Proceed with scaling
        try:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            # Ensure data is numeric and handle potential NaNs from coercion before scaling.
            # We create a copy to avoid modifying the original df in case of errors during this stage.
            df_to_scale = self.df[columns].copy()
            for col in columns:
                # Attempt conversion, coercing errors to NaN
                df_to_scale[col] = pd.to_numeric(df_to_scale[col], errors='coerce')
                # Check if any new NaNs were introduced by coercion that weren't caught before
                if df_to_scale[col].isnull().sum() > self.df[columns][col].isnull().sum() and col not in numeric_conversion_issues:
                     print(f"Warning: Column '{col}' developed NaNs during final numeric conversion before scaling. This might indicate mixed types not caught earlier.")

            # Scalers generally cannot handle NaNs. Check for any NaNs in the data to be scaled.
            if df_to_scale.isnull().sum().sum() > 0:
                nan_cols = df_to_scale.columns[df_to_scale.isnull().any()].tolist()
                raise ValueError(f"Columns {nan_cols} contain NaN values after attempting numeric conversion. Please handle missing values before scaling.")
            
            # Scale and update the dataframe
            scaled_values = scaler.fit_transform(df_to_scale)
            scaled_df = pd.DataFrame(scaled_values, columns=columns, index=self.df.index)
            
            # Replace the original columns with scaled values
            for col in columns:
                self.df[col] = scaled_df[col]
            
            print(f"Successfully scaled {len(columns)} columns using {scaler_type} scaling.")
            print("Descriptive statistics of scaled columns:")
            print(self.df[columns].describe())
            
        except ValueError as ve: # Catch ValueErrors specifically to provide better context
            raise ValueError(f"Error during scaling preparation (e.g. non-numeric data, NaNs): {str(ve)}")
        except Exception as e:
            # General catch for other scaling issues
            raise RuntimeError(f"An unexpected error occurred during scaling: {str(e)}")

    def train_model(self, target_column, problem_type='classification', test_size=0.2, model_type='random_forest', 
                    proceed_with_id_target=False, id_feature_handling='exclude', 
                    proceed_high_cardinality_classification=False, impute_missing_features=True):
        """
        Train a ML model and return metrics.
        
        Args:
            target_column (str): Target column for prediction
            problem_type (str): Either 'classification' or 'regression'
            test_size (float): Proportion of data to use for testing
            model_type (str): Type of model to train (random_forest, knn, logistic_regression, etc.)
            proceed_with_id_target (bool): Whether to proceed if target column appears to be an ID. Defaults to False.
            id_feature_handling (str): How to handle ID columns in features ('exclude' or 'keep'). Defaults to 'exclude'.
            proceed_high_cardinality_classification (bool): Whether to proceed with classification if target has high cardinality. Defaults to False.
            impute_missing_features (bool): Whether to impute missing values in features using mean. Defaults to True.
            
        Returns:
            dict: Dictionary containing model metrics, the trained model, and any generated plot objects.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, classification_report # Added f1_score
        from sklearn.model_selection import train_test_split
        
        returned_plots = {}
        model_results = {}

        # Validate target column existence
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
            
        # Check if target column is an ID column
        is_id, reason = self.is_likely_id_column(target_column)
        if is_id:
            print(f"Warning: The selected target column '{target_column}' appears to be an ID or code column: {reason}")
            print("ID columns typically don't have meaningful patterns for ML models to learn.")
            if not proceed_with_id_target:
                print("Modeling aborted: Target column appears to be an ID and 'proceed_with_id_target' is False.")
                # Consider returning a specific status or raising an exception
                return {"error": "Target column identified as ID and proceed_with_id_target is False."}
            
        features = [col for col in self.df.columns if col != target_column]
        
        # Handle ID columns in features
        id_features_identified = []
        for col in features:
            is_id, id_reason = self.is_likely_id_column(col)
            if is_id:
                id_features_identified.append((col, id_reason))
                
        if id_features_identified:
            print("Warning: The following feature columns appear to be IDs or codes:")
            for col, id_reason in id_features_identified:
                print(f"- {col}: {id_reason}")
            print("ID columns can lead to overfitting and poor model generalization.")
            
            if id_feature_handling == 'exclude':
                features_to_exclude = [id_col for id_col, _ in id_features_identified]
                features = [col for col in features if col not in features_to_exclude]
                print(f"Excluded {len(features_to_exclude)} ID columns from features based on id_feature_handling='exclude'.")
            elif id_feature_handling == 'keep':
                print("Keeping all columns including identified ID-like features based on id_feature_handling='keep'.")
            else:
                raise ValueError(f"Invalid value for id_feature_handling: '{id_feature_handling}'. Choose 'exclude' or 'keep'.")

        target_col_data = self.df[target_column]
        print("\n--- Target Column Analysis ---")
        if pd.api.types.is_numeric_dtype(target_col_data):
            print(f"Target '{target_column}' (Numeric): Statistics - {target_col_data.describe().to_dict()}")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(target_col_data.dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {target_column}")
            returned_plots['target_distribution_histogram'] = fig 
            plt.close(fig)
        else:
            value_counts = target_col_data.value_counts()
            unique_count = len(value_counts)
            print(f"Target '{target_column}' (Categorical): {unique_count} unique values.")
            
            if problem_type == 'classification' and unique_count > 50:
                print(f"Warning: Target has {unique_count} unique values, very high for classification.")
                if not proceed_high_cardinality_classification:
                    print("Modeling aborted: High cardinality in classification target and 'proceed_high_cardinality_classification' is False.")
                    return {"error": "High cardinality target for classification and proceed_high_cardinality_classification is False."}
            
            display_count = min(10, unique_count)
            print(f"Top {display_count} most frequent values for '{target_column}':")
            print(value_counts.head(display_count).reset_index().rename(
                columns={'index': target_column, target_column: 'Count'}))
            
            fig, ax = plt.subplots(figsize=(10, 4))
            value_counts.head(display_count).plot.bar(ax=ax)
            ax.set_title(f"Top values of {target_column}")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            returned_plots['target_top_values_barchart'] = fig
            plt.close(fig)
        
        print("\n--- Feature Preparation ---")
        numeric_features = []
        non_numeric_to_exclude = []
        for col in features:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_features.append(col)
            else:
                # Attempt to convert to numeric, if this fails, it's truly non-numeric for ML
                try:
                    pd.to_numeric(self.df[col])
                    numeric_features.append(col) # If convertible, treat as numeric
                    print(f"Info: Feature '{col}' was converted to numeric for modeling.")
                except ValueError:
                     non_numeric_to_exclude.append(col)

        if non_numeric_to_exclude:
            print(f"Warning: The following non-numeric features will be excluded: {non_numeric_to_exclude}")
            features = [f for f in numeric_features if f not in non_numeric_to_exclude]
        else:
            features = numeric_features
            
        if not features:
            raise ValueError("No suitable (numeric or convertible) features available for modeling after exclusions.")
        print(f"Using {len(features)} features for modeling: {features}")
                
        X = self.df[features].copy()
        y = self.df[target_column].copy()

        # Convert all selected feature columns in X to numeric, coercing errors to NaN.
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce') # Coerce ensures conversion
        
        # Handle missing values in features (X)
        if X.isnull().sum().sum() > 0:
            if impute_missing_features:
                print(f"Features data (X) contains {X.isnull().sum().sum()} missing values. Imputing with mean.")
                imputer_X = SimpleImputer(strategy='mean')
                X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns, index=X.index)
                model_results['feature_imputation_strategy'] = 'mean'
            else:
                print(f"Features data (X) contains {X.isnull().sum().sum()} missing values. Dropping rows with any NaNs in X as impute_missing_features=False.")
                original_len = len(X)
                X = X.dropna()
                y = y.loc[X.index] # Align y with X after dropping NaNs
                print(f"Dropped {original_len - len(X)} rows from X and y due to NaNs in X.")
                model_results['feature_imputation_strategy'] = 'dropna'
        else:
            print("No missing values detected in features (X).")
        
        # Handle missing values in target (y)
        if y.isnull().sum() > 0:
            print(f"Target variable '{target_column}' contains {y.isnull().sum()} missing values. Rows with missing target values will be dropped.")
            original_len = len(y)
            valid_indices = y.dropna().index
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            print(f"Dropped {original_len - len(y)} rows due to NaNs in target. Modeling with {len(X)} rows.")

        if X.empty or y.empty:
            raise ValueError("No data available for modeling after handling missing values and feature selection.")

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
                    raise ValueError(f"Unknown model type: {model_type}")
                
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
                    print("Error: Target values are not suitable for classification. Try regression instead.")
                else:
                    print(f"Classification model error: {str(e)}")
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
            raise ValueError(f"Unknown problem type: {problem_type}")
            return None

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            # Replace st.info with print
            print(f"Info: Removed {removed_rows} duplicate rows.")

    def convert_datatypes(self, column_types):
        """
        Converts columns to specified datatypes.

        Args:
            column_types (dict): A dictionary where keys are column names and values are datatypes.
        """
        for column, dtype in column_types.items():
            if column not in self.df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame. Skipping conversion.")
                continue
            try:
                current_dtype = self.df[column].dtype
                # Avoid unnecessary conversions if already the target type (simplistic check)
                if (dtype == 'int' and pd.api.types.is_integer_dtype(current_dtype)) or \
                   (dtype == 'float' and pd.api.types.is_float_dtype(current_dtype)) or \
                   (dtype == 'category' and pd.api.types.is_categorical_dtype(current_dtype)) or \
                   (dtype == 'str' and pd.api.types.is_string_dtype(current_dtype)):
                    # print(f"Info: Column '{column}' is already effectively of type '{dtype}'. Skipping conversion.")
                    continue

                if dtype == 'int':
                    # Attempt to convert to numeric first, then to Int64 to handle NaNs gracefully
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype(float)
                elif dtype == 'category':
                    self.df[column] = self.df[column].astype('category')
                elif dtype == 'str':
                    self.df[column] = self.df[column].astype(str)
                elif dtype == 'datetime':
                    self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                else:
                    print(f"Warning: Unsupported dtype '{dtype}' for column '{column}'. Skipping.")
            except Exception as e:
                # Replace st.warning with print
                print(f"Warning: Error converting column '{column}' to type '{dtype}': {str(e)}. Column remains {self.df[column].dtype}.")

    def standardize_column_names(self):
        """
        Standardizes column names by converting to lowercase and replacing spaces with underscores.
        """
        self.df.columns = [str(col).lower().replace(' ', '_') for col in self.df.columns]
        # Replace st.info with print
        print("Info: Column names have been standardized.")

    def get_cleaned_data(self):
        """
        Returns the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        return self.df.copy()
        
    def reset(self):
        """
        Reset the DataFrame to its original state.
        """
        self.df = self.original_df.copy()
        print("Info: Data has been reset to original state.")

    def plot_distribution(self, column):
        """
        Create a distribution plot for a given column.
        For numeric columns, it generates a histogram and descriptive statistics.
        For categorical columns, it generates a bar chart and a frequency table.

        Args:
            column (str): The column to plot.

        Returns:
            dict: A dictionary containing the plot object ('plot') and 
                  relevant data ('statistics' for numeric, 'frequency_table' for categorical).
                  Returns None if the column is not found.
        """
        if column not in self.df.columns:
            print(f"Error: Column '{column}' not found in the dataset.")
            return None

        plot_data = {}
        fig = None

        if pd.api.types.is_numeric_dtype(self.df[column]):
            fig = plt.figure(figsize=(10, 6))
            plt.hist(self.df[column].dropna(), bins=30, edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            # Removed st.pyplot(fig) and plt.close()
            plot_data['plot'] = fig

            # Add descriptive statistics
            # Removed st.write("#### Descriptive Statistics:")
            stats_series = self.df[column].describe()
            # Removed st.write(stats)
            plot_data['statistics'] = stats_series
            plot_data['frequency_table'] = None # No frequency table for numeric

        else:
            # For categorical columns, create a bar chart
            value_counts = self.df[column].value_counts()
            fig = plt.figure(figsize=(10, 6))
            # Ensure index is string for plotting, especially if it contains mixed types or numbers
            plt.bar(value_counts.index.astype(str), value_counts.values)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(column)
            plt.ylabel('Count')
            # Removed st.pyplot(fig) and plt.close()
            plot_data['plot'] = fig

            # Add frequency table
            # Removed st.write("#### Frequency Table:")
            freq_df = pd.DataFrame({
                'Value': value_counts.index.astype(str),
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(self.df[column].dropna()) * 100).round(2) # Use dropna for correct percentage
            })
            # Removed st.write(freq_df)
            plot_data['frequency_table'] = freq_df
            plot_data['statistics'] = None # No descriptive stats like .describe() for categorical in this format
        
        return plot_data

    def plot_pie_chart(self, column, proceed_if_numeric_high_unique=False):
        """
        Create a pie chart for categorical columns.

        Args:
            column (str): The column to plot.
            proceed_if_numeric_high_unique (bool): Whether to proceed if column is numeric with many unique values. Defaults to False.

        Returns:
            matplotlib.figure.Figure or None: The pie chart figure, or None if plotting is aborted or an error occurs.
        """
        if column not in self.df.columns:
            print(f"Error: Column '{column}' not found in the dataset.")
            return None
            
        unique_count = self.df[column].nunique()
        
        if unique_count > 15:
            print(f"Error: Column '{column}' has {unique_count} unique values, which is too many for a meaningful pie chart.")
            print("Pie charts work best with 5-10 categories. Consider using a bar chart instead, or selecting a different column.")
            return None
            
        if pd.api.types.is_numeric_dtype(self.df[column]) and unique_count > 10:
            print(f"Warning: Column '{column}' appears to be numeric with {unique_count} unique values. A pie chart may not be the best visualization.")
            if not proceed_if_numeric_high_unique:
                print("Plotting of pie chart aborted as 'proceed_if_numeric_high_unique' is False.")
                return None
                
        value_counts = self.df[column].value_counts()
        plot_data_series = value_counts # Renamed for clarity
        
        if len(value_counts) > 8:
            top_n = 7
            top_values = value_counts.nlargest(top_n)
            other_sum = value_counts.iloc[top_n:].sum() # Use iloc for position based slicing
            
            # Create a new series with top categories + "Other"
            plot_values = list(top_values.values) + [other_sum]
            plot_indices = list(top_values.index.astype(str)) + ['Other']
            plot_data_series = pd.Series(plot_values, index=plot_indices)
            
            print(f"Info: Showing top {top_n} categories for column '{column}'. {len(value_counts) - top_n} smaller categories grouped as 'Other'.")
        
        fig = plt.figure(figsize=(10, 8))
        plt.pie(plot_data_series.values, labels=plot_data_series.index.astype(str), 
               autopct='%1.1f%%', startangle=90, 
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        plt.title(f'Pie Chart of {column}')
        plt.axis('equal')
        # Removed st.pyplot(fig) and plt.close()
        return fig
            
    # For backward compatibility, maintain the plot_pie_chart function
    # This implementation seems to refer to a plot_categorical_distribution method which is not visible yet.
    # If plot_categorical_distribution is also Streamlit-heavy, this will need further refactoring.
    # For now, just removing the st.warning.
    def plot_pie_chart_compatibility(self, column):
        """
        Create a pie chart for categorical columns.
        This function is maintained for backward compatibility.
        Consider using plot_categorical_distribution instead.
        
        Args:
            column (str): The column to plot.
        """
        print("Info: The plot_pie_chart_compatibility method is a wrapper. Consider using plot_pie_chart or plot_distribution directly.")
        # Assuming plot_categorical_distribution exists and is/will be refactored.
        # If it returns a figure, this wrapper should also return it.
        # return self.plot_categorical_distribution(column, plot_type='pie')
        # For now, let it call the primary plot_pie_chart to avoid NameError if plot_categorical_distribution doesn't exist or isn't refactored yet.
        return self.plot_pie_chart(column) 

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
            tuple: (forecast_results DataFrame, plotly figure) or (None, None) if error.
        """
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False

        if not PLOTLY_AVAILABLE:
            # Instead of st.error, raise an ImportError or print a message and return None
            print("Error: Plotly is required for forecasting visualizations. Please install with 'pip install plotly'")
            raise ImportError("Plotly is required for forecasting visualizations but not installed.")
        
        forecast_df = self.df.copy()
        
        # Date conversion
        if date_column not in forecast_df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        if demand_column not in forecast_df.columns:
            raise ValueError(f"Demand column '{demand_column}' not found.")

        # Try to detect format first, then general conversion
        detected_format = self.detect_date_format(forecast_df[date_column].astype(str).head())
        try:
            if detected_format:
                forecast_df[date_column] = pd.to_datetime(forecast_df[date_column], format=detected_format, errors='coerce')
            else:
                forecast_df[date_column] = pd.to_datetime(forecast_df[date_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Could not convert date column '{date_column}' to datetime: {e}")

        invalid_dates_mask = forecast_df[date_column].isna()
        if invalid_dates_mask.any():
            num_invalid_dates = invalid_dates_mask.sum()
            print(f"Warning: {num_invalid_dates} rows have invalid dates in column '{date_column}' and will be dropped.")
            forecast_df = forecast_df.dropna(subset=[date_column])
        
        if forecast_df.empty:
            raise ValueError("DataFrame is empty after dropping rows with invalid dates.")

        forecast_df = forecast_df.sort_values(by=date_column)
        
        if forecast_df[demand_column].isnull().any():
            print(f"Warning: Demand column '{demand_column}' contains missing values. Filling with forward fill then backward fill.")
            forecast_df[demand_column] = forecast_df[demand_column].ffill().bfill()
            if forecast_df[demand_column].isnull().any(): # If still null (e.g. all values were null)
                 raise ValueError(f"Demand column '{demand_column}' still contains NaNs after ffill/bfill. Please clean data.")
        
        results = None
        fig = None # Plotly figure
        
        try:
            if algorithm == 'prophet':
                from prophet import Prophet
                prophet_df = forecast_df.rename(columns={date_column: 'ds', demand_column: 'y'})[['ds', 'y']] # Select only ds, y
                
                prophet_config = {
                    'yearly_seasonality': seasonality == 'yearly' or seasonality == 'auto',
                    'weekly_seasonality': seasonality == 'weekly' or seasonality == 'auto',
                    'daily_seasonality': seasonality == 'daily' or seasonality == 'auto',
                }
                if seasonality not in ['auto', 'yearly', 'weekly', 'daily']:
                     print(f"Warning: Non-standard seasonality '{seasonality}' provided. Prophet will use its defaults or auto-detection.")
                     # Let Prophet decide defaults if a non-explicitly supported string is passed other than auto
                     model = Prophet()
                else:
                    # Pass only True seasonalities. Prophet handles False/None internally if not specified.
                    active_seasonalities = {k:v for k,v in prophet_config.items() if v}
                    model = Prophet(**active_seasonalities)

                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_periods)
                forecast = model.predict(future)
                
                # Ensure 'ds' is in forecast output for date reference
                if 'ds' not in forecast.columns:
                    raise ValueError("Prophet forecast output missing 'ds' column.")

                results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy() # Use .copy()
                results.rename(columns={'ds': 'date', 'yhat': 'demand'}, inplace=True)
                
                # Create the Plotly figure
                fig = go.Figure()
                
                # Historical values
                fig.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    name='Historical',
                    mode='lines',
                    line=dict(color='blue')
                ))
                # Forecast values
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
                    y=results['yhat_upper'], 
                    name='Upper Bound', 
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=results['date'], 
                    y=results['yhat_lower'], 
                    name='Lower Bound', 
                    mode='lines', 
                    line=dict(width=0), 
                    fillcolor='rgba(255, 0, 0, 0.2)', # light red transparent fill
                    fill='tonexty', 
                    showlegend=False
                ))
                
                # Single call to update_layout for the main forecast plot
                fig.update_layout(
                    title=f'Demand Forecast ({forecast_periods} periods) - Prophet', 
                    xaxis_title='Date', 
                    yaxis_title='Demand', 
                    hovermode='x unified'
                )
                
                # The Prophet components plot (which used st.plotly_chart) is omitted here.
                # If needed, model.plot_components(forecast) could be returned or handled separately.

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
                    title=f'Demand Forecast ({forecast_periods} periods) - ARIMA',
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
                    title=f'Demand Forecast ({forecast_periods} periods) - Exponential Smoothing',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    hovermode='x unified'
                )
                
            return results, fig
        
        except Exception as e:
            # Changed st.error to print for logging
            print(f"Error: Forecasting error encountered: {str(e)}")
            import traceback
            # Changed st.error to print for logging traceback
            print(traceback.format_exc())
            # Consider re-raising a custom exception or ensuring the calling code expects (None, None) on error
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
        
        Returns:
            plotly.graph_objects.Figure or None: The Plotly figure object, or None if Plotly is not available.
        """
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False

        if components is None:
            components = ['Trend', 'Seasonality', 'Promotions']
            
        if not PLOTLY_AVAILABLE:
            # Changed st.error to raise ImportError
            print("Error: Plotly is required for plot_demand_simulation. Please install with 'pip install plotly'")
            raise ImportError("Plotly is required for plot_demand_simulation but not installed.")
            
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
        return fig # Return the figure object

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
            print(f"Error: Missing required columns: {', '.join(missing)}")
            return None, None, None

        # Create working copy
        clv_df = self.df[required_cols].copy()

        # Convert date column to datetime
        try:
            clv_df[invoice_date_col] = pd.to_datetime(clv_df[invoice_date_col])
        except Exception as e:
            print(f"Error: Error converting {invoice_date_col} to datetime: {str(e)}")
            return None, None, None

        # Convert numeric columns
        try:
            clv_df[quantity_col] = pd.to_numeric(clv_df[quantity_col], errors='coerce')
            clv_df[unit_price_col] = pd.to_numeric(clv_df[unit_price_col], errors='coerce')
        except Exception as e:
            print(f"Error: Error converting quantity/price to numeric: {str(e)}")
            return None, None, None

        # Calculate total price
        clv_df['total_price'] = clv_df[quantity_col] * clv_df[unit_price_col]

        # Remove invalid transactions
        initial_rows = len(clv_df)
        clv_df = clv_df.dropna(subset=[customer_id_col, 'total_price'])
        clv_df = clv_df[clv_df['total_price'] > 0]
        
        if len(clv_df) < initial_rows:
            print(f"Warning: Removed {initial_rows - len(clv_df)} invalid transactions")

        if clv_df.empty:
            print("Error: No valid transactions remaining after cleaning")
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
            print(f"Error: Missing required columns: {', '.join(missing)}")
            return None
            
        # Convert columns to numeric if needed
        for col in [actual_column, forecast_column]:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    print(f"Error: Could not convert {col} to numeric values")
                    return None
                    
        # Create working copy
        accuracy_df = self.df[required_cols].copy()
        
        # Remove rows with missing values
        initial_rows = len(accuracy_df)
        accuracy_df = accuracy_df.dropna()
        
        if len(accuracy_df) < initial_rows:
            print(f"Warning: Removed {initial_rows - len(accuracy_df)} rows with missing values")
            
        if accuracy_df.empty:
            print("Error: No valid data remaining after cleaning")
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
    def inventory_turnover_simulation(self, turnover_company1=8, turnover_company2=16, initial_inventory=15, days=60):
        """
        Simulate inventory turnover for two companies over a given number of days.
        Args:
            turnover_company1 (int): Turnover rate for Company 1 (times per year)
            turnover_company2 (int): Turnover rate for Company 2 (times per year)
            initial_inventory (int): Initial inventory level for both companies
            days (int): Number of days to simulate
        Returns:
            dict: Dictionary with daily inventory levels for both companies
        """
        inventory_levels_1 = []
        inventory_levels_2 = []
        inventory_1 = initial_inventory
        inventory_2 = initial_inventory
        
        # Calculate daily depletion rates
        daily_depletion_1 = turnover_company1 / 365 * initial_inventory
        daily_depletion_2 = turnover_company2 / 365 * initial_inventory

        for day in range(days + 1):
            inventory_levels_1.append(max(0, inventory_1))
            inventory_levels_2.append(max(0, inventory_2))
            inventory_1 -= daily_depletion_1
            inventory_2 -= daily_depletion_2
            if inventory_1 < 0 and inventory_2 < 0:
                break

        return {
            'days': list(range(len(inventory_levels_1))),
            'company1_inventory': inventory_levels_1,
            'company2_inventory': inventory_levels_2
        }