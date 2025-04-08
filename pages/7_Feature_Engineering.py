import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import io

# Add parent directory to path to import from data_cleaner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import safe_display_dataframe
except ImportError:
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        # Fix PyArrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🔧",
    layout="wide"
)

st.markdown("# Feature Engineering")
st.sidebar.header("Feature Engineering")
st.write(
    """
    This page provides tools for feature engineering:
    - Transform variables
    - Create new features 
    - Normalize/standardize data
    - Encode categorical variables
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

# Get data from session state
df = get_data()

if df is None:
    st.info("No data loaded. Please upload data on the Home page.")
else:
    # Initialize transformed dataframe in session state if needed
    if 'transformed_df' not in st.session_state:
        st.session_state.transformed_df = df.copy()
    
    transformed_df = st.session_state.transformed_df
    
    st.write("### Data Preview")
    st.write(f"Dataset has {transformed_df.shape[0]} rows and {transformed_df.shape[1]} columns.")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        safe_display_dataframe(transformed_df.head(10))
    
    # Feature Engineering Operations
    st.write("### Feature Engineering Operations")
    
    # Choose operation type
    operation = st.selectbox(
        "Select Operation", 
        ["Variable Transformation", "Feature Creation", "Scaling/Normalization", "Categorical Encoding", "Binning"]
    )
    
    # Get column types
    numeric_cols = transformed_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = transformed_df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in transformed_df.columns if pd.api.types.is_datetime64_dtype(transformed_df[col])]
    all_cols = transformed_df.columns.tolist()
    
    # Variable Transformation
    if operation == "Variable Transformation":
        st.write("#### Transform Variables")
        st.write("Apply mathematical transformations to numeric variables.")
        
        # Select columns to transform
        transform_cols = st.multiselect(
            "Select columns to transform",
            options=numeric_cols,
            default=[]
        )
        
        if transform_cols:
            # Select transformation type
            transform_type = st.selectbox(
                "Select transformation type",
                ["Log", "Square Root", "Square", "Cube", "Box-Cox", "Yeo-Johnson", "Inverse", "Z-Score"]
            )
            
            if st.button("Apply Transformation"):
                with st.spinner("Transforming data..."):
                    # Apply transformation
                    if transform_type == "Log":
                        # Handle zero/negative values
                        for col in transform_cols:
                            min_val = transformed_df[col].min()
                            if min_val <= 0:
                                offset = abs(min_val) + 1
                                transformed_df[f"{col}_log"] = np.log(transformed_df[col] + offset)
                                st.info(f"Added offset of {offset} to {col} before log transform to handle non-positive values.")
                            else:
                                transformed_df[f"{col}_log"] = np.log(transformed_df[col])
                    
                    elif transform_type == "Square Root":
                        for col in transform_cols:
                            min_val = transformed_df[col].min()
                            if min_val < 0:
                                offset = abs(min_val) + 1
                                transformed_df[f"{col}_sqrt"] = np.sqrt(transformed_df[col] + offset)
                                st.info(f"Added offset of {offset} to {col} before square root to handle negative values.")
                            else:
                                transformed_df[f"{col}_sqrt"] = np.sqrt(transformed_df[col])
                    
                    elif transform_type == "Square":
                        for col in transform_cols:
                            transformed_df[f"{col}_squared"] = transformed_df[col] ** 2
                    
                    elif transform_type == "Cube":
                        for col in transform_cols:
                            transformed_df[f"{col}_cubed"] = transformed_df[col] ** 3
                    
                    elif transform_type == "Box-Cox":
                        from scipy import stats
                        for col in transform_cols:
                            min_val = transformed_df[col].min()
                            if min_val <= 0:
                                offset = abs(min_val) + 1
                                transformed_col, lambda_val = stats.boxcox(transformed_df[col] + offset)
                                transformed_df[f"{col}_boxcox"] = transformed_col
                                st.info(f"Added offset of {offset} to {col} before Box-Cox. Lambda: {lambda_val:.4f}")
                            else:
                                transformed_col, lambda_val = stats.boxcox(transformed_df[col])
                                transformed_df[f"{col}_boxcox"] = transformed_col
                                st.info(f"Lambda value for {col}: {lambda_val:.4f}")
                    
                    elif transform_type == "Yeo-Johnson":
                        pt = PowerTransformer(method='yeo-johnson')
                        for col in transform_cols:
                            transformed_df[f"{col}_yeojohnson"] = pt.fit_transform(transformed_df[[col]])
                    
                    elif transform_type == "Inverse":
                        for col in transform_cols:
                            # Handle zero values
                            if (transformed_df[col] == 0).any():
                                st.warning(f"Column {col} contains zeros. Adding small constant (0.001) to avoid division by zero.")
                                transformed_df[f"{col}_inverse"] = 1 / (transformed_df[col] + 0.001)
                            else:
                                transformed_df[f"{col}_inverse"] = 1 / transformed_df[col]
                    
                    elif transform_type == "Z-Score":
                        for col in transform_cols:
                            transformed_df[f"{col}_zscore"] = (transformed_df[col] - transformed_df[col].mean()) / transformed_df[col].std()
                    
                    # Update session state
                    st.session_state.transformed_df = transformed_df
                    
                    # Show before and after plots
                    st.write("#### Transformation Results")
                    
                    for col in transform_cols:
                        transformed_col = [c for c in transformed_df.columns if c.startswith(f"{col}_")][0]
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original distribution
                        sns.histplot(transformed_df[col], kde=True, ax=axes[0])
                        axes[0].set_title(f"Original: {col}")
                        
                        # Transformed distribution
                        sns.histplot(transformed_df[transformed_col], kde=True, ax=axes[1])
                        axes[1].set_title(f"Transformed: {transformed_col}")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Feature Creation
    elif operation == "Feature Creation":
        st.write("#### Create New Features")
        st.write("Create new features from existing ones.")
        
        # Feature creation methods
        creation_method = st.selectbox(
            "Select creation method",
            ["Arithmetic Operation", "Date Features", "Polynomial Features", "Binning", "Custom Formula"]
        )
        
        if creation_method == "Arithmetic Operation":
            col1 = st.selectbox("Select first column", numeric_cols)
            operation = st.selectbox("Select operation", ["+", "-", "*", "/"])
            col2_option = st.radio("Second operand type", ["Column", "Constant"])
            
            if col2_option == "Column":
                col2 = st.selectbox("Select second column", numeric_cols)
                new_name = st.text_input("New feature name", f"{col1}_{operation}_{col2}")
                
                if st.button("Create Feature"):
                    with st.spinner("Creating feature..."):
                        if operation == "+":
                            transformed_df[new_name] = transformed_df[col1] + transformed_df[col2]
                        elif operation == "-":
                            transformed_df[new_name] = transformed_df[col1] - transformed_df[col2]
                        elif operation == "*":
                            transformed_df[new_name] = transformed_df[col1] * transformed_df[col2]
                        elif operation == "/":
                            # Handle division by zero
                            if (transformed_df[col2] == 0).any():
                                st.warning(f"Column {col2} contains zeros. Adding small constant to avoid division by zero.")
                                transformed_df[new_name] = transformed_df[col1] / (transformed_df[col2] + 0.001)
                            else:
                                transformed_df[new_name] = transformed_df[col1] / transformed_df[col2]
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Created new feature: {new_name}")
            else:
                constant = st.number_input("Enter constant value", value=1.0)
                new_name = st.text_input("New feature name", f"{col1}_{operation}_{constant}")
                
                if st.button("Create Feature"):
                    with st.spinner("Creating feature..."):
                        if operation == "+":
                            transformed_df[new_name] = transformed_df[col1] + constant
                        elif operation == "-":
                            transformed_df[new_name] = transformed_df[col1] - constant
                        elif operation == "*":
                            transformed_df[new_name] = transformed_df[col1] * constant
                        elif operation == "/":
                            if constant == 0:
                                st.error("Cannot divide by zero.")
                            else:
                                transformed_df[new_name] = transformed_df[col1] / constant
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Created new feature: {new_name}")
        
        elif creation_method == "Date Features":
            if datetime_cols:
                date_col = st.selectbox("Select date column", datetime_cols)
                
                date_features = st.multiselect(
                    "Select date features to extract",
                    ["Year", "Month", "Day", "Day of Week", "Quarter", "Week of Year", "Is Weekend", "Hour", "Minute"]
                )
                
                if date_features and st.button("Extract Date Features"):
                    with st.spinner("Extracting date features..."):
                        # Extract selected features
                        if "Year" in date_features:
                            transformed_df[f"{date_col}_year"] = transformed_df[date_col].dt.year
                        
                        if "Month" in date_features:
                            transformed_df[f"{date_col}_month"] = transformed_df[date_col].dt.month
                        
                        if "Day" in date_features:
                            transformed_df[f"{date_col}_day"] = transformed_df[date_col].dt.day
                        
                        if "Day of Week" in date_features:
                            transformed_df[f"{date_col}_dayofweek"] = transformed_df[date_col].dt.dayofweek
                        
                        if "Quarter" in date_features:
                            transformed_df[f"{date_col}_quarter"] = transformed_df[date_col].dt.quarter
                        
                        if "Week of Year" in date_features:
                            transformed_df[f"{date_col}_weekofyear"] = transformed_df[date_col].dt.isocalendar().week
                        
                        if "Is Weekend" in date_features:
                            transformed_df[f"{date_col}_is_weekend"] = transformed_df[date_col].dt.dayofweek >= 5
                        
                        if "Hour" in date_features:
                            if hasattr(transformed_df[date_col].dt, 'hour'):
                                transformed_df[f"{date_col}_hour"] = transformed_df[date_col].dt.hour
                            else:
                                st.warning(f"Column {date_col} does not have hour information.")
                        
                        if "Minute" in date_features:
                            if hasattr(transformed_df[date_col].dt, 'minute'):
                                transformed_df[f"{date_col}_minute"] = transformed_df[date_col].dt.minute
                            else:
                                st.warning(f"Column {date_col} does not have minute information.")
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Extracted date features from {date_col}")
            else:
                st.warning("No datetime columns found. Convert a column to datetime first.")
        
        elif creation_method == "Polynomial Features":
            st.write("Generate polynomial and interaction features.")
            
            poly_cols = st.multiselect("Select columns for polynomial features", numeric_cols)
            degree = st.slider("Polynomial degree", 2, 5, 2)
            include_interactions = st.checkbox("Include interaction terms", True)
            
            if poly_cols and st.button("Generate Polynomial Features"):
                with st.spinner("Generating polynomial features..."):
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    # Create polynomial features
                    poly = PolynomialFeatures(
                        degree=degree, 
                        include_bias=False, 
                        interaction_only=not include_interactions
                    )
                    
                    poly_features = poly.fit_transform(transformed_df[poly_cols])
                    feature_names = poly.get_feature_names_out(poly_cols)
                    
                    # Add new features to dataframe
                    for i, name in enumerate(feature_names):
                        # Skip original features
                        if name in poly_cols:
                            continue
                        
                        # Clean up feature name
                        clean_name = name.replace(' ', '_').replace('^', '_pow_')
                        transformed_df[clean_name] = poly_features[:, i]
                    
                    # Update session state
                    st.session_state.transformed_df = transformed_df
                    st.success(f"Generated {len(feature_names) - len(poly_cols)} new polynomial features")
        
        elif creation_method == "Custom Formula":
            st.write("Create a feature using a custom formula.")
            st.write("Example formulas: `col1 * col2 + col3`, `np.log(col1) - col2`")
            
            formula = st.text_area("Enter formula (use column names as variables)")
            new_name = st.text_input("New feature name", "custom_feature")
            
            if formula and new_name and st.button("Create Feature"):
                with st.spinner("Creating feature..."):
                    try:
                        # Define a dictionary of variables for evaluation
                        variables = {}
                        for col in transformed_df.columns:
                            # Make column name safe for eval
                            safe_name = col.replace(' ', '_').replace('-', '_')
                            variables[safe_name] = transformed_df[col]
                        
                        # Add numpy for calculations
                        variables['np'] = np
                        
                        # Make formula safe by replacing column names
                        safe_formula = formula
                        for col in transformed_df.columns:
                            safe_name = col.replace(' ', '_').replace('-', '_')
                            safe_formula = safe_formula.replace(col, safe_name)
                        
                        # Evaluate formula
                        result = eval(safe_formula, {"__builtins__": {}}, variables)
                        transformed_df[new_name] = result
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Created new feature: {new_name}")
                    except Exception as e:
                        st.error(f"Error evaluating formula: {str(e)}")
    
    # Scaling/Normalization
    elif operation == "Scaling/Normalization":
        st.write("#### Scale or Normalize Features")
        st.write("Transform features to a common scale.")
        
        # Select columns to scale
        scale_cols = st.multiselect(
            "Select columns to scale",
            options=numeric_cols,
            default=[]
        )
        
        if scale_cols:
            # Select scaling method
            scaling_method = st.selectbox(
                "Select scaling method",
                ["StandardScaler (Z-score normalization)", 
                 "MinMaxScaler (0-1 scaling)", 
                 "RobustScaler (robust to outliers)",
                 "MaxAbsScaler", 
                 "Custom Range"]
            )
            
            if scaling_method == "Custom Range":
                min_val = st.number_input("Minimum value", value=0.0)
                max_val = st.number_input("Maximum value", value=1.0)
                
                if min_val >= max_val:
                    st.error("Maximum value must be greater than minimum value.")
            
            if st.button("Apply Scaling"):
                with st.spinner("Scaling data..."):
                    # Apply scaling
                    if scaling_method == "StandardScaler (Z-score normalization)":
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(transformed_df[scale_cols])
                        suffix = "_scaled"
                    
                    elif scaling_method == "MinMaxScaler (0-1 scaling)":
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(transformed_df[scale_cols])
                        suffix = "_minmax"
                    
                    elif scaling_method == "RobustScaler (robust to outliers)":
                        scaler = RobustScaler()
                        scaled_data = scaler.fit_transform(transformed_df[scale_cols])
                        suffix = "_robust"
                    
                    elif scaling_method == "MaxAbsScaler":
                        from sklearn.preprocessing import MaxAbsScaler
                        scaler = MaxAbsScaler()
                        scaled_data = scaler.fit_transform(transformed_df[scale_cols])
                        suffix = "_maxabs"
                    
                    elif scaling_method == "Custom Range":
                        scaler = MinMaxScaler(feature_range=(min_val, max_val))
                        scaled_data = scaler.fit_transform(transformed_df[scale_cols])
                        suffix = f"_{min_val}to{max_val}"
                    
                    # Add scaled features to dataframe
                    for i, col in enumerate(scale_cols):
                        transformed_df[f"{col}{suffix}"] = scaled_data[:, i]
                    
                    # Update session state
                    st.session_state.transformed_df = transformed_df
                    st.success(f"Scaled {len(scale_cols)} columns using {scaling_method}")
                    
                    # Show before and after plots for first scaled column
                    if scale_cols:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Original distribution
                        sns.histplot(transformed_df[scale_cols[0]], kde=True, ax=axes[0])
                        axes[0].set_title(f"Original: {scale_cols[0]}")
                        
                        # Scaled distribution
                        sns.histplot(transformed_df[f"{scale_cols[0]}{suffix}"], kde=True, ax=axes[1])
                        axes[1].set_title(f"Scaled: {scale_cols[0]}{suffix}")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Categorical Encoding
    elif operation == "Categorical Encoding":
        st.write("#### Encode Categorical Variables")
        st.write("Convert categorical variables to numeric form.")
        
        # Select columns to encode
        encode_cols = st.multiselect(
            "Select categorical columns to encode",
            options=categorical_cols,
            default=[]
        )
        
        if encode_cols:
            # Select encoding method
            encoding_method = st.selectbox(
                "Select encoding method",
                ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding", "Target Encoding", "Frequency Encoding"]
            )
            
            if encoding_method == "Ordinal Encoding":
                st.info("Ordinal encoding requires a specific order for categories.")
                
                # For each column, let user specify order
                ordinal_orders = {}
                for col in encode_cols:
                    unique_values = transformed_df[col].dropna().unique().tolist()
                    
                    # Let user reorder categories
                    st.write(f"**Order for {col}**")
                    reordered = st.multiselect(
                        f"Arrange categories from lowest to highest (first is 0, next is 1, etc.)",
                        options=unique_values,
                        default=unique_values
                    )
                    
                    # Store order if user has selected all categories
                    if len(reordered) == len(unique_values):
                        ordinal_orders[col] = reordered
                    else:
                        st.warning(f"Please select all categories for {col} to define the order.")
            
            elif encoding_method == "Target Encoding":
                target_options = transformed_df.columns.tolist()
                target_col = st.selectbox(
                    "Select target column for encoding",
                    options=target_options
                )
                
                cv_folds = st.slider("Number of cross-validation folds", 2, 10, 5)
            
            if st.button("Apply Encoding"):
                with st.spinner("Encoding data..."):
                    # Apply encoding
                    if encoding_method == "One-Hot Encoding":
                        # Use pandas get_dummies for one-hot encoding
                        for col in encode_cols:
                            dummies = pd.get_dummies(transformed_df[col], prefix=col, prefix_sep='_', drop_first=False)
                            transformed_df = pd.concat([transformed_df, dummies], axis=1)
                        
                        st.success(f"One-hot encoded {len(encode_cols)} columns")
                    
                    elif encoding_method == "Label Encoding":
                        from sklearn.preprocessing import LabelEncoder
                        
                        for col in encode_cols:
                            le = LabelEncoder()
                            # Handle NaN values
                            if transformed_df[col].isna().any():
                                # Create a copy to avoid modifying the original
                                transformed_df[f"{col}_label"] = transformed_df[col].fillna('NaN_placeholder')
                                transformed_df[f"{col}_label"] = le.fit_transform(transformed_df[f"{col}_label"])
                            else:
                                transformed_df[f"{col}_label"] = le.fit_transform(transformed_df[col])
                        
                        st.success(f"Label encoded {len(encode_cols)} columns")
                    
                    elif encoding_method == "Ordinal Encoding":
                        for col in encode_cols:
                            # Check if we have an order for this column
                            if col in ordinal_orders and ordinal_orders[col]:
                                # Create mapping dictionary
                                order_dict = {cat: i for i, cat in enumerate(ordinal_orders[col])}
                                
                                # Apply mapping
                                transformed_df[f"{col}_ordinal"] = transformed_df[col].map(order_dict)
                                
                                # Handle missing values
                                if transformed_df[f"{col}_ordinal"].isna().any():
                                    st.warning(f"Some values in {col} were not in the specified order. NaN values assigned.")
                            else:
                                st.warning(f"Skipping {col} as no valid order was provided.")
                        
                        st.success("Applied ordinal encoding")
                    
                    elif encoding_method == "Target Encoding":
                        try:
                            from category_encoders import TargetEncoder
                            
                            # Create and fit the encoder
                            encoder = TargetEncoder(cols=encode_cols, cv=cv_folds)
                            
                            # Get target values
                            y = transformed_df[target_col]
                            
                            # Fit and transform
                            encoded_data = encoder.fit_transform(transformed_df[encode_cols], y)
                            
                            # Add encoded columns to dataframe
                            for col in encode_cols:
                                transformed_df[f"{col}_target"] = encoded_data[col]
                            
                            st.success(f"Target encoded {len(encode_cols)} columns using {target_col} as target")
                        except ImportError:
                            st.error("Target encoding requires category_encoders package. Please install it with `pip install category_encoders`.")
                    
                    elif encoding_method == "Frequency Encoding":
                        for col in encode_cols:
                            # Calculate frequency of each category
                            freq = transformed_df[col].value_counts(normalize=True)
                            
                            # Apply frequency mapping
                            transformed_df[f"{col}_freq"] = transformed_df[col].map(freq)
                        
                        st.success(f"Frequency encoded {len(encode_cols)} columns")
                    
                    # Update session state
                    st.session_state.transformed_df = transformed_df
    
    # Binning
    elif operation == "Binning":
        st.write("#### Bin Continuous Variables")
        st.write("Convert continuous variables into discrete bins.")
        
        # Select column to bin
        bin_col = st.selectbox(
            "Select column to bin",
            options=numeric_cols,
            index=0 if numeric_cols else None
        )
        
        if bin_col:
            # Get column min, max, and summary
            col_min = float(transformed_df[bin_col].min())
            col_max = float(transformed_df[bin_col].max())
            
            # Binning method
            bin_method = st.selectbox(
                "Select binning method",
                ["Equal Width", "Equal Frequency", "Custom Bins"]
            )
            
            if bin_method == "Equal Width":
                num_bins = st.slider("Number of bins", 2, 20, 5)
                
                if st.button("Apply Binning"):
                    with st.spinner("Binning data..."):
                        # Create bin edges
                        bins = np.linspace(col_min, col_max, num_bins + 1)
                        
                        # Apply binning
                        bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                        transformed_df[f"{bin_col}_bin"] = pd.cut(
                            transformed_df[bin_col], 
                            bins=bins, 
                            labels=bin_labels,
                            include_lowest=True
                        )
                        
                        # Visualize bins
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(transformed_df[bin_col], bins=bins, kde=False, ax=ax)
                        
                        # Add bin edges
                        for edge in bins:
                            ax.axvline(edge, color='red', linestyle='--', alpha=0.7)
                        
                        ax.set_title(f"Equal Width Binning for {bin_col}")
                        st.pyplot(fig)
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Created {num_bins} equal-width bins for {bin_col}")
            
            elif bin_method == "Equal Frequency":
                num_bins = st.slider("Number of bins", 2, 20, 5)
                
                if st.button("Apply Binning"):
                    with st.spinner("Binning data..."):
                        # Apply binning
                        bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                        transformed_df[f"{bin_col}_bin"] = pd.qcut(
                            transformed_df[bin_col], 
                            q=num_bins, 
                            labels=bin_labels,
                            duplicates='drop'
                        )
                        
                        # Get bin edges for visualization
                        bin_edges = pd.qcut(transformed_df[bin_col], q=num_bins, duplicates='drop').cat.categories
                        bins = [float(str(edge).split(",")[0].strip("([]")) for edge in bin_edges]
                        bins.append(float(str(bin_edges[-1]).split(",")[1].strip("([])")))
                        
                        # Visualize bins
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(transformed_df[bin_col], bins=20, kde=False, ax=ax)
                        
                        # Add bin edges
                        for edge in bins:
                            ax.axvline(edge, color='red', linestyle='--', alpha=0.7)
                        
                        ax.set_title(f"Equal Frequency Binning for {bin_col}")
                        st.pyplot(fig)
                        
                        # Update session state
                        st.session_state.transformed_df = transformed_df
                        st.success(f"Created {num_bins} equal-frequency bins for {bin_col}")
            
            elif bin_method == "Custom Bins":
                st.write(f"Column range: {col_min} to {col_max}")
                
                # Let user input custom bin edges
                bin_str = st.text_input(
                    "Enter bin edges separated by commas (e.g., 0,10,20,30,40,50)",
                    value=",".join([str(round(col_min + i*(col_max-col_min)/5, 2)) for i in range(6)])
                )
                
                if bin_str and st.button("Apply Binning"):
                    with st.spinner("Binning data..."):
                        try:
                            # Parse bin edges
                            bins = [float(x.strip()) for x in bin_str.split(",")]
                            
                            if len(bins) < 2:
                                st.error("Please provide at least 2 bin edges.")
                            elif bins[0] > col_min:
                                st.warning(f"First bin edge {bins[0]} is greater than column minimum {col_min}. Some values may be excluded.")
                            elif bins[-1] < col_max:
                                st.warning(f"Last bin edge {bins[-1]} is less than column maximum {col_max}. Some values may be excluded.")
                            else:
                                # Apply binning
                                num_bins = len(bins) - 1
                                bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                                transformed_df[f"{bin_col}_bin"] = pd.cut(
                                    transformed_df[bin_col], 
                                    bins=bins, 
                                    labels=bin_labels,
                                    include_lowest=True
                                )
                                
                                # Visualize bins
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(transformed_df[bin_col], bins=20, kde=False, ax=ax)
                                
                                # Add bin edges
                                for edge in bins:
                                    ax.axvline(edge, color='red', linestyle='--', alpha=0.7)
                                
                                ax.set_title(f"Custom Binning for {bin_col}")
                                st.pyplot(fig)
                                
                                # Update session state
                                st.session_state.transformed_df = transformed_df
                                st.success(f"Created {num_bins} custom bins for {bin_col}")
                        
                        except ValueError as e:
                            st.error(f"Error parsing bin edges: {str(e)}")
    
    # Export options
    st.write("### Export Transformed Data")
    
    export_cols = st.multiselect(
        "Select columns to keep in the exported data",
        options=transformed_df.columns.tolist(),
        default=transformed_df.columns.tolist()
    )
    
    if export_cols:
        export_df = transformed_df[export_cols]
        
        # Update main session data
        if st.button("Update Session Data"):
            st.session_state.data = export_df
            st.success("Updated main session data with transformed features.")
        
        # Download option
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Export format", ["CSV", "Excel", "JSON"])
        
        with col2:
            if export_format == "CSV":
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="transformed_data.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Transformed Data')
                    writer.close()
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="transformed_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:  # JSON
                json_str = export_df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="transformed_data.json",
                    mime="application/json"
                ) 