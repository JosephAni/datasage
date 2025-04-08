import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error, classification_report,
    confusion_matrix
)
import time

# Add parent directory to path to import from data_cleaner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import DataCleaner, safe_display_dataframe
except ImportError:
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        # Convert all object types to string to avoid PyArrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.markdown("# Machine Learning Models")
st.sidebar.header("Machine Learning")
st.write(
    """
    This page allows you to build and train machine learning models on your data.
    Select your target variable, features, and model type to get started.
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

# Main machine learning function
def build_and_train_model(df, target_col, feature_cols, model_type, problem_type,
                          test_size=0.2, random_state=42, model_params=None):
    """
    Build and train a machine learning model
    
    Args:
        df: DataFrame with the data
        target_col: Target column name
        feature_cols: List of feature column names
        model_type: Type of model to use
        problem_type: 'classification' or 'regression'
        test_size: Test set size (0-1)
        random_state: Random seed for reproducibility
        model_params: Dictionary of model parameters
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode categorical target for classification
    if problem_type == 'classification':
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_
        else:
            class_names = sorted(y.unique())
    
    # Handle categorical features
    cat_cols = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            cat_cols.append(col)
    
    # For simplicity, use label encoding for categorical features
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if problem_type == 'classification':
        if model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=random_state, **(model_params or {}))
        elif model_type == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=random_state, max_iter=1000, **(model_params or {}))
        elif model_type == 'Support Vector Machine':
            from sklearn.svm import SVC
            model = SVC(random_state=random_state, probability=True, **(model_params or {}))
        elif model_type == 'Gradient Boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=random_state, **(model_params or {}))
        elif model_type == 'XGBoost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(random_state=random_state, **(model_params or {}))
            except ImportError:
                st.warning("XGBoost not installed. Using Random Forest instead.")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=random_state, **(model_params or {}))
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=random_state, **(model_params or {}))
    else:  # Regression
        if model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=random_state, **(model_params or {}))
        elif model_type == 'Linear Regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**(model_params or {}))
        elif model_type == 'Support Vector Regression':
            from sklearn.svm import SVR
            model = SVR(**(model_params or {}))
        elif model_type == 'Gradient Boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=random_state, **(model_params or {}))
        elif model_type == 'XGBoost':
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(random_state=random_state, **(model_params or {}))
            except ImportError:
                st.warning("XGBoost not installed. Using Random Forest instead.")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=random_state, **(model_params or {}))
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=random_state, **(model_params or {}))
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # For classification, also get probability predictions if available
    if problem_type == 'classification' and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)
    else:
        y_proba = None
    
    # Evaluate model
    if problem_type == 'classification':
        # For multi-class, use weighted averages
        average = 'weighted' if len(np.unique(y)) > 2 else 'binary'
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_test, y_pred, average=average, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:  # Regression
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }
        cm = None
        report = None
    
    # Feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For linear models
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': np.abs(model.coef_) if len(model.coef_.shape) == 1 else np.mean(np.abs(model.coef_), axis=0)
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = None
    
    # Cross-validation scores
    try:
        if problem_type == 'classification':
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
    except Exception as e:
        st.warning(f"Cross-validation failed: {str(e)}")
        cv_scores = None
        cv_mean = None
        cv_std = None
    
    # Prepare results
    results = {
        'model': model,
        'model_type': model_type,
        'problem_type': problem_type,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
    }
    
    if problem_type == 'classification':
        results['class_names'] = class_names
    
    return results

# Get data from session state
df = get_data()

if df is None:
    st.info("No data loaded. Please upload data on the Home page.")
else:
    st.write("### Data Preview")
    st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        safe_display_dataframe(df.head(10))
    
    # Machine Learning Configuration
    st.write("### Model Configuration")
    
    # Select columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Problem type
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"],
        horizontal=True,
        help="Classification predicts categories, Regression predicts continuous values"
    )
    
    # Target variable selection
    if problem_type == "Classification":
        # For classification, prefer categorical columns as default
        target_options = categorical_cols + numeric_cols
    else:
        # For regression, prefer numeric columns as default
        target_options = numeric_cols + categorical_cols
    
    target_col = st.selectbox(
        "Select Target Variable",
        options=target_options,
        help="The variable you want to predict"
    )
    
    # Feature selection
    feature_options = [col for col in all_cols if col != target_col]
    feature_cols = st.multiselect(
        "Select Features",
        options=feature_options,
        default=feature_options[:min(5, len(feature_options))],
        help="The variables to use for prediction"
    )
    
    # Model selection
    if problem_type == "Classification":
        model_options = [
            "Random Forest",
            "Logistic Regression",
            "Support Vector Machine",
            "Gradient Boosting",
            "XGBoost"
        ]
    else:
        model_options = [
            "Random Forest",
            "Linear Regression",
            "Support Vector Regression",
            "Gradient Boosting",
            "XGBoost"
        ]
    
    model_type = st.selectbox(
        "Select Model Type",
        options=model_options,
        help="The machine learning algorithm to use"
    )
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        random_state = st.number_input(
            "Random Seed",
            value=42,
            help="Seed for reproducible results"
        )
        
        # Model-specific parameters
        st.write("#### Model Parameters")
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
            max_depth = st.slider("Max Depth", 1, 50, 10, 1)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            }
        elif model_type in ["Logistic Regression", "Linear Regression"]:
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
            model_params = {'C': C} if model_type == "Logistic Regression" else {}
        elif model_type in ["Support Vector Machine", "Support Vector Regression"]:
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], 1)
            model_params = {'C': C, 'kernel': kernel}
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of Estimators", 10, 500, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Max Depth", 1, 20, 3, 1)
            model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth
            }
        elif model_type == "XGBoost":
            n_estimators = st.slider("Number of Estimators", 10, 500, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Max Depth", 1, 20, 3, 1)
            model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth
            }
        else:
            model_params = {}
    
    # Validation
    if not feature_cols:
        st.warning("Please select at least one feature.")
    elif target_col in feature_cols:
        st.warning("Target variable should not be included in features.")
    else:
        # Train model button
        if st.button("Train Model"):
            if len(feature_cols) > 0:
                with st.spinner("Training model..."):
                    start_time = time.time()
                    
                    results = build_and_train_model(
                        df=df,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        model_type=model_type,
                        problem_type=problem_type.lower(),
                        test_size=test_size,
                        random_state=int(random_state),
                        model_params=model_params
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Store model in session state
                    st.session_state.ml_results = results
                    
                    st.success(f"Model trained successfully in {training_time:.2f} seconds!")
            else:
                st.warning("Please select at least one feature.")
    
    # Display results if model has been trained
    if 'ml_results' in st.session_state:
        results = st.session_state.ml_results
        
        st.write("### Model Results")
        
        # Metrics
        st.write("#### Performance Metrics")
        metrics = results['metrics']
        
        # Create metrics display
        metric_cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(label=metric_name.upper(), value=f"{metric_value:.4f}")
        
        # Cross-validation results
        if results['cv_scores'] is not None:
            st.write("#### Cross-Validation Results")
            st.write(f"Mean Score: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
        
        # Feature importance
        if results['feature_importance'] is not None:
            st.write("#### Feature Importance")
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importance = results['feature_importance']
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
            ax.set_title("Top 10 Feature Importance")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display feature importance table
            safe_display_dataframe(feature_importance)
        
        # Classification-specific visualizations
        if results['problem_type'] == 'classification':
            st.write("#### Classification Results")
            
            # Confusion Matrix
            st.write("##### Confusion Matrix")
            cm = results['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            sns.heatmap(
                cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax
            )
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Normalized Confusion Matrix')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Classification report
            if results['classification_report'] is not None:
                st.write("##### Classification Report")
                report_df = pd.DataFrame(results['classification_report']).T
                safe_display_dataframe(report_df)
            
            # ROC Curve for binary classification
            if len(results['class_names']) == 2 and results['y_proba'] is not None:
                st.write("##### ROC Curve")
                from sklearn.metrics import roc_curve, auc
                
                y_test = results['y_test']
                y_proba = results['y_proba'][:, 1]  # Probability of positive class
                
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
        
        # Regression-specific visualizations
        else:
            st.write("#### Regression Results")
            
            # Predicted vs Actual
            st.write("##### Predicted vs Actual Values")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_test'], results['y_pred'], alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(results['y_test'].min(), results['y_pred'].min())
            max_val = max(results['y_test'].max(), results['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Predicted vs Actual Values')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Residuals Plot
            st.write("##### Residuals Plot")
            
            residuals = results['y_test'] - results['y_pred']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_pred'], residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals Plot')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Distribution of residuals
            st.write("##### Distribution of Residuals")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(residuals, kde=True, ax=ax)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Residuals')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Model Prediction
        st.write("### Model Prediction")
        st.write("Use the trained model to make predictions on new data.")
        
        with st.expander("Make Predictions", expanded=False):
            # Input method selection
            input_method = st.radio("Select Input Method", ["Sample from Test Set", "Custom Input"])
            
            if input_method == "Sample from Test Set":
                # Select a random sample from the test set
                sample_index = st.slider(
                    "Select Sample Index", 
                    0, 
                    len(results['X_test']) - 1, 
                    0
                )
                
                # Display sample features
                sample_features = results['X_test'].iloc[sample_index].to_dict()
                
                st.write("#### Sample Features")
                sample_df = pd.DataFrame([sample_features])
                safe_display_dataframe(sample_df)
                
                # Make prediction
                if st.button("Predict Sample"):
                    # Scale features
                    X_sample = results['scaler'].transform(sample_df)
                    
                    # Predict
                    model = results['model']
                    
                    if results['problem_type'] == 'classification':
                        prediction = model.predict(X_sample)[0]
                        
                        # Get class name
                        predicted_class = results['class_names'][prediction]
                        actual_class = results['class_names'][results['y_test'].iloc[sample_index]]
                        
                        st.write(f"#### Prediction: {predicted_class}")
                        st.write(f"#### Actual: {actual_class}")
                        
                        # Show probabilities if available
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_sample)[0]
                            proba_df = pd.DataFrame({
                                'Class': results['class_names'],
                                'Probability': proba
                            })
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x='Class', y='Probability', data=proba_df, ax=ax)
                            ax.set_ylim(0, 1)
                            ax.set_title("Prediction Probabilities")
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        prediction = model.predict(X_sample)[0]
                        actual = results['y_test'].iloc[sample_index]
                        
                        st.write(f"#### Predicted Value: {prediction:.4f}")
                        st.write(f"#### Actual Value: {actual:.4f}")
                        st.write(f"#### Error: {abs(prediction - actual):.4f}")
            
            else:  # Custom input
                st.write("#### Enter Feature Values")
                
                # Create input fields for each feature
                custom_input = {}
                
                for feature in results['feature_cols']:
                    # Get feature type and range
                    if feature in numeric_cols:
                        feature_min = float(df[feature].min())
                        feature_max = float(df[feature].max())
                        feature_mean = float(df[feature].mean())
                        
                        custom_input[feature] = st.slider(
                            f"{feature}",
                            min_value=feature_min,
                            max_value=feature_max,
                            value=feature_mean
                        )
                    else:
                        # For categorical features, show dropdown
                        options = df[feature].unique().tolist()
                        custom_input[feature] = st.selectbox(
                            f"{feature}",
                            options=options
                        )
                
                # Make prediction
                if st.button("Predict"):
                    # Convert input to DataFrame
                    input_df = pd.DataFrame([custom_input])
                    
                    # Handle categorical features
                    for col in input_df.columns:
                        if col in categorical_cols:
                            # For simplicity, convert to string
                            input_df[col] = input_df[col].astype(str)
                    
                    # Use the same preprocessing as during training
                    # This is a simplified approach - in a real app, we'd save the preprocessing pipeline
                    for col in input_df.columns:
                        if col in categorical_cols:
                            # For simplicity, if category not seen during training, assign 0
                            try:
                                le = LabelEncoder()
                                le.fit(df[col].astype(str))
                                input_df[col] = le.transform(input_df[col])
                            except:
                                input_df[col] = 0
                    
                    # Scale features
                    X_input = results['scaler'].transform(input_df)
                    
                    # Predict
                    model = results['model']
                    
                    if results['problem_type'] == 'classification':
                        prediction = model.predict(X_input)[0]
                        
                        # Get class name
                        predicted_class = results['class_names'][prediction]
                        
                        st.write(f"#### Prediction: {predicted_class}")
                        
                        # Show probabilities if available
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_input)[0]
                            proba_df = pd.DataFrame({
                                'Class': results['class_names'],
                                'Probability': proba
                            })
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x='Class', y='Probability', data=proba_df, ax=ax)
                            ax.set_ylim(0, 1)
                            ax.set_title("Prediction Probabilities")
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        prediction = model.predict(X_input)[0]
                        
                        st.write(f"#### Predicted Value: {prediction:.4f}")
        
        # Save model
        st.write("### Save Model")
        st.write("Download the trained model for later use.")
        
        # Model export options
        export_format = st.selectbox("Select Export Format", ["Pickle", "ONNX", "PMML"])
        
        if st.button("Export Model"):
            import pickle
            import io
            
            # Pickle the model
            if export_format == "Pickle":
                model_pkl = pickle.dumps(results['model'])
                
                model_info = {
                    'model': results['model'],
                    'model_type': results['model_type'],
                    'problem_type': results['problem_type'],
                    'feature_cols': results['feature_cols'],
                    'target_col': results['target_col'],
                    'metrics': results['metrics'],
                    'scaler': results['scaler'],
                }
                
                export_data = pickle.dumps(model_info)
                
                st.download_button(
                    label="Download Model",
                    data=export_data,
                    file_name=f"{results['model_type'].replace(' ', '_').lower()}_{results['target_col']}_model.pkl",
                    mime="application/octet-stream"
                )
            else:
                st.info(f"{export_format} export format coming soon!")
        
        # Model documentation
        st.write("### Model Documentation")
        
        # Generate simple model report
        model_report = {
            'Model Type': results['model_type'],
            'Problem Type': results['problem_type'].capitalize(),
            'Target Variable': results['target_col'],
            'Features': ", ".join(results['feature_cols']),
            'Number of Features': len(results['feature_cols']),
            'Model Parameters': str(model_params),
            'Metrics': ", ".join([f"{k}: {v:.4f}" for k, v in results['metrics'].items()]),
        }
        
        if results['cv_mean'] is not None:
            model_report['Cross-Validation Score'] = f"{results['cv_mean']:.4f} (±{results['cv_std']:.4f})"
        
        # Display model report
        model_report_df = pd.DataFrame([model_report]).T.reset_index()
        model_report_df.columns = ['Property', 'Value']
        safe_display_dataframe(model_report_df)
        
        # Download model report
        if st.button("Download Model Report"):
            report_text = "\n".join([f"{k}: {v}" for k, v in model_report.items()])
            
            st.download_button(
                label="Download Report Text",
                data=report_text,
                file_name=f"{results['model_type'].replace(' ', '_').lower()}_{results['target_col']}_report.txt",
                mime="text/plain"
            ) 