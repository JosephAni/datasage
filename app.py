from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file, flash
import pandas as pd
import numpy as np
import os
import json
import io
import uuid
import datetime
import tempfile
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from flask_wtf.csrf import CSRFProtect, generate_csrf
from dotenv import load_dotenv

# Import scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import math
from data_manager import DataManager
from functools import wraps
from scipy.stats import norm

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Basic configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///inventory.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Configure server-side sessions
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=os.path.join(tempfile.gettempdir(), 'flask_session'),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7),
    SESSION_USE_SIGNER=False,  # Set to False to avoid bytes/string type issues
    SESSION_COOKIE_SECURE=False,  # Set to False for local development
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_NAME='inventory_session'  # Set session cookie name here
)

# Initialize Session before other extensions
Session(app)

# Configure CSRF protection
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_TIME_LIMIT'] = None  # No time limit for CSRF tokens
app.config['WTF_CSRF_SSL_STRICT'] = False  # Allow CSRF tokens over HTTP during development
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable automatic checking to handle it explicitly where needed

# Initialize extensions
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize DataManager
data_manager = DataManager()

# Define database models
class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    date_uploaded = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    session_id = db.Column(db.String(100), nullable=False)
    
    def __repr__(self):
        return f'<UploadedFile {self.filename}>'

class ProcessedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'))
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    process_type = db.Column(db.String(50)) # e.g., 'cleaned', 'converted'
    date_processed = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    session_id = db.Column(db.String(100), nullable=False)
    
    def __repr__(self):
        return f'<ProcessedFile {self.filename}>'

# Create database tables
with app.app_context():
    db.create_all()

# Utility functions
def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def load_sample_data():
    """Load a sample dataset for demonstration"""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df
    except:
        return pd.DataFrame()

def get_session_data():
    """Get dataframe from current session with performance optimization"""
    try:
        # Get filepath from session, with a default of None
        filepath = session.get('filepath', None)
        
        # Check if filepath exists and is valid
        if not filepath or not isinstance(filepath, str) or not os.path.exists(filepath):
            # Try cleaned filepath
            cleaned_filepath = session.get('cleaned_filepath', None)
            if cleaned_filepath and isinstance(cleaned_filepath, str) and os.path.exists(cleaned_filepath):
                filepath = cleaned_filepath
            else:
                # Try to find most recent file for this session from database
                try:
                    session_id = get_session_id()
                    latest_file = ProcessedFile.query.filter_by(session_id=session_id).order_by(ProcessedFile.date_processed.desc()).first()
                    
                    if not latest_file:
                        latest_file = UploadedFile.query.filter_by(session_id=session_id).order_by(UploadedFile.date_uploaded.desc()).first()
                    
                    if latest_file and os.path.exists(latest_file.filepath):
                        filepath = latest_file.filepath
                        session['filepath'] = filepath
                        session['filename'] = latest_file.filename
                    else:
                        return None
                except Exception as e:
                    print(f"Database error in get_session_data: {str(e)}")
                    return None
        
        # Load and return the data
        try:
            if filepath.endswith('.csv'):
                # Check file size
                file_size = os.path.getsize(filepath)
                if file_size > 50 * 1024 * 1024:  # 50MB
                    return pd.read_csv(filepath, nrows=1000)
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                return pd.read_json(filepath)
            else:
                print(f"Unsupported file type for {filepath}")
                return None
        except Exception as e:
            print(f"Error loading data file {filepath}: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Session error in get_session_data: {str(e)}")
        return None

def apply_cleaning(df, cleaning_actions):
    """Apply cleaning actions to dataframe"""
    if df is None or df.empty:
        return df, ["No data to clean"]
    
    summary = []
    initial_rows, initial_cols = df.shape
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    if cleaning_actions.get('drop_na', False):
        before_rows = len(df_clean)
        df_clean = df_clean.dropna()
        after_rows = len(df_clean)
        rows_dropped = before_rows - after_rows
        if rows_dropped > 0:
            summary.append(f"Dropped {rows_dropped} rows with missing values")
    
    if cleaning_actions.get('fill_na_mean', False):
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if df_clean[col].isna().sum() > 0:
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
                    summary.append(f"Filled {col} missing values with mean ({mean_val:.2f})")
    
    if cleaning_actions.get('fill_na_mode', False):
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df_clean[col].isna().sum() > 0:
                    mode_val = df_clean[col].mode().iloc[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    summary.append(f"Filled {col} missing values with mode ({mode_val})")
    
    # Handle duplicates
    if cleaning_actions.get('drop_duplicates', False):
        before_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_rows = len(df_clean)
        rows_dropped = before_rows - after_rows
        if rows_dropped > 0:
            summary.append(f"Removed {rows_dropped} duplicate rows")
    
    # Handle outliers
    if cleaning_actions.get('remove_outliers', False):
        outliers_removed = 0
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Calculate z-scores
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            
            # Mark outliers
            outliers = z_scores > 3
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                df_clean = df_clean[~outliers]
                outliers_removed += outlier_count
        
        if outliers_removed > 0:
            summary.append(f"Removed {outliers_removed} outliers (beyond 3 standard deviations)")
    
    # Handle data types
    if cleaning_actions.get('convert_numeric', False):
        converted = 0
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try converting to numeric
                try:
                    new_col = pd.to_numeric(df_clean[col], errors='coerce')
                    # Only convert if it doesn't introduce too many NaNs
                    if new_col.isna().sum() <= df_clean[col].isna().sum() * 1.1:
                        df_clean[col] = new_col
                        converted += 1
                except:
                    pass
        
        if converted > 0:
            summary.append(f"Converted {converted} columns to numeric type")
    
    if cleaning_actions.get('convert_datetime', False):
        converted = 0
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try converting to datetime
                try:
                    new_col = pd.to_datetime(df_clean[col], errors='coerce')
                    # Only convert if it doesn't introduce too many NaNs
                    if new_col.isna().sum() <= df_clean[col].isna().sum() * 1.1:
                        df_clean[col] = new_col
                        converted += 1
                except:
                    pass
        
        if converted > 0:
            summary.append(f"Converted {converted} columns to datetime type")
    
    # Final summary
    final_rows, final_cols = df_clean.shape
    if initial_rows != final_rows:
        summary.append(f"Overall: {initial_rows - final_rows} rows removed, {final_rows} rows remaining")
    
    return df_clean, summary

# Routes
@app.route('/')
def index():
    # Initialize dashboard stats if they don't exist
    if 'total_uploads' not in session:
        session['total_uploads'] = 0
    if 'cleaning_operations' not in session:
        session['cleaning_operations'] = 0
    if 'visualizations' not in session:
        session['visualizations'] = 0
    if 'ml_models' not in session:
        session['ml_models'] = 0
    
    # Initialize progress stats if they don't exist
    if 'cleaning_progress' not in session:
        session['cleaning_progress'] = 0
    if 'analysis_progress' not in session:
        session['analysis_progress'] = 0
    if 'visualization_progress' not in session:
        session['visualization_progress'] = 0
    if 'ml_progress' not in session:
        session['ml_progress'] = 0
    
    # Sample recent files for display
    if 'recent_files' not in session:
        session['recent_files'] = [
            {'name': 'Sample Sales Data.csv', 'type': 'csv', 'date': 'Today'},
            {'name': 'Inventory Analysis.xlsx', 'type': 'excel', 'date': 'Yesterday'},
            {'name': 'Customer Data.json', 'type': 'json', 'date': '3 days ago'}
        ]
    
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file:
            try:
                # Generate a unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save file to disk
                file.save(filepath)
                
                # Store in database
                session_id = get_session_id()
                uploaded_file = UploadedFile(
                    filename=filename,
                    filepath=filepath,
                    session_id=session_id
                )
                db.session.add(uploaded_file)
                db.session.commit()
                
                # Update session
                session['filepath'] = filepath
                session['filename'] = filename
                
                # Load data using DataManager
                if data_manager.load_data(filepath, filename):
                    flash('File uploaded and loaded successfully', 'success')
                    return redirect(url_for('data_cleaning'))
                else:
                    # If data loading fails, clean up
                    os.remove(filepath)
                    db.session.delete(uploaded_file)
                    db.session.commit()
                    flash('Error loading data from file', 'error')
                    return redirect(request.url)
                    
            except Exception as e:
                print(f"Error in file upload: {str(e)}")
                flash(f'Error uploading file: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/sample')
def use_sample_data():
    try:
        # Load sample data using DataManager
        if data_manager.load_data(sample_name="retail_inventory"):
            flash('Sample data loaded successfully', 'success')
            return redirect(url_for('data_cleaning'))
        else:
            flash('Error loading sample data', 'danger')
            return redirect(url_for('index'))
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        flash('Error loading sample data', 'danger')
        return redirect(url_for('index'))

@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('data_cleaning.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in data cleaning: {str(e)}")
        return render_template('data_cleaning.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/advanced_data_cleaning', methods=['GET', 'POST'])
def advanced_data_cleaning():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('advanced_data_cleaning.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in advanced data cleaning: {str(e)}")
        return render_template('advanced_data_cleaning.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/convert_data_types', methods=['GET', 'POST'])
def convert_data_types():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('convert_data_types.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in convert data types: {str(e)}")
        return render_template('convert_data_types.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/data_interpretation', methods=['GET', 'POST'])
def data_interpretation():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('data_interpretation.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in data interpretation: {str(e)}")
        return render_template('data_interpretation.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/advanced_data_analysis', methods=['GET', 'POST'])
def advanced_data_analysis():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('advanced_data_analysis.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in advanced data analysis: {str(e)}")
        return render_template('advanced_data_analysis.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/feature_engineering', methods=['GET', 'POST'])
def feature_engineering():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('feature_engineering.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        return render_template('feature_engineering.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/machine_learning', methods=['GET', 'POST'])
def machine_learning():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('machine_learning.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in machine learning: {str(e)}")
        return render_template('machine_learning.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/time_series', methods=['GET', 'POST'])
def time_series():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('time_series.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in time series: {str(e)}")
        return render_template('time_series.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/demand_forecasting', methods=['GET', 'POST'])
def demand_forecasting():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('demand_forecasting.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in demand forecasting: {str(e)}")
        return render_template('demand_forecasting.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/inventory_turnover', methods=['GET', 'POST'])
def inventory_turnover():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('inventory_turnover.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in inventory turnover: {str(e)}")
        return render_template('inventory_turnover.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/cost_of_inventory', methods=['GET', 'POST'])
def cost_of_inventory():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('cost_of_inventory.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in cost of inventory: {str(e)}")
        return render_template('cost_of_inventory.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/eoq_simulation')
def eoq_simulation():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('eoq_simulation.html', 
                              has_data=has_data, 
                              message="No data loaded. You can still use the simulation with manual inputs." if not has_data else "")
    except Exception as e:
        print(f"Error in EOQ simulation: {str(e)}")
        return render_template('eoq_simulation.html', 
                              has_data=False, 
                              message="An error occurred. You can still use the simulation with manual inputs.")

@app.route('/newsvendor_simulation')
def newsvendor_simulation():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('newsvendor_simulation.html', 
                              has_data=has_data, 
                              message="No data loaded. You can still use the simulation with manual inputs." if not has_data else "")
    except Exception as e:
        print(f"Error in newsvendor simulation: {str(e)}")
        return render_template('newsvendor_simulation.html', 
                              has_data=False, 
                              message="An error occurred. You can still use the simulation with manual inputs.")

@app.route('/clv_analysis', methods=['GET', 'POST'])
def clv_analysis():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('clv_analysis.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in CLV analysis: {str(e)}")
        return render_template('clv_analysis.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

@app.route('/data_visualization', methods=['GET', 'POST'])
def data_visualization():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('data_visualization.html', 
                              has_data=has_data, 
                              message="No data loaded. Please upload a dataset first or load sample data." if not has_data else "")
    except Exception as e:
        print(f"Error in data visualization: {str(e)}")
        return render_template('data_visualization.html', 
                              has_data=False, 
                              message="An error occurred. Please upload a dataset first or load sample data.")

# API endpoints for data processing
@app.route('/api/data_preview', methods=['GET'])
def api_data_preview():
    try:
        df = get_session_data()
        if df is None:
            return jsonify({"error": "No data loaded"}), 404
        
        # Limit number of rows for preview to improve performance
        preview_rows = min(10, len(df))
        
        # Convert to simple types for JSON serialization
        preview_data = []
        for idx, row in df.head(preview_rows).iterrows():
            row_dict = {}
            for col, val in row.items():
                # Handle different data types for JSON serialization
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    row_dict[col] = float(val)
                elif isinstance(val, (np.datetime64, pd.Timestamp)):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = str(val)
            preview_data.append(row_dict)
        
        # Return limited preview and schema info
        response = {
            "preview": preview_data,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in data preview: {str(e)}")
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

@app.route('/api/clean_data', methods=['POST'])
@csrf.exempt
def api_clean_data():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    cleaning_actions = request.json
    df_clean, summary = apply_cleaning(df, cleaning_actions)
    
    # Generate a unique filename
    original_filename = session.get('filename', 'data.csv')
    base_name, file_ext = os.path.splitext(original_filename)
    unique_filename = f"cleaned_{base_name}_{uuid.uuid4().hex}{file_ext}"
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save cleaned dataframe
    if file_ext.lower() == '.csv':
        df_clean.to_csv(cleaned_filepath, index=False)
    elif file_ext.lower() in ['.xls', '.xlsx']:
        df_clean.to_excel(cleaned_filepath, index=False)
    elif file_ext.lower() == '.json':
        df_clean.to_json(cleaned_filepath, orient='records')
    
    # Update session
    session['cleaned_filepath'] = cleaned_filepath
    session['filepath'] = cleaned_filepath  # Use cleaned data as current
    
    # Store in database
    session_id = get_session_id()
    # Get original file id if exists
    original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
    original_id = original_file.id if original_file else None
    
    processed_file = ProcessedFile(
        original_file_id=original_id,
        filename=f"cleaned_{original_filename}",
        filepath=cleaned_filepath,
        process_type='cleaned',
        session_id=session_id
    )
    db.session.add(processed_file)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "summary": summary,
        "preview": df_clean.head(10).to_dict(orient='records'),
        "shape": df_clean.shape
    })

@app.route('/api/data_types', methods=['GET'])
def api_data_types():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    data_types = {}
    for col in df.columns:
        data_types[col] = str(df[col].dtype)
    
    return jsonify(data_types)

@app.route('/api/convert_types', methods=['POST'])
@csrf.exempt
def api_convert_types():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    conversions = request.json
    df_converted = df.copy()
    
    results = []
    for col, new_type in conversions.items():
        try:
            if new_type == 'numeric':
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            elif new_type == 'datetime':
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
            elif new_type == 'category':
                df_converted[col] = df_converted[col].astype('category')
            elif new_type == 'string':
                df_converted[col] = df_converted[col].astype(str)
            
            results.append({
                "column": col,
                "new_type": str(df_converted[col].dtype),
                "success": True
            })
        except Exception as e:
            results.append({
                "column": col,
                "error": str(e),
                "success": False
            })
    
    # Generate a unique filename
    original_filename = session.get('filename', 'data.csv')
    base_name, file_ext = os.path.splitext(original_filename)
    unique_filename = f"converted_{base_name}_{uuid.uuid4().hex}{file_ext}"
    converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save converted dataframe
    if file_ext.lower() == '.csv':
        df_converted.to_csv(converted_filepath, index=False)
    elif file_ext.lower() in ['.xls', '.xlsx']:
        df_converted.to_excel(converted_filepath, index=False)
    elif file_ext.lower() == '.json':
        df_converted.to_json(converted_filepath, orient='records')
    
    # Update session
    session['filepath'] = converted_filepath  # Update session to use converted data
    
    # Store in database
    session_id = get_session_id()
    # Get original file id if exists
    original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
    original_id = original_file.id if original_file else None
    
    processed_file = ProcessedFile(
        original_file_id=original_id,
        filename=f"converted_{original_filename}",
        filepath=converted_filepath,
        process_type='converted',
        session_id=session_id
    )
    db.session.add(processed_file)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "results": results,
        "preview": df_converted.head(10).to_dict(orient='records')
    })

@app.route('/api/data_stats', methods=['GET'])
def api_data_stats():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    stats = {}
    # Basic dataset info
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)
    stats['missing_values'] = df.isna().sum().sum()
    stats['duplicate_rows'] = df.duplicated().sum()
    
    # Column-wise statistics
    column_stats = {}
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isna().sum()),
            'unique_values': int(df[col].nunique())
        }
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
            })
        
        column_stats[col] = col_stats
    
    stats['columns_stats'] = column_stats
    
    return jsonify(stats)

@app.route('/api/download_data', methods=['GET'])
def api_download_data():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    file_format = request.args.get('format', 'csv')
    
    if file_format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            download_name='cleaned_data.csv',
            as_attachment=True
        )
        
    elif file_format == 'excel':
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name='cleaned_data.xlsx',
            as_attachment=True
        )
        
    elif file_format == 'json':
        output = io.StringIO()
        output.write(df.to_json(orient='records'))
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='application/json',
            download_name='cleaned_data.json',
            as_attachment=True
        )
        
    else:
        return jsonify({"error": "Unsupported file format"})

# Machine Learning Routes

@app.route('/api/ml/data/preview', methods=['GET'])
def get_ml_data_preview():
    """Return a preview of the data for the ML page with improved performance"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 404
        
        # Determine proper number of rows to show (5 is often enough for preview)
        preview_rows = min(5, len(df))
        
        # Prepare a clean JSON response with properly serialized values
        preview_data = []
        for idx, row in df.head(preview_rows).iterrows():
            row_dict = {}
            for col, val in row.items():
                # Handle different data types for JSON serialization
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    row_dict[col] = float(val)
                elif isinstance(val, (np.datetime64, pd.Timestamp)):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = str(val)
            preview_data.append(row_dict)
        
        # Include data types to help with feature selection
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Determine suitable feature columns (numeric columns are typically best for ML)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return jsonify({
            'data': preview_data,
            'columns': df.columns.tolist(),
            'dtypes': dtypes,
            'row_count': len(df),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns
        })
    except Exception as e:
        print(f"Error in ML data preview: {str(e)}")
        return jsonify({'error': f'Failed to load ML data: {str(e)}'}), 500

@app.route('/api/ml/train', methods=['POST'])
@csrf.exempt
def train_model():
    try:
        data = request.json
        problem_type = data.get('problem_type')
        model_type = data.get('model_type')
        features = data.get('features', [])
        test_size = data.get('test_size', 0.2)
        random_seed = data.get('random_seed', 42)
        
        # Get the data from session
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available. Please upload or load a dataset first.'}), 400
            
        # For classification/regression, we need a target variable
        if problem_type in ['classification', 'regression']:
            target_variable = data.get('target_variable')
            if not target_variable:
                return jsonify({'error': 'Target variable is required for classification/regression'}), 400
            if target_variable not in df.columns:
                return jsonify({'error': f'Target variable {target_variable} not found in dataset'}), 400
            if not features:
                return jsonify({'error': 'At least one feature must be selected'}), 400
            if target_variable in features:
                return jsonify({'error': 'Target variable cannot be used as a feature'}), 400
            
            # Prepare features and target
            X = df[features]
            y = df[target_variable]
            
            # Handle categorical target for classification
            if problem_type == 'classification':
                if y.dtype == 'object' or y.dtype.name == 'category':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    class_names = le.classes_.tolist()
                    session['ml_reverse_mapping'] = {i: label for i, label in enumerate(class_names)}
                else:
                    class_names = sorted(y.unique().tolist())
                    session['ml_reverse_mapping'] = {i: label for i, label in enumerate(class_names)}
        
        # Handle categorical features
        categorical_features = []

        numeric_features = []

        # First, perform more robust type checking
        for col in features:
            # Check sample values to determine true type
            sample = df[col].dropna().head(100)  # Get non-null sample values
            
            # If the column is already numeric type, check if it actually contains strings
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    # Try to convert the whole column to float to catch any string values
                    pd.to_numeric(df[col])
                    numeric_features.append(col)
                except (ValueError, TypeError):
                    # If conversion fails, it contains some strings
                    if df[col].nunique() < 50:
                        print(f"Converting mixed-type column {col} to categorical")
                        categorical_features.append(col)
                    else:
                        print(f"Skipping column {col} - mixed types with too many unique values")
            
            # If it's an object/string type, check if it actually contains only numbers
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Check for actual string content and unique counts
                if sample.apply(lambda x: isinstance(x, str)).any():
                    # It contains strings, check number of unique values
                    if df[col].nunique() < 50:
                        print(f"Adding {col} as categorical - contains strings with acceptable unique count")
                        categorical_features.append(col)
                    else:
                        print(f"Skipping categorical encoding for {col} - too many unique values ({df[col].nunique()})")
                else:
                    # Might be numeric stored as object, try to convert
                    try:
                        pd.to_numeric(df[col])
                        print(f"Converting {col} from object to numeric")
                        numeric_features.append(col)
                    except (ValueError, TypeError):
                        # Mixed type with non-convertible values
                        if df[col].nunique() < 50:
                            categorical_features.append(col)
                        else:
                            print(f"Skipping column {col} - unconvertible with too many unique values")
            
            # Handle other data types (datetime, etc.)
            else:
                print(f"Skipping unsupported column type: {col} ({df[col].dtype})")

        # Safety check - force-convert numeric features to float before modeling
        for col in numeric_features:
            try:
                # Convert to float and replace column in dataframe
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove from numeric features if too many NaNs were introduced
                if df[col].isna().mean() > 0.3:  # If over 30% NaN after conversion
                    print(f"Removing {col} from features - too many values couldn't be converted to numeric")
                    numeric_features.remove(col)
            except Exception as e:
                print(f"Error converting {col} to numeric: {str(e)}")
                numeric_features.remove(col)

        print(f"Final feature sets - Numeric: {numeric_features}, Categorical: {categorical_features}")

        # Add safer preprocessing for numeric features
        transformers = []

        # Only add numeric transformer if there are numeric features
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))

        # Only add categorical transformer if there are categorical features
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))

        # If no valid transformers, use a simple passthrough
        if not transformers:
            transformers.append(('pass', 'passthrough', features))

        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Split data for supervised learning
        if problem_type in ['classification', 'regression']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )
            
            # Fit preprocessor
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
        
        # Initialize model based on type
        if problem_type == 'classification':
            if model_type == 'logistic_regression':
                model = LogisticRegression(random_state=random_seed, max_iter=1000)
            elif model_type == 'decision_tree_classifier':
                model = DecisionTreeClassifier(random_state=random_seed)
            elif model_type == 'random_forest_classifier':
                model = RandomForestClassifier(random_state=random_seed)
            elif model_type == 'svm_classifier':
                model = SVC(probability=True, random_state=random_seed)
            elif model_type == 'knn_classifier':
                model = KNeighborsClassifier()
            else:
                return jsonify({'error': f'Invalid classification model type: {model_type}'}), 400
                
        elif problem_type == 'regression':
            if model_type == 'linear_regression':
                model = LinearRegression()
            elif model_type == 'decision_tree_regressor':
                model = DecisionTreeRegressor(random_state=random_seed)
            elif model_type == 'random_forest_regressor':
                model = RandomForestRegressor(random_state=random_seed)
            elif model_type == 'svm_regressor':
                model = SVR()
            else:
                return jsonify({'error': f'Invalid regression model type: {model_type}'}), 400
                
        elif problem_type == 'clustering':
            if model_type == 'kmeans':
                model = KMeans(random_state=random_seed)
            elif model_type == 'hierarchical':
                model = AgglomerativeClustering()
            elif model_type == 'dbscan':
                model = DBSCAN()
            else:
                return jsonify({'error': f'Invalid clustering model type: {model_type}'}), 400
        else:
            return jsonify({'error': f'Invalid problem type: {problem_type}'}), 400
            
        # Add any model-specific parameters
        model_params = {k: v for k, v in data.items() if k not in 
                       ['problem_type', 'model_type', 'features', 'target_variable', 'test_size', 'random_seed']}
        if model_params:
            model.set_params(**model_params)
            
        # Train model
        if problem_type in ['classification', 'regression']:
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            
            # Calculate metrics
            if problem_type == 'classification':
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_pred, average='weighted')),
                    'f1': float(f1_score(y_test, y_pred, average='weighted'))
                }
            else:  # regression
                metrics = {
                    'r2': float(r2_score(y_test, y_pred)),
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred))
                }
        else:  # clustering
            X_processed = preprocessor.fit_transform(X)
            model.fit(X_processed)
            labels = model.labels_
            
            if hasattr(model, 'inertia_'):
                metrics = {'inertia': float(model.inertia_)}
            else:
                metrics = {'clusters': len(set(labels))}
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = {
                'features': features,
                'values': importance.tolist()
            }
        elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
            importance = np.abs(model.coef_)
            feature_importance = {
                'features': features,
                'values': importance.tolist()
            }
            
        # Store model and related info in session
        session['ml_model'] = model
        session['ml_model_type'] = problem_type
        session['ml_features'] = features
        session['ml_preprocessor'] = preprocessor
        
        response = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'features': features
        }
        
        if problem_type == 'classification':
            response['class_names'] = class_names
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/predict', methods=['POST'])
@csrf.exempt
def make_prediction():
    """Make predictions using the trained model"""
    try:
        # Check if model exists in session
        if 'ml_model' not in session:
            return jsonify({'error': 'No trained model available. Train a model first.'}), 400
        
        # Get the model and related info from session
        model = session['ml_model']
        model_type = session['ml_model_type']
        features = session['ml_features']
        preprocessor = session['ml_preprocessor']
        
        # Get the input data
        data = request.json
        
        # Validate input features
        for feature in features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        try:
            # Prepare input data as DataFrame
            input_data = pd.DataFrame([{feature: data[feature] for feature in features}])
            
            # Preprocess input data
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            if model_type == 'classification':
                # Get prediction
                prediction = model.predict(input_processed)[0]
                
                # Convert prediction to original label if mapping exists
                if 'ml_reverse_mapping' in session:
                    prediction = session['ml_reverse_mapping'][int(prediction)]
                
                # Get class probabilities if available
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_processed)[0]
                    
                    # If we have a target mapping, use original labels
                    if 'ml_reverse_mapping' in session:
                        reverse_mapping = session['ml_reverse_mapping']
                        probabilities = {reverse_mapping[i]: float(proba[i]) for i in range(len(proba))}
                    else:
                        probabilities = {str(i): float(proba[i]) for i in range(len(proba))}
                
                return jsonify({
                    'prediction': prediction,
                    'probabilities': probabilities
                })
            
            elif model_type == 'regression':
                prediction = float(model.predict(input_processed)[0])
                return jsonify({'prediction': prediction})
            
            elif model_type == 'clustering':
                cluster = int(model.predict(input_processed)[0])
                return jsonify({'prediction': cluster})
            
            else:
                return jsonify({'error': f'Unsupported model type: {model_type}'}), 400
        
        except Exception as e:
            return jsonify({'error': f'Error making prediction: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_model', methods=['POST'])
def export_model():
    """Export the trained model"""
    # Check if model exists in session
    if 'ml_model' not in session:
        return jsonify({'error': 'No trained model available. Train a model first.'}), 400
    
    # Import joblib for model serialization
    try:
        import joblib
    except ImportError:
        return jsonify({'error': 'Required libraries not installed. Install joblib.'}), 500
    
    # Get model info
    model = session.get('ml_model')
    model_type = session.get('ml_model_type')
    features = session.get('ml_features')
    scaler = session.get('ml_scaler')
    
    # Get model name and description from request
    data = request.json
    model_name = data.get('model_name', 'untitled_model')
    model_description = data.get('model_description', '')
    
    try:
        # Create a directory for models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Create a dictionary with all model info
        model_info = {
            'model': model,
            'type': model_type,
            'features': features,
            'scaler': scaler,
            'description': model_description,
            'date_created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add target info for classification/regression
        if model_type in ['classification', 'regression']:
            model_info['target'] = session.get('ml_target')
        
        # Add target mapping for classification
        if model_type == 'classification' and 'ml_target_mapping' in session:
            model_info['target_mapping'] = session.get('ml_target_mapping')
            model_info['reverse_mapping'] = session.get('ml_reverse_mapping')
        
        # Save the model
        filename = f"models/{model_name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model_info, filename)
        
        return jsonify({'success': True, 'filename': filename})
    
    except Exception as e:
        return jsonify({'error': f'Error exporting model: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a report for the trained model"""
    # Check if model exists in session
    if 'ml_model' not in session:
        return jsonify({'error': 'No trained model available. Train a model first.'}), 400
    
    try:
        # Get model info
        model_type = session.get('ml_model_type')
        
        # Get additional info from request
        data = request.json
        model_name = data.get('model_name', 'Untitled Model')
        model_description = data.get('model_description', '')
        
        # Create a simple HTML report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Model Report: {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Machine Learning Model Report</h1>
                <h2>{model_name}</h2>
                <p>{model_description}</p>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>Model Information</h3>
                <table>
                    <tr><th>Model Type</th><td>{model_type}</td></tr>
                    <tr><th>Features</th><td>{', '.join(session.get('ml_features', []))}</td></tr>
        """
        
        # Add target for classification/regression
        if model_type in ['classification', 'regression']:
            report_html += f"""
                    <tr><th>Target</th><td>{session.get('ml_target', '')}</td></tr>
            """
        
        # Close the table and add more sections
        report_html += """
                </table>
            </div>
            
            <div class="section">
                <h3>Model Performance</h3>
                <p>For detailed model performance metrics and visualizations, please use the interactive dashboard.</p>
            </div>
            
            <div class="section">
                <h3>Model Usage</h3>
                <p>To use this model for predictions, you can:</p>
                <ol>
                    <li>Export the model using the "Export Model" button</li>
                    <li>Load the model using joblib: <code>model_info = joblib.load('model_filename.joblib')</code></li>
                    <li>Access the model and make predictions</li>
                </ol>
            </div>
        </body>
        </html>
        """
        
        # Create report directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Save the report
        report_filename = f"reports/{model_name.replace(' ', '_').lower()}_report.html"
        with open(report_filename, 'w') as f:
            f.write(report_html)
        
        # Create a URL for the report
        report_url = url_for('static', filename='../' + report_filename)
        
        return jsonify({'success': True, 'report_url': report_url})
        
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

@app.route('/manage_files')
def manage_files():
    # Get all files for this session
    session_id = get_session_id()
    
    # Get uploaded files
    uploaded_files = UploadedFile.query.filter_by(session_id=session_id).order_by(UploadedFile.date_uploaded.desc()).all()
    
    # Get processed files
    processed_files = ProcessedFile.query.filter_by(session_id=session_id).order_by(ProcessedFile.date_processed.desc()).all()
    
    return render_template('manage_files.html', 
                          uploaded_files=uploaded_files, 
                          processed_files=processed_files,
                          current_file=session.get('filepath'))

@app.route('/switch_file/<int:file_id>/<file_type>')
def switch_file(file_id, file_type):
    # Get the file from database
    if file_type == 'uploaded':
        file_obj = UploadedFile.query.get_or_404(file_id)
    else:
        file_obj = ProcessedFile.query.get_or_404(file_id)
    
    # Check if file exists
    if not os.path.exists(file_obj.filepath):
        flash('File not found on disk', 'danger')
        return redirect(url_for('manage_files'))
    
    # Update session
    session['filepath'] = file_obj.filepath
    session['filename'] = file_obj.filename
    
    flash(f'Switched to file: {file_obj.filename}', 'success')
    return redirect(url_for('data_cleaning'))

@app.route('/delete_file/<int:file_id>/<file_type>')
def delete_file(file_id, file_type):
    # Get the file from database
    if file_type == 'uploaded':
        file_obj = UploadedFile.query.get_or_404(file_id)
    else:
        file_obj = ProcessedFile.query.get_or_404(file_id)
    
    # Delete file from disk if exists
    if os.path.exists(file_obj.filepath):
        os.remove(file_obj.filepath)
    
    # Delete from database
    db.session.delete(file_obj)
    db.session.commit()
    
    # If this was the current file, clear session
    if session.get('filepath') == file_obj.filepath:
        session.pop('filepath', None)
        session.pop('filename', None)
        session.pop('columns', None)
    
    flash(f'Deleted file: {file_obj.filename}', 'success')
    return redirect(url_for('manage_files'))

@app.route('/api/ts/data/preview', methods=['GET'])
def get_ts_data_preview():
    try:
        # Get data from session
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available. Please upload or load a dataset first.'}), 400
            
        # Get column information
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        
        # Identify datetime columns
        date_columns = [col for col, dtype in dtypes.items() 
                       if 'datetime' in dtype.lower() or df[col].dtype == 'datetime64[ns]']
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return jsonify({
            'columns': columns,
            'dtypes': dtypes,
            'date_columns': date_columns,
            'numeric_columns': numeric_columns,
            'rows': len(df),
            'dataset_name': session.get('filename', 'Unknown')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ts/convert-datetime', methods=['POST'])
def convert_to_datetime():
    try:
        data = request.json
        column = data.get('column')
        format = data.get('format')  # Optional format string
        
        if not column:
            return jsonify({'error': 'Column name is required'}), 400
            
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available'}), 400
            
        # Try to convert to datetime
        try:
            if format:
                df[column] = pd.to_datetime(df[column], format=format)
            else:
                df[column] = pd.to_datetime(df[column])
                
            # Update session data
            session['data'] = df.to_dict('records')
            
            return jsonify({'success': True, 'message': f'Successfully converted {column} to datetime'})
            
        except Exception as e:
            return jsonify({'error': f'Failed to convert column: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ts/plot', methods=['POST'])
@csrf.exempt
def generate_ts_plot():
    try:
        data = request.json
        date_column = data.get('date_column')
        value_column = data.get('value_column')
        resample = data.get('resample', False)
        freq = data.get('freq', 'D')
        agg = data.get('agg', 'mean')
        
        if not date_column or not value_column:
            return jsonify({'error': 'Date column and value column are required'}), 400
            
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available'}), 400
            
        # Prepare time series data
        ts_data = df[[date_column, value_column]].copy()
        ts_data = ts_data.sort_values(by=date_column)
        
        # Handle resampling if requested
        if resample:
            ts_data = ts_data.set_index(date_column)
            ts_data = ts_data.resample(freq).agg(agg)
            ts_data = ts_data.reset_index()
            
        # Convert to list format for JSON
        plot_data = {
            'dates': ts_data[date_column].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': ts_data[value_column].tolist()
        }
        
        return jsonify(plot_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ts/decompose', methods=['POST'])
@csrf.exempt
def decompose_time_series():
    try:
        data = request.json
        date_column = data.get('date_column')
        value_column = data.get('value_column')
        period = data.get('period')
        model = data.get('model', 'additive')
        
        if not date_column or not value_column:
            return jsonify({'error': 'Date column and value column are required'}), 400
            
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available'}), 400
            
        # Prepare time series data
        ts_data = df[[date_column, value_column]].copy()
        ts_data = ts_data.sort_values(by=date_column)
        ts_data = ts_data.set_index(date_column)
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data[value_column], 
                                         period=int(period), 
                                         model=model)
                                         
        # Convert components to lists for JSON
        result = {
            'dates': ts_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'observed': decomposition.observed.tolist(),
            'trend': decomposition.trend.tolist(),
            'seasonal': decomposition.seasonal.tolist(),
            'residual': decomposition.resid.tolist()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ts/forecast', methods=['POST'])
@csrf.exempt
def forecast_time_series():
    try:
        data = request.json
        date_column = data.get('date_column')
        value_column = data.get('value_column')
        method = data.get('method', 'sma')
        periods = int(data.get('periods', 10))
        params = data.get('params', {})
        
        if not date_column or not value_column:
            return jsonify({'error': 'Date column and value column are required'}), 400
            
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available'}), 400
            
        # Prepare time series data
        ts_data = df[[date_column, value_column]].copy()
        ts_data = ts_data.sort_values(by=date_column)
        
        # Generate forecast based on method
        if method == 'sma':
            window = int(params.get('window', 3))
            # Simple Moving Average
            values = ts_data[value_column].values
            ma = pd.Series(values).rolling(window=window).mean()
            forecast = [ma.iloc[-1]] * periods
            fitted = ma.tolist()
            
        elif method == 'ema':
            alpha = float(params.get('alpha', 0.3))
            # Exponential Moving Average
            values = ts_data[value_column].values
            ema = pd.Series(values).ewm(alpha=alpha).mean()
            forecast = [ema.iloc[-1]] * periods
            fitted = ema.tolist()
            
        else:  # linear trend
            # Simple linear regression
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data[value_column].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            fitted = model.predict(X)
            future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            
        # Generate future dates
        last_date = ts_data[date_column].iloc[-1]
        freq = pd.infer_freq(ts_data[date_column])
        if not freq:
            freq = 'D'  # Default to daily if can't infer
            
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        
        result = {
            'dates': ts_data[date_column].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': ts_data[value_column].tolist(),
            'fitted': fitted,
            'forecast_dates': future_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'forecast': forecast.tolist()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/calculate', methods=['POST'])
@csrf.exempt
def calculate_inventory_metrics():
    try:
        data = request.json
        
        # Extract input values
        purchase_cost = float(data.get('purchase_cost', 0))
        order_cost = float(data.get('order_cost', 0))
        physical_holding_rate = float(data.get('physical_holding_rate', 0))
        financial_holding_rate = float(data.get('financial_holding_rate', 0))
        avg_inventory_value = float(data.get('avg_inventory_value', 0))
        annual_sales = float(data.get('annual_sales', 0))
        cogs = float(data.get('cogs', 0))
        
        # Calculate holding costs
        physical_holding_cost = avg_inventory_value * (physical_holding_rate / 100)
        financial_holding_cost = avg_inventory_value * (financial_holding_rate / 100)
        total_holding_cost = physical_holding_cost + financial_holding_cost
        
        # Calculate inventory turnover metrics
        inventory_turnover = cogs / avg_inventory_value if avg_inventory_value > 0 else 0
        days_inventory = 365 / inventory_turnover if inventory_turnover > 0 else 0
        
        # Calculate asset utilization
        asset_turnover = annual_sales / (avg_inventory_value + order_cost) if (avg_inventory_value + order_cost) > 0 else 0
        
        # Calculate total inventory cost
        total_inventory_cost = purchase_cost + order_cost + total_holding_cost
        
        return jsonify({
            'physical_holding_cost': physical_holding_cost,
            'financial_holding_cost': financial_holding_cost,
            'total_holding_cost': total_holding_cost,
            'inventory_turnover': inventory_turnover,
            'days_inventory': days_inventory,
            'asset_turnover': asset_turnover,
            'total_inventory_cost': total_inventory_cost
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/eoq/calculate', methods=['POST'])
@csrf.exempt
def calculate_eoq():
    try:
        data = request.get_json()
        
        # Extract parameters
        annual_demand = float(data.get('annual_demand', 400))
        order_cost = float(data.get('order_cost', 500))
        unit_cost = float(data.get('unit_cost', 200))
        holding_rate = float(data.get('holding_rate', 0.2))
        custom_order = float(data.get('custom_order', 0))
        
        # Calculate optimal order quantity
        optimal_order = math.sqrt((2 * annual_demand * order_cost) / (unit_cost * holding_rate))
        
        # Calculate costs for optimal order
        optimal_costs = calculate_costs(optimal_order, annual_demand, order_cost, unit_cost, holding_rate)
        
        # Calculate costs for custom order if provided
        custom_costs = calculate_costs(custom_order, annual_demand, order_cost, unit_cost, holding_rate) if custom_order > 0 else None
        
        # Generate data points for cost curves
        order_sizes = [optimal_order * (0.5 + i * 0.1) for i in range(20)]
        order_costs = [(annual_demand / size) * order_cost for size in order_sizes]
        holding_costs = [(size / 2) * unit_cost * holding_rate for size in order_sizes]
        total_costs = [order_costs[i] + holding_costs[i] for i in range(len(order_sizes))]
        
        # Generate inventory pattern data
        time_periods = 12  # One year
        time_points = [i * 0.5 for i in range(time_periods * 2 + 1)]
        
        # Calculate optimal inventory pattern
        optimal_inventory = []
        for t in time_points:
            cycle = t % (optimal_order / (annual_demand / 12))
            inventory = optimal_order - (cycle * (annual_demand / 12))
            optimal_inventory.append(max(0, inventory))
        
        # Calculate custom inventory pattern if provided
        custom_inventory = []
        if custom_order > 0:
            for t in time_points:
                cycle = t % (custom_order / (annual_demand / 12))
                inventory = custom_order - (cycle * (annual_demand / 12))
                custom_inventory.append(max(0, inventory))
        
        return jsonify({
            'optimal_order': optimal_order,
            'optimal_costs': optimal_costs,
            'custom_costs': custom_costs,
            'order_sizes': order_sizes,
            'order_costs': order_costs,
            'holding_costs': holding_costs,
            'total_costs': total_costs,
            'time_points': time_points,
            'optimal_inventory': optimal_inventory,
            'custom_inventory': custom_inventory
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def calculate_costs(order_size, annual_demand, order_cost, unit_cost, holding_rate):
    try:
        num_orders = annual_demand / order_size
        annual_order_cost = num_orders * order_cost
        avg_inventory = order_size / 2
        annual_holding_cost = avg_inventory * unit_cost * holding_rate
        total_cost = annual_order_cost + annual_holding_cost
        
        return {
            'order_cost': annual_order_cost,
            'holding_cost': annual_holding_cost,
            'total_cost': total_cost,
            'num_orders': num_orders
        }
    except:
        return {
            'order_cost': 0,
            'holding_cost': 0,
            'total_cost': 0,
            'num_orders': 0
        }

@app.route('/api/clv/data/preview', methods=['GET'])
def get_clv_data_preview():
    """Return a preview of the data for CLV analysis"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data available. Please upload or load a dataset first.'}), 400
            
        # Get column information
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()
        
        # Identify potential columns based on common names
        id_cols = [col for col in columns if any(term in col.lower() for term in ['customer', 'client', 'user', 'id'])]
        date_cols = [col for col in columns if any(term in col.lower() for term in ['date', 'time', 'invoice', 'purchase'])]
        quantity_cols = [col for col in columns if any(term in col.lower() for term in ['quantity', 'qty', 'amount', 'count'])]
        price_cols = [col for col in columns if any(term in col.lower() for term in ['price', 'value', 'revenue', 'sales'])]
        
        return jsonify({
            'columns': columns,
            'dtypes': dtypes,
            'suggested_columns': {
                'customer_id': id_cols,
                'invoice_date': date_cols,
                'quantity': quantity_cols,
                'price': price_cols
            },
            'rows': len(df),
            'dataset_name': session.get('filename', 'Unknown')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clv/calculate', methods=['POST'])
@csrf.exempt
def calculate_clv():
    """Calculate Customer Lifetime Value"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 404
        
        data = request.get_json()
        customer_id_col = data.get('customer_id_col')
        date_col = data.get('date_col')
        amount_col = data.get('amount_col')
        
        if not all([customer_id_col, date_col, amount_col]):
            return jsonify({'error': 'Missing required columns'}), 400
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate recency, frequency, monetary value
        today = df[date_col].max()
        
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (today - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
        
        # Calculate CLV
        # Using a simple formula: CLV = Average Order Value × Purchase Frequency × Customer Lifespan
        avg_lifespan = 365  # Assuming 1 year for this example
        
        # Handle zero recency values to prevent division by zero
        rfm['recency'] = rfm['recency'].replace(0, 1)  # Replace zeros with 1 to avoid division by zero
        
        rfm['clv'] = (rfm['monetary'] / rfm['frequency']) * rfm['frequency'] * (avg_lifespan / rfm['recency'])
        
        # Prepare results
        results = {
            'customers': rfm.to_dict(orient='records'),
            'summary': {
                'total_customers': len(rfm),
                'average_clv': float(rfm['clv'].mean()),
                'total_revenue': float(rfm['monetary'].sum()),
                'average_order_value': float(rfm['monetary'].sum() / rfm['frequency'].sum())
            }
        }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in CLV calculation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/info', methods=['GET'])
def get_data_info():
    """Get information about the loaded dataset"""
    df = get_session_data()
    if df is None:
        return jsonify({'error': 'No data loaded'}), 404
    
    # Create basic metadata
    metadata = {
        'filename': session.get('filename', 'Unknown'),
        'last_updated': datetime.datetime.now().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Get column types
    column_types = {col: str(df[col].dtype) for col in df.columns}
    
    # Calculate basic stats for each column
    column_stats = {}
    for col in df.columns:
        stats = {
            'missing': int(df[col].isna().sum()),
            'unique_values': int(df[col].nunique())
        }
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            stats.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
            })
        
        column_stats[col] = stats
    
    return jsonify({
        'metadata': metadata,
        'column_types': column_types,
        'column_stats': column_stats
    })

def require_data(f):
    """Decorator to check if data is available"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not data_manager.has_data():
            flash('Please upload a dataset first', 'warning')
            return redirect(url_for('upload'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/visualization')
@require_data
def visualization():
    return render_template('visualization.html')

@app.route('/api/data/preview', methods=['GET'])
def get_data_preview():
    """Get a preview of the loaded data"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 404
            
        # Convert data to JSON-serializable format
        preview_data = []
        for idx, row in df.head(10).iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    row_dict[col] = float(val)
                elif isinstance(val, (np.datetime64, pd.Timestamp)):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = str(val)
            preview_data.append(row_dict)
        
        # Create metadata
        metadata = {
            'filename': session.get('filename', 'Unknown'),
            'last_updated': datetime.datetime.now().isoformat(),
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        return jsonify({
            'data': preview_data,
            'columns': df.columns.tolist(),
            'row_count': len(df),
            'column_count': len(df.columns),
            'filename': metadata.get('filename', 'Unknown'),
            'last_updated': metadata.get('last_updated'),
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
    except Exception as e:
        print(f"Error in data preview: {str(e)}")
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/safety_stock')
def safety_stock():
    try:
        data = get_session_data()
        has_data = data is not None
        return render_template('safety_stock.html', 
                              has_data=has_data, 
                              message="No data loaded. You can still use the simulation with manual inputs." if not has_data else "")
    except Exception as e:
        print(f"Error in safety stock calculation: {str(e)}")
        return render_template('safety_stock.html', 
                              has_data=False, 
                              message="An error occurred. You can still use the simulation with manual inputs.")

@app.route('/api/safety-stock/simulate', methods=['POST'])
@csrf.exempt  # Exempt this route from CSRF protection since we're handling it in the frontend
def simulate_safety_stock():
    try:
        data = request.get_json()
        
        # Extract parameters
        service_level = float(data['service_level']) / 100
        sigma = float(data['sigma'])
        lead_time = float(data['lead_time'])
        expected_demand = float(data['expected_demand'])
        unit_profit = float(data['unit_profit'])
        holding_rate = float(data['holding_rate'])
        weeks = int(data['weeks'])
        
        # Calculate safety stock
        z = norm.ppf(service_level)
        safety_stock = z * sigma * math.sqrt(lead_time)
        
        # Run simulation
        stock_outs = 0
        lost_units = 0
        total_excess = 0
        current_stock = safety_stock + expected_demand  # Start with safety stock + expected demand
        
        # Simulate weekly demand
        np.random.seed()  # Reset random seed
        for _ in range(weeks):
            # Generate random demand with normal distribution
            demand = np.random.normal(expected_demand, sigma)
            
            if demand > current_stock:
                # Stock out occurred
                stock_outs += 1
                lost_units += (demand - current_stock)
                current_stock = 0
            else:
                # Record excess before demand
                excess = max(0, current_stock - demand)
                total_excess += excess
                current_stock = excess
            
            # Replenish inventory back to safety stock + expected demand
            current_stock = safety_stock + expected_demand
        
        # Calculate metrics
        avg_excess = total_excess / weeks
        lost_profit = lost_units * unit_profit
        holding_cost = avg_excess * holding_rate * weeks
        
        # Determine action based on stockouts
        stockout_rate = stock_outs / weeks
        if stockout_rate > 0.05:  # More than 5% stockout rate
            action = "Raise Service"
        elif stockout_rate < 0.02:  # Less than 2% stockout rate
            action = "Lower Service"
        else:
            action = "Maintain Service"
        
        return jsonify({
            'stock_outs': stock_outs,
            'lost_units': int(lost_units),
            'lost_profit': lost_profit,
            'avg_excess': avg_excess,
            'holding_cost': holding_cost,
            'action': action
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/inventory', methods=['GET'])
def inventory():
    return render_template('inventory.html')

@app.route('/api/inventory/analyze', methods=['POST'])
@csrf.exempt
def analyze_inventory():
    try:
        data = request.get_json()
        
        # Extract parameters
        annual_demand = float(data['annual_demand'])
        unit_cost = float(data['unit_cost'])
        holding_cost = float(data['holding_cost']) / 100  # Convert percentage to decimal
        order_cost = float(data['order_cost'])
        lead_time = float(data['lead_time'])
        service_level = float(data['service_level']) / 100  # Convert percentage to decimal
        
        # Calculate EOQ
        eoq = math.sqrt((2 * annual_demand * order_cost) / (unit_cost * holding_cost))
        
        # Calculate safety stock
        z = norm.ppf(service_level)
        daily_demand = annual_demand / 365
        demand_std = daily_demand * 0.2  # Assuming standard deviation is 20% of daily demand
        safety_stock = z * demand_std * math.sqrt(lead_time)
        
        # Calculate reorder point
        rop = (daily_demand * lead_time) + safety_stock
        
        # Calculate average inventory
        avg_inventory = (eoq / 2) + safety_stock
        
        # Calculate costs
        annual_holding_cost = avg_inventory * unit_cost * holding_cost
        annual_order_cost = (annual_demand / eoq) * order_cost
        total_annual_cost = annual_holding_cost + annual_order_cost
        
        # Generate inventory pattern for visualization
        days = 90  # Show 90 days pattern
        time_points = list(range(days))
        inventory_levels = []
        
        current_inventory = eoq
        daily_demand_values = np.random.normal(daily_demand, demand_std, days)
        
        for day in range(days):
            # If inventory hits reorder point, place an order
            if current_inventory <= rop:
                current_inventory += eoq
            
            # Subtract daily demand
            current_inventory = max(0, current_inventory - daily_demand_values[day])
            inventory_levels.append(current_inventory)
        
        return jsonify({
            'eoq': eoq,
            'rop': rop,
            'safety_stock': safety_stock,
            'avg_inventory': avg_inventory,
            'annual_holding_cost': annual_holding_cost,
            'annual_order_cost': annual_order_cost,
            'total_annual_cost': total_annual_cost,
            'inventory_pattern': {
                'time_points': time_points,
                'inventory_levels': inventory_levels
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.before_request
def before_request():
    """Initialize DataManager with the current session before each request"""
    data_manager.init_app(session)

# Add CSRF token route for AJAX requests
@app.route('/get-csrf-token')
def get_csrf_token():
    token = generate_csrf()
    return jsonify({'csrf_token': token})

# Add CSRF token context processor for templates
@app.context_processor
def inject_csrf_token():
    return {'csrf_token': generate_csrf()}

@app.route('/api/data/dtypes', methods=['GET'])
def api_data_dtypes():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    data_types = {}
    for col in df.columns:
        data_types[col] = str(df[col].dtype)
    
    return jsonify(data_types)

@app.route('/api/data/columns', methods=['GET'])
def api_data_columns():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    columns = df.columns.tolist()
    return jsonify({
        'columns': columns,
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    })

@app.route('/api/data/overview', methods=['GET'])
def api_data_overview():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    # Basic dataset statistics
    overview = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'columns': df.columns.tolist(),
    }
    
    # Column statistics
    column_stats = {}
    for col in df.columns:
        stats = {
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_percentage': float(df[col].isna().mean() * 100),
            'unique_count': int(df[col].nunique()),
        }
        
        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            stats.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
            })
        
        column_stats[col] = stats
    
    overview['column_stats'] = column_stats
    
    return jsonify(overview)

@app.route('/api/data/skewness', methods=['GET'])
def api_data_skewness():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    # Calculate skewness for numeric columns
    skewness = {}
    skewness_data = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        try:
            skew_value = float(df[col].skew())
            skewness[col] = {
                'skew': skew_value,
                'interpretation': 'Highly right-skewed' if skew_value > 1 else
                                'Moderately right-skewed' if skew_value > 0.5 else
                                'Approximately symmetric' if abs(skew_value) <= 0.5 else
                                'Moderately left-skewed' if skew_value > -1 else
                                'Highly left-skewed'
            }
            # Add to the array format needed by the frontend
            skewness_data.append({
                'column': col,
                'skewness': skew_value,
                'interpretation': skewness[col]['interpretation']
            })
        except Exception as e:
            skewness[col] = {
                'skew': None,
                'error': str(e)
            }
    
    # Sort skewness data by absolute skewness value (descending)
    skewness_data.sort(key=lambda x: abs(x['skewness']), reverse=True)
    
    return jsonify({
        'skewness': skewness,
        'skewness_data': skewness_data,
        'columns': df.columns.tolist()
    })

@app.route('/api/data/transform', methods=['POST'])
@csrf.exempt
def transform_data():
    """Apply transformations to the dataset (advanced data cleaning)"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({"error": "No data loaded"}), 404
        
        # Get transformation parameters
        data = request.get_json()
        print(f"Received transform request data: {data}")
        
        if not data:
            return jsonify({"error": "No transformation parameters provided. Make sure to send a valid JSON body."}), 400
            
        # Extract parameters with flexible parsing for multiple possible frontend formats
        # Handle the transform_form from advanced_data_cleaning.html
        if 'method' in data:
            transformation = data.get('method')
            if transformation == 'yeo-johnson':
                transformation = 'yeo_johnson'
            elif transformation == 'box-cox':
                transformation = 'boxcox'
        else:
            transformation = data.get('transformation') or data.get('transform_type') or data.get('type')
            
        columns = data.get('columns', []) or data.get('column', [])
        
        # Convert single column to list if needed
        if isinstance(columns, str):
            columns = [columns]
        
        print(f"Parsed transformation: {transformation}, columns: {columns}")
            
        if not transformation:
            return jsonify({
                "error": "Transformation type not specified",
                "received_data": data,
                "help": "Request must include 'transformation', 'transform_type', or 'method' field"
            }), 400
            
        if not columns:
            return jsonify({
                "error": "No columns selected for transformation",
                "received_data": data, 
                "help": "Request must include 'columns' field with at least one column name"
            }), 400
        
        # Make a copy of the dataframe to avoid modifying the original
        df_transformed = df.copy()
        
        # Apply the transformation
        applied_transforms = []
        
        if transformation == 'log':
            # Log transformation (with handling of negative and zero values)
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Log transformation can only be applied to numeric columns'
                    })
                    continue
                
                # Handle negative and zero values
                min_val = df[col].min()
                if min_val <= 0:
                    # Shift all values to make the minimum positive
                    shift = abs(min_val) + 1 if min_val < 0 else 1
                    df_transformed[col] = np.log(df[col] + shift)
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': f'Applied log transformation with shift of {shift}'
                    })
                else:
                    df_transformed[col] = np.log(df[col])
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': 'Applied log transformation'
                    })
        
        elif transformation == 'sqrt':
            # Square root transformation (handles negative values by squaring them first)
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Square root transformation can only be applied to numeric columns'
                    })
                    continue
                
                # Handle negative values
                neg_mask = df[col] < 0
                if neg_mask.any():
                    # For negative values, square them first to make them positive
                    df_transformed.loc[neg_mask, col] = -np.sqrt(df[col][neg_mask] ** 2)
                    df_transformed.loc[~neg_mask, col] = np.sqrt(df[col][~neg_mask])
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': 'Applied square root transformation (negative values handled)'
                    })
                else:
                    df_transformed[col] = np.sqrt(df[col])
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': 'Applied square root transformation'
                    })
        
        elif transformation == 'standard_scale':
            # Standardization (mean=0, std=1)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Standardization can only be applied to numeric columns'
                    })
                    continue
                
                # Standardize
                df_transformed[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                applied_transforms.append({
                    'column': col,
                    'success': True,
                    'message': 'Applied standardization (mean=0, std=1)'
                })
        
        elif transformation == 'min_max_scale':
            # Min-Max scaling (0 to 1)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Min-Max scaling can only be applied to numeric columns'
                    })
                    continue
                
                # Scale to 0-1 range
                df_transformed[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                applied_transforms.append({
                    'column': col,
                    'success': True,
                    'message': 'Applied Min-Max scaling (0 to 1)'
                })
        
        elif transformation == 'robust_scale':
            # Robust scaling (using quantiles)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Robust scaling can only be applied to numeric columns'
                    })
                    continue
                
                # Apply robust scaling
                df_transformed[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                applied_transforms.append({
                    'column': col,
                    'success': True,
                    'message': 'Applied robust scaling (using quantiles)'
                })
        
        elif transformation == 'boxcox':
            # Box-Cox transformation
            from scipy import stats
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Box-Cox transformation can only be applied to numeric columns'
                    })
                    continue
                
                # Box-Cox requires all positive values
                min_val = df[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    try:
                        df_transformed[col], _ = stats.boxcox(df[col] + shift)
                        applied_transforms.append({
                            'column': col,
                            'success': True,
                            'message': f'Applied Box-Cox transformation with shift of {shift}'
                        })
                    except Exception as e:
                        applied_transforms.append({
                            'column': col,
                            'success': False,
                            'message': f'Error in Box-Cox transformation: {str(e)}'
                        })
                else:
                    try:
                        df_transformed[col], _ = stats.boxcox(df[col])
                        applied_transforms.append({
                            'column': col,
                            'success': True,
                            'message': 'Applied Box-Cox transformation'
                        })
                    except Exception as e:
                        applied_transforms.append({
                            'column': col,
                            'success': False,
                            'message': f'Error in Box-Cox transformation: {str(e)}'
                        })
        
        elif transformation == 'yeo_johnson':
            # Yeo-Johnson transformation
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson')
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Yeo-Johnson transformation can only be applied to numeric columns'
                    })
                    continue
                
                try:
                    df_transformed[col] = pt.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': 'Applied Yeo-Johnson transformation'
                    })
                except Exception as e:
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': f'Error in Yeo-Johnson transformation: {str(e)}'
                    })
        
        elif transformation == 'winsorize':
            # Winsorize outliers
            from scipy import stats
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Winsorization can only be applied to numeric columns'
                    })
                    continue
                
                try:
                    # Winsorize values at 5th and 95th percentiles
                    df_transformed[col] = stats.mstats.winsorize(df[col], limits=[0.05, 0.05])
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': 'Applied winsorization (5% on each tail)'
                    })
                except Exception as e:
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': f'Error in winsorization: {str(e)}'
                    })
        
        elif transformation == 'one_hot_encode':
            # One-hot encoding
            for col in columns:
                if col not in df.columns:
                    continue
                
                # Don't one-hot encode columns with too many categories
                if df[col].nunique() > 50:
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': f'Column has too many unique values ({df[col].nunique()}) for one-hot encoding'
                    })
                    continue
                
                try:
                    # Get dummies and add them to the dataframe
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_transformed = pd.concat([df_transformed, dummies], axis=1)
                    
                    # Keep track of added dummy columns
                    dummy_cols = dummies.columns.tolist()
                    
                    # Drop the original column if all dummies were created successfully
                    df_transformed = df_transformed.drop(columns=[col])
                    
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': f'Applied one-hot encoding, added {len(dummy_cols)} dummy variables'
                    })
                except Exception as e:
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': f'Error in one-hot encoding: {str(e)}'
                    })
        
        elif transformation == 'bin':
            # Binning (equal width)
            num_bins = data.get('num_bins', 5)
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if not pd.api.types.is_numeric_dtype(df[col]):
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': 'Binning can only be applied to numeric columns'
                    })
                    continue
                
                try:
                    # Create bin edges
                    bins = pd.cut(df[col], bins=num_bins, labels=False)
                    df_transformed[f"{col}_binned"] = bins
                    
                    applied_transforms.append({
                        'column': col,
                        'success': True,
                        'message': f'Applied equal-width binning with {num_bins} bins'
                    })
                except Exception as e:
                    applied_transforms.append({
                        'column': col,
                        'success': False,
                        'message': f'Error in binning: {str(e)}'
                    })
        
        else:
            return jsonify({"error": f"Unsupported transformation: {transformation}"}), 400
        
        # Generate a unique filename for the transformed data
        original_filename = session.get('filename', 'data.csv')
        base_name, file_ext = os.path.splitext(original_filename)
        unique_filename = f"transformed_{base_name}_{uuid.uuid4().hex}{file_ext}"
        transformed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the transformed dataframe
        if file_ext.lower() == '.csv':
            df_transformed.to_csv(transformed_filepath, index=False)
        elif file_ext.lower() in ['.xls', '.xlsx']:
            df_transformed.to_excel(transformed_filepath, index=False)
        elif file_ext.lower() == '.json':
            df_transformed.to_json(transformed_filepath, orient='records')
        
        # Update session to use transformed data
        session['filepath'] = transformed_filepath
        session['filename'] = f"transformed_{original_filename}"
        
        # Store in database
        session_id = get_session_id()
        # Get original file id if exists
        original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
        original_id = original_file.id if original_file else None
        
        processed_file = ProcessedFile(
            original_file_id=original_id,
            filename=f"transformed_{original_filename}",
            filepath=transformed_filepath,
            process_type='transformed',
            session_id=session_id
        )
        db.session.add(processed_file)
        db.session.commit()
        
        # Return the result
        return jsonify({
            'success': True,
            'transformations': applied_transforms,
            'preview': df_transformed.head(10).to_dict(orient='records'),
            'columns': df_transformed.columns.tolist(),
            'shape': df_transformed.shape
        })
    
    except Exception as e:
        print(f"Error in data transformation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/feature-selection', methods=['POST'])
@csrf.exempt
def api_feature_selection():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"}), 404
    
    try:
        # Get request data
        data = request.get_json()
        method = data.get('method', 'correlation')
        target = data.get('target')
        n_features = int(data.get('n_features', 5))
        
        # Validate inputs
        if target and target not in df.columns:
            return jsonify({"error": f"Target column {target} not found"}), 400
        
        # Handle different feature selection methods
        if method == 'correlation':
            # Select features based on correlation with target
            if not target:
                return jsonify({"error": "Target column required for correlation method"}), 400
                
            # Get only numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            # Check if target is in numeric columns
            if target not in numeric_df.columns:
                return jsonify({"error": f"Target column {target} must be numeric"}), 400
                
            # Calculate correlation with target
            correlations = numeric_df.corr()[target].sort_values(ascending=False)
            
            # Remove target itself from correlations
            correlations = correlations.drop(target)
            
            # Get top features
            top_features = correlations.head(n_features)
            bottom_features = correlations.tail(n_features)
            
            return jsonify({
                "top_positively_correlated": top_features.to_dict(),
                "top_negatively_correlated": bottom_features.to_dict()
            })
            
        elif method == 'variance':
            # Select features based on variance
            numeric_df = df.select_dtypes(include=['number'])
            
            # Calculate variance
            variances = numeric_df.var().sort_values(ascending=False)
            
            # Get top features
            top_features = variances.head(n_features)
            
            return jsonify({
                "top_variance_features": top_features.to_dict()
            })
        
        else:
            return jsonify({"error": f"Unsupported method: {method}"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/clustering', methods=['POST'])
@csrf.exempt
def api_clustering():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"}), 404
    
    try:
        # Get request data
        data = request.get_json()
        method = data.get('method', 'kmeans')
        features = data.get('features', [])
        n_clusters = int(data.get('n_clusters', 3))
        
        # Validate inputs
        if not features:
            return jsonify({"error": "Features list required"}), 400
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return jsonify({"error": f"Features not found: {missing_features}"}), 400
        
        # Get data for clustering
        X = df[features].select_dtypes(include=['number'])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on method
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return jsonify({"error": f"Unsupported method: {method}"}), 400
            
        # Fit model
        labels = model.fit_predict(X_scaled)
        
        # Get cluster statistics
        df_result = df.copy()
        df_result['cluster'] = labels
        
        # Compute cluster statistics
        cluster_stats = []
        for i in range(max(labels) + 1):
            cluster_data = df_result[df_result['cluster'] == i]
            stats = {
                'cluster': int(i),
                'size': len(cluster_data),
                'percentage': float(len(cluster_data) / len(df) * 100)
            }
            
            # Add mean/std for numeric features
            for col in features:
                if col in X.columns:
                    stats[f'{col}_mean'] = float(cluster_data[col].mean())
                    stats[f'{col}_std'] = float(cluster_data[col].std())
            
            cluster_stats.append(stats)
        
        return jsonify({
            "cluster_stats": cluster_stats,
            "cluster_labels": labels.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feature-engineering', methods=['POST'])
@csrf.exempt
def api_feature_engineering():
    df = get_session_data()
    if df is None:
        return jsonify({"error": "No data loaded"}), 404
    
    try:
        # Get request data
        data = request.get_json()
        operation = data.get('operation')
        columns = data.get('columns', [])
        new_column_name = data.get('new_column_name')
        
        # Validate inputs
        if not operation:
            return jsonify({"error": "Operation required"}), 400
            
        if not columns:
            return jsonify({"error": "Columns required"}), 400
            
        if not new_column_name:
            return jsonify({"error": "New column name required"}), 400
            
        # Check if columns exist
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Columns not found: {missing_columns}"}), 400
            
        # Perform operation
        df_result = df.copy()
        
        if operation == 'sum':
            df_result[new_column_name] = df_result[columns].sum(axis=1)
        elif operation == 'mean':
            df_result[new_column_name] = df_result[columns].mean(axis=1)
        elif operation == 'max':
            df_result[new_column_name] = df_result[columns].max(axis=1)
        elif operation == 'min':
            df_result[new_column_name] = df_result[columns].min(axis=1)
        elif operation == 'product':
            df_result[new_column_name] = df_result[columns].product(axis=1)
        elif operation == 'log':
            if len(columns) != 1:
                return jsonify({"error": "Log operation requires exactly one column"}), 400
            df_result[new_column_name] = np.log(df_result[columns[0]])
        elif operation == 'square':
            if len(columns) != 1:
                return jsonify({"error": "Square operation requires exactly one column"}), 400
            df_result[new_column_name] = np.square(df_result[columns[0]])
        elif operation == 'sqrt':
            if len(columns) != 1:
                return jsonify({"error": "Square root operation requires exactly one column"}), 400
            df_result[new_column_name] = np.sqrt(df_result[columns[0]])
        else:
            return jsonify({"error": f"Unsupported operation: {operation}"}), 400
            
        # Update session data
        session['data'] = df_result.to_dict('records')
        
        # Generate a unique filename
        original_filename = session.get('filename', 'data.csv')
        base_name, file_ext = os.path.splitext(original_filename)
        unique_filename = f"processed_{base_name}_{uuid.uuid4().hex}{file_ext}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save processed dataframe
        if file_ext.lower() == '.csv':
            df_result.to_csv(processed_filepath, index=False)
        elif file_ext.lower() in ['.xls', '.xlsx']:
            df_result.to_excel(processed_filepath, index=False)
        elif file_ext.lower() == '.json':
            df_result.to_json(processed_filepath, orient='records')
        
        # Update session
        session['filepath'] = processed_filepath
        session_id = get_session_id()
        
        # Get original file id if exists
        original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
        original_id = original_file.id if original_file else None
        
        processed_file = ProcessedFile(
            original_file_id=original_id,
            filename=f"processed_{original_filename}",
            filepath=processed_filepath,
            process_type='feature_engineered',
            session_id=session_id
        )
        db.session.add(processed_file)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": f"Created new column: {new_column_name}",
            "new_column": df_result[new_column_name].head(5).tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/column-stats', methods=['GET'])
def get_column_stats():
    """Get statistics for a specific column"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 404
            
        column = request.args.get('column')
        if not column:
            return jsonify({'error': 'Column parameter is required'}), 400
            
        if column not in df.columns:
            return jsonify({'error': f'Column {column} not found'}), 404
            
        # Calculate basic stats
        stats = {
            'count': len(df),
            'missing': int(df[column].isna().sum()),
            'missing_percent': float(df[column].isna().mean() * 100),
            'unique': int(df[column].nunique())
        }
        
        # Calculate type-specific stats
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric column stats
            stats.update({
                'min': float(df[column].min()) if not pd.isna(df[column].min()) else None,
                'max': float(df[column].max()) if not pd.isna(df[column].max()) else None,
                'mean': float(df[column].mean()) if not pd.isna(df[column].mean()) else None,
                'median': float(df[column].median()) if not pd.isna(df[column].median()) else None,
                'std': float(df[column].std()) if not pd.isna(df[column].std()) else None
            })
        elif pd.api.types.is_datetime64_dtype(df[column]) or pd.api.types.is_datetime64_ns_dtype(df[column]):
            # Date column stats
            min_date = df[column].min()
            max_date = df[column].max()
            stats.update({
                'min': min_date.isoformat() if min_date is not pd.NaT else None,
                'max': max_date.isoformat() if max_date is not pd.NaT else None,
                'range': (max_date - min_date).days if max_date is not pd.NaT and min_date is not pd.NaT else None
            })
        else:
            # Categorical column stats - get top values
            value_counts = df[column].value_counts().head(5)
            top_values = []
            
            for value, count in value_counts.items():
                top_values.append({
                    'value': str(value),
                    'count': int(count),
                    'percentage': float(count / len(df) * 100)
                })
                
            stats['top_values'] = top_values
            
        return jsonify(stats)
            
    except Exception as e:
        print(f"Error getting column stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/handle-cardinality', methods=['POST'])
@csrf.exempt
def handle_cardinality():
    """Handle high cardinality in categorical variables"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({"error": "No data loaded"}), 404
        
        # Get parameters
        data = request.get_json()
        if not data:
            return jsonify({"error": "No parameters provided"}), 400
        
        max_categories = int(data.get('max_categories', 10))
        method = data.get('method', 'group_small')
        
        # Make a copy of the dataframe
        df_transformed = df.copy()
        
        # Get categorical columns with high cardinality
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        processed_cols = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > max_categories:
                if method == 'group_small':
                    # Group small categories into 'Other'
                    value_counts = df[col].value_counts()
                    top_categories = value_counts.index[:max_categories].tolist()
                    
                    # Replace less frequent categories with 'Other'
                    df_transformed[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                    processed_cols.append(col)
                    
                elif method == 'target_encoding':
                    # For target encoding, we'd need a target variable
                    # This is a simplified version that uses a mean encoding of some column
                    # In a real implementation, you'd use the actual target variable
                    
                    # Find a numeric column to use as target
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        target_col = numeric_cols[0]
                        # Get mean of target for each category
                        means = df.groupby(col)[target_col].mean().to_dict()
                        # Replace categories with their mean target value
                        df_transformed[col + '_encoded'] = df[col].map(means)
                        processed_cols.append(col)
        
        # Generate a unique filename
        original_filename = session.get('filename', 'data.csv')
        base_name, file_ext = os.path.splitext(original_filename)
        unique_filename = f"cardinality_{base_name}_{uuid.uuid4().hex}{file_ext}"
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save processed dataframe
        if file_ext.lower() == '.csv':
            df_transformed.to_csv(processed_filepath, index=False)
        elif file_ext.lower() in ['.xls', '.xlsx']:
            df_transformed.to_excel(processed_filepath, index=False)
        elif file_ext.lower() == '.json':
            df_transformed.to_json(processed_filepath, orient='records')
        
        # Update session
        session['filepath'] = processed_filepath
        session['filename'] = f"cardinality_{original_filename}"
        
        # Store in database
        session_id = get_session_id()
        original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
        original_id = original_file.id if original_file else None
        
        processed_file = ProcessedFile(
            original_file_id=original_id,
            filename=f"cardinality_{original_filename}",
            filepath=processed_filepath,
            process_type='cardinality',
            session_id=session_id
        )
        db.session.add(processed_file)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'processed_cols': processed_cols,
            'preview': df_transformed.head(5).to_dict(orient='records'),
            'columns': df_transformed.columns.tolist()
        })
        
    except Exception as e:
        print(f"Error handling cardinality: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/impute', methods=['POST'])
@csrf.exempt
def impute_missing_values():
    """Impute missing values using advanced methods"""
    try:
        df = get_session_data()
        if df is None:
            return jsonify({"error": "No data loaded"}), 404
        
        # Get parameters
        data = request.get_json()
        print(f"Received imputation request data: {data}")
        
        if not data:
            return jsonify({"error": "No parameters provided"}), 400
        
        strategy = data.get('strategy', 'knn')
        n_neighbors = int(data.get('n_neighbors', 5))
        columns = data.get('columns', [])
        
        # Convert single column to list if needed
        if isinstance(columns, str):
            columns = [columns]
        
        print(f"Imputation request: strategy={strategy}, columns={columns}")
        
        if not columns:
            return jsonify({"error": "No columns selected for imputation"}), 400
        
        # Make a copy of the dataframe
        df_transformed = df.copy()
        
        # Check all column missing values
        missing_values_info = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_values_info[col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(df) * 100)
            }
        
        print(f"Missing values in dataset: {missing_values_info}")
        
        # Verify that columns exist and check for missing values
        missing_cols = []
        no_missing_cols = []
        for col in columns:
            if col not in df.columns:
                return jsonify({"error": f"Column {col} not found in dataset"}), 400
            
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_cols.append(col)
            else:
                no_missing_cols.append(col)
        
        # If none of the selected columns have missing values, give a helpful message
        if not missing_cols:
            print(f"No missing values found in columns: {columns}")
            return jsonify({
                "message": "Selected columns don't have missing values. No imputation needed.",
                "imputed_cols": [],
                "no_missing_cols": no_missing_cols,
                "success": True
            })
        
        # Log what we're imputing
        print(f"Columns with missing values to impute: {missing_cols}")
        print(f"Columns without missing values (skipping): {no_missing_cols}")
        
        # Apply imputation
        imputed_cols = []
        
        # Add a special case for if there are missing values in the dataset but not in selected columns
        all_missing = sum(df[col].isna().sum() for col in df.columns)
        if all_missing > 0 and not missing_cols:
            # There are missing values but not in the selected columns
            return jsonify({
                "message": f"Selected columns don't have missing values, but dataset has {all_missing} missing values in other columns.",
                "missing_info": missing_values_info,
                "success": True
            })
        
        if strategy == 'knn':
            try:
                from sklearn.impute import KNNImputer
                
                # Select only numeric columns for KNN imputation
                numeric_cols = [col for col in missing_cols if pd.api.types.is_numeric_dtype(df[col])]
                non_numeric = [col for col in missing_cols if col not in numeric_cols]
                
                if non_numeric:
                    print(f"Skipping non-numeric columns for KNN imputation: {non_numeric}")
                
                if len(numeric_cols) > 0:
                    # Create and fit the imputer
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    # Impute values
                    imputed_data = imputer.fit_transform(df[numeric_cols])
                    
                    # Update dataframe with imputed values
                    for i, col in enumerate(numeric_cols):
                        df_transformed[col] = imputed_data[:, i]
                        imputed_cols.append(col)
                
                # For non-numeric columns, use simple imputation
                for col in non_numeric:
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                        # Use most frequent value for categorical
                        most_freq = df[col].mode()[0]
                        df_transformed[col] = df[col].fillna(most_freq)
                        imputed_cols.append(col)
            except Exception as e:
                print(f"KNN imputation error: {str(e)}")
                return jsonify({"error": f"Error in KNN imputation: {str(e)}"}), 500
                
        elif strategy == 'iterative':
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                
                # Select only numeric columns for iterative imputation
                numeric_cols = [col for col in missing_cols if pd.api.types.is_numeric_dtype(df[col])]
                non_numeric = [col for col in missing_cols if col not in numeric_cols]
                
                if non_numeric:
                    print(f"Skipping non-numeric columns for iterative imputation: {non_numeric}")
                
                if len(numeric_cols) > 0:
                    # Create and fit the imputer
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    # Impute values
                    imputed_data = imputer.fit_transform(df[numeric_cols])
                    
                    # Update dataframe with imputed values
                    for i, col in enumerate(numeric_cols):
                        df_transformed[col] = imputed_data[:, i]
                        imputed_cols.append(col)
                
                # For non-numeric columns, use simple imputation
                for col in non_numeric:
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                        # Use most frequent value for categorical
                        most_freq = df[col].mode()[0]
                        df_transformed[col] = df[col].fillna(most_freq)
                        imputed_cols.append(col)
            except Exception as e:
                print(f"Iterative imputation error: {str(e)}")
                return jsonify({"error": f"Error in iterative imputation: {str(e)}"}), 500
                
        else:
            return jsonify({"error": f"Unsupported imputation strategy: {strategy}"}), 400
        
        # Only save file if imputation actually happened
        if imputed_cols:
            # Generate a unique filename
            original_filename = session.get('filename', 'data.csv')
            base_name, file_ext = os.path.splitext(original_filename)
            unique_filename = f"imputed_{base_name}_{uuid.uuid4().hex}{file_ext}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save processed dataframe
            if file_ext.lower() == '.csv':
                df_transformed.to_csv(processed_filepath, index=False)
            elif file_ext.lower() in ['.xls', '.xlsx']:
                df_transformed.to_excel(processed_filepath, index=False)
            elif file_ext.lower() == '.json':
                df_transformed.to_json(processed_filepath, orient='records')
            
            # Update session
            session['filepath'] = processed_filepath
            session['filename'] = f"imputed_{original_filename}"
            
            # Store in database
            session_id = get_session_id()
            original_file = UploadedFile.query.filter_by(filepath=session.get('filepath')).first()
            original_id = original_file.id if original_file else None
            
            processed_file = ProcessedFile(
                original_file_id=original_id,
                filename=f"imputed_{original_filename}",
                filepath=processed_filepath,
                process_type='imputed',
                session_id=session_id
            )
            db.session.add(processed_file)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'imputed_cols': imputed_cols,
                'no_missing_cols': no_missing_cols,
                'preview': df_transformed.head(5).to_dict(orient='records'),
                'message': f"Successfully imputed {len(imputed_cols)} columns"
            })
        else:
            return jsonify({
                'success': True,
                'message': "No imputation needed or no suitable columns for imputation",
                'imputed_cols': [],
                'no_missing_cols': no_missing_cols
            })
        
    except Exception as e:
        print(f"Error in data imputation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 