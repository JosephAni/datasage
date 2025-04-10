from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
import pandas as pd
import numpy as np
import os
import json
import io
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create Flask app
app = Flask(__name__)
app.secret_key = 'data-cleaner-secret-key'  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Utility functions
def load_sample_data():
    """Load a sample dataset for demonstration"""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df
    except:
        return pd.DataFrame()

def get_session_data():
    """Get dataframe from current session"""
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return None
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return None
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
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
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'})
        
        # Store data info in session
        session['filename'] = filename
        session['filepath'] = filepath
        session['columns'] = df.columns.tolist()
        session['row_count'] = len(df)
        
        return redirect(url_for('data_cleaning'))

@app.route('/sample')
def use_sample_data():
    # Load sample data
    df = load_sample_data()
    
    # Store data info in session
    session['filename'] = 'sample_data.csv'
    session['filepath'] = None
    session['columns'] = df.columns.tolist()
    session['row_count'] = len(df)
    
    # Save sample data to a file
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_data.csv')
    df.to_csv(sample_path, index=False)
    session['filepath'] = sample_path
    
    return redirect(url_for('data_cleaning'))

@app.route('/data_cleaning')
def data_cleaning():
    return render_template('data_cleaning.html')

@app.route('/inventory_turnover')
def inventory_turnover():
    return render_template('inventory_turnover.html')

@app.route('/data_visualization')
def data_visualization():
    return render_template('data_visualization.html')

@app.route('/time_series')
def time_series():
    return render_template('time_series.html')

@app.route('/clv_analysis')
def clv_analysis():
    return render_template('clv_analysis.html')

@app.route('/machine_learning')
def machine_learning():
    return render_template('machine_learning.html')

@app.route('/newsvendor_simulation')
def newsvendor_simulation():
    return render_template('newsvendor_simulation.html')

@app.route('/eoq_simulation')
def eoq_simulation():
    return render_template('eoq_simulation.html')

# API endpoints for data processing
@app.route('/api/data/preview', methods=['GET'])
def get_data_preview():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No data available'})
    
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'})
        
        # Return preview (first 100 rows)
        return jsonify({
            'filename': session.get('filename', 'Unknown'),
            'columns': df.columns.tolist(),
            'data': df.head(100).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/data/clean', methods=['POST'])
def clean_data():
    # Get cleaning actions from request
    cleaning_actions = request.json
    
    # Get current dataframe
    df = get_session_data()
    if df is None:
        return jsonify({'error': 'No data available'})
    
    # Apply cleaning
    cleaned_df, summary = apply_cleaning(df, cleaning_actions)
    
    # Save cleaned dataframe
    cleaned_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_data.csv')
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # Update session
    session['filepath'] = cleaned_path
    session['filename'] = 'cleaned_data.csv'
    session['columns'] = cleaned_df.columns.tolist()
    session['row_count'] = len(cleaned_df)
    
    return jsonify({
        'success': True,
        'summary': summary,
        'rows': len(cleaned_df),
        'columns': len(cleaned_df.columns)
    })

@app.route('/api/data/download', methods=['GET'])
def download_data():
    format_type = request.args.get('format', 'csv')
    
    # Get current dataframe
    df = get_session_data()
    if df is None:
        return jsonify({'error': 'No data available'})
    
    if format_type == 'csv':
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Create response
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            download_name='cleaned_data.csv',
            as_attachment=True
        )
    
    elif format_type == 'excel':
        # Create Excel file in memory
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        
        # Create response
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name='cleaned_data.xlsx',
            as_attachment=True
        )
    
    else:
        return jsonify({'error': 'Unsupported format'})

if __name__ == '__main__':
    app.run(debug=True) 