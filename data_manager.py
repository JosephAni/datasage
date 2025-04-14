import pandas as pd
import numpy as np
from flask import session
import os
import tempfile
from datetime import datetime

class DataManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the data manager"""
        self.data = None
        self.metadata = {
            'last_updated': None,
            'filename': None,
            'row_count': 0,
            'column_count': 0
        }
        self.temp_dir = tempfile.mkdtemp()
        
    def load_data(self, file_path, file_name):
        """Load data from file and store metadata"""
        try:
            # Load data based on file extension
            if file_name.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_name.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            
            # Update metadata
            self.metadata.update({
                'last_updated': datetime.now().isoformat(),
                'filename': file_name,
                'row_count': len(self.data),
                'column_count': len(self.data.columns)
            })
            
            # Store in session
            self._save_to_session()
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def _save_to_session(self):
        """Save data state to session"""
        if self.data is not None:
            # Save metadata to session
            session['data_metadata'] = self.metadata
            
            # Save data to temporary parquet file
            temp_file = os.path.join(self.temp_dir, 'temp_data.parquet')
            self.data.to_parquet(temp_file)
            session['data_path'] = temp_file
    
    def get_data(self):
        """Retrieve data, loading from session if necessary"""
        if self.data is None:
            self._load_from_session()
        return self.data
    
    def _load_from_session(self):
        """Load data from session storage"""
        if 'data_path' in session:
            try:
                self.data = pd.read_parquet(session['data_path'])
                self.metadata = session.get('data_metadata', {})
                return True
            except Exception as e:
                print(f"Error loading data from session: {str(e)}")
        return False
    
    def clear_data(self):
        """Clear all data and session storage"""
        self.data = None
        self.metadata = {
            'last_updated': None,
            'filename': None,
            'row_count': 0,
            'column_count': 0
        }
        if 'data_path' in session:
            try:
                os.remove(session['data_path'])
            except:
                pass
        session.pop('data_path', None)
        session.pop('data_metadata', None)
    
    def has_data(self):
        """Check if data is available"""
        return self.data is not None or self._load_from_session()
    
    def get_column_types(self):
        """Get column types for the loaded data"""
        if not self.has_data():
            return {}
        return self.data.dtypes.astype(str).to_dict()
    
    def get_column_stats(self):
        """Get basic statistics for each column"""
        if not self.has_data():
            return {}
        
        stats = {}
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                stats[col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max())
                }
            else:
                stats[col] = {
                    'unique_count': int(self.data[col].nunique()),
                    'missing_count': int(self.data[col].isnull().sum())
                }
        return stats 