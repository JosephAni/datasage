# data_manager.py
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime

class DataManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialize()
            cls._instance._check_dependencies()
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
        self._session = None
    
    def init_app(self, session):
        """Initialize with Flask session"""
        self._session = session
        
    def load_data(self, file_path=None, file_obj=None, file_ext=None, sample_name=None):
        """Load data from a file or sample"""
        try:
            if sample_name:
                success = self._load_sample(sample_name)
                if not success:
                    print(f"Failed to load sample data: {sample_name}")
                    return False
            elif file_path or file_obj:
                success = self._load_file(file_path, file_obj, file_ext)
                if not success:
                    print(f"Failed to load file data from {file_path or 'uploaded file'}")
                    return False
            else:
                success = self._load_from_session()
                if not success:
                    print("Failed to load data from session and no alternative data source provided")
                    return False

            # Generate column metadata after loading
            if self.data is not None and not self.data.empty:
                self._generate_column_metadata()

                # --- Add this section to save data to Flask session ---
                if self._session is not None:
                    try:
                        # Convert DataFrame to JSON and store in session
                        self._session['current_dataset'] = self.data.to_json()
                        print("DataManager: Successfully saved DataFrame to session['current_dataset']")
                        # Also store filename in session
                        if file_path:
                            self._session['filename'] = os.path.basename(file_path)
                        elif sample_name:
                            self._session['filename'] = f"sample_{sample_name}.csv" # Or appropriate extension
                        print(f"DataManager: Saved filename '{self._session.get('filename')}' to session")

                    except Exception as e:
                        print(f"DataManager: Error saving DataFrame to session: {str(e)}")
                        import traceback
                        traceback.print_exc()
                # -----------------------------------------------------

                return True
            else:
                print("Data loaded but appears to be empty")
                return False

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    
    def _save_to_session(self):
        """Save current data to session storage with optimized performance"""
        if self._session is None or self.data is None:
            return False
            
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Save as parquet if available for better performance and type preservation
            if hasattr(self, 'has_pyarrow') and self.has_pyarrow:
                try:
                    parquet_path = os.path.join(self.temp_dir, 'temp_data.parquet')
                    # Use compression for better performance
                    self.data.to_parquet(parquet_path, index=False, compression='gzip')
                except Exception as e:
                    print(f"Error saving to Parquet, falling back to CSV: {str(e)}")
                    # Fall back to CSV with compression
                    csv_path = os.path.join(self.temp_dir, 'temp_data.csv.gz')
                    self.data.to_csv(csv_path, index=False, compression='gzip')
            else:
                # Save as compressed CSV
                csv_path = os.path.join(self.temp_dir, 'temp_data.csv.gz')
                self.data.to_csv(csv_path, index=False, compression='gzip')
            
            # Update session with metadata
            self._session['data_metadata'] = self.metadata
            
            return True
        except Exception as e:
            print(f"Error saving data to session: {str(e)}")
            return False
    
    def get_data(self):
        """Retrieve data, loading from session if necessary"""
        if self.data is None:
            self._load_from_session()
        return self.data
    
    def _load_from_session(self):
        """Load data from session storage with optimized performance"""
        if self._session is None:
            return False
            
        try:
            # Try to load from parquet first if pyarrow is available
            if hasattr(self, 'has_pyarrow') and self.has_pyarrow:
                parquet_path = os.path.join(self.temp_dir, 'temp_data.parquet')
                if os.path.exists(parquet_path):
                    try:
                        # Use chunking for large files
                        file_size = os.path.getsize(parquet_path)
                        if file_size > 50 * 1024 * 1024:  # If file > 50MB
                            print("Large Parquet file detected, using chunked reading...")
                            chunks = []
                            for chunk in pd.read_parquet(parquet_path, chunksize=100000):
                                chunks.append(chunk)
                            self.data = pd.concat(chunks, ignore_index=True)
                        else:
                            self.data = pd.read_parquet(parquet_path)
                        return True
                    except Exception as e:
                        print(f"Failed to load Parquet data from session: {str(e)}")
                        # Fall back to CSV
            
            # Try to load from compressed CSV
            csv_path = os.path.join(self.temp_dir, 'temp_data.csv.gz')
            if os.path.exists(csv_path):
                # Use chunking for large files
                file_size = os.path.getsize(csv_path)
                if file_size > 50 * 1024 * 1024:  # If file > 50MB
                    print("Large CSV file detected, using chunked reading...")
                    chunks = []
                    for chunk in pd.read_csv(csv_path, chunksize=100000, compression='gzip'):
                        chunks.append(chunk)
                    self.data = pd.concat(chunks, ignore_index=True)
                else:
                    self.data = pd.read_csv(csv_path, compression='gzip')
                return True
            
            return False
        except Exception as e:
            print(f"Error loading data from session: {str(e)}")
            return False
            
    def _load_file(self, file_path=None, file_obj=None, file_ext=None):
        """Load data from a file path or file object with optimized performance"""
        try:
            print(f"Attempting to load file: {file_path or 'file object'}")
            print(f"File extension: {file_ext}")
            
            # Determine file extension
            if file_path:
                file_extension = os.path.splitext(file_path.lower())[1]
                print(f"Detected file extension: {file_extension}")
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    return False
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size} bytes")
            elif file_obj and file_ext:
                file_extension = file_ext.lower()
                print(f"Using provided file extension: {file_extension}")
                file_size = file_obj.seek(0, 2)  # Get file size
                file_obj.seek(0)  # Reset file pointer
            else:
                print("Either file_path or both file_obj and file_ext must be provided")
                return False
                
            # For large files, use chunking
            CHUNK_SIZE = 100000  # Process 100k rows at a time
            chunks = []
            
            # Load data based on file extension
            if file_extension == '.csv':
                print("Loading CSV file...")
                if file_path:
                    # Use chunking for large CSV files
                    if file_size > 50 * 1024 * 1024:  # If file > 50MB
                        print("Large file detected, using chunked reading...")
                        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
                            chunks.append(chunk)
                        self.data = pd.concat(chunks, ignore_index=True)
                    else:
                        self.data = pd.read_csv(file_path)
                else:
                    print("Loading CSV from file object...")
                    self.data = pd.read_csv(file_obj)
                
            elif file_extension in ['.xls', '.xlsx']:
                print("Loading Excel file...")
                if file_path:
                    print(f"Reading Excel file from path: {file_path}")
                    # For large Excel files, read only the first sheet
                    self.data = pd.read_excel(file_path, sheet_name=0)
                else:
                    print("Reading Excel file from file object")
                    self.data = pd.read_excel(file_obj, sheet_name=0)
                
            elif file_extension == '.json':
                print("Loading JSON file...")
                if file_path:
                    # For large JSON files, use chunking
                    if file_size > 50 * 1024 * 1024:
                        print("Large JSON file detected, using chunked reading...")
                        with open(file_path, 'r') as f:
                            for chunk in pd.read_json(f, lines=True, chunksize=CHUNK_SIZE):
                                chunks.append(chunk)
                        self.data = pd.concat(chunks, ignore_index=True)
                    else:
                        self.data = pd.read_json(file_path)
                else:
                    self.data = pd.read_json(file_obj)
                
            elif hasattr(self, 'has_pyarrow') and self.has_pyarrow and file_extension == '.parquet':
                print("Loading Parquet file...")
                if file_path:
                    self.data = pd.read_parquet(file_path)
                else:
                    self.data = pd.read_parquet(file_obj)
            else:
                print(f"Unsupported file format: {file_extension}")
                return False
                
            # Update metadata
            print("Updating metadata...")
            self.metadata.update({
                'last_updated': datetime.now().isoformat(),
                'filename': os.path.basename(file_path) if file_path else "uploaded_file",
                'row_count': len(self.data),
                'column_count': len(self.data.columns)
            })
            
            print(f"Successfully loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Store in session if available
            if self._session is not None:
                print("Saving to session...")
                self._save_to_session()
                
            return True
            
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def _load_csv_file(self, file_path):
        """Load CSV file with multiple encoding and delimiter options"""
        try:
            print(f"Attempting to load CSV file: {file_path}")
            print("Trying UTF-8 encoding...")
            # Try different encodings and delimiters
            try:
                self.data = pd.read_csv(file_path, encoding='utf-8')
                print("Successfully loaded with UTF-8 encoding")
                return
            except UnicodeDecodeError as e:
                print(f"UTF-8 failed: {str(e)}")
                print("Trying Latin-1 encoding...")
                try:
                    self.data = pd.read_csv(file_path, encoding='latin1')
                    print("Successfully loaded with Latin-1 encoding")
                    return
                except Exception as e:
                    print(f"Latin-1 failed: {str(e)}")
                    print("Trying different delimiters...")
                    # Try with different delimiters
                    for delimiter in [',', ';', '\t', '|']:
                        print(f"Trying delimiter: {delimiter}")
                        try:
                            self.data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                            if len(self.data.columns) > 1:  # If more than one column, we found the right delimiter
                                print(f"Successfully loaded with delimiter: {delimiter}")
                                return
                        except Exception as e:
                            print(f"Failed with delimiter {delimiter}: {str(e)}")
                            continue
                            
            raise Exception("Failed to load CSV with all encoding and delimiter combinations")
        except Exception as e:
            raise Exception(f"Error reading CSV: {str(e)}")
            
    def _load_sample(self, sample_name):
        """Load a sample dataset"""
        try:
            print(f"Attempting to load sample dataset: {sample_name}")
            
            if sample_name == "retail_inventory":
                print("Creating retail inventory sample data...")
                # Create a sample retail inventory dataset
                data = {
                    'item_id': [f'ITEM{i}' for i in range(1, 101)],
                    'product_name': [f'Product {i}' for i in range(1, 101)],
                    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Office'], 100),
                    'quantity': np.random.randint(0, 200, 100),
                    'unit_cost': np.round(np.random.uniform(5, 100, 100), 2),
                    'unit_price': np.round(np.random.uniform(10, 200, 100), 2),
                    'supplier': np.random.choice(['Supplier A', 'Supplier B', 'Supplier C'], 100),
                    'last_restock_date': pd.date_range(start='2023-01-01', periods=100),
                    'min_stock_level': np.random.randint(5, 50, 100),
                    'max_stock_level': np.random.randint(100, 300, 100)
                }
                print("Converting retail data to DataFrame...")
                self.data = pd.DataFrame(data)
                print("Retail data created successfully")
                
            elif sample_name == "manufacturing_inventory":
                print("Creating manufacturing inventory sample data...")
                # Create a sample manufacturing inventory dataset
                data = {
                    'part_id': [f'PART{i}' for i in range(1, 101)],
                    'part_name': [f'Component {i}' for i in range(1, 101)],
                    'category': np.random.choice(['Raw Material', 'Component', 'Finished Good', 'Packaging'], 100),
                    'quantity': np.random.randint(0, 500, 100),
                    'unit_cost': np.round(np.random.uniform(1, 50, 100), 2),
                    'supplier': np.random.choice(['Supplier X', 'Supplier Y', 'Supplier Z'], 100),
                    'lead_time_days': np.random.randint(1, 30, 100),
                    'last_order_date': pd.date_range(start='2023-01-01', periods=100),
                    'reorder_point': np.random.randint(50, 150, 100),
                    'economic_order_quantity': np.random.randint(100, 400, 100)
                }
                print("Converting manufacturing data to DataFrame...")
                self.data = pd.DataFrame(data)
                print("Manufacturing data created successfully")
                
            else:
                print(f"Unknown sample dataset: {sample_name}")
                return False
                
            # Update metadata
            print("Updating metadata...")
            self.metadata.update({
                'last_updated': datetime.now().isoformat(),
                'filename': f"sample_{sample_name}",
                'row_count': len(self.data),
                'column_count': len(self.data.columns),
                'is_sample': True
            })
            
            # Store in session if available
            if self._session is not None:
                print("Saving to session...")
                self._save_to_session()
                
            print(f"Successfully loaded {sample_name} sample data with {len(self.data)} rows")
            return True
            
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _convert_to_python_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_to_python_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(x) for x in obj]
        return obj

    def _generate_column_metadata(self):
        """Generate metadata about columns in the dataset"""
        if self.data is None or self.data.empty:
            return
            
        columns_meta = {}
        for col in self.data.columns:
            col_type = str(self.data[col].dtype)
            
            # Determine column type category
            if col_type.startswith('int') or col_type.startswith('float'):
                type_category = 'numeric'
            elif col_type.startswith('datetime'):
                type_category = 'datetime'
            elif col_type == 'bool':
                type_category = 'boolean'
            else:
                type_category = 'categorical' if self.data[col].nunique() < len(self.data) * 0.5 else 'text'
                
            # Get basic stats
            stats = {
                'dtype': col_type,
                'type_category': type_category,
                'count': int(self.data[col].count()),
                'null_count': int(self.data[col].isna().sum()),
                'unique_count': int(self.data[col].nunique())
            }
            
            # Add numeric stats if applicable
            if type_category == 'numeric':
                stats.update({
                    'min': float(self.data[col].min()) if not pd.isna(self.data[col].min()) else None,
                    'max': float(self.data[col].max()) if not pd.isna(self.data[col].max()) else None,
                    'mean': float(self.data[col].mean()) if not pd.isna(self.data[col].mean()) else None,
                    'median': float(self.data[col].median()) if not pd.isna(self.data[col].median()) else None
                })
                
            # Convert all values to Python native types
            columns_meta[col] = self._convert_to_python_types(stats)
            
        self.metadata['columns'] = columns_meta
        
    def clear_data(self):
        """Clear all data and session storage"""
        self.data = None
        self.metadata = {
            'last_updated': None,
            'filename': None,
            'row_count': 0,
            'column_count': 0
        }
        if self._session is not None and 'data_path' in self._session:
            try:
                os.remove(self._session['data_path'])
            except:
                pass
            self._session.pop('data_path', None)
            self._session.pop('data_metadata', None)
    
    def has_data(self):
        """Check if data is available"""
        return self.data is not None or self._load_from_session()
        
    def get_column_types(self):
        """Get the data types of each column"""
        if not self.has_data():
            return {}
            
        df = self.get_data()
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
        
    def get_column_stats(self):
        """Get basic statistics for each column"""
        if not self.has_data():
            return {}
            
        df = self.get_data()
        stats = {}
        
        for col in df.columns:
            col_stats = {
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
                
            # Convert all values to Python native types
            stats[col] = self._convert_to_python_types(col_stats)
                
        return stats
    
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            # Check for pyarrow (needed for parquet)
            import importlib.util
            pyarrow_spec = importlib.util.find_spec("pyarrow")
            self.has_pyarrow = pyarrow_spec is not None
            
            if not self.has_pyarrow:
                print("WARNING: pyarrow is not installed. Parquet support will be disabled.")
                print("To enable parquet support, install pyarrow: pip install pyarrow")
        except Exception as e:
            print(f"Error checking dependencies: {str(e)}")
            self.has_pyarrow = False