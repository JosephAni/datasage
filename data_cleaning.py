import pandas as pd

class DataCleaner:
    def __init__(self, df):
        """Initialize the DataCleaner with a pandas DataFrame."""
        self.original_df = df.copy()
        self.df = df.copy()

    def convert_datatypes(self, column_types):
        """Convert column datatypes according to the specified types."""
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
                print(f"Error converting {column} to {dtype}: {str(e)}")

    def get_cleaned_data(self):
        """Return the cleaned DataFrame."""
        return self.df.copy()

    def reset(self):
        """Reset the DataFrame to its original state."""
        self.df = self.original_df.copy() 