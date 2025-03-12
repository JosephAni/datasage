# AI-Powered Data Cleaning Platform 🖨️💎🛁📊🤖

A powerful, user-friendly data cleaning and analysis platform built with Python, Streamlit, and various machine learning libraries. This tool provides an interactive web interface for data preprocessing, analysis, visualization, and basic machine learning tasks.

## Features

### 1. Basic Data Cleaning 🧹

- **Column Name Standardization**: Automatically convert column names to lowercase and replace spaces with underscores
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, or drop)
- **Duplicate Row Removal**: Identify and remove duplicate entries
- **Data Type Conversion**: Convert columns to appropriate data types (int, float, category, string)

### 2. AI/ML Features 🤖

- **Outlier Detection**: Using Isolation Forest algorithm with adjustable contamination factor
- **Feature Engineering**:
  - Automatic encoding of categorical variables
  - Feature scaling using StandardScaler
- **Basic Machine Learning**:
  - Support for both classification and regression problems
  - Random Forest models with performance metrics
  - Feature importance visualization

### 3. Data Interpretation 📊

- **Dataset Overview**: Total records, features, and data type distribution
- **Missing Values Analysis**: Detailed breakdown of missing values by column
- **Numeric Features Analysis**:
  - Statistical summaries (mean, std, min, max, quartiles)
  - Skewness analysis
- **Categorical Features Analysis**:
  - Unique value counts
  - Frequency distributions
- **Correlation Analysis**: Identification of strongly correlated features
- **Automated Insights**: Key patterns and potential data quality issues
- **Smart Recommendations**: Suggested actions for data preprocessing

### 4. Data Visualization 📈

- **Distribution Plots**: Histograms for numeric columns, bar charts for categorical
- **Pie Charts**: For categorical columns with limited unique values
- **Box Plots**: For identifying outliers in numeric columns
- **Correlation Heatmaps**: Visual representation of feature correlations

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd data-cleaner
```

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.24.3
- streamlit >= 1.22.0
- scikit-learn >= 1.0.2
- matplotlib >= 3.10.1
- seaborn >= 0.13.2
- openpyxl >= 3.0.9 (for Excel file support)

## Usage

1. Start the application:

```bash
streamlit run data_cleaner.py
```

1. Open your web browser and navigate to the provided URL (typically `http://localhost:8502`)

1. Upload your dataset (CSV or Excel file)

1. Use the sidebar to access different features:
   - Basic Cleaning
   - AI/ML Features
   - Data Interpretation
   - Data Visualization

1. Download the cleaned dataset when finished

## Data Cleaning Workflow

1. **Upload Data**: Support for CSV and Excel files
1. **Explore**: Use Data Interpretation to understand your dataset
1. **Clean**: Apply basic cleaning operations as needed
1. **Analyze**: Utilize visualization tools to explore patterns
1. **Process**: Apply AI/ML features for advanced preprocessing
1. **Export**: Download the cleaned dataset

## Features in Detail

### Basic Cleaning

- **Standardize Column Names**: Converts all column names to a consistent format
- **Handle Missing Values**:  
  - Mean imputation for numeric columns
  - Median imputation for skewed distributions
  - Mode imputation for categorical data
  - Option to drop rows with missing values
- **Remove Duplicates**: Identify and remove duplicate records
- **Convert Data Types**: Ensure columns have appropriate data types

### AI/ML Features

- **Outlier Detection**:
  - Uses Isolation Forest algorithm
  - Adjustable contamination factor
  - Visual representation of outliers
- **Feature Engineering**:
  - Automatic label encoding for categorical variables
  - Standard scaling for numeric features
- **Machine Learning**:
  - Supports classification and regression
  - Provides model performance metrics
  - Shows feature importance rankings

### Data Interpretation

- **Comprehensive Analysis**:
  - Dataset overview and statistics
  - Missing value analysis
  - Distribution analysis
  - Correlation detection
- **Automated Insights**:
  - Identifies potential data quality issues
  - Highlights important patterns
  - Provides actionable recommendations

### Visualization Tools

- **Distribution Analysis**:
  - Histograms for numeric data
  - Bar charts for categorical data
- **Relationship Analysis**:
  - Correlation heatmaps
  - Box plots for outlier detection
- **Interactive Visualizations**:
  - Zoomable plots
  - Hover information
  - Downloadable figures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by scikit-learn
- Visualization using matplotlib and seaborn

## Support

For support, please open an issue in the repository or contact the maintainers.
