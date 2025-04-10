# AI-Powered Data Cleaning Platform 🖨️💎🛁📊🤖

A powerful, user-friendly data cleaning and analysis platform built with Python, Flask, and various machine learning libraries. This tool provides an interactive web interface for data preprocessing, analysis, visualization, and advanced machine learning tasks.

## Features

### 1. Advanced Data Cleaning 🧹

- **Column Name Standardization**: Automatically convert column names to lowercase and replace spaces with underscores
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, KNN imputation, or drop)
- **Duplicate Row Removal**: Identify and remove duplicate entries
- **Data Type Conversion**: Convert columns to appropriate data types (int, float, category, string, datetime)
- **High Cardinality Handling**: Target encoding and frequency encoding for categorical variables
- **Skewed Feature Transformation**: Yeo-Johnson and Box-Cox transformations
- **Date Format Detection and Conversion**: Automatic detection and standardization of date formats

### 2. AI/ML Features 🤖

- **Outlier Detection**: Using Isolation Forest algorithm with adjustable contamination factor
- **Feature Engineering**:
  - Automatic encoding of categorical variables (Label, Target, Frequency encoding)
  - Feature scaling using StandardScaler
  - Power transformations for skewed features
- **Advanced Machine Learning**:
  - Support for both classification and regression problems
  - Multiple model types (Random Forest, Decision Tree, KNN, Logistic Regression, SVM, LDA, Naive Bayes)
  - Model performance metrics and feature importance visualization
  - Automated feature selection
  - Clustering analysis (K-means)

### 3. Time Series Analysis 📈

- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Stationarity Testing**: Augmented Dickey-Fuller test
- **Demand Forecasting**: Using Prophet and ARIMA models
- **Demand Simulation**: Generate synthetic demand data with seasonality and promotions
- **Forecast Accuracy Analysis**: Multiple accuracy metrics and visualizations

### 4. Business Analytics 💼

- **Customer Lifetime Value (CLV)**: Calculate and predict customer value
- **RFM Analysis**: Customer segmentation based on Recency, Frequency, Monetary value
- **Inventory Turnover Simulation**: Visual inventory management
- **Forecast Aggregation**: Multi-store demand forecasting

### 5. Data Visualization 📊

- **Interactive Plots**: Using Plotly and Chart.js for dynamic visualizations
- **Distribution Analysis**:
  - Histograms for numeric columns
  - Bar charts, treemaps, and lollipop charts for categorical data
  - Box plots for outlier detection
- **Time Series Visualization**: Seasonal patterns and trends
- **Correlation Analysis**: Interactive heatmaps
- **Business Metrics Visualization**: CLV, RFM segments, inventory levels

## Backend Frameworks

This project has two backend options:

1. **Original Streamlit Backend**: The `main` branch uses Streamlit for a simple, quick deployment
2. **Flask Backend**: The `new-frontend` branch provides a more customizable Flask-based interface

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd data-cleaner
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

For the Flask backend (`new-frontend` branch):
- Python 3.8+
- flask==2.3.3
- pandas==2.2.0
- numpy==1.26.3
- scikit-learn==1.4.0
- matplotlib==3.8.2
- seaborn==0.13.2
- scipy==1.12.0
- category_encoders==2.6.3
- statsmodels==0.14.1
- prophet==1.1.5
- plotly==5.18.0
- pmdarima==2.0.4
- flask-wtf==1.2.1
- jinja2==3.1.2
- werkzeug==2.3.7
- gunicorn==21.2.0

For the Streamlit backend (`main` branch):
- Python 3.8+
- streamlit==1.31.1
- pandas==2.2.0
- numpy==1.26.3
- scikit-learn==1.4.0
- matplotlib==3.8.2
- seaborn==0.13.2
- scipy==1.12.0
- category_encoders==2.6.3
- statsmodels==0.14.1
- prophet==1.1.5
- plotly==5.18.0
- pmdarima==2.0.4

## Usage

### Flask Backend (new-frontend branch)

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

### Streamlit Backend (main branch)

1. Start the application:

```bash
streamlit run Home.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. For both backends:
   - Upload your dataset (CSV, Excel, or JSON file)
   - Use the navigation to access different features
   - Download the cleaned dataset when finished

## Data Cleaning Workflow

1. **Upload Data**: Support for CSV, Excel, and JSON files
2. **Explore**: Use automated EDA to understand your dataset
3. **Clean**: Apply advanced cleaning operations as needed
4. **Analyze**: Utilize visualization tools to explore patterns
5. **Process**: Apply AI/ML features for advanced preprocessing
6. **Export**: Download the cleaned dataset

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

- Flask backend with Bootstrap UI
- Original version built with Streamlit
- Powered by scikit-learn and Prophet
- Visualization using Plotly, Chart.js, matplotlib, and seaborn

## Support

For support, please open an issue in the repository or contact the maintainers.
