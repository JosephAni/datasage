import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path to import DataCleaner class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import DataCleaner, safe_display_dataframe
except ImportError:
    st.error("Unable to import DataCleaner class. Please ensure data_cleaner.py is in the project root.")
    
    # Define a basic version for demo purposes if import fails
    class DataCleaner:
        def __init__(self, df):
            self.df = df.copy()
            self.original_df = df.copy()
            
        def interpret_data(self):
            """
            Basic data interpretation
            """
            st.write("## 📊 Data Interpretation")
            
            # Basic Dataset Information
            st.write("### 📌 Dataset Overview")
            total_rows, total_cols = self.df.shape
            st.write(f"- Total Records: {total_rows:,}")
            st.write(f"- Total Features: {total_cols}")
            
            # Missing Values Analysis
            st.write("### ❓ Missing Values Analysis")
            missing_vals = self.df.isnull().sum()
            missing_percent = (missing_vals / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Missing Values': missing_vals,
                'Percent Missing': missing_percent
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percent Missing', ascending=False)
            
            if not missing_df.empty:
                st.write("Columns with missing values:")
                st.dataframe(missing_df)
            else:
                st.write("No missing values in the dataset.")
            
            # Numeric column statistics
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("### 📈 Numeric Column Statistics")
                st.dataframe(self.df[numeric_cols].describe())
            
            # Categorical column analysis
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.write("### 📊 Categorical Column Analysis")
                for col in cat_cols:
                    st.write(f"**{col}**: {self.df[col].nunique()} unique values")
        
        def perform_hypothesis_test(self, column1, column2=None, test_type='ttest'):
            result = {"test_name": test_type}
            
            # Check if columns exist
            if column1 not in self.df.columns:
                return {"error": f"Column '{column1}' not found"}
            if column2 and column2 not in self.df.columns:
                return {"error": f"Column '{column2}' not found"}
                
            # Basic t-test
            if test_type == 'ttest' and pd.api.types.is_numeric_dtype(self.df[column1]):
                stat, pvalue = stats.ttest_1samp(self.df[column1].dropna(), 0)
                result['statistic'] = stat
                result['p_value'] = pvalue
                result['interpretation'] = "Mean significantly different from 0" if pvalue < 0.05 else "Mean not significantly different from 0"
            
            return result
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Data Interpretation",
    page_icon="🔍",
    layout="wide"
)

st.markdown("# Data Interpretation")
st.sidebar.header("Data Interpretation")
st.write(
    """
    This page provides comprehensive interpretation of your dataset including:
    
    - Statistical summaries and insights
    - Distribution analysis
    - Correlation detection
    - Hypothesis testing
    - Anomaly identification
    
    Get deeper insights into your data to drive better decisions.
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

# Initialize session state for the cleaner
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None

# Get data from session state
df = get_data()

if df is None:
    st.info("No data loaded. Please upload data on the Home page.")
else:
    # Initialize DataCleaner if not already done
    if st.session_state.cleaner is None:
        st.session_state.cleaner = DataCleaner(df)
    
    cleaner = st.session_state.cleaner
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        st.write("### Data Sample")
        safe_display_dataframe(df.head(5), cleaner)
    
    # Basic dataset information
    with st.expander("Dataset Overview", expanded=True):
        total_rows, total_cols = df.shape
        st.write(f"- Total Records: {total_rows:,}")
        st.write(f"- Total Features: {total_cols}")
        
        # Data types analysis
        st.write("### Data Type Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dtype_counts.index.astype(str), dtype_counts.values)
        ax.set_title('Distribution of Data Types')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Missing Values Analysis
        st.write("### Missing Values Analysis")
        missing_vals = df.isnull().sum()
        missing_percent = (missing_vals / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_vals,
            'Percent Missing': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percent Missing', ascending=False)
        
        if not missing_df.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_df)
            
            # Plot missing values
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(missing_df.index, missing_df['Percent Missing'])
            ax.set_ylabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Column')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No missing values in the dataset.")
    
    # Statistical Analysis
    with st.expander("Statistical Analysis", expanded=True):
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.write("### Numeric Column Statistics")
            st.dataframe(df[numeric_cols].describe())
            
            # Skewness and Kurtosis
            skew_data = df[numeric_cols].skew()
            kurt_data = df[numeric_cols].kurtosis()
            
            stat_df = pd.DataFrame({
                'Mean': df[numeric_cols].mean(),
                'Median': df[numeric_cols].median(),
                'Std Dev': df[numeric_cols].std(),
                'Skewness': skew_data,
                'Kurtosis': kurt_data
            })
            
            st.write("### Additional Statistics")
            st.dataframe(stat_df)
            
            # Highlight highly skewed columns
            high_skew_cols = skew_data[abs(skew_data) > 1].index.tolist()
            if high_skew_cols:
                st.write("### Highly Skewed Columns")
                st.write("These columns have skewness > 1 or < -1 and might benefit from transformation:")
                for col in high_skew_cols:
                    st.write(f"- **{col}**: Skewness = {skew_data[col]:.2f}")
        
        # Categorical column analysis
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            st.write("### Categorical Column Analysis")
            
            cat_summary = {}
            for col in cat_cols:
                value_counts = df[col].value_counts()
                cat_summary[col] = {
                    'unique_values': df[col].nunique(),
                    'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
                    'top_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'top_percent': (value_counts.iloc[0] / len(df) * 100) if not value_counts.empty else 0
                }
            
            cat_df = pd.DataFrame.from_dict(cat_summary, orient='index')
            st.dataframe(cat_df)
            
            # High cardinality warning
            high_card_cols = [col for col in cat_cols if df[col].nunique() > 10]
            if high_card_cols:
                st.write("### High Cardinality Columns")
                st.write("These categorical columns have many unique values and might need encoding or grouping:")
                for col in high_card_cols:
                    st.write(f"- **{col}**: {df[col].nunique()} unique values")
    
    # Distribution Analysis
    with st.expander("Distribution Analysis", expanded=False):
        st.write("### Distribution of Numeric Columns")
        
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {selected_col}')
                st.pyplot(fig)
                
                # Distribution statistics
                st.write("#### Distribution Statistics")
                st.write(f"Mean: {df[selected_col].mean():.4f}")
                st.write(f"Median: {df[selected_col].median():.4f}")
                st.write(f"Standard Deviation: {df[selected_col].std():.4f}")
                st.write(f"Skewness: {df[selected_col].skew():.4f}")
                st.write(f"Kurtosis: {df[selected_col].kurtosis():.4f}")
            
            with col2:
                # Box plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(y=df[selected_col].dropna(), ax=ax)
                ax.set_title(f'Box Plot of {selected_col}')
                st.pyplot(fig)
                
                # Q-Q plot to check normality
                from scipy import stats
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(df[selected_col].dropna(), plot=ax)
                ax.set_title(f'Q-Q Plot of {selected_col}')
                st.pyplot(fig)
        else:
            st.info("No numeric columns available for distribution analysis.")
    
    # Correlation Analysis
    with st.expander("Correlation Analysis", expanded=False):
        if len(numeric_cols) > 1:
            st.write("### Correlation Matrix")
            
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap of correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            
            # Highlight strong correlations
            strong_corrs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append((col1, col2, corr_val))
            
            if strong_corrs:
                st.write("### Strong Correlations (|r| > 0.7)")
                corr_df = pd.DataFrame(strong_corrs, columns=['Variable 1', 'Variable 2', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(corr_df)
            else:
                st.write("No strong correlations (|r| > 0.7) found between variables.")
            
            # Select two variables for detailed correlation analysis
            st.write("### Detailed Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("Select X variable", numeric_cols, key="x_var")
            
            with col2:
                y_var = st.selectbox("Select Y variable", numeric_cols, key="y_var")
            
            if x_var != y_var:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
                ax.set_title(f'Scatter Plot: {x_var} vs {y_var}')
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                
                # Add regression line
                sns.regplot(x=df[x_var], y=df[y_var], scatter=False, ax=ax, line_kws={"color":"red"})
                
                st.pyplot(fig)
                
                # Calculate correlation statistics
                corr, p_value = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
                
                st.write("#### Correlation Statistics")
                st.write(f"Pearson correlation coefficient: {corr:.4f}")
                st.write(f"P-value: {p_value:.4f}")
                
                # Interpret correlation
                if p_value < 0.05:
                    significance = "statistically significant"
                else:
                    significance = "not statistically significant"
                    
                if abs(corr) < 0.3:
                    strength = "weak"
                elif abs(corr) < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                direction = "positive" if corr > 0 else "negative"
                
                st.write(f"Interpretation: {strength} {direction} correlation that is {significance}.")
        else:
            st.info("Need at least two numeric columns for correlation analysis.")
    
    # Hypothesis Testing
    with st.expander("Hypothesis Testing", expanded=False):
        st.write("### Hypothesis Testing")
        st.write("""
        Test statistical hypotheses about your data to validate assumptions and compare groups.
        """)
        
        test_type = st.selectbox(
            "Select test type",
            ["t-test", "correlation", "chi-square", "ANOVA"]
        )
        
        if test_type == "t-test":
            # One sample or two sample t-test
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns available for t-test.")
            else:
                st.write("#### T-Test")
                
                test_mode = st.radio("Test type", ["One sample", "Two sample"])
                
                col1 = st.selectbox("Select variable", numeric_cols)
                
                if test_mode == "One sample":
                    test_value = st.number_input("Test against value", value=0.0)
                    
                    if st.button("Run t-test"):
                        with st.spinner("Running t-test..."):
                            # Perform one-sample t-test
                            t_stat, p_val = stats.ttest_1samp(df[col1].dropna(), test_value)
                            
                            st.write("#### T-Test Results")
                            st.write(f"Testing if mean of '{col1}' differs from {test_value}")
                            st.write(f"t-statistic: {t_stat:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success(f"Result: The mean of '{col1}' is significantly different from {test_value} (p < 0.05)")
                            else:
                                st.info(f"Result: No significant evidence that the mean of '{col1}' differs from {test_value} (p ≥ 0.05)")
                else:
                    col2 = st.selectbox("Select second variable", [c for c in numeric_cols if c != col1])
                    
                    if st.button("Run t-test"):
                        with st.spinner("Running t-test..."):
                            # Perform two-sample t-test
                            t_stat, p_val = stats.ttest_ind(
                                df[col1].dropna(),
                                df[col2].dropna(),
                                equal_var=False  # Welch's t-test
                            )
                            
                            st.write("#### T-Test Results")
                            st.write(f"Testing if means of '{col1}' and '{col2}' are equal")
                            st.write(f"t-statistic: {t_stat:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success(f"Result: The means of '{col1}' and '{col2}' are significantly different (p < 0.05)")
                            else:
                                st.info(f"Result: No significant evidence that the means of '{col1}' and '{col2}' differ (p ≥ 0.05)")
        
        elif test_type == "correlation":
            # Correlation test
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for correlation test.")
            else:
                st.write("#### Correlation Test")
                
                col1 = st.selectbox("Select first variable", numeric_cols, key="corr_var1")
                col2 = st.selectbox("Select second variable", [c for c in numeric_cols if c != col1], key="corr_var2")
                
                if st.button("Run correlation test"):
                    with st.spinner("Running correlation test..."):
                        # Perform correlation test
                        corr, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                        
                        st.write("#### Correlation Test Results")
                        st.write(f"Testing correlation between '{col1}' and '{col2}'")
                        st.write(f"Correlation coefficient: {corr:.4f}")
                        st.write(f"p-value: {p_val:.4f}")
                        
                        # Interpret correlation strength
                        if abs(corr) < 0.3:
                            strength = "weak"
                        elif abs(corr) < 0.7:
                            strength = "moderate"
                        else:
                            strength = "strong"
                        
                        direction = "positive" if corr > 0 else "negative"
                        
                        if p_val < 0.05:
                            st.success(f"Result: {strength.capitalize()} {direction} correlation detected (p < 0.05)")
                        else:
                            st.info(f"Result: No significant correlation detected (p ≥ 0.05)")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
                        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        
                        # Add regression line
                        sns.regplot(x=df[col1], y=df[col2], scatter=False, ax=ax, line_kws={"color":"red"})
                        
                        st.pyplot(fig)
        
        elif test_type == "chi-square":
            # Chi-square test of independence
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(cat_cols) < 2:
                st.warning("Need at least two categorical columns for chi-square test.")
            else:
                st.write("#### Chi-Square Test of Independence")
                
                col1 = st.selectbox("Select first categorical variable", cat_cols, key="chi_var1")
                col2 = st.selectbox("Select second categorical variable", [c for c in cat_cols if c != col1], key="chi_var2")
                
                if st.button("Run chi-square test"):
                    with st.spinner("Running chi-square test..."):
                        # Create contingency table
                        contingency = pd.crosstab(df[col1], df[col2])
                        
                        # Perform chi-square test
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        
                        st.write("#### Chi-Square Test Results")
                        st.write(f"Testing independence between '{col1}' and '{col2}'")
                        st.write(f"Chi-square statistic: {chi2:.4f}")
                        st.write(f"Degrees of freedom: {dof}")
                        st.write(f"p-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success(f"Result: '{col1}' and '{col2}' are dependent (p < 0.05)")
                        else:
                            st.info(f"Result: No significant evidence that '{col1}' and '{col2}' are dependent (p ≥ 0.05)")
                        
                        # Show contingency table
                        st.write("#### Contingency Table")
                        st.dataframe(contingency)
                        
                        # Visualize
                        fig, ax = plt.subplots(figsize=(10, 6))
                        contingency.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_title(f'Stacked Bar Chart: {col1} vs {col2}')
                        ax.set_xlabel(col1)
                        ax.set_ylabel('Count')
                        plt.legend(title=col2)
                        st.pyplot(fig)
        
        elif test_type == "ANOVA":
            # One-way ANOVA
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not numeric_cols or not cat_cols:
                st.warning("Need at least one numeric column and one categorical column for ANOVA.")
            else:
                st.write("#### One-way ANOVA")
                
                numeric_var = st.selectbox("Select numeric variable", numeric_cols)
                cat_var = st.selectbox("Select categorical variable (groups)", cat_cols)
                
                if st.button("Run ANOVA"):
                    with st.spinner("Running ANOVA..."):
                        # Group data
                        groups = []
                        group_names = []
                        
                        for name, group in df.groupby(cat_var):
                            if len(group[numeric_var].dropna()) > 0:
                                groups.append(group[numeric_var].dropna())
                                group_names.append(name)
                        
                        if len(groups) < 2:
                            st.error("Need at least two non-empty groups for ANOVA.")
                        else:
                            # Perform ANOVA
                            f_stat, p_val = stats.f_oneway(*groups)
                            
                            st.write("#### ANOVA Results")
                            st.write(f"Testing if means of '{numeric_var}' are equal across groups in '{cat_var}'")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success(f"Result: At least one group mean is significantly different (p < 0.05)")
                            else:
                                st.info(f"Result: No significant difference between group means (p ≥ 0.05)")
                            
                            # Visualize
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x=cat_var, y=numeric_var, data=df, ax=ax)
                            ax.set_title(f'Box Plot: {numeric_var} by {cat_var}')
                            ax.set_xlabel(cat_var)
                            ax.set_ylabel(numeric_var)
                            plt.xticks(rotation=45)
                            st.pyplot(fig) 