import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import DataCleaner class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import DataCleaner, safe_display_dataframe
except ImportError:
    st.error("Unable to import DataCleaner class. Please ensure data_cleaner.py is in the project root.")
    
    class DataCleaner:
        def __init__(self, df):
            self.df = df.copy()
            self.original_df = df.copy()
    
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Advanced Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.markdown("# Advanced Data Analysis")
st.sidebar.header("Advanced Data Analysis")
st.write(
    """
    This page provides advanced data analysis features including:
    
    - Automated exploratory data analysis (EDA)
    - Statistical hypothesis testing
    - Feature selection
    - Clustering analysis
    - Correlation analysis
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
    
    # Automated EDA
    with st.expander("Automated EDA", expanded=True):
        st.write("### Automated Exploratory Data Analysis")
        st.write("""
        Get a quick overview of your dataset with automatic visualizations and insights.
        """)
        
        if st.button("Run Automated EDA"):
            with st.spinner("Running automated EDA..."):
                # Basic dataset info
                st.write("#### Dataset Information")
                st.write(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
                
                # Data types
                st.write("#### Data Types")
                dtype_counts = df.dtypes.value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(dtype_counts.index.astype(str), dtype_counts.values)
                ax.set_title('Distribution of Data Types')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Numeric summary
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write("#### Numeric Columns Summary")
                    st.dataframe(df[numeric_cols].describe())
                    
                    # Correlation heatmap
                    if len(numeric_cols) > 1:
                        st.write("#### Correlation Heatmap")
                        corr = df[numeric_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
                        ax.set_title('Correlation Matrix')
                        st.pyplot(fig)
                
                # Categorical summary
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    st.write("#### Categorical Columns Summary")
                    
                    for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                        st.write(f"**{col}**")
                        value_counts = df[col].value_counts().head(10)
                        
                        # Create horizontal bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        value_counts.plot.barh(ax=ax)
                        ax.set_title(f'Top 10 values in {col}')
                        ax.set_xlabel('Count')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Feature Selection
    with st.expander("Feature Selection", expanded=False):
        st.write("### Feature Selection")
        st.write("""
        Identify the most important features for your target variable using various methods.
        """)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns available for feature selection.")
        else:
            target_column = st.selectbox(
                "Select target variable",
                numeric_cols
            )
            
            feature_selection_method = st.selectbox(
                "Feature selection method",
                ["correlation", "random_forest", "mutual_info"]
            )
            
            n_features = st.slider(
                "Number of top features to select",
                min_value=1,
                max_value=min(10, len(numeric_cols)-1),
                value=min(5, len(numeric_cols)-1)
            )
            
            if st.button("Run Feature Selection"):
                with st.spinner("Running feature selection..."):
                    features = [col for col in numeric_cols if col != target_column]
                    
                    if feature_selection_method == "correlation":
                        # Use correlation for feature selection
                        corr_scores = []
                        for col in features:
                            corr = df[col].corr(df[target_column])
                            corr_scores.append((col, abs(corr)))
                        
                        # Sort by absolute correlation
                        corr_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # Display results
                        st.write("#### Top Features by Correlation")
                        
                        results_df = pd.DataFrame({
                            'Feature': [feat for feat, score in corr_scores[:n_features]],
                            'Absolute Correlation': [score for feat, score in corr_scores[:n_features]]
                        })
                        
                        st.dataframe(results_df)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(
                            [feat for feat, score in corr_scores[:n_features]],
                            [score for feat, score in corr_scores[:n_features]]
                        )
                        ax.set_title(f'Top {n_features} Features by Correlation with {target_column}')
                        ax.set_ylabel('Absolute Correlation')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    elif feature_selection_method == "random_forest":
                        try:
                            from sklearn.ensemble import RandomForestRegressor
                            
                            # Prepare the data
                            X = df[features]
                            y = df[target_column]
                            
                            # Train a random forest
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X, y)
                            
                            # Get feature importances
                            importances = rf.feature_importances_
                            
                            # Sort features by importance
                            feature_importance = list(zip(features, importances))
                            feature_importance.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display results
                            st.write("#### Top Features by Random Forest Importance")
                            
                            results_df = pd.DataFrame({
                                'Feature': [feat for feat, score in feature_importance[:n_features]],
                                'Importance': [score for feat, score in feature_importance[:n_features]]
                            })
                            
                            st.dataframe(results_df)
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.bar(
                                [feat for feat, score in feature_importance[:n_features]],
                                [score for feat, score in feature_importance[:n_features]]
                            )
                            ax.set_title(f'Top {n_features} Features by Random Forest Importance')
                            ax.set_ylabel('Importance')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error in Random Forest feature selection: {str(e)}")
                    
                    elif feature_selection_method == "mutual_info":
                        try:
                            from sklearn.feature_selection import mutual_info_regression
                            
                            # Prepare the data
                            X = df[features]
                            y = df[target_column]
                            
                            # Calculate mutual information
                            mi_scores = mutual_info_regression(X, y)
                            
                            # Sort features by mutual information
                            mi_feature_scores = list(zip(features, mi_scores))
                            mi_feature_scores.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display results
                            st.write("#### Top Features by Mutual Information")
                            
                            results_df = pd.DataFrame({
                                'Feature': [feat for feat, score in mi_feature_scores[:n_features]],
                                'Mutual Information': [score for feat, score in mi_feature_scores[:n_features]]
                            })
                            
                            st.dataframe(results_df)
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.bar(
                                [feat for feat, score in mi_feature_scores[:n_features]],
                                [score for feat, score in mi_feature_scores[:n_features]]
                            )
                            ax.set_title(f'Top {n_features} Features by Mutual Information')
                            ax.set_ylabel('Mutual Information')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error in Mutual Information feature selection: {str(e)}")
    
    # Clustering Analysis
    with st.expander("Clustering Analysis", expanded=False):
        st.write("### Clustering Analysis")
        st.write("""
        Discover natural groupings in your data using clustering algorithms.
        """)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are needed for clustering.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                feature1 = st.selectbox("Select first feature", numeric_cols, index=0)
            
            with col2:
                remaining_cols = [col for col in numeric_cols if col != feature1]
                feature2 = st.selectbox("Select second feature", remaining_cols, index=0)
                
            clustering_method = st.selectbox(
                "Clustering algorithm",
                ["kmeans", "dbscan", "hierarchical"]
            )
            
            if clustering_method == "kmeans":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
            elif clustering_method == "dbscan":
                eps = st.slider("DBSCAN epsilon (neighborhood size)", 0.1, 2.0, 0.5)
                min_samples = st.slider("Minimum samples in core neighborhood", 2, 10, 5)
            elif clustering_method == "hierarchical":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                
            if st.button("Run Clustering"):
                with st.spinner("Running clustering analysis..."):
                    # Prepare the data
                    X = df[[feature1, feature2]].dropna()
                    
                    try:
                        if clustering_method == "kmeans":
                            from sklearn.cluster import KMeans
                            
                            # Run KMeans
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = kmeans.fit_predict(X)
                            
                            # Add cluster labels to the data
                            X_with_clusters = X.copy()
                            X_with_clusters['Cluster'] = cluster_labels
                            
                            # Plot the clusters
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(
                                X_with_clusters[feature1],
                                X_with_clusters[feature2],
                                c=X_with_clusters['Cluster'],
                                cmap='viridis',
                                alpha=0.7
                            )
                            
                            # Plot cluster centers
                            centers = kmeans.cluster_centers_
                            ax.scatter(
                                centers[:, 0],
                                centers[:, 1],
                                c='red',
                                marker='X',
                                s=200,
                                label='Cluster Centers'
                            )
                            
                            ax.set_title(f'KMeans Clustering ({n_clusters} clusters)')
                            ax.set_xlabel(feature1)
                            ax.set_ylabel(feature2)
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show cluster statistics
                            st.write("#### Cluster Statistics")
                            cluster_stats = X_with_clusters.groupby('Cluster').agg({
                                feature1: ['mean', 'std', 'min', 'max'],
                                feature2: ['mean', 'std', 'min', 'max'],
                                'Cluster': 'size'
                            })
                            
                            cluster_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in cluster_stats.columns]
                            cluster_stats.rename(columns={'Cluster_size': 'Size'}, inplace=True)
                            
                            st.dataframe(cluster_stats)
                            
                        elif clustering_method == "dbscan":
                            from sklearn.cluster import DBSCAN
                            
                            # Run DBSCAN
                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            cluster_labels = dbscan.fit_predict(X)
                            
                            # Add cluster labels to the data
                            X_with_clusters = X.copy()
                            X_with_clusters['Cluster'] = cluster_labels
                            
                            # Plot the clusters
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(
                                X_with_clusters[feature1],
                                X_with_clusters[feature2],
                                c=X_with_clusters['Cluster'],
                                cmap='viridis',
                                alpha=0.7
                            )
                            
                            ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
                            ax.set_xlabel(feature1)
                            ax.set_ylabel(feature2)
                            
                            # Add colorbar to show cluster labels
                            cbar = plt.colorbar(scatter, ax=ax)
                            cbar.set_label('Cluster')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show number of clusters and noise points
                            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                            n_noise = list(cluster_labels).count(-1)
                            
                            st.write(f"Number of clusters: {n_clusters}")
                            st.write(f"Number of noise points: {n_noise}")
                            
                            # Show cluster statistics (excluding noise points)
                            if n_clusters > 0:
                                st.write("#### Cluster Statistics")
                                non_noise = X_with_clusters[X_with_clusters['Cluster'] != -1]
                                
                                if not non_noise.empty:
                                    cluster_stats = non_noise.groupby('Cluster').agg({
                                        feature1: ['mean', 'std', 'min', 'max'],
                                        feature2: ['mean', 'std', 'min', 'max'],
                                        'Cluster': 'size'
                                    })
                                    
                                    cluster_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in cluster_stats.columns]
                                    cluster_stats.rename(columns={'Cluster_size': 'Size'}, inplace=True)
                                    
                                    st.dataframe(cluster_stats)
                                
                        elif clustering_method == "hierarchical":
                            from sklearn.cluster import AgglomerativeClustering
                            
                            # Run hierarchical clustering
                            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                            cluster_labels = hierarchical.fit_predict(X)
                            
                            # Add cluster labels to the data
                            X_with_clusters = X.copy()
                            X_with_clusters['Cluster'] = cluster_labels
                            
                            # Plot the clusters
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(
                                X_with_clusters[feature1],
                                X_with_clusters[feature2],
                                c=X_with_clusters['Cluster'],
                                cmap='viridis',
                                alpha=0.7
                            )
                            
                            ax.set_title(f'Hierarchical Clustering ({n_clusters} clusters)')
                            ax.set_xlabel(feature1)
                            ax.set_ylabel(feature2)
                            
                            # Add colorbar to show cluster labels
                            cbar = plt.colorbar(scatter, ax=ax)
                            cbar.set_label('Cluster')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show cluster statistics
                            st.write("#### Cluster Statistics")
                            cluster_stats = X_with_clusters.groupby('Cluster').agg({
                                feature1: ['mean', 'std', 'min', 'max'],
                                feature2: ['mean', 'std', 'min', 'max'],
                                'Cluster': 'size'
                            })
                            
                            cluster_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in cluster_stats.columns]
                            cluster_stats.rename(columns={'Cluster_size': 'Size'}, inplace=True)
                            
                            st.dataframe(cluster_stats)
                        
                    except Exception as e:
                        st.error(f"Error in clustering analysis: {str(e)}")
    
    # Statistical Testing
    with st.expander("Statistical Testing", expanded=False):
        st.write("### Statistical Hypothesis Testing")
        st.write("""
        Perform statistical tests to validate hypotheses about your data.
        """)
        
        test_type = st.selectbox(
            "Test type",
            ["t-test", "correlation", "chi-square"]
        )
        
        if test_type == "t-test":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns available for t-test.")
            else:
                col1 = st.selectbox("Select variable", numeric_cols)
                
                test_mode = st.radio("Test type", ["One sample", "Two sample"])
                
                if test_mode == "One sample":
                    test_value = st.number_input("Test against value", value=0.0)
                    
                    if st.button("Run t-test"):
                        with st.spinner("Running t-test..."):
                            from scipy import stats
                            
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
                            from scipy import stats
                            
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for correlation test.")
            else:
                col1 = st.selectbox("Select first variable", numeric_cols, key="corr1")
                col2 = st.selectbox("Select second variable", [c for c in numeric_cols if c != col1], key="corr2")
                
                if st.button("Run correlation test"):
                    with st.spinner("Running correlation test..."):
                        from scipy import stats
                        
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
                        ax.scatter(df[col1], df[col2], alpha=0.5)
                        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        
                        # Add best fit line
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(df[col1].dropna(), df[col2].dropna())
                        x = np.array([df[col1].min(), df[col1].max()])
                        y = intercept + slope * x
                        ax.plot(x, y, color='red', label=f'y = {slope:.4f}x + {intercept:.4f}')
                        ax.legend()
                        
                        st.pyplot(fig)
                            
        elif test_type == "chi-square":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(cat_cols) < 2:
                st.warning("Need at least two categorical columns for chi-square test.")
            else:
                col1 = st.selectbox("Select first categorical variable", cat_cols, key="chi1")
                col2 = st.selectbox("Select second categorical variable", [c for c in cat_cols if c != col1], key="chi2")
                
                if st.button("Run chi-square test"):
                    with st.spinner("Running chi-square test..."):
                        from scipy import stats
                        
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