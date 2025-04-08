import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import from data_cleaner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_cleaner import safe_display_dataframe
except ImportError:
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Data Visualization",
    page_icon="📊",
    layout="wide"
)

st.markdown("# Data Visualization")
st.sidebar.header("Data Visualization")
st.write(
    """
    This page provides tools for visualizing your data using various charts and plots.
    Select your visualization type and configure the options to gain insights from your data.
    """
)

# Function to get session data
def get_data():
    if 'data' in st.session_state:
        return st.session_state.data
    return None

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
    
    # Visualization options
    st.write("### Visualization Options")
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Histogram", "Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Heatmap", "Pair Plot"]
    )
    
    # Column selectors based on viz type
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
    
    if viz_type == "Histogram":
        st.write("#### Histogram Configuration")
        hist_col = st.selectbox("Select Column for Histogram", numeric_cols) if numeric_cols else st.error("No numeric columns available for histogram")
        
        if numeric_cols:
            bins = st.slider("Number of Bins", min_value=5, max_value=100, value=20)
            color_by = st.selectbox("Color by Category (optional)", ["None"] + categorical_cols)
            
            st.write("#### Histogram Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if color_by != "None" and categorical_cols:
                # Limit to top 10 categories to avoid too many colors
                top_cats = df[color_by].value_counts().nlargest(10).index
                for cat in top_cats:
                    subset = df[df[color_by] == cat]
                    sns.histplot(subset[hist_col], bins=bins, label=cat, alpha=0.5, ax=ax)
                ax.legend()
            else:
                sns.histplot(df[hist_col], bins=bins, ax=ax)
            
            ax.set_title(f"Histogram of {hist_col}")
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show descriptive statistics
            st.write("#### Descriptive Statistics")
            desc_stats = df[hist_col].describe().to_frame().T
            safe_display_dataframe(desc_stats)
    
    elif viz_type == "Scatter Plot":
        st.write("#### Scatter Plot Configuration")
        
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis Column", numeric_cols)
            y_col = st.selectbox("Select Y-axis Column", numeric_cols, index=min(1, len(numeric_cols)-1))
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            size_by = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
            
            st.write("#### Scatter Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with or without color and size variations
            if color_by != "None" and size_by != "None":
                scatter = ax.scatter(
                    df[x_col], df[y_col],
                    c=pd.factorize(df[color_by])[0] if color_by in categorical_cols else df[color_by],
                    s=df[size_by] * 20 / df[size_by].max(),
                    alpha=0.6
                )
                if color_by in categorical_cols:
                    categories = df[color_by].unique()
                    legend1 = ax.legend(scatter.legend_elements()[0], categories, title=color_by, loc="upper left")
                    ax.add_artist(legend1)
                else:
                    plt.colorbar(scatter, label=color_by)
            elif color_by != "None":
                scatter = ax.scatter(
                    df[x_col], df[y_col],
                    c=pd.factorize(df[color_by])[0] if color_by in categorical_cols else df[color_by],
                    alpha=0.6
                )
                if color_by in categorical_cols:
                    categories = df[color_by].unique()
                    legend1 = ax.legend(scatter.legend_elements()[0], categories, title=color_by, loc="upper left")
                    ax.add_artist(legend1)
                else:
                    plt.colorbar(scatter, label=color_by)
            elif size_by != "None":
                ax.scatter(
                    df[x_col], df[y_col],
                    s=df[size_by] * 20 / df[size_by].max(),
                    alpha=0.6
                )
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
            
            ax.set_title(f"Scatter Plot of {y_col} vs {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show correlation
            st.write("#### Correlation")
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation Coefficient", f"{corr:.4f}")
        else:
            st.error("Need at least 2 numeric columns for scatter plot")
    
    elif viz_type == "Line Chart":
        st.write("#### Line Chart Configuration")
        
        if date_cols:
            x_col = st.selectbox("Select X-axis (Date) Column", date_cols)
            y_cols = st.multiselect("Select Y-axis Column(s)", numeric_cols)
            
            if y_cols:
                st.write("#### Line Chart")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for col in y_cols:
                    # Sort by date and plot
                    plot_df = df.sort_values(by=x_col)
                    ax.plot(plot_df[x_col], plot_df[col], marker='o', linestyle='-', alpha=0.7, label=col)
                
                ax.set_title(f"Line Chart over Time")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Value")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least one numeric column for Y-axis")
        elif numeric_cols:
            # If no date columns, let user select a numeric column for X-axis
            x_col = st.selectbox("Select X-axis Column", numeric_cols)
            y_cols = st.multiselect("Select Y-axis Column(s)", 
                                  [col for col in numeric_cols if col != x_col])
            
            if y_cols:
                st.write("#### Line Chart")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for col in y_cols:
                    # Sort by X column and plot
                    plot_df = df.sort_values(by=x_col)
                    ax.plot(plot_df[x_col], plot_df[col], marker='o', linestyle='-', alpha=0.7, label=col)
                
                ax.set_title(f"Line Chart")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Value")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least one numeric column for Y-axis")
        else:
            st.error("Need at least one date column and one numeric column for line chart")
    
    elif viz_type == "Bar Chart":
        st.write("#### Bar Chart Configuration")
        
        if categorical_cols and numeric_cols:
            x_col = st.selectbox("Select X-axis (Category) Column", categorical_cols)
            y_col = st.selectbox("Select Y-axis (Value) Column", numeric_cols)
            
            # Options for aggregation
            agg_func = st.selectbox(
                "Aggregation Function",
                ["Mean", "Sum", "Count", "Median", "Min", "Max"]
            )
            
            # Option to limit categories
            max_categories = min(20, df[x_col].nunique())
            limit_cats = st.slider("Limit to top N categories", 
                                 min_value=5, 
                                 max_value=max_categories,
                                 value=min(10, max_categories))
            
            # Prepare data - aggregate and limit categories
            agg_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Count": "count",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            # Group by category and apply aggregation
            grouped = df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
            # Sort and limit to top N categories
            grouped = grouped.sort_values(by=y_col, ascending=False).head(limit_cats)
            
            st.write("#### Bar Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=x_col, y=y_col, data=grouped, ax=ax)
            ax.set_title(f"{agg_func} of {y_col} by {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(f"{agg_func} of {y_col}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show data table
            st.write("#### Aggregated Data")
            safe_display_dataframe(grouped)
        else:
            st.error("Need at least one categorical column and one numeric column for bar chart")
    
    elif viz_type == "Box Plot":
        st.write("#### Box Plot Configuration")
        
        if numeric_cols:
            y_col = st.selectbox("Select Column for Box Plot", numeric_cols)
            group_by = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            
            st.write("#### Box Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if group_by != "None":
                # Limit to top 10 categories to avoid too wide plot
                top_cats = df[group_by].value_counts().nlargest(10).index
                plot_df = df[df[group_by].isin(top_cats)]
                sns.boxplot(x=group_by, y=y_col, data=plot_df, ax=ax)
                ax.set_title(f"Box Plot of {y_col} by {group_by}")
                plt.xticks(rotation=45, ha='right')
            else:
                sns.boxplot(y=df[y_col], ax=ax)
                ax.set_title(f"Box Plot of {y_col}")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Need at least one numeric column for box plot")
    
    elif viz_type == "Heatmap":
        st.write("#### Heatmap Configuration")
        
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect(
                "Select Columns for Correlation Heatmap", 
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_cols) > 1:
                st.write("#### Correlation Heatmap")
                corr_matrix = df[selected_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                heatmap = sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap="coolwarm", 
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title("Correlation Heatmap")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show correlation matrix as a table
                st.write("#### Correlation Matrix")
                safe_display_dataframe(corr_matrix)
            else:
                st.warning("Please select at least 2 columns for heatmap")
        else:
            st.error("Need at least 2 numeric columns for heatmap")
    
    elif viz_type == "Pair Plot":
        st.write("#### Pair Plot Configuration")
        
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect(
                "Select Columns for Pair Plot", 
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            hue_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            
            if len(selected_cols) > 1:
                st.write("#### Generating Pair Plot...")
                
                # Limit rows if dataset is large
                sample_size = 1000
                if len(df) > sample_size:
                    st.info(f"Dataset is large. Using a random sample of {sample_size} rows for pair plot.")
                    plot_df = df.sample(sample_size, random_state=42)
                else:
                    plot_df = df
                
                # Generate pair plot
                if hue_col != "None":
                    # Get top categories if too many
                    if plot_df[hue_col].nunique() > 5:
                        top_cats = plot_df[hue_col].value_counts().nlargest(5).index
                        plot_df = plot_df[plot_df[hue_col].isin(top_cats)]
                        st.info(f"Limited to top 5 categories of {hue_col} for clarity.")
                    
                    g = sns.pairplot(plot_df[selected_cols + [hue_col]], 
                                    hue=hue_col, 
                                    plot_kws={'alpha': 0.6})
                else:
                    g = sns.pairplot(plot_df[selected_cols], 
                                    plot_kws={'alpha': 0.6})
                
                g.fig.suptitle("Pair Plot", y=1.02)
                st.pyplot(g.fig)
            else:
                st.warning("Please select at least 2 columns for pair plot")
        else:
            st.error("Need at least 2 numeric columns for pair plot")
    
    # Add download options for plots
    st.write("### Export Visualization")
    export_format = st.selectbox("Select Export Format", ["PNG", "SVG", "PDF"])
    
    if st.button("Download Visualization"):
        # This is a placeholder - in a real app you would save the figure and provide a download link
        st.info("Feature coming soon! In a complete implementation, this would allow downloading the current visualization.") 