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
    from data_cleaner import DataCleaner, safe_display_dataframe
except ImportError:
    st.error("Unable to import DataCleaner class. Please ensure data_cleaner.py is in the project root.")
    
    # Define a basic placeholder if import fails
    class DataCleaner:
        def __init__(self, df):
            self.df = df.copy()
            
        def calculate_clv(self, customer_id_col, invoice_date_col, quantity_col, price_col, segment_by=None):
            st.error("CLV calculation requires the full DataCleaner class from data_cleaner.py")
            return None, None
            
    def safe_display_dataframe(df, cleaner=None, **kwargs):
        if df is None or df.empty:
            return st.dataframe(pd.DataFrame(), **kwargs)
        return st.dataframe(df, **kwargs)

# Set page config
st.set_page_config(
    page_title="Customer Lifetime Value Analysis",
    page_icon="👥",
    layout="wide"
)

st.markdown("# Customer Lifetime Value Analysis")
st.sidebar.header("CLV Analysis")
st.write(
    """
    This page allows you to calculate Customer Lifetime Value (CLV) for your customer data.
    Select the appropriate columns and parameters to analyze customer spending patterns and segment your customers.
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
    
    # Initialize DataCleaner if not already done
    if 'clv_cleaner' not in st.session_state:
        st.session_state.clv_cleaner = DataCleaner(df)
    
    cleaner = st.session_state.clv_cleaner
    
    # CLV Analysis options
    st.write("### Customer Lifetime Value Analysis")
    st.write("""
    Customer Lifetime Value (CLV) is a metric that represents the total value a customer brings to your business over their entire relationship.
    It helps identify your most valuable customers and understand your customer acquisition costs in relation to customer value.
    """)
    
    # Column selectors for CLV analysis
    all_cols = df.columns.tolist()
    
    # Get column suggestions based on names
    customer_id_suggestions = [col for col in all_cols if any(term in col.lower() for term in ['customer', 'client', 'user', 'id'])]
    invoice_date_suggestions = [col for col in all_cols if any(term in col.lower() for term in ['date', 'time', 'invoice', 'purchase'])]
    quantity_suggestions = [col for col in all_cols if any(term in col.lower() for term in ['quantity', 'qty', 'amount', 'count'])]
    price_suggestions = [col for col in all_cols if any(term in col.lower() for term in ['price', 'value', 'revenue', 'sales'])]
    segment_suggestions = [col for col in all_cols if any(term in col.lower() for term in ['segment', 'category', 'type', 'group', 'class'])]
    
    # Column selection with smart defaults
    customer_id_col = st.selectbox(
        "Customer ID Column",
        all_cols, 
        index=all_cols.index(customer_id_suggestions[0]) if customer_id_suggestions else 0
    )
    
    invoice_date_col = st.selectbox(
        "Invoice Date Column",
        all_cols,
        index=all_cols.index(invoice_date_suggestions[0]) if invoice_date_suggestions else 0
    )
    
    quantity_col = st.selectbox(
        "Quantity Column",
        all_cols,
        index=all_cols.index(quantity_suggestions[0]) if quantity_suggestions else 0
    )
    
    price_col = st.selectbox(
        "Price Column",
        all_cols,
        index=all_cols.index(price_suggestions[0]) if price_suggestions else 0
    )
    
    # Optional segmentation
    use_segmentation = st.checkbox("Segment Customers")
    
    if use_segmentation:
        segment_by = st.selectbox(
            "Segment By Column",
            ["None"] + all_cols,
            index=all_cols.index(segment_suggestions[0])+1 if segment_suggestions else 0
        )
        if segment_by == "None":
            segment_by = None
    else:
        segment_by = None
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options", expanded=False):
        # Date format if needed
        date_format = st.text_input(
            "Date Format (leave empty for auto-detection)",
            value="",
            help="e.g., '%Y-%m-%d' for YYYY-MM-DD format"
        )
        
        # Filter options
        min_purchase = st.number_input(
            "Minimum Purchase Value",
            value=0.0,
            help="Filter out transactions below this amount"
        )
        
        # Prediction period (for future CLV)
        prediction_period = st.number_input(
            "Prediction Period (months)",
            value=12,
            min_value=1,
            max_value=60,
            help="Period for projecting future CLV"
        )
    
    # Calculate CLV button
    if st.button("Calculate CLV"):
        with st.spinner("Calculating Customer Lifetime Value..."):
            try:
                # Check if the invoice date column is datetime type
                if not pd.api.types.is_datetime64_dtype(df[invoice_date_col]):
                    if date_format:
                        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], format=date_format)
                    else:
                        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], errors='coerce')
                
                # Update the cleaner with the latest data
                cleaner.df = df
                
                # Calculate CLV
                clv_results, segment_stats = cleaner.calculate_clv(
                    customer_id_col=customer_id_col,
                    invoice_date_col=invoice_date_col,
                    quantity_col=quantity_col,
                    price_col=price_col,
                    segment_by=segment_by
                )
                
                if clv_results is not None:
                    st.success("CLV calculation completed successfully!")
                    
                    # Display CLV results
                    st.write("### Customer Lifetime Value Results")
                    safe_display_dataframe(clv_results, cleaner)
                    
                    # CLV Distribution
                    st.write("### CLV Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(clv_results['CLV'], kde=True, ax=ax)
                    ax.set_title("Distribution of Customer Lifetime Value")
                    ax.set_xlabel("Customer Lifetime Value")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    
                    # Top customers
                    st.write("### Top 10 Customers by CLV")
                    top_customers = clv_results.sort_values(by='CLV', ascending=False).head(10)
                    safe_display_dataframe(top_customers, cleaner)
                    
                    # Customer segmentation if available
                    if segment_stats is not None and segment_by is not None:
                        st.write(f"### Customer Segmentation by {segment_by}")
                        safe_display_dataframe(segment_stats, cleaner)
                        
                        # Segment comparison chart
                        st.write(f"### Segment Comparison")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(x=segment_stats.index, y='Average CLV', data=segment_stats, ax=ax)
                        ax.set_title(f"Average CLV by {segment_by}")
                        ax.set_xlabel(segment_by)
                        ax.set_ylabel("Average CLV")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Additional segment insights
                        st.write("### Segment Insights")
                        
                        # Frequency by segment
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(x=segment_stats.index, y='Average Frequency', data=segment_stats, ax=ax)
                        ax.set_title(f"Average Purchase Frequency by {segment_by}")
                        ax.set_xlabel(segment_by)
                        ax.set_ylabel("Average Frequency")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # AOV by segment
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(x=segment_stats.index, y='Average Order Value', data=segment_stats, ax=ax)
                        ax.set_title(f"Average Order Value by {segment_by}")
                        ax.set_xlabel(segment_by)
                        ax.set_ylabel("Average Order Value")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # RFM Analysis
                    st.write("### RFM Analysis")
                    
                    rfm_cols = ['Recency', 'Frequency', 'Monetary Value', 'CLV']
                    if all(col in clv_results.columns for col in rfm_cols[:3]):  # Check if RFM data is available
                        # RFM correlation heatmap
                        rfm_corr = clv_results[rfm_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(rfm_corr, annot=True, cmap="coolwarm", ax=ax)
                        ax.set_title("Correlation between RFM Metrics and CLV")
                        st.pyplot(fig)
                        
                        # RFM scatter plots
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # Recency vs CLV
                        sns.scatterplot(x='Recency', y='CLV', data=clv_results, ax=axes[0])
                        axes[0].set_title("Recency vs CLV")
                        
                        # Frequency vs CLV
                        sns.scatterplot(x='Frequency', y='CLV', data=clv_results, ax=axes[1])
                        axes[1].set_title("Frequency vs CLV")
                        
                        # Monetary vs CLV
                        sns.scatterplot(x='Monetary Value', y='CLV', data=clv_results, ax=axes[2])
                        axes[2].set_title("Monetary Value vs CLV")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download options
                        st.write("### Download Results")
                        csv = clv_results.to_csv(index=False)
                        st.download_button(
                            label="Download CLV Results CSV",
                            data=csv,
                            file_name="clv_results.csv",
                            mime="text/csv"
                        )
                        
                        if segment_stats is not None:
                            segment_csv = segment_stats.to_csv()
                            st.download_button(
                                label="Download Segment Stats CSV",
                                data=segment_csv,
                                file_name="segment_stats.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning("RFM metrics not available in the results.")
                else:
                    st.error("CLV calculation failed. Please check your data and parameters.")
            
            except Exception as e:
                st.error(f"Error calculating CLV: {str(e)}")
                st.warning("Make sure your data has the correct format and column types for CLV calculation.")
    
    # CLV Information section
    with st.expander("About Customer Lifetime Value", expanded=False):
        st.markdown("""
        ### What is Customer Lifetime Value (CLV)?
        
        Customer Lifetime Value is a prediction of the total value a business can expect from a customer throughout their entire relationship. It helps businesses make decisions about:
        
        - How much to spend on customer acquisition
        - Which customer segments to prioritize
        - How to optimize marketing strategies
        
        ### CLV Formula
        
        The basic formula for CLV is:
        
        ```
        CLV = Average Order Value × Purchase Frequency × Customer Lifespan
        ```
        
        Where:
        - Average Order Value = Total Revenue / Number of Orders
        - Purchase Frequency = Number of Orders / Number of Unique Customers
        - Customer Lifespan = Time between first and last purchase
        
        ### RFM Analysis
        
        RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to analyze customer behavior:
        
        - Recency: How recently did the customer make a purchase?
        - Frequency: How often does the customer make purchases?
        - Monetary Value: How much does the customer spend?
        
        These metrics are strong predictors of a customer's future behavior and value.
        """) 