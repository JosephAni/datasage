import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Cost of Inventory Analysis",
    page_icon="💰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .cost-card {
        background-color: #2C3E50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #4B70E2;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #34495E;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

def calculate_inventory_metrics(purchase_cost, order_cost, physical_holding_rate, financial_holding_rate, 
                             avg_inventory_value, annual_sales, cogs):
    """
    Calculate comprehensive inventory metrics based on input parameters
    """
    # Basic holding costs
    physical_holding_cost = avg_inventory_value * (physical_holding_rate / 100)
    financial_holding_cost = avg_inventory_value * (financial_holding_rate / 100)
    total_holding_cost = physical_holding_cost + financial_holding_cost
    
    # Inventory turnover metrics
    inventory_turnover = cogs / avg_inventory_value if avg_inventory_value > 0 else 0
    days_inventory = 365 / inventory_turnover if inventory_turnover > 0 else 0
    
    # Asset utilization
    asset_turnover = annual_sales / (avg_inventory_value + order_cost) if (avg_inventory_value + order_cost) > 0 else 0
    
    # Total inventory cost
    total_inventory_cost = purchase_cost + order_cost + total_holding_cost
    
    return {
        'physical_holding_cost': physical_holding_cost,
        'financial_holding_cost': financial_holding_cost,
        'total_holding_cost': total_holding_cost,
        'inventory_turnover': inventory_turnover,
        'days_inventory': days_inventory,
        'asset_turnover': asset_turnover,
        'total_inventory_cost': total_inventory_cost
    }

def main():
    st.title("Cost of Inventory Analysis 💰")
    
    # Add reset button in the top right
    col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
    with col4:
        if st.button("🔄 Reset", help="Reset all values to defaults"):
            # Clear all the input values from session state
            for key in st.session_state.keys():
                if key.startswith('cost_inventory_'):
                    del st.session_state[key]
            st.rerun()
    
    st.markdown("""
    <div class='cost-card'>
    <h3>Calculate Your Inventory Costs</h3>
    <p>This tool helps you analyze different components of inventory costs and efficiency metrics to optimize your inventory management strategy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Direct Costs")
        purchase_cost = st.number_input(
            "Purchase Cost per Unit ($)",
            min_value=0.0,
            value=75.0,
            step=1.0,
            help="The direct cost of acquiring one unit of inventory",
            key="cost_inventory_purchase_cost"
        )
        
        order_cost = st.number_input(
            "Cost of Placing an Order ($)",
            min_value=0.0,
            value=250.0,
            step=10.0,
            help="The expenses associated with ordering more inventory",
            key="cost_inventory_order_cost"
        )
    
    with col2:
        st.subheader("Holding Costs")
        physical_holding_rate = st.number_input(
            "Physical Holding Cost Rate (%)",
            min_value=0.0,
            value=15.0,
            step=0.1,
            help="Percentage of inventory value for physical storage costs",
            key="cost_inventory_physical_rate"
        )
        
        financial_holding_rate = st.number_input(
            "Financial Holding Cost Rate (%)",
            min_value=0.0,
            value=12.0,
            step=0.1,
            help="Percentage of inventory value for opportunity cost of capital",
            key="cost_inventory_financial_rate"
        )
    
    with col3:
        st.subheader("Financial Metrics")
        annual_sales = st.number_input(
            "Annual Sales ($)",
            min_value=0.0,
            value=5000000.0,
            step=10000.0,
            help="Total annual sales revenue",
            key="cost_inventory_annual_sales"
        )
        
        cogs = st.number_input(
            "Cost of Goods Sold ($)",
            min_value=0.0,
            value=3500000.0,
            step=10000.0,
            help="Total cost of goods sold annually",
            key="cost_inventory_cogs"
        )
        
        avg_inventory_value = st.number_input(
            "Average Inventory Value ($)",
            min_value=0.0,
            value=875000.0,
            step=5000.0,
            help="The average value of inventory held during the period",
            key="cost_inventory_avg_value"
        )
    
    # Calculate metrics
    metrics = calculate_inventory_metrics(
        purchase_cost,
        order_cost,
        physical_holding_rate,
        financial_holding_rate,
        avg_inventory_value,
        annual_sales,
        cogs
    )
    
    # Display results in two sections
    st.markdown("<h3>Cost Analysis</h3>", unsafe_allow_html=True)
    
    # Cost metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Physical Holding Cost",
            f"${metrics['physical_holding_cost']:,.2f}",
            help="Costs related to physically storing the inventory"
        )
    
    with col2:
        st.metric(
            "Financial Holding Cost",
            f"${metrics['financial_holding_cost']:,.2f}",
            help="Opportunity cost of capital invested in inventory"
        )
    
    with col3:
        st.metric(
            "Total Holding Cost",
            f"${metrics['total_holding_cost']:,.2f}",
            help="Sum of physical and financial holding costs"
        )
    
    with col4:
        st.metric(
            "Total Inventory Cost",
            f"${metrics['total_inventory_cost']:,.2f}",
            help="Total cost including purchase, order, and holding costs"
        )
    
    # Efficiency metrics
    st.markdown("<h3>Efficiency Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Inventory Turnover",
            f"{metrics['inventory_turnover']:.2f}x",
            help="Number of times inventory is sold and replaced over a period"
        )
    
    with col2:
        st.metric(
            "Days Inventory",
            f"{metrics['days_inventory']:.2f} days",
            help="Average number of days it takes to sell inventory"
        )
    
    with col3:
        st.metric(
            "Asset Turnover",
            f"{metrics['asset_turnover']:.2f}x",
            help="Efficiency of asset use in generating sales"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost breakdown pie chart
        fig1 = go.Figure(data=[go.Pie(
            labels=['Physical Holding Cost', 'Financial Holding Cost', 'Order Cost', 'Purchase Cost'],
            values=[metrics['physical_holding_cost'], metrics['financial_holding_cost'], 
                   order_cost, purchase_cost],
            hole=.3
        )])
        
        fig1.update_layout(
            title="Total Cost Breakdown",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Efficiency metrics gauge
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['inventory_turnover'],
            title = {'text': "Inventory Turnover Rate"},
            gauge = {
                'axis': {'range': [None, 70]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 50], 'color': "gray"},
                    {'range': [50, 70], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recommendations
    st.subheader("Optimization Recommendations")
    
    recommendations = []
    
    # Based on Apple's metrics as benchmarks
    if metrics['days_inventory'] > 12:  # Apple's max is around 11.81
        recommendations.append("❗ Days Inventory is higher than optimal. Consider improving inventory turnover.")
    
    if metrics['inventory_turnover'] < 30:  # Apple's min is around 30.90
        recommendations.append("❗ Inventory Turnover is below industry benchmark. Look into improving sales or reducing average inventory.")
    
    if metrics['asset_turnover'] < 0.65:  # Apple's min is around 0.66
        recommendations.append("❗ Asset Turnover is below optimal levels. Consider strategies to increase sales or reduce asset investment.")
    
    if metrics['financial_holding_cost'] > metrics['physical_holding_cost'] * 1.5:
        recommendations.append("💡 Financial holding costs are significantly high. Consider reducing inventory levels or negotiating better financing terms.")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Your inventory metrics are within optimal ranges. Continue monitoring and maintaining current practices.")
    
    for rec in recommendations:
        st.markdown(f"{rec}")

if __name__ == "__main__":
    main() 