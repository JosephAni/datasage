import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="EOQ Simulation",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .simulation-card {
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
</style>
""", unsafe_allow_html=True)

def calculate_eoq(annual_demand, order_cost, unit_cost, holding_rate):
    """Calculate the Economic Order Quantity"""
    try:
        eoq = np.sqrt((2 * annual_demand * order_cost) / (unit_cost * holding_rate))
        return eoq
    except:
        return 0

def calculate_costs(order_size, annual_demand, order_cost, unit_cost, holding_rate):
    """Calculate various costs for a given order size"""
    try:
        # Number of orders per year
        num_orders = annual_demand / order_size
        
        # Order costs
        annual_order_cost = num_orders * order_cost
        
        # Holding costs
        avg_inventory = order_size / 2
        annual_holding_cost = avg_inventory * unit_cost * holding_rate
        
        # Total cost
        total_cost = annual_order_cost + annual_holding_cost
        
        return {
            'order_cost': annual_order_cost,
            'holding_cost': annual_holding_cost,
            'total_cost': total_cost
        }
    except:
        return {
            'order_cost': 0,
            'holding_cost': 0,
            'total_cost': 0
        }

def main():
    st.title("EOQ Simulation 📊")
    
    st.markdown("""
    <div class='simulation-card'>
    <h3>Economic Order Quantity (EOQ) Simulator</h3>
    <p>Visualize how order size affects total inventory costs and find the optimal order quantity that minimizes total costs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add How to Use section in an expander
    with st.expander("📖 How to Use This Simulator", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Input Parameters**:
           - **Annual Demand**: Enter the total number of units needed per year
           - **Cost per Order**: Enter the fixed cost associated with placing each order
           - **Unit Cost**: Enter the cost of each individual unit
           - **Cost of Capital**: Enter the annual holding cost as a percentage of unit cost
        
        2. **Understanding the Results**:
           - The **Cost Curves** graph shows how different order sizes affect:
             * Order Costs (blue line)
             * Holding Costs (orange line)
             * Total Costs (green line)
           - The vertical dashed line shows the optimal order quantity
        
        3. **Interactive Comparison**:
           - The comparison table shows costs for both optimal and custom order sizes
           - Use the number input in the "Custom Order" row to experiment with different order quantities
           - See how your custom order size compares to the optimal solution
        
        4. **Inventory Pattern**:
           - The sawtooth diagram shows inventory levels over time
           - Compare how different order sizes affect inventory patterns
           - Optimal order pattern (solid line) vs Custom order pattern (dashed line)
        
        5. **Best Practices**:
           - Start with the suggested optimal order quantity
           - Experiment with different order sizes to understand cost trade-offs
           - Consider practical constraints that might affect your final decision
        """)
    
    # Create two columns - one for inputs and one for the main chart
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        
        annual_demand = st.number_input(
            "Annual Demand (units)",
            min_value=1,
            value=400,
            step=1,
            help="Total number of units demanded per year"
        )
        
        order_cost = st.number_input(
            "Cost per Order ($)",
            min_value=0.0,
            value=500.0,
            step=10.0,
            help="Fixed cost of placing an order"
        )
        
        unit_cost = st.number_input(
            "Unit Cost ($)",
            min_value=0.0,
            value=200.0,
            step=1.0,
            help="Cost per unit of inventory"
        )
        
        holding_rate = st.number_input(
            "Cost of Capital (%)",
            min_value=0.0,
            value=20.0,
            step=0.1,
            help="Annual holding cost as a percentage of unit cost"
        ) / 100  # Convert percentage to decimal
        
        # Calculate EOQ
        optimal_order = calculate_eoq(annual_demand, order_cost, unit_cost, holding_rate)
        optimal_costs = calculate_costs(optimal_order, annual_demand, order_cost, unit_cost, holding_rate)
        
        st.markdown("""
        <div class='result-card'>
        <h4>Optimal Order Quantity</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Optimal Order Size", f"{optimal_order:.0f} units")
        
    # Generate cost curves
    order_sizes = np.linspace(10, max(optimal_order * 3, 400), 100)
    costs_data = []
    
    for size in order_sizes:
        costs = calculate_costs(size, annual_demand, order_cost, unit_cost, holding_rate)
        costs_data.append({
            'order_size': size,
            'order_cost': costs['order_cost'],
            'holding_cost': costs['holding_cost'],
            'total_cost': costs['total_cost']
        })
    
    df = pd.DataFrame(costs_data)
    
    with col2:
        # Create the main cost curves plot
        fig = go.Figure()
        
        # Add the three cost curves
        fig.add_trace(go.Scatter(
            x=df['order_size'],
            y=df['order_cost'],
            name='Order Cost',
            line=dict(color='#1f77b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['order_size'],
            y=df['holding_cost'],
            name='Holding Cost',
            line=dict(color='#ff7f0e')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['order_size'],
            y=df['total_cost'],
            name='Total Cost',
            line=dict(color='#2ca02c', width=3)
        ))
        
        # Add a vertical line at the optimal order quantity
        fig.add_vline(
            x=optimal_order,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal Order: {optimal_order:.0f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="Cost Curves by Order Size",
            xaxis_title="Order Size",
            yaxis_title="Annual Cost ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a comparison table
        st.subheader("Cost Comparison")
        
        # Initialize custom order with optimal order
        if 'custom_order' not in st.session_state:
            st.session_state.custom_order = int(optimal_order)
        
        # Create comparison table with interactive input
        custom_order = st.number_input(
            "Custom Order Size",
            min_value=1,
            value=st.session_state.custom_order,
            step=1,
            key="custom_order_input",
            label_visibility="visible"
        )
        
        # Update session state
        st.session_state.custom_order = custom_order
        
        # Calculate costs for custom order
        custom_costs = calculate_costs(custom_order, annual_demand, order_cost, unit_cost, holding_rate)
        
        # Create comparison table
        comparison_data = {
            'Order Size': ['Optimal Order', 'Custom Order'],
            'Size': [f"{optimal_order:.0f}", custom_order],
            'Order Costs': [f"${optimal_costs['order_cost']:,.2f}", f"${custom_costs['order_cost']:,.2f}"],
            'Holding Costs': [f"${optimal_costs['holding_cost']:,.2f}", f"${custom_costs['holding_cost']:,.2f}"],
            'Total Costs': [f"${optimal_costs['total_cost']:,.2f}", f"${custom_costs['total_cost']:,.2f}"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)
        
        # Add the sawtooth diagram
        st.subheader("Inventory Level Over Time")
        
        # Generate sawtooth plot data
        time_periods = 4
        time = np.linspace(0, time_periods, 1000)
        
        # Create sawtooth wave for optimal order
        sawtooth_optimal = optimal_order * (0.5 - (time % 1))
        
        # Create sawtooth wave for custom order
        sawtooth_custom = custom_order * (0.5 - (time % 1))
        
        # Create the sawtooth plot
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=time,
            y=sawtooth_optimal,
            name='Optimal Order Pattern',
            line=dict(color='#1f77b4')
        ))
        
        if custom_order != optimal_order:
            fig2.add_trace(go.Scatter(
                x=time,
                y=sawtooth_custom,
                name='Custom Order Pattern',
                line=dict(color='#ff7f0e', dash='dash')
            ))
        
        fig2.update_layout(
            title="Inventory Level Pattern",
            xaxis_title="Time (Years)",
            yaxis_title="Inventory Level",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main() 