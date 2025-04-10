import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

# Set page config
st.set_page_config(
    page_title="NewsVendor Simulation",
    page_icon="📰",
    layout="wide"
)

# Initialize session state
if 'newsvendor_params' not in st.session_state:
    st.session_state.newsvendor_params = {
        'distribution': 'Normal',
        'expected_demand': 100,
        'std_dev': 10.0,
        'cost': 5.0,
        'price': 10.0,
        'salvage_price': 2.5,
        'lock_range': False,
        'wins': 0,
        'last_range': 10.0
    }

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
    .stButton>button {
        background-color: #4B70E2;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

def calculate_newsvendor_metrics(demand_mean, demand_std, price, cost, salvage):
    """Calculate optimal order quantity and other metrics for the newsvendor model"""
    try:
        # Calculate critical ratio
        critical_ratio = (price - cost) / (price - salvage)
        
        # Calculate optimal order quantity
        if st.session_state.newsvendor_params['distribution'] == 'Normal':
            optimal_q = demand_mean + demand_std * stats.norm.ppf(critical_ratio)
        else:  # Uniform
            a = demand_mean - demand_std * np.sqrt(3)  # Lower bound
            b = demand_mean + demand_std * np.sqrt(3)  # Upper bound
            optimal_q = a + (b - a) * critical_ratio
        
        # Calculate expected sales
        expected_sales = price * demand_mean
        
        # Calculate expected salvage
        expected_excess = max(0, optimal_q - demand_mean)
        salvage_revenue = salvage * expected_excess
        
        # Calculate costs
        total_cost = cost * optimal_q
        
        # Calculate expected profit
        expected_profit = expected_sales + salvage_revenue - total_cost
        
        return {
            'optimal_q': optimal_q,
            'critical_ratio': critical_ratio,
            'service_level': critical_ratio * 100,
            'sales': expected_sales,
            'salvage': salvage_revenue,
            'costs': total_cost,
            'expected_profit': expected_profit
        }
    except:
        return {
            'optimal_q': 0,
            'critical_ratio': 0,
            'service_level': 0,
            'sales': 0,
            'salvage': 0,
            'costs': 0,
            'expected_profit': 0
        }

def simulate_newsvendor(n_periods, order_quantity, demand_mean, demand_std, price, cost, salvage):
    """Simulate the newsvendor problem"""
    np.random.seed(42)  # For reproducibility
    
    if st.session_state.newsvendor_params['distribution'] == 'Normal':
        demands = np.random.normal(demand_mean, demand_std, n_periods)
    else:  # Uniform
        a = demand_mean - demand_std * np.sqrt(3)
        b = demand_mean + demand_std * np.sqrt(3)
        demands = np.random.uniform(a, b, n_periods)
    
    total_profit = 0
    for demand in demands:
        sales = min(demand, order_quantity)
        leftover = max(0, order_quantity - demand)
        
        revenue = sales * price
        salvage_revenue = leftover * salvage
        ordering_cost = order_quantity * cost
        
        profit = revenue + salvage_revenue - ordering_cost
        total_profit += profit
    
    return total_profit

def plot_distribution(mean, std, optimal_q):
    """Create distribution plot with optimal quantity marker"""
    if st.session_state.newsvendor_params['distribution'] == 'Normal':
        x = np.linspace(mean - 4*std, mean + 4*std, 200)
        y = stats.norm.pdf(x, mean, std)
        
        # Split the distribution at optimal_q
        mask = x <= optimal_q
        
        fig = go.Figure()
        
        # Add the left part (red)
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red'),
            showlegend=False
        ))
        
        # Add the right part (blue)
        fig.add_trace(go.Scatter(
            x=x[~mask],
            y=y[~mask],
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.3)',
            line=dict(color='blue'),
            showlegend=False
        ))
    else:  # Uniform
        a = mean - std * np.sqrt(3)
        b = mean + std * np.sqrt(3)
        x = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 200)
        y = np.where((x >= a) & (x <= b), 1/(b-a), 0)
        
        fig = go.Figure()
        
        # Split the distribution at optimal_q
        mask = x <= optimal_q
        
        # Add the left part (red)
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red'),
            showlegend=False
        ))
        
        # Add the right part (blue)
        fig.add_trace(go.Scatter(
            x=x[~mask],
            y=y[~mask],
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.3)',
            line=dict(color='blue'),
            showlegend=False
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Demand",
        yaxis_title="Probability Density",
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def main():
    st.title("NewsVendor Simulation 📰")
    
    st.markdown("""
    <div class='simulation-card'>
    <h3>NewsVendor Model Simulator</h3>
    <p>Experiment with different order quantities and see how they perform against the optimal strategy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add How to Use section in an expander
    with st.expander("📖 How to Use This Simulator", expanded=False):
        st.markdown("""
        ### The NewsVendor Problem
        
        The NewsVendor problem is a classic inventory management model that helps determine the optimal order quantity for products with uncertain demand. It's called the "NewsVendor" because it models scenarios similar to selling newspapers - you must decide how many to order before knowing exactly how many will be demanded.
        
        ### Step-by-Step Guide
        
        1. **Set Distribution Parameters**:
           - **Distribution Type**: Choose between Normal (bell curve) or Uniform (flat) probability distributions
           - **Expected Demand**: The average number of units demanded per period
           - **Standard Deviation**: How variable the demand is (higher means more uncertainty)
           - **Cost**: The cost per unit that you pay the supplier
           - **Price**: What you sell each unit for to customers
           - **Salvage Price**: What you can recover per unsold unit
        
        2. **Understand the Results Table**:
           - **Optimum Column**: Shows the theoretically optimal order quantity and expected results
           - **Custom Column**: Enter your own order quantity to see how it compares
           - **Order**: The number of units to order (optimal vs. your choice)
           - **Critical Fractal**: The critical ratio determining the optimal solution (price-cost)/(price-salvage)
           - **Service Level**: Probability of not stocking out
           - **Sales $**: Expected revenue from sales
           - **Salvage $**: Expected revenue from salvaging unsold units
           - **Costs**: Total cost of ordering units
           - **Expected Profit**: Overall expected profit
        
        3. **Experiment with the Simulation**:
           - Set the number of order periods to simulate (e.g., 1000 periods)
           - Your order quantity is taken from the Custom column
           - Click SUBMIT to run the simulation and see how your order quantity performs against the optimal
           - The emoji 😊 indicates which strategy won in the simulation
        
        4. **Analyze the Distribution Plot**:
           - The graph shows the probability distribution of demand
           - Red area: Portion of distribution below the optimal order quantity
           - Blue area: Portion of distribution above the optimal order quantity
           - The "Lock Range" checkbox maintains a consistent visual scale when changing the standard deviation
        
        5. **Tips for Learning**:
           - Start by using the optimal order quantity to understand baseline performance
           - Try different order quantities to see how they affect profit
           - Notice how the optimal quantity changes when you modify parameters
           - For short simulation periods, sometimes you can "get lucky" and beat the optimal strategy
           - Try extreme values to understand how the model behaves
        """)
    
    # Create two columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        
        # Distribution selection
        distribution = st.selectbox(
            "Distribution",
            ["Normal", "Uniform"],
            key="distribution",
            index=0 if st.session_state.newsvendor_params['distribution'] == 'Normal' else 1
        )
        st.session_state.newsvendor_params['distribution'] = distribution
        
        # Input parameters
        expected_demand = st.number_input(
            "Expected Demand",
            min_value=1.0,
            value=float(st.session_state.newsvendor_params['expected_demand']),
            step=1.0,
            format="%.1f"
        )
        st.session_state.newsvendor_params['expected_demand'] = float(expected_demand)
        
        std_dev = st.number_input(
            "Standard Deviation",
            min_value=0.1,
            value=float(st.session_state.newsvendor_params['std_dev']),
            step=0.1,
            format="%.1f"
        )
        
        cost = st.number_input(
            "Cost",
            min_value=0.0,
            value=float(st.session_state.newsvendor_params['cost']),
            step=0.1,
            format="%.1f"
        )
        st.session_state.newsvendor_params['cost'] = float(cost)
        
        price = st.number_input(
            "Price",
            min_value=float(cost),
            value=max(float(st.session_state.newsvendor_params['price']), float(cost)),
            step=0.1,
            format="%.1f"
        )
        st.session_state.newsvendor_params['price'] = float(price)
        
        salvage_price = st.number_input(
            "Salvage Price",
            min_value=0.0,
            max_value=float(cost),
            value=min(float(st.session_state.newsvendor_params['salvage_price']), float(cost)),
            step=0.1,
            format="%.1f"
        )
        st.session_state.newsvendor_params['salvage_price'] = float(salvage_price)
        
        st.markdown("---")  # Add a separator line
        
        # Handle lock range feature
        lock_range = st.checkbox(
            "Lock Range",
            value=st.session_state.newsvendor_params['lock_range']
        )
        st.session_state.newsvendor_params['lock_range'] = lock_range
        
        if lock_range:
            # If range is locked, maintain the same visual range when std_dev changes
            if std_dev != st.session_state.newsvendor_params['std_dev']:
                st.session_state.newsvendor_params['last_range'] = float(st.session_state.newsvendor_params['std_dev'])
        else:
            st.session_state.newsvendor_params['last_range'] = float(std_dev)
        
        st.session_state.newsvendor_params['std_dev'] = float(std_dev)
    
    # Calculate optimal values
    metrics = calculate_newsvendor_metrics(
        float(expected_demand),
        float(std_dev),
        float(price),
        float(cost),
        float(salvage_price)
    )
    
    with col2:
        # Calculate optimal values for table
        metrics = calculate_newsvendor_metrics(
            float(expected_demand),
            float(std_dev),
            float(price),
            float(cost),
            float(salvage_price)
        )
        
        # Custom CSS for the table layout
        st.markdown("""
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table th {
            background-color: #f2f2f2;
        }
        .input-cell {
            padding: 0 !important;
        }
        .order-input {
            margin: 0;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create header row
        col_header1, col_header2, col_header3 = st.columns([3, 2, 2])
        with col_header1:
            st.write("") # Empty space for first column
        with col_header2:
            st.subheader("Optimum")
        with col_header3:
            st.subheader("Custom")
        
        # Create Order row
        col_order1, col_order2, col_order3 = st.columns([3, 2, 2])
        with col_order1:
            st.write("**Order**")
        with col_order2:
            st.write(f"**{metrics['optimal_q']:.0f} units**")
        with col_order3:
            # Place the number input directly in this cell
            custom_order = st.number_input(
                "",
                min_value=-100,  # Changed from 0 to allow negative values
                value=0,
                step=1,
                key="custom_order_input",
                label_visibility="collapsed"
            )
            st.write(f"{custom_order} units")
        
        # Calculate custom metrics if order quantity is provided
        if custom_order > 0:
            # Calculate sales for custom order
            expected_sales = float(price) * min(float(expected_demand), float(custom_order))
            
            # Calculate salvage for custom order
            expected_excess = max(0, float(custom_order) - float(expected_demand))
            salvage_revenue = float(salvage_price) * expected_excess
            
            # Calculate costs for custom order
            total_cost = float(cost) * float(custom_order)
            
            # Calculate expected profit for custom order
            custom_profit = expected_sales + salvage_revenue - total_cost
            
            # Calculate service level for custom order
            if st.session_state.newsvendor_params['distribution'] == 'Normal':
                service_level = 100 * (1 - stats.norm.cdf(custom_order, expected_demand, std_dev))
            else:  # Uniform
                a = expected_demand - std_dev * np.sqrt(3)
                b = expected_demand + std_dev * np.sqrt(3)
                service_level = 100 * ((b - custom_order) / (b - a)) if custom_order < b else 0
            
            # Calculate critical ratio
            critical_ratio = (float(price) - float(cost)) / (float(price) - float(salvage_price))
        else:
            critical_ratio = (float(price) - float(cost)) / (float(price) - float(salvage_price))
            service_level = 0
            expected_sales = 0
            salvage_revenue = 0
            total_cost = 0
            custom_profit = 0
            
        # Create Critical Fractal row
        col_cf1, col_cf2, col_cf3 = st.columns([3, 2, 2])
        with col_cf1:
            st.write("**Critical Fractal**")
        with col_cf2:
            st.write(f"**{metrics['critical_ratio']:.2f}**")
        with col_cf3:
            st.write(f"{critical_ratio:.2f}")
            
        # Create Service Level row
        col_sl1, col_sl2, col_sl3 = st.columns([3, 2, 2])
        with col_sl1:
            st.write("**Service Level**")
        with col_sl2:
            st.write(f"**{metrics['service_level']:.2f}%**")
        with col_sl3:
            st.write(f"{service_level:.2f}%")
            
        # Create Sales row
        col_sales1, col_sales2, col_sales3 = st.columns([3, 2, 2])
        with col_sales1:
            st.write("**Sales $**")
        with col_sales2:
            st.write(f"**${metrics['sales']:,.2f}**")
        with col_sales3:
            st.write(f"${expected_sales:,.2f}")
            
        # Create Salvage row
        col_salvage1, col_salvage2, col_salvage3 = st.columns([3, 2, 2])
        with col_salvage1:
            st.write("**Salvage $**")
        with col_salvage2:
            st.write(f"**${metrics['salvage']:,.2f}**")
        with col_salvage3:
            st.write(f"${salvage_revenue:,.2f}")
            
        # Create Costs row
        col_costs1, col_costs2, col_costs3 = st.columns([3, 2, 2])
        with col_costs1:
            st.write("**Costs**")
        with col_costs2:
            st.write(f"**${metrics['costs']:,.2f}**")
        with col_costs3:
            st.write(f"${total_cost:,.2f}")
            
        # Create Expected Profit row
        col_profit1, col_profit2, col_profit3 = st.columns([3, 2, 2])
        with col_profit1:
            st.write("**Expected Profit**")
        with col_profit2:
            st.write(f"**${metrics['expected_profit']:,.2f}**")
        with col_profit3:
            st.write(f"${custom_profit:,.2f}")
            
        st.markdown("""---""")  # Separator line
        
        st.markdown("""
        The NewsVendor model will result in the optimal order quantity over time. 
        However, for short time horizons, it is possible to get lucky and beat the optimal order. 
        You can experiment with this effect below by changing the # of order periods and the order quantity.
        """)
        
        # Create simulation inputs
        sim_col1, sim_col2, sim_col3 = st.columns([2, 2, 1])
        
        with sim_col1:
            n_periods = st.number_input(
                "# of Order Periods",
                min_value=1,
                value=1000,
                step=100,
                format="%d"
            )
        
        with sim_col2:
            # Use the same custom_order value for simulation
            order_qty = custom_order
        
        with sim_col3:
            submit = st.button("SUBMIT", use_container_width=True)
        
        if submit:
            # Run simulations
            user_profit = simulate_newsvendor(
                int(n_periods),
                int(order_qty),
                float(expected_demand),
                float(std_dev),
                float(price),
                float(cost),
                float(salvage_price)
            )
            
            model_profit = simulate_newsvendor(
                int(n_periods),
                float(metrics['optimal_q']),
                float(expected_demand),
                float(std_dev),
                float(price),
                float(cost),
                float(salvage_price)
            )
            
            # Update win counter
            if user_profit > model_profit:
                st.session_state.newsvendor_params['wins'] += 1
            
            # Create results table
            results_data = {
                '': ['You', 'Model'],
                'Profit': [f"${user_profit:,.0f}", f"${model_profit:,.0f}"],
                'Score': [
                    "1 😊" if user_profit > model_profit else "0",
                    "1 😊" if model_profit >= user_profit else "0"
                ]
            }
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, hide_index=True)
        
        # Plot the distribution
        fig = plot_distribution(
            float(expected_demand),
            float(st.session_state.newsvendor_params['last_range'] if lock_range else std_dev),
            float(metrics['optimal_q'])
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 