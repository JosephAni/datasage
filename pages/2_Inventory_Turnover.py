import pandas as pd
import numpy as np
import streamlit as st
import sys
import os
import time
import math
import matplotlib.pyplot as plt

# Add parent directory to path to import DataCleaner class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config
st.set_page_config(
    page_title="Inventory Turnover Simulation",
    page_icon="📦",
    layout="wide"
)

st.markdown("# Inventory Turnover Simulation")
st.sidebar.header("Inventory Turnover Simulation")
st.write(
    """
    This simulation demonstrates the concept of inventory turnover by visualizing 
    the rate at which inventory is sold for two different companies.
    """
)

# Add How to Use instructions
with st.expander("How to Use", expanded=True):
    st.markdown("""
    **What is Inventory Turnover?**
    Inventory turnover measures how many times a company sells and replaces its inventory in a given period. 
    A higher turnover rate indicates more efficient inventory management.
    
    **How to use this simulation:**
    1. **Adjust turnover rates** using the +/- buttons or input fields:
       - Higher turnover = faster inventory depletion
       - Default values: Company 1 (8), Company 2 (16)
    
    2. **View days inventory** values which show how many days inventory stays in stock:
       - Days Inventory = 365 / Turnover Rate
       - Lower days inventory = more efficient inventory management
    
    3. **Click START SELLING** to begin the simulation:
       - Watch as inventory boxes disappear in real-time
       - The company with higher turnover will deplete inventory faster
       - Click again to pause/reset the simulation
    
    4. **Compare companies** to see the impact of different turnover rates
       - Try making Company 1 twice as fast as Company 2
       - Observe how changing turnover affects inventory depletion speed
    """)

# Initialize session state for simulation if not exists
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
    st.session_state.inventory_company1 = 15  # Number of boxes in pyramid
    st.session_state.inventory_company2 = 15  # Number of boxes in pyramid
    st.session_state.last_update_time = None

# Create control inputs with increment/decrement buttons
col1, col2 = st.columns(2)

with col1:
    st.write("#### Company 1")
    
    # Turnover rate for Company 1
    st.write("Turnover Company 1")
    turnover_col1_minus, turnover_col1_value, turnover_col1_plus = st.columns([1, 3, 1])
    with turnover_col1_minus:
        if st.button("-", key="turnover1_minus"):
            if 'turnover_company1' in st.session_state and st.session_state.turnover_company1 > 1:
                st.session_state.turnover_company1 -= 1
    
    with turnover_col1_value:
        if 'turnover_company1' not in st.session_state:
            st.session_state.turnover_company1 = 8
        turnover_company1 = st.number_input("", 
                                            min_value=1, 
                                            max_value=100, 
                                            value=st.session_state.turnover_company1,
                                            key="turnover_company1_input",
                                            label_visibility="collapsed")
        st.session_state.turnover_company1 = turnover_company1
    
    with turnover_col1_plus:
        if st.button("+", key="turnover1_plus"):
            if 'turnover_company1' in st.session_state:
                st.session_state.turnover_company1 += 1
    
    # Days Inventory for Company 1
    st.write("Days Inventory 1")
    days_col1_minus, days_col1_value, days_col1_plus = st.columns([1, 3, 1])
    with days_col1_minus:
        if st.button("-", key="days1_minus"):
            if 'days_inventory1' in st.session_state and st.session_state.days_inventory1 > 1:
                st.session_state.days_inventory1 -= 1
    
    with days_col1_value:
        if 'days_inventory1' not in st.session_state:
            st.session_state.days_inventory1 = 46
        days_inventory1 = st.number_input("", 
                                          min_value=1.0, 
                                          max_value=365.0, 
                                          value=float(st.session_state.days_inventory1),
                                          key="days_inventory1_input",
                                          label_visibility="collapsed")
        st.session_state.days_inventory1 = days_inventory1
    
    with days_col1_plus:
        if st.button("+", key="days1_plus"):
            if 'days_inventory1' in st.session_state:
                st.session_state.days_inventory1 += 1

with col2:
    st.write("#### Company 2")
    
    # Turnover rate for Company 2
    st.write("Turnover Company 2")
    turnover_col2_minus, turnover_col2_value, turnover_col2_plus = st.columns([1, 3, 1])
    with turnover_col2_minus:
        if st.button("-", key="turnover2_minus"):
            if 'turnover_company2' in st.session_state and st.session_state.turnover_company2 > 1:
                st.session_state.turnover_company2 -= 1
    
    with turnover_col2_value:
        if 'turnover_company2' not in st.session_state:
            st.session_state.turnover_company2 = 16
        turnover_company2 = st.number_input("", 
                                            min_value=1, 
                                            max_value=100, 
                                            value=st.session_state.turnover_company2,
                                            key="turnover_company2_input",
                                            label_visibility="collapsed")
        st.session_state.turnover_company2 = turnover_company2
    
    with turnover_col2_plus:
        if st.button("+", key="turnover2_plus"):
            if 'turnover_company2' in st.session_state:
                st.session_state.turnover_company2 += 1
    
    # Days Inventory for Company 2
    st.write("Days Inventory 2")
    days_col2_minus, days_col2_value, days_col2_plus = st.columns([1, 3, 1])
    with days_col2_minus:
        if st.button("-", key="days2_minus"):
            if 'days_inventory2' in st.session_state and st.session_state.days_inventory2 > 1:
                st.session_state.days_inventory2 -= 1
    
    with days_col2_value:
        if 'days_inventory2' not in st.session_state:
            st.session_state.days_inventory2 = 22.8
        days_inventory2 = st.number_input("", 
                                          min_value=1.0, 
                                          max_value=365.0, 
                                          value=float(st.session_state.days_inventory2),
                                          key="days_inventory2_input",
                                          label_visibility="collapsed")
        st.session_state.days_inventory2 = days_inventory2
    
    with days_col2_plus:
        if st.button("+", key="days2_plus"):
            if 'days_inventory2' in st.session_state:
                st.session_state.days_inventory2 += 1

# START SELLING button
start_button_style = """
<style>
div.stButton > button {
    background-color: #4B70E2;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    width: 100%;
}
</style>
"""
st.markdown(start_button_style, unsafe_allow_html=True)

if st.button("START SELLING", key="start_selling"):
    st.session_state.simulation_running = not st.session_state.simulation_running
    # Reset inventory if simulation was off and turning on
    if st.session_state.simulation_running:
        st.session_state.inventory_company1 = 15
        st.session_state.inventory_company2 = 15
        st.session_state.last_update_time = time.time()

# Add metrics display
metrics_col1, metrics_col2 = st.columns(2)
with metrics_col1:
    st.metric(
        "Company 1 Turnover", 
        f"{st.session_state.turnover_company1}x",
        f"{(st.session_state.turnover_company1 / st.session_state.turnover_company2 * 100 - 100):.1f}% vs Company 2"
    )
with metrics_col2:
    st.metric(
        "Company 2 Turnover", 
        f"{st.session_state.turnover_company2}x",
        f"{(st.session_state.turnover_company2 / st.session_state.turnover_company1 * 100 - 100):.1f}% vs Company 1"
    )

# Display the inventory visualization
col1, col2 = st.columns(2)

# Calculate inventory levels based on turnover rates if simulation is running
if st.session_state.simulation_running and st.session_state.last_update_time is not None:
    current_time = time.time()
    elapsed_time = current_time - st.session_state.last_update_time
    
    # Scale elapsed time for faster visualization (1 second = 1 day)
    scaled_time = elapsed_time * 10
    
    # Calculate boxes to remove based on turnover rates
    company1_rate = st.session_state.turnover_company1 / 365  # Daily sales rate
    company2_rate = st.session_state.turnover_company2 / 365  # Daily sales rate
    
    # Calculate inventory to remove
    company1_remove = scaled_time * company1_rate 
    company2_remove = scaled_time * company2_rate
    
    # Update inventory levels
    st.session_state.inventory_company1 = max(0, st.session_state.inventory_company1 - company1_remove)
    st.session_state.inventory_company2 = max(0, st.session_state.inventory_company2 - company2_remove)
    
    # Update last update time
    st.session_state.last_update_time = current_time

# Function to create pyramid visualization
def create_pyramid(inventory_left, max_inventory=15, color="#3178c6"):
    # Calculate how many boxes to show based on inventory left
    boxes_to_show = math.ceil(inventory_left)
    boxes_to_show = min(boxes_to_show, max_inventory)  # Cap at max inventory
    
    # Create HTML for pyramid
    pyramid_html = "<div style='display: flex; flex-direction: column; align-items: center;'>"
    
    # Each row of the pyramid
    boxes_per_row = [1, 2, 3, 4, 5]  # 5 rows with increasing boxes
    boxes_used = 0
    
    for row_boxes in boxes_per_row:
        row_html = "<div style='display: flex; flex-direction: row;'>"
        for i in range(row_boxes):
            if boxes_used < boxes_to_show:
                # Box is visible
                row_html += f"<div style='width: 30px; height: 30px; margin: 2px; background-color: {color};'></div>"
            else:
                # Box is invisible (removed)
                row_html += "<div style='width: 30px; height: 30px; margin: 2px;'></div>"
            boxes_used += 1
        row_html += "</div>"
        pyramid_html += row_html
    
    pyramid_html += "</div>"
    return pyramid_html

# Display company names
with col1:
    st.markdown("<h4 style='text-align: center;'>Company 1</h4>", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='text-align: center;'>Company 2</h4>", unsafe_allow_html=True)

# Display pyramids
with col1:
    company1_pyramid = create_pyramid(st.session_state.inventory_company1)
    st.markdown(company1_pyramid, unsafe_allow_html=True)
    
    # Add inventory left indicator
    st.progress(min(1.0, max(0.0, st.session_state.inventory_company1 / 15)))
    st.write(f"Inventory left: {st.session_state.inventory_company1:.1f} / 15 units")

with col2:
    company2_pyramid = create_pyramid(st.session_state.inventory_company2)
    st.markdown(company2_pyramid, unsafe_allow_html=True)
    
    # Add inventory left indicator
    st.progress(min(1.0, max(0.0, st.session_state.inventory_company2 / 15)))
    st.write(f"Inventory left: {st.session_state.inventory_company2:.1f} / 15 units")

# Display some additional information about inventory turnover
with st.expander("About Inventory Turnover", expanded=False):
    st.markdown("""
    ### Understanding Inventory Turnover

    **Definition:**
    Inventory turnover is a ratio that measures how many times a company has sold and replaced its inventory during a given period. A high turnover rate indicates that a company is efficiently managing its inventory.

    **Formula:**
    ```
    Inventory Turnover Ratio = Cost of Goods Sold / Average Inventory
    ```

    **Days Inventory Outstanding (DIO):**
    ```
    DIO = 365 / Inventory Turnover Ratio
    ```
    This metric represents the average number of days it takes to sell the inventory.

    **Why it matters:**
    - **Cash Flow**: Higher turnover frees up cash by reducing the amount tied up in inventory
    - **Storage Costs**: Lower inventory levels reduce warehouse and storage expenses
    - **Obsolescence**: Faster turnover reduces risk of inventory becoming obsolete or deteriorating
    - **Profitability**: More efficient inventory management generally leads to higher profitability

    **Industry Benchmarks:**
    - Retail: 4-6x per year
    - Grocery: 12-25x per year
    - Auto: 8-10x per year
    - Technology: 5-7x per year
    - Luxury goods: 1-2x per year
    """)

# Add comparison chart
with st.expander("Turnover Efficiency Comparison", expanded=False):
    # Calculate days inventory
    days_inventory1 = 365 / st.session_state.turnover_company1
    days_inventory2 = 365 / st.session_state.turnover_company2
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Comparison data
    companies = ['Company 1', 'Company 2']
    turnover_rates = [st.session_state.turnover_company1, st.session_state.turnover_company2]
    days_inventories = [days_inventory1, days_inventory2]
    
    # Create bar charts
    x = np.arange(len(companies))
    width = 0.35
    
    ax.bar(x - width/2, turnover_rates, width, label='Turnover Rate (x/year)')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, days_inventories, width, color='orange', label='Days Inventory')
    
    # Add labels and legend
    ax.set_xlabel('Company')
    ax.set_ylabel('Turnover Rate (times/year)')
    ax2.set_ylabel('Days Inventory')
    ax.set_xticks(x)
    ax.set_xticklabels(companies)
    
    # Add combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Inventory Turnover Efficiency Comparison')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add efficiency analysis
    st.markdown("### Efficiency Analysis")
    
    if st.session_state.turnover_company1 > st.session_state.turnover_company2:
        st.success(f"Company 1 is more efficient with {st.session_state.turnover_company1}x turnover vs Company 2's {st.session_state.turnover_company2}x turnover.")
        st.write(f"Company 1 sells its inventory every {days_inventory1:.1f} days, while Company 2 takes {days_inventory2:.1f} days.")
    elif st.session_state.turnover_company2 > st.session_state.turnover_company1:
        st.success(f"Company 2 is more efficient with {st.session_state.turnover_company2}x turnover vs Company 1's {st.session_state.turnover_company1}x turnover.")
        st.write(f"Company 2 sells its inventory every {days_inventory2:.1f} days, while Company 1 takes {days_inventory1:.1f} days.")
    else:
        st.info(f"Both companies have the same turnover rate of {st.session_state.turnover_company1}x and sell their inventory every {days_inventory1:.1f} days.")

# Add auto-refresh for animation (every 100ms)
if st.session_state.simulation_running:
    st.empty()
    time.sleep(0.1)
    st.rerun() 