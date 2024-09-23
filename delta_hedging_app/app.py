import pandas as pd
import streamlit as st
import numpy as np
#from delta_hedging.delta_hedge import option_delta  # Import the Black-Scholes logic
#from delta_hedging.stock_simulation import simulate_stock_price
from delta_hedging_app.delta_hedging.delta_hedge import simulate_delta_hedge
from delta_hedging_app.components.header import display_header     # Import the header component
import seaborn as sns
import matplotlib.pyplot as plt
# Display Header
display_header()

# Sidebar Input
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input('Initial Stock Price (S0)', min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input('Strike Price', min_value=0.01, value=80.0, step=1.0)
T = st.sidebar.number_input('Time to Maturity (T in years)', min_value=0.01, value=1.0, step=0.1)
r = st.sidebar.number_input('Risk-free Rate (r)', min_value=0.00, value=0.05, step=0.01)
sigma = st.sidebar.slider('Volatility (σ)', min_value=0.01, value=0.2, step=0.01)
steps = st.sidebar.number_input("Number of time steps",min_value=1,value = 100,step =1)


# Generate strike prices for the price matrix
time_points,pnl,stock_prices,hedge_positions = simulate_delta_heding(S0,K,T,r,sigma,steps)

# Create a DataFrame for Streamlit plotting
data = pd.DataFrame({
    'Time to Maturity': time_points,
    'PnL': pnl,
    'Stock Price': stock_prices,
    'Delta Hedge': hedge_positions
})

# Plot the PnL Over Time using Streamlit's line_chart
st.subheader("PnL Evolution Over Time")
st.line_chart(data[['Time to Maturity', 'PnL']].set_index('Time to Maturity'))

# Plot the Stock Prices Over Time using Streamlit's line_chart
st.subheader("Stock Price Evolution Over Time")
st.line_chart(data[['Time to Maturity', 'Stock Price']].set_index('Time to Maturity'))

# Plot the Delta Hedge Positions Over Time using Streamlit's line_chart
st.subheader("Delta Hedge Position Over Time")
st.line_chart(data[['Time to Maturity', 'Delta Hedge']].set_index('Time to Maturity'))


# Display the heatmap in Streamlit

st.write("""
    ### What is Delta Hedging?
    Delta hedging is a strategy used to mitigate the directional risk associated with price movements in the underlying asset. 
    For example, a call option has a positive delta, which means its price moves in the same direction as the underlying stock. 
    By dynamically adjusting your position in the underlying stock, you can neutralize the delta and reduce risk exposure.
""")


