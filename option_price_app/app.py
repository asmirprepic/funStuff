import streamlit as st
import numpy as np
from pricing.black_scholes import black_scholes  # Import the Black-Scholes logic
from components.header import display_header     # Import the header component

# Display Header
display_header()

# Sidebar Input
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input('Stock Price (S)', min_value=0.01, value=100.0, step=1.0)
K_min = st.sidebar.number_input('Minimum Strike Price (K_min)', min_value=0.01, value=80.0, step=1.0)
K_max = st.sidebar.number_input('Maximum Strike Price (K_max)', min_value=0.01, value=120.0, step=1.0)
K_step = st.sidebar.number_input('Strike Price Step (K_step)', min_value=0.01, value=5.0, step=0.1)
T = st.sidebar.number_input('Time to Maturity (T in years)', min_value=0.01, value=1.0, step=0.1)
r = st.sidebar.number_input('Risk-free Rate (r)', min_value=0.00, value=0.05, step=0.01)
sigma = st.sidebar.slider('Volatility (σ)', min_value=0.01, value=0.2, step=0.01)

# Generate strike prices for the price matrix
strike_prices = np.arange(K_min, K_max +K_step, K_step)
call_prices = [black_scholes(S, K, T, r, sigma, 'call') for K in strike_prices]
put_prices = [black_scholes(S, K, T, r, sigma, 'put') for K in strike_prices]
# Plot Call and Put Prices
st.subheader("Call and Put Option Prices")
st.line_chart({
    'Strike Price': strike_prices,
    'Call Price': call_prices,
    'Put Price': put_prices
})
# Display Price Matrix
st.subheader("Option Price Matrix")
price_matrix = np.column_stack((strike_prices, call_prices, put_prices))
st.dataframe({
    "Strike Price (K)": strike_prices,
    "Call Price": call_prices,
    "Put Price": put_prices
})


