import streamlit as st

def display_header():
    st.title("Delta Hedging Simulation")
    st.markdown("""
        This app simulates delta hedging over time using the Black-Scholes model. 
        Adjust the parameters and see how delta evolves over time as the stock price fluctuates.
    """)
