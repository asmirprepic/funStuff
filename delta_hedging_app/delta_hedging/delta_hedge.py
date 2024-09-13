from math import log,sqrt,exp
from scipy.stats import norm
from delta_hedging.stock_simulation import simulate_stock_price
from delta_hedging.black_scholes import black_scholes
import numpy as np
def option_delta(S,K,T,r,sigma,type):

    d1 = (log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    
    if type == "call":
        return norm.cdf(d1)
    if type == "put":
        return norm.cdf(d1)-1
    
def simulate_delta_heding(S0,K,T,r,sigma,steps):
    """
    Simulates the PnL from delta hedging over time using stock prices from GBM.
    
    Parameters:
    ------------
    S0 : float : Initial stock price
    K : float : Strike price
    T : float : Time to maturity (years)
    r : float : Risk-free rate
    sigma : float : Volatility
    steps : int : Number of simulation steps
    
    Returns:
    ------------
    time_points : np.array : Time points between 0 and T
    pnl : np.array : PnL from delta hedging over time
    stock_prices : np.array : Simulated stock prices over time
    """

    # Simulate stock prices
    time_points,stock_prices = simulate_stock_price(S0,T,r,sigma,steps)

    # Initalize variables for PnL
    initial_option_price = black_scholes(S0,K,T,r,sigma,'call')

    pnl = np.zeros(steps)
    hedge_positions = np.zeros(steps)
    cash_balance = np.zeros(steps)

    # Initial conditions
    hedge_positions[0] = option_delta(S0,K,T,r,sigma,type = 'call')
    cash_balance[0] = initial_option_price - hedge_positions[0]*S0

    for t in range(1,steps):
        option_value =black_scholes(stock_prices[t],K,T-time_points[t],r,sigma,"call")
        current_delta = option_delta(stock_prices[t],K,T-time_points[t],r,sigma,'call')

        # Adjust hedge
        hedge_positions[t]= current_delta
        cash_balance[t] = cash_balance[t-1] - (hedge_positions[t]-hedge_positions[t-1])*stock_prices[t]

        pnl[t] = option_value + cash_balance[t] - hedge_positions[t]*stock_prices[t]
    return time_points,pnl,stock_prices,hedge_positions