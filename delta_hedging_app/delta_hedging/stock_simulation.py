import numpy as np

def simulate_stock_price(S0,T,r,sigma,steps):
    """
    Simulate stock prices using GBM

    Parameters:
    ------------
    S0: initial stock price
    T: total time
    sigma: Volatility
    steps: number of steps to simulate

    Returns:
    Time points and stock price 
    
    """

    dt = T/steps
    time_points = np.linspace(0,T,steps)
    stock_prices = np.zeros(steps)
    stock_prices[0] = S0

    Z = np.random.normal(size = steps)

    # Drift and diffusion
    drift =(r-0.5*sigma**2)*dt
    diffusion = sigma*np.sqrt(dt)*Z

    # Compute prices
    stock_prices=S0*np.exp(np.cumsum(drift + diffusion))
    
    return time_points,stock_prices