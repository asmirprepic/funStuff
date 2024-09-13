from math import log,sqrt,exp
from scipy.stats import norm

def black_scholes(S,K,T,r,sigma,type):

    d1 = (log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    if type == 'call':
        price = S*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
    elif type == 'put':
        price= K*exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    
    return price