from scipy.stats import norm
import math
import numpy as np

# geometric Asian option

def geometric_asia_option (S0,sigma,r,T,K,n,option_type):
    sigmaA = sigma*math.sqrt((n+1)*(2*n+1)/(6*n*n))
    muA = (r - 0.5*sigma**2)*(n+1)/(2*n) + 0.5*sigmaA**2
    d1 = (math.log(S0/K) + (muA+0.5*sigmaA**2)*T)/(sigmaA*math.sqrt(T))
    d2 = d1 - sigmaA*math.sqrt(T)
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        value_geo = np.exp(-r*T)*(S0*np.exp(muA*T)*norm.cdf(d1) - K*norm.cdf(d2))
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        value_geo = np.exp(-r*T)*(K*norm.cdf(-d2) - S0*np.exp(muA*T)*norm.cdf(-d1))
    
    return value_geo


