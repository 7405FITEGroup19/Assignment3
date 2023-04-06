from scipy.stats import norm
import math
import numpy as np

# geometric basket option

def geometric_bascket_option (S1,S2,sigma1,sigma2,r,T,K,rho,option_type):
    sigmaBg = math.sqrt(sigma1**2 + 2*sigma1*sigma2*rho + sigma2**2)/2
    muBg = r - 0.5*(sigma1**2 + sigma2**2)/2 + 0.5*sigmaBg**2
    S = (S1*S2)**0.5
    d1 = (math.log(S/K) + (muBg+0.5*sigmaBg**2)*T)/(sigmaBg*math.sqrt(T))
    d2 = d1 - sigmaBg*math.sqrt(T)
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        value_geo = np.exp(-r*T)*(S*np.exp(muBg*T)*norm.cdf(d1) - K*norm.cdf(d2))
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        value_geo = np.exp(-r*T)*(K*norm.cdf(-d2) - S*np.exp(muBg*T)*norm.cdf(-d1))
    
    return value_geo
