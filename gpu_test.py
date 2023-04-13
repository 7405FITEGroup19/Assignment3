import cupy as cp
import math
from scipy.stats import norm

def geometric_basket_option(S1, S2, sigma1, sigma2, r, T, K, rho, option_type):
    sigmaBg = cp.sqrt(sigma1**2 + 2*sigma1*sigma2*rho + sigma2**2) / 2
    muBg = r - 0.5 * (sigma1**2 + sigma2**2) / 2 + 0.5 * sigmaBg**2
    S = (S1 * S2)**0.5
    d1 = (cp.log(S / K) + (muBg + 0.5 * sigmaBg**2) * T) / (sigmaBg * cp.sqrt(T))
    d2 = d1 - sigmaBg * cp.sqrt(T)
    
    norm_cdf_d1 = cp.array(norm.cdf(d1.get())) # Transfer d1 to CPU for scipy's norm.cdf
    norm_cdf_d2 = cp.array(norm.cdf(d2.get())) # Transfer d2 to CPU for scipy's norm.cdf
    
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        value_geo = cp.exp(-r * T) * (S * cp.exp(muBg * T) * norm_cdf_d1 - K * norm_cdf_d2)
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        value_geo = cp.exp(-r * T) * (K * norm.cdf(-d2.get()) - S * cp.exp(muBg * T) * norm_cdf_d1)
    
    return value_geo.get() # Transfer result back to CPU
