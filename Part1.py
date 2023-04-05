import numpy as np
import scipy.stats as si


def black_scholes(S, K, T, t, sigma, r, q=0, call=True):

    d1 = (np.log(S / K) + (r-q)*(T-t))/ (sigma * np.sqrt(T - t)) + 1/2*sigma* np.sqrt(T - t)
    d2 = (np.log(S / K) + (r-q)*(T-t))/ (sigma * np.sqrt(T - t)) - 1/2*sigma* np.sqrt(T - t)
    # print('111',(sigma))
    # print('d1, d2', d1,d2)
    if call:
        call = S * np.exp(-q * (T-t)) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * (T-t)) * si.norm.cdf(d2, 0.0, 1.0)
        return call
    else:
        put = K * np.exp(-r * (T-t)) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * (T-t)) * si.norm.cdf(-d1, 0.0, 1.0)
        return put