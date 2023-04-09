import numpy as np
import scipy.stats as si


def black_scholes(S, K, T, sigma, r, t = 0,  q=0, call=True):

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

# print(black_scholes(100,100,0.5, 0.2, 0.01, call=False), black_scholes(100,100,0.5, 0.2, 0.01, call=True))
# print(black_scholes(100,120,0.5, 0.2, 0.01, call=False), black_scholes(100,120,0.5, 0.2, 0.01, call=True))
# print(black_scholes(100,100,1, 0.2, 0.01, call=False), black_scholes(100,100,1, 0.2, 0.01, call=True))
# print(black_scholes(100,100,0.5, 0.3, 0.01, call=False), black_scholes(100,100,0.5, 0.3, 0.01, call=True))
# print(black_scholes(100,100,0.5, 0.2, 0.02, call=False), black_scholes(100,100,0.5, 0.2, 0.02, call=True))
# print(black_scholes(100,100,0.5, 0.2, 0.01, q=0.1, call=False), black_scholes(100,100,0.5, 0.2, 0.01, q=0.1, call=True))
