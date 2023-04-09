import math
from European_Option_Black_Scholes_Formulas import black_scholes as bs
import scipy.stats as si
import numpy as np

N_prime = si.norm.pdf
# pi = 3.141592653589793
def vega(S, K, T, r, sigma, q, boundary = True):

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r - q) * (T)) / (sigma * np.sqrt(T)) + 1 / 2 * sigma * np.sqrt(T)
    # print(d1,si.norm.cdf(d1, 0, 1.0))
    # print(d1, si.norm.cdf(d1))
    # print(d1, N_prime( d1))
    if boundary:
        vega = max(S * np.sqrt(T) * np.exp(-q * (T)) * N_prime(d1), np.finfo(np.float64).eps)
           # (1/np.sqrt(2*pi)* np.exp(-1/2 * (d1*d1) ))
           # N_prime(d1)
    return vega

def Newton_Raphson4Ivolatility(C_true, S, K, T, r, q, call=True, boundary = True):
    learning_rate = 1
    sigmahat = math.sqrt(2 * abs((math.log(S / K) + (r-q) * T) / T))
    # print('sigmahat',sigmahat)
    tol = 1e-8
    nmax = 100
    sigmadiff = 1
    n = 1
    sigma = sigmahat
    while sigmadiff >= tol and n < nmax:
        if boundary:
            C = max(bs(S, K, T, sigma, r, q=q, t=0, call=call),0)
            if call:
                # print(C , S * np.exp(-q * (T)) - K * np.exp(-r * (T)))
                C = max(S * np.exp(-q * (T)) - K * np.exp(-r * (T)),C)
                C = min(S * np.exp(-q * (T)), C)
            else:
                C = max(K * np.exp(-r * (T)) - S * np.exp(-q * (T)), C)
                C = min(K * np.exp(-r * (T)), C)
        else:
            C = bs(S, K, T, sigma, r, q=q, t=0, call=call)
        Cvega = vega(S, K, T, r, sigma, q, boundary = boundary)
        increment = (C - C_true) / Cvega
        # print(C, C_true, Cvega)
        sigma = sigma - increment * learning_rate
        n = n + 1
        sigmadiff = abs(increment)
        # print(sigmadiff)
    if n == nmax:
        sigma = None
    return sigma

print(Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.01, 0, call=False), Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.01, 0, call=True))
print(Newton_Raphson4Ivolatility(10, 100, 80, 0.5, 0.01, 0, call=False), Newton_Raphson4Ivolatility(10, 100, 120, 0.5, 0.01, 0, call=True))
print(Newton_Raphson4Ivolatility(10, 100, 100, 1, 0.01, 0, call=False), Newton_Raphson4Ivolatility(10, 100, 100, 1, 0.01, 0, call=True))
print(Newton_Raphson4Ivolatility(20, 100, 100, 0.5, 0.01, 0, call=False), Newton_Raphson4Ivolatility(20, 100, 100, 0.5, 0.01, 0, call=True))
print(Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.02, 0, call=False), Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.02, 0, call=True))
print(Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.01, 0.1, call=False), Newton_Raphson4Ivolatility(10, 100, 100, 0.5, 0.01, 0.1, call=True))