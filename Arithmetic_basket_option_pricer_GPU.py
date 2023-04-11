from scipy.stats import norm
import math
import numpy as np
import cupy as cp
from Geometric_basket_option_pricer import geometric_bascket_option

def arithmetic_bascket_option(S1, S2, sigma1, sigma2, r, T, K, rho, option_type, M, cv):
    value_geo = geometric_bascket_option(S1, S2, sigma1, sigma2, r, T, K, rho, option_type)
    Spath1 = cp.zeros(M)
    Spath2 = cp.zeros(M)
    cp.random.seed(1)
    x = cp.random.normal(loc=0, scale=1, size=2*M)
    Z1 = x[:M]
    Z2 = rho * x[:M] + cp.sqrt(1 - rho**2) * x[M:]
    
    for i in range(M):
        Spath1[i] = S1 * cp.exp((r - 0.5 * sigma1**2) * T + sigma1 * cp.sqrt(T) * Z1[i])
        Spath2[i] = S2 * cp.exp((r - 0.5 * sigma2**2) * T + sigma2 * cp.sqrt(T) * Z2[i])

    arithpayoff = cp.zeros(M)
    geopayoff = cp.zeros(M)
    arithmean = (Spath1 + Spath2) / 2
    geomean = cp.sqrt(Spath1 * Spath2)

    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        payoff1 = arithmean - K
        payoff1[payoff1 < 0] = 0
        arithpayoff = cp.exp(-r * T) * payoff1
        payoff2 = geomean - K
        payoff2[payoff2 < 0] = 0
        geopayoff = cp.exp(-r * T) * payoff2

    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        payoff1 = K - arithmean
        payoff1[payoff1 < 0] = 0
        arithpayoff = cp.exp(-r * T) * payoff1
        payoff2 = K - geomean
        payoff2[payoff2 < 0] = 0
        geopayoff = cp.exp(-r * T) * payoff2

    if cv == 'control variate':
        cov = cp.mean(arithpayoff * geopayoff) - cp.mean(arithpayoff) * cp.mean(geopayoff)
        theta = cov / cp.var(geopayoff)
        Z = arithpayoff + theta * (value_geo - geopayoff)
        Zmean = cp.mean(Z)
        Zstd = cp.std(Z)
        conf = [Zmean - 1.96 * Zstd / cp.sqrt(M), Zmean + 1.96 * Zstd / cp.sqrt(M)]

    else:
        Pmean = cp.mean(arithpayoff)
        Pstd = cp.std(arithpayoff)
        conf = Pmean - 1.96 * Pstd / cp.sqrt(M), Pmean + 1.96 * Pstd / cp.sqrt(M)]

    # Convert the result back to NumPy array before returning
    return cp.asnumpy(conf)
