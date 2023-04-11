import cupy as cp
from scipy.stats import norm
import math
from Geometric_Asian_option_pricer import geometric_asia_option

# arithmatic Asian option
def arithmetic_asia_option(S0, sigma, r, T, K, n, option_type, M, cv):
    # the input cv should be 'control variate' meaning using control variate, or others meaning not using the general Monte Carlo
    value_geo = geometric_asia_option(S0, sigma, r, T, K, n, option_type)
    
    Dt = T / n
    cp.random.seed(1)
    Z = cp.random.normal(loc=0, scale=1, size=(M, n))

    Spath = cp.zeros((M, n))
    Spath[:, 0] = S0 * cp.exp((r - 0.5 * sigma ** 2) * Dt + sigma * cp.sqrt(Dt) * Z[:, 0])
    for j in range(1, n):
        Spath[:, j] = Spath[:, j - 1] * cp.exp((r - 0.5 * sigma ** 2) * Dt + sigma * cp.sqrt(Dt) * Z[:, j])

    arithmean = cp.mean(Spath, axis=1)
    geomean = cp.exp((1 / n) * cp.sum(cp.log(Spath), axis=1))
    
    arithpayoff, geopayoff = calculate_payoffs(arithmean, geomean, K, r, T, option_type)

    if cv == 'control variate':  # monte carlo with control variate
        conf = control_variate(arithpayoff, geopayoff, value_geo, M)
    else:  # general monte carlo
        conf = general_monte_carlo(arithpayoff, M)

    return conf

def calculate_payoffs(arithmean, geomean, K, r, T, option_type):
    if option_type.lower() == 'call':
        arithpayoff = cp.maximum(arithmean - K, 0) * cp.exp(-r * T)
        geopayoff = cp.maximum(geomean - K, 0) * cp.exp(-r * T)
    elif option_type.lower() == 'put':
        arithpayoff = cp.maximum(K - arithmean, 0) * cp.exp(-r * T)
        geopayoff = cp.maximum(K - geomean, 0) * cp.exp(-r * T)
    return arithpayoff, geopayoff

def control_variate(arithpayoff, geopayoff, value_geo, M):
    cov = cp.mean(arithpayoff * geopayoff) - cp.mean(arithpayoff) * cp.mean(geopayoff)
    theta = cov / cp.var(geopayoff)

    Z = arithpayoff + theta * (value_geo - geopayoff)
    Zmean = cp.mean(Z)
    Zstd = cp.std(Z)
    conf = [Zmean - 1.96 * Zstd / cp.sqrt(M), Zmean + 1.96 * Zstd / cp.sqrt(M)]
    return conf


def general_monte_carlo(arithpayoff, M):
    Pmean = cp.mean(arithpayoff)
    Pstd = cp.std(arithpayoff)
    conf = [Pmean - 1.96 * Pstd / cp.sqrt(M), Pmean + 1.96 * Pstd / cp.sqrt(M)]
    return conf
