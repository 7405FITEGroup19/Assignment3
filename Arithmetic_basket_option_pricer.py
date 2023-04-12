from scipy.stats import norm
import math
import numpy as np
from Geometric_basket_option_pricer import geometric_bascket_option


# arithmatic basket option

def arithmetic_bascket_option (S1,S2,sigma1,sigma2,r,T,K,rho,option_type,M,cv):
    # the input cv should be 'control variate' meaning using control variate, or others meaning not using the general Monte Carlo
    value_geo = geometric_bascket_option (S1,S2,sigma1,sigma2,r,T,K,rho,option_type)
    Spath1 = np.zeros(M)
    Spath2 = np.zeros(M)
    np.random.seed(1)
    x = np.random.normal(loc=0, scale=1, size=2*M)
    # according to rho, generate 2 different standard normal random variables for S1 and S2 respectively
    Z1 = x[:M]
    Z2 = rho*x[:M] + math.sqrt(1-rho**2)*x[M:]
    for i in range(M):
        Spath1[i] = S1*math.exp((r-0.5*sigma1**2)*T + sigma1*math.sqrt(T)*Z1[i])
        Spath2[i] = S2*math.exp((r-0.5*sigma2**2)*T + sigma2*math.sqrt(T)*Z2[i]) 
    
    arithpayoff = np.zeros(M)
    geopayoff = np.zeros(M)
    arithmean = (Spath1 + Spath2)/2
    geomean = (Spath1*Spath2)**0.5
    
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        payoff1 = arithmean-K
        payoff1[payoff1 < 0] = 0
        arithpayoff = np.exp(-r*T)*payoff1
        payoff2 = geomean-K
        payoff2[payoff2 < 0] = 0
        geopayoff = np.exp(-r*T)*payoff2
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        payoff1 = K-arithmean
        payoff1[payoff1 < 0] = 0
        arithpayoff = np.exp(-r*T)*payoff1
        payoff2 = K-geomean
        payoff2[payoff2 < 0] = 0
        geopayoff = np.exp(-r*T)*payoff2
    
    if cv == 'control variate': # monte carlo with control variate
        cov = np.mean(arithpayoff*geopayoff) - np.mean(arithpayoff)*np.mean(geopayoff)
        theta = cov/np.var(geopayoff)
        
        Z = arithpayoff + theta*(value_geo-geopayoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        conf = [Zmean-1.96*Zstd/math.sqrt(M), Zmean+1.96*Zstd/math.sqrt(M)]
    else: # general monte carlo 
        Pmean = np.mean(arithpayoff)
        Pstd = np.std(arithpayoff)
        conf = [Pmean-1.96*Pstd/math.sqrt(M), Pmean+1.96*Pstd/math.sqrt(M)]
    
    return conf
